import asyncio
import json
from pathlib import Path

from ..core import app_keys
from ..features.universe import RULES, classify, summarise
from aiohttp import web

# The static assets live in `web/` at the project root, two levels up from this file.
# This was `parent.parent`, which was correct while this module sat at `app/web.py` and
# silently became wrong when it moved to `app/web/web.py` -- the extra directory made it
# resolve to `app/web/`, where nothing is published. Nothing failed loudly: `/` answered
# 404 and every `/assets/*` request 404'd, so the console was a blank page with no clue
# as to why. Resolved from the package root instead of by counting parents, so a future
# move cannot reintroduce it.
WEB_ROOT = Path(__file__).resolve().parents[2] / 'web'

async def index(request):
    return web.FileResponse(WEB_ROOT/'index.html')

async def state(request): return web.json_response(request.app[app_keys.STATE]())

async def risk_reset(request):
    reset_fn=request.app.get(app_keys.RESET_RISK)
    if reset_fn is None: return web.json_response({'error':'risk_reset_unavailable'},status=503)
    return web.json_response(reset_fn())

async def risk_profiles_state(request):
    """Available presets plus the profile currently in force."""
    set_fn=request.app.get(app_keys.SET_RISK_PROFILE)
    if set_fn is None: return web.json_response({'error':'risk_profile_unavailable'},status=503)
    return web.json_response(set_fn(None, None, switch=False))

async def edge_symbols_state(request):
    get_fn=request.app.get(app_keys.SYMBOL_EDGE_STATE)
    if get_fn is None: return web.json_response({'error':'symbol_edge_unavailable'},status=503)
    return web.json_response(get_fn())

async def exit_policies_state(request):
    get_fn=request.app.get(app_keys.SET_EXIT_POLICY)
    if get_fn is None: return web.json_response({'error':'exit_policy_unavailable'},status=503)
    return web.json_response(get_fn(None, None, switch=False))

async def exit_policy_select(request):
    set_fn=request.app.get(app_keys.SET_EXIT_POLICY)
    if set_fn is None: return web.json_response({'error':'exit_policy_unavailable'},status=503)
    try:
        payload=await request.json() if request.can_read_body else {}
    except (ValueError, TypeError):
        payload={}
    try:
        return web.json_response(set_fn(payload.get('policy') or payload.get('exit_policy'),
                                        payload.get('overrides') or payload.get('exit_overrides'),
                                        switch=True))
    except ValueError as exc:
        return web.json_response({'error':str(exc)},status=400)

async def risk_profile_select(request):
    """Switch the live profile, optionally with per-session overrides."""
    set_fn=request.app.get(app_keys.SET_RISK_PROFILE)
    if set_fn is None: return web.json_response({'error':'risk_profile_unavailable'},status=503)
    try:
        payload=await request.json() if request.can_read_body else {}
    except (ValueError, TypeError):
        payload={}
    try:
        return web.json_response(set_fn(payload.get('profile') or payload.get('risk_profile'),
                                        payload.get('overrides') or payload.get('risk_overrides'),
                                        switch=True))
    except ValueError as exc:
        return web.json_response({'error':str(exc)},status=400)

async def simulation_control(request):
    session=request.app[app_keys.SIMULATION]; action=request.match_info['action']
    try:
        payload=await request.json() if request.can_read_body else {}
        if action=='start':
            # Default to the trained universe, not the gainers list. The model's serving
            # gate refuses any symbol whose features fall outside its training range, and
            # that gate is right to: the model was fit on twelve majors, and a movers list
            # is by construction full of markets it has never seen. Defaulting to 'gainers'
            # therefore started sessions that could not open a position by design -- every
            # selected micro-cap was blocked as out-of-distribution (ATR 0.027-0.034
            # against a training ceiling of 0.015) and the reason shown was a feature list
            # rather than 'this model was never trained on this market'. The other sources
            # remain available; they are simply no longer the silent default.
            session.start(payload.get('initial_cash',10000),payload.get('leverage',3),
                          payload.get('source','trained'),payload.get('symbol_count',3))
            # The profile a session trades under is part of starting it, not a separate
            # step, so a session cannot begin between the two and inherit stale limits.
            try:
                request.app[app_keys.ON_SESSION_START](session.session_id,
                                                       payload.get('risk_profile'),
                                                       payload.get('risk_overrides'))
            except Exception:
                # A rejected profile (an out-of-range override, an unknown name) used to
                # return 400 while leaving the new session running with the *previous*
                # session's risk state: the daily-loss breaker was never reset, so a fresh
                # session could start already halted. Undo the start before reporting.
                try: session.end('start_failed')
                except Exception: pass
                raise
        elif action=='pause': session.pause()
        elif action=='resume': session.resume()
        elif action=='end': session.end('manual')
        else: raise ValueError('unknown_action')
        return web.json_response(session.snapshot())
    except (ValueError,TypeError) as exc: return web.json_response({'error':str(exc)},status=400)

async def market_detail(request):
    data=request.app[app_keys.STATE](); symbol=request.match_info['symbol'].upper(); return web.json_response({'symbol':symbol,'market':next((x for x in data.get('all',[]) if x.get('symbol')==symbol),None)})

async def orders(request): return web.json_response({'orders':request.app[app_keys.STATE]().get('open_orders',[])})

async def close_position(request):
    symbol=request.match_info['symbol'].upper()
    close_fn=request.app.get(app_keys.CLOSE_POSITION)
    if close_fn is None: return web.json_response({'error':'position_close_unavailable'},status=503)
    try:
        result=close_fn(symbol)
    except (KeyError,ValueError) as exc:
        return web.json_response({'error':str(exc)},status=400)
    except Exception as exc:
        # Anything else used to escape as a bare 500 with an empty body: the dashboard
        # could only say the close failed, and the reason was in the server log if it was
        # anywhere. A close that fails is exactly the moment an operator needs to know why.
        import logging
        logging.getLogger('web').exception('close_position failed for %s', symbol)
        return web.json_response({'error':'%s: %s' % (type(exc).__name__, exc)},status=500)
    if result is None: return web.json_response({'error':'position_not_found'},status=404)
    if request.app.get(app_keys.BROADCAST): await request.app[app_keys.BROADCAST]()
    return web.json_response(result)

async def cancel_order(request):
    broker=request.app.get(app_keys.BROKER); order_id=request.match_info['order_id']
    if broker is None: return web.json_response({'error':'broker_unavailable'},status=503)
    order,reason=broker.cancel(order_id)
    if order is None: return web.json_response({'error':reason},status=404)
    if request.app.get(app_keys.BROADCAST): await request.app[app_keys.BROADCAST]()
    return web.json_response({'order':order.__dict__,'reason':reason})

async def realtime(request):
    socket=web.WebSocketResponse(heartbeat=20); await socket.prepare(request); request.app[app_keys.SUBSCRIBERS].add(socket)
    try:
        await socket.send_json(request.app[app_keys.STATE]())
        async for message in socket:
            if message.type==web.WSMsgType.ERROR: break
    finally: request.app[app_keys.SUBSCRIBERS].discard(socket)
    return socket

async def chart_data(request):
    symbol=request.match_info['symbol'].upper()
    fn=request.app.get(app_keys.CHART_DATA)
    return web.json_response(fn(symbol) if fn else {'symbol':symbol,'candles':[],'trades':[],'equity_curve':[]})

async def research_report(request):
    path=Path(__file__).resolve().parent.parent/'data'/'research_report.json'; return web.json_response(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status':'not_generated'})
async def training_state(request):
    runner = request.app.get(app_keys.TRAINING)
    if runner is None:
        return web.json_response({'current': None, 'history': [], 'running': False})
    return web.json_response(runner.state())

async def training_start(request):
    runner = request.app.get(app_keys.TRAINING)
    if runner is None:
        raise web.HTTPServiceUnavailable(text='training_unavailable')
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    tier = payload.get('tier') or 'mainstream'
    options = {key: payload[key] for key in
               ('interval', 'days', 'horizon', 'folds', 'rounds', 'cost_bps', 'max_symbols')
               if key in payload}
    try:
        job = runner.start(tier, **options)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    rows = await request.app[app_keys.MARKET_ROWS]()
    runner.task = asyncio.create_task(runner.execute(job, rows))
    return web.json_response({'run_id': job.run_id, 'tier': job.tier,
                              'symbols': job.symbols, 'status': job.status})

async def universe(request):
    rows = await request.app[app_keys.MARKET_ROWS]()
    graded = classify(rows)
    return web.json_response({'counts': summarise(graded), 'symbols': graded,
                              'rules': RULES})

async def models(request):
    state = request.app.get(app_keys.MODEL_STATE)
    return web.json_response(state() if state else {'production': None, 'models': [], 'mode': 'unavailable'})
async def health(request):
    """Liveness plus a verdict on the inputs.

    A liveness probe must not assemble the dashboard payload: that is the most expensive
    thing this server builds, and /health is the endpoint a monitor hits. It must also not
    report a constant 'ok' -- which it did, alongside five counters that were all zero in
    the states worth alerting on. app/health.py builds the verdict from state that is
    already in memory, so it stays as cheap as the counters were.
    """
    from ..ops.health import report

    runtime = request.app[app_keys.RUNTIME]
    return web.json_response(report(runtime))


async def ready(request):
    """Readiness, as distinct from liveness.

    /health answers "is this process serving?" and must return 200 whenever it does: an
    orchestrator that restarts a container on a failing liveness probe would restart it
    forever during a venue outage, which is the one time a running process that is merely
    unable to trade is still worth keeping. This endpoint answers the other question --
    can entries be trusted right now -- and fails with 503 when they cannot.
    """
    from ..ops.health import report

    payload = report(request.app[app_keys.RUNTIME])
    return web.json_response(payload, status=503 if payload['status'] == 'down' else 200)
async def metrics(request):
    """Prometheus exposition.

    This served a single hand-written line, paper_equity, while app/prometheus.py held an
    unused counter store. Everything a monitor would alert on was reachable only through
    the JSON dashboard payload, which is not a scrape target.
    """
    from ..ops.prometheus import text

    state = request.app[app_keys.STATE]()
    # aiohttp rejects a charset inside content_type; it must be passed separately. This
    # raised ValueError on every request, so the Prometheus endpoint answered 500 to a
    # scraper for as long as it has existed -- and a scrape failure is silent by design.
    return web.Response(text=text(state),
                        content_type='text/plain',
                        charset='utf-8',
                        headers={'X-Prometheus-Text-Version': '0.0.4'})
