"""HTTP application assembly for the paper dashboard."""
import asyncio
import json

from aiohttp import web

from ..core import app_keys
from .web import WEB_ROOT, cancel_order, edge_symbols_state, exit_policies_state, exit_policy_select, health, index, market_detail, metrics, models, orders, ready, realtime, research_report, risk_profile_select, risk_profiles_state, risk_reset, simulation_control, training_start, training_state, universe
from .web import chart_data as chart_data_handler
from .web import close_position as close_position_handler
from .web import state as state_handler


def build_app(runtime, state_fn, realtime_state_fn, chart_fn, on_session_start_fn,
              close_position_fn, reset_risk_fn, model_state_fn, set_risk_profile_fn=None,
              set_exit_policy_fn=None, symbol_edge_state_fn=None):
    # Parameter names carry a _fn suffix because this function previously shadowed
    # the imported route handlers: a lambda bound to a name like "state" replaced the
    # /api/state view and every request failed with a TypeError.
    app = web.Application()
    app[app_keys.STATE] = state_fn
    app[app_keys.REALTIME_STATE] = realtime_state_fn
    app[app_keys.CHART_DATA] = chart_fn
    app[app_keys.SIMULATION] = runtime.session
    app[app_keys.BROKER] = runtime.broker
    app[app_keys.SUBSCRIBERS] = set()
    app[app_keys.ON_SESSION_START] = on_session_start_fn
    app[app_keys.CLOSE_POSITION] = close_position_fn
    app[app_keys.RESET_RISK] = reset_risk_fn
    app[app_keys.MODEL_STATE] = model_state_fn
    app[app_keys.RUNTIME] = runtime
    app[app_keys.SET_RISK_PROFILE] = set_risk_profile_fn
    app[app_keys.SET_EXIT_POLICY] = set_exit_policy_fn
    app[app_keys.SYMBOL_EDGE_STATE] = symbol_edge_state_fn
    broadcast_lock = asyncio.Lock()

    async def broadcast():
        if broadcast_lock.locked() or not app[app_keys.SUBSCRIBERS]:
            return
        async with broadcast_lock:
            payload = json.dumps(app[app_keys.REALTIME_STATE](), ensure_ascii=False,
                                 allow_nan=False, separators=(",", ":"))

            async def send(socket):
                try:
                    await asyncio.wait_for(socket.send_str(payload), timeout=.25)
                except (asyncio.TimeoutError, ConnectionError, RuntimeError):
                    app[app_keys.SUBSCRIBERS].discard(socket)
                    try:
                        await asyncio.wait_for(socket.close(), timeout=.25)
                    except (asyncio.TimeoutError, ConnectionError, RuntimeError):
                        pass

            await asyncio.gather(*(send(socket) for socket in list(app[app_keys.SUBSCRIBERS])))

    app[app_keys.BROADCAST] = broadcast
    app.router.add_get('/', index)
    app.router.add_static('/assets', WEB_ROOT)
    app.router.add_get('/ws', realtime)
    app.router.add_get('/api/state', state_handler)
    app.router.add_post('/api/simulation/{action}', simulation_control)
    app.router.add_post('/api/risk/reset', risk_reset)
    app.router.add_get('/api/risk/profiles', risk_profiles_state)
    app.router.add_post('/api/risk/profile', risk_profile_select)
    app.router.add_get('/api/edge/symbols', edge_symbols_state)
    app.router.add_get('/api/exit/policies', exit_policies_state)
    app.router.add_post('/api/exit/policy', exit_policy_select)
    app.router.add_get('/api/market/{symbol}', market_detail)
    app.router.add_get('/api/chart/{symbol}', chart_data_handler)
    app.router.add_get('/api/orders', orders)
    app.router.add_post('/api/orders/{order_id}/cancel', cancel_order)
    app.router.add_post('/api/positions/{symbol}/close', close_position_handler)
    app.router.add_get('/api/research/report', research_report)
    app.router.add_get('/api/models', models)
    app.router.add_get('/api/universe', universe)
    app.router.add_get('/api/training/state', training_state)
    app.router.add_post('/api/training/start', training_start)
    app.router.add_get('/health', health)
    app.router.add_get('/ready', ready)
    app.router.add_get('/metrics', metrics)
    return app
