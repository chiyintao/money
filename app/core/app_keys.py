"""Typed application keys.

aiohttp warns on string keys because a typo silently creates a new slot. Using
AppKey instances turns those mistakes into failures at import time.
"""
from aiohttp import web

STATE = web.AppKey('state', object)
REALTIME_STATE = web.AppKey('realtime_state', object)
CHART_DATA = web.AppKey('chart_data', object)
SIMULATION = web.AppKey('simulation', object)
BROKER = web.AppKey('broker', object)
SUBSCRIBERS = web.AppKey('subscribers', set)
BROADCAST = web.AppKey('broadcast', object)
ON_SESSION_START = web.AppKey('on_session_start', object)
CLOSE_POSITION = web.AppKey('close_position', object)
RESET_RISK = web.AppKey('reset_risk', object)
MODEL_STATE = web.AppKey('model_state', object)
TRAINING = web.AppKey('training', object)
MARKET_ROWS = web.AppKey('market_rows', object)
RUNTIME = web.AppKey('runtime', object)
SET_RISK_PROFILE = web.AppKey('set_risk_profile', object)
SET_EXIT_POLICY = web.AppKey('set_exit_policy', object)
SYMBOL_EDGE_STATE = web.AppKey('symbol_edge_state', object)
