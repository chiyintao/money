// HTTP client for the dashboard API.

async function request(path, options) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
  return data;
}

const post = (path, payload) => request(path, payload === undefined ? {method: "POST"} : {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify(payload),
});

export const getState = () => request("/api/state?ts=" + Date.now());
export const getChart = symbol => request("/api/chart/" + encodeURIComponent(symbol) + "?ts=" + Date.now());
export const cancelOrder = id => post("/api/orders/" + encodeURIComponent(id) + "/cancel");
export const closePosition = symbol => post("/api/positions/" + encodeURIComponent(symbol) + "/close");
export const resetRisk = () => post("/api/risk/reset");
export const getRiskProfiles = () => request("/api/risk/profiles?ts=" + Date.now());
export const setRiskProfile = (profile, overrides) =>
  post("/api/risk/profile", overrides ? {profile, overrides} : {profile});
export const getExitPolicies = () => request("/api/exit/policies?ts=" + Date.now());
export const setExitPolicy = (policy, overrides) =>
  post("/api/exit/policy", overrides ? {policy, overrides} : {policy});
export const control = (action, payload = {}) => post("/api/simulation/" + action, payload);
export const getUniverse = () => request("/api/universe");
export const getTrainingState = () => request("/api/training/state?ts=" + Date.now());
export const startTraining = payload => post("/api/training/start", payload);
