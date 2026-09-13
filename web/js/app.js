// Application entry: state store, websocket, polling and event wiring.

import {getState, cancelOrder, closePosition, resetRisk, control, getUniverse, getTrainingState, startTraining, getRiskProfiles, setRiskProfile, getExitPolicies, setExitPolicy} from "./api.js";
import {renderRiskProfile, fillProfileSelect, renderExitPolicy, fillPolicySelect} from "./views/session.js";
import {renderTraining, renderUniverse, renderEstimate} from "./views/training.js";
import {initTerminal} from "./terminal.js";
import {renderModels} from "./operations.js";
import {createStore} from "./state.js";
import {render, showTradeDetail, showSessionReview} from "./render.js";
import {shell} from "./shell.js";
import {value} from "./dom.js";

const store = createStore();
const root = document.getElementById("app-shell");
root.innerHTML = shell();

let socket, pollTimer, framePending = false, pendingState = null, lastPaint = 0, currentView = "overview";
// Risk profiles are configuration, not streaming state, so they are fetched on load and
// after a change rather than on every frame.
let riskCatalogue = null;
let exitCatalogue = null;

const VIEW_TITLES = {
  overview: ["资产总览", "模拟账户净值与合约风险"],
  trade: ["模拟交易", "持仓、委托与详细成交记录"],
  markets: ["市场行情", "USDT 永续合约实时快照"],
  analytics: ["绩效分析", "权益表现与策略结果"],
  audit: ["审计记录", "模拟会话与事件"],
  models: ["模型中心", "真实模型在线决策 · 影子预测"],
  training: ["模型训练", "抓取历史 · 滚动验证 · 训练候选模型"],
  system: ["系统健康", "行情 · 风控 · 持久化"],
};

const SESSION_TOASTS = {
  start: "模拟会话已启动", pause: "模拟会话已暂停",
  resume: "模拟会话已继续", end: "模拟会话已结束",
};

// Realtime frames arrive at 10Hz; coalesce them into one paint per ~250ms so the
// DOM is not rebuilt on every tick.
function scheduleState(next) {
  pendingState = next;
  if (framePending) return;
  framePending = true;
  const wait = Math.max(0, 250 - (performance.now() - lastPaint));
  setTimeout(() => requestAnimationFrame(() => {
    framePending = false;
    const state = pendingState;
    pendingState = null;
    lastPaint = performance.now();
    if (!state) return;
    const previous = store.get();
    if (state.realtime) {
      const merged = new Map((previous.all || []).map(x => [x.symbol, x]));
      (state.symbols || []).forEach(x => merged.set(x.symbol, {...merged.get(x.symbol), ...x}));
      store.set({...previous, ...state, all: [...merged.values()]});
    } else {
      store.set(state);
    }
  }), wait);
}

function connection(live, text) {
  document.querySelectorAll(".live-dot").forEach(x => x.classList.toggle("live", live));
  const node = document.getElementById("connText");
  if (node) node.textContent = text;
}

function error(message) {
  const box = document.getElementById("errorBox");
  box.textContent = message;
  box.classList.toggle("show", !!message);
}

function toast(message) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 2200);
}

function selectTradeTab(tab = "positions") {
  document.querySelectorAll("#tradeTabs [data-tab]").forEach(x => x.classList.toggle("active", x.dataset.tab === tab));
  document.querySelectorAll("[data-pane]").forEach(x => x.classList.toggle("active", x.dataset.pane === tab));
}

function view(name, tab) {
  currentView = name;
  document.querySelectorAll(".view").forEach(x => x.classList.toggle("active", x.id === name));
  document.querySelectorAll("[data-view]").forEach(x => x.classList.toggle("active", x.dataset.view === name && !x.dataset.tradeTab));
  const [title, subtitle] = VIEW_TITLES[name] || VIEW_TITLES.overview;
  document.getElementById("pageTitle").textContent = title;
  document.getElementById("pageSubtitle").textContent = subtitle;
  if (name === "trade") selectTradeTab(tab || "positions");
  render(store.get());
}

async function refresh() {
  try {
    store.set(await getState());
    error("");
    connection(true, "实时连接");
  } catch (e) {
    connection(false, "连接异常");
    error("数据刷新失败：" + e.message);
  }
}

async function loadRiskProfiles() {
  try {
    riskCatalogue = await getRiskProfiles();
    // render.js draws the panel on every frame; publishing the catalogue keeps the
    // fetch in one place instead of duplicating it per render.
    window.__riskCatalogue = riskCatalogue;
    fillProfileSelect("riskProfileSelect", riskCatalogue);
    fillProfileSelect("liveProfileSelect", riskCatalogue);
    renderRiskProfile(riskCatalogue);
  } catch (e) {
    error("风险档位加载失败：" + e.message);
  }
}

async function loadExitPolicies() {
  try {
    exitCatalogue = await getExitPolicies();
    window.__exitCatalogue = exitCatalogue;
    fillPolicySelect("liveExitSelect", exitCatalogue);
    renderExitPolicy(exitCatalogue);
  } catch (e) {
    error("止盈止损策略加载失败：" + e.message);
  }
}

async function applyExitPolicy() {
  const button = document.getElementById("applyExit");
  const name = value("liveExitSelect");
  if (!name) return;
  button.disabled = true;
  try {
    exitCatalogue = await setExitPolicy(name);
    window.__exitCatalogue = exitCatalogue;
    renderExitPolicy(exitCatalogue);
    toast("止盈止损策略已切换为 " + (exitCatalogue.active?.label || name));
  } catch (e) {
    error(e.message);
  } finally {
    button.disabled = false;
  }
}

async function applyRiskProfile() {
  const button = document.getElementById("applyProfile");
  const name = value("liveProfileSelect");
  if (!name) return;
  button.disabled = true;
  try {
    riskCatalogue = await setRiskProfile(name);
    // render() draws the panel from this shared reference, so updating only the local
    // variable left the next frame drawing the previous profile and the switch looked
    // like it had not taken effect.
    window.__riskCatalogue = riskCatalogue;
    renderRiskProfile(riskCatalogue);
    toast("风险档位已切换为 " + (riskCatalogue.active?.label || name));
  } catch (e) {
    error(e.message);
  } finally {
    button.disabled = false;
  }
}

async function sessionAction(action) {
  try {
    const payload = action === "start" ? {
      initial_cash: Number(value("initialCash")),
      leverage: Number(value("sessionLeverage")),
      source: value("sessionSource"),
      symbol_count: Number(value("symbolCount")),
      // Read at click time so the session starts under the profile shown in the form.
      risk_profile: value("riskProfileSelect") || undefined,
    } : {};
    await control(action, payload);
    toast(SESSION_TOASTS[action]);
    // Starting a session may have applied a different profile, so re-read it.
    if (action === "start") await loadRiskProfiles();
    await refresh();
  } catch (e) {
    error(e.message);
  }
}

function closeDrawer() {
  document.getElementById("tradeDrawer").classList.remove("open");
  document.getElementById("drawerBackdrop").classList.remove("open");
}

async function handleClick(event) {
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.view) return view(target.dataset.view, target.dataset.tradeTab);
  if (target.dataset.tab) return selectTradeTab(target.dataset.tab);
  if (target.dataset.tradeIndex !== undefined) return showTradeDetail(Number(target.dataset.tradeIndex));
  if (target.dataset.sessionIndex !== undefined) return showSessionReview(Number(target.dataset.sessionIndex));

  if (target.classList.contains("cancel-btn")) {
    try {
      await cancelOrder(target.dataset.id);
      toast("订单已撤销");
      await refresh();
    } catch (err) { error(err.message); }
    return;
  }

  if (target.classList.contains("close-position-btn")) {
    const symbol = target.dataset.symbol;
    if (!confirm("确认立即平仓 " + symbol + "？将按当前标记价成交。")) return;
    target.disabled = true;
    try {
      const result = await closePosition(symbol);
      toast(symbol + " 已平仓，净盈亏 " + Number(result.pnl || 0).toFixed(4) + " USDT");
      await refresh();
    } catch (err) {
      target.disabled = false;
      error(err.message);
    }
  }
}

async function handleRiskReset() {
  if (!store.get().risk_state?.halted) return;
  if (!confirm("确认重置风控基准？这不会修改资金或删除历史记录。")) return;
  try {
    await resetRisk();
    toast("风控基准已重置");
    await refresh();
  } catch (e) { error(e.message); }
}

function wire() {
  document.addEventListener("click", handleClick);
  document.getElementById("refreshBtn").onclick = refresh;
  document.getElementById("privacyBtn").onclick = () => document.body.classList.toggle("privacy");
  document.getElementById("riskStatus").onclick = handleRiskReset;
  document.getElementById("startSession").onclick = () => sessionAction("start");
  document.getElementById("pauseSession").onclick = () => sessionAction("pause");
  document.getElementById("resumeSession").onclick = () => sessionAction("resume");
  document.getElementById("endSession").onclick = () => sessionAction("end");
  document.getElementById("applyProfile").onclick = applyRiskProfile;
  document.getElementById("applyExit").onclick = applyExitPolicy;
  document.getElementById("marketSearch").oninput = () => render(store.get());
  document.getElementById("marketSort").onchange = () => render(store.get());
  document.getElementById("ledgerSearch").oninput = () => render(store.get());
  document.getElementById("ledgerSide").onchange = () => render(store.get());
  document.getElementById("drawerClose").onclick = closeDrawer;
  document.getElementById("drawerBackdrop").onclick = closeDrawer;
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
}

function connect() {
  if (socket && socket.readyState <= 1) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(protocol + "://" + location.host + "/ws");
  socket.onopen = () => {
    connection(true, "实时连接");
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  };
  socket.onmessage = event => {
    try {
      scheduleState(JSON.parse(event.data));
      error("");
    } catch { /* ignore malformed frame */ }
  };
  socket.onerror = () => connection(false, "连接异常");
  socket.onclose = () => {
    connection(false, "重连中");
    if (!pollTimer) pollTimer = setInterval(refresh, 5000);
    setTimeout(connect, 3000);
  };
}

async function refreshModels() {
  try {
    const response = await fetch("/api/models", {cache: "no-store"});
    if (!response.ok) throw Error("HTTP " + response.status);
    renderModels(await response.json());
  } catch (e) {
    renderModels({}, e.message);
  }
}

async function refreshTraining() {
  try {
    const [training, universe] = await Promise.all([getTrainingState(), getUniverse()]);
    renderTraining(training);
    renderUniverse(universe);
    return training;
  } catch (e) {
    return null;
  }
}

function trainingOptions() {
  const read = (id, fallback) => {
    const node = document.getElementById(id);
    const value = node ? Number(node.value) : NaN;
    return Number.isFinite(value) ? value : fallback;
  };
  const tierNode = document.getElementById("trainTier");
  return {
    tier: tierNode ? tierNode.value : "mainstream",
    interval: document.getElementById("trainInterval").value,
    days: read("trainDays", 365),
    folds: read("trainFolds", 12),
    cost_bps: read("trainCost", 12),
    max_symbols: read("trainMaxSymbols", 15),
    horizon: read("trainHorizon", 12),
  };
}

async function beginTraining() {
  const button = document.getElementById("startTraining");
  button.disabled = true;
  try {
    await startTraining(trainingOptions());
    toast("训练已开始");
    await refreshTraining();
  } catch (e) {
    error("训练启动失败：" + e.message);
    button.disabled = false;
  }
}

async function pollTraining() {
  if (document.hidden) return;
  const training = await refreshTraining();
  const running = !!(training && training.running);
  clearTimeout(trainingTimer);
  trainingTimer = setTimeout(pollTraining, running ? 4000 : 15000);
}

let trainingTimer = null;

store.subscribe(value => render(value));
wire();
initTerminal();
view("trade");
connect();
loadRiskProfiles();
loadExitPolicies();
refresh();
refreshModels();
document.getElementById("refreshModels").onclick = refreshModels;
document.getElementById("startTraining").onclick = beginTraining;
for (const id of ["trainInterval", "trainDays", "trainMaxSymbols"]) {
  const node = document.getElementById(id);
  if (node) node.oninput = node.onchange = renderEstimate;
}
renderEstimate();
refreshTraining();
pollTraining();
setInterval(() => {
  if (!document.hidden) {
    refresh();
    refreshModels();
  }
}, 10000);
