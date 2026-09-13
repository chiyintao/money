// Contract bar, candlestick chart and order book.

import {getChart} from "./api.js";
import {set} from "./dom.js";

const fmt = v => v == null ? "—" : Number(v).toLocaleString("zh-CN", {maximumFractionDigits: 6});

let chart, series, selected = "BTCUSDT", generation = 0, lastState = {};

export function terminalShell() {
  return `
  <div class="contract-bar">
    <label>USDT 永续<select id="contractSelect"><option>BTCUSDT</option><option>ETHUSDT</option></select></label>
    <div><small>最新价</small><strong id="contractPrice">—</strong></div>
    <div><small>标记价格</small><b id="contractMark">—</b></div>
    <div><small>24h 涨跌</small><b id="contractChange">—</b></div>
    <div><small>资金费率</small><b id="contractFunding">—</b></div>
    <span class="paper-pill">PAPER</span>
  </div>
  <div class="terminal-grid">
    <div class="terminal-chart">
      <div class="chart-toolbar"><b>K 线</b><span id="chartStatus">加载中</span><button id="fitChart" title="适应图表">适应</button></div>
      <div id="candleChart"></div>
      <a class="chart-credit" href="https://www.tradingview.com/" target="_blank" rel="noopener">TradingView Lightweight Charts</a>
    </div>
    <div class="terminal-book">
      <h2>盘口</h2>
      <div class="book-header"><span>价格 (USDT)</span><span>数量</span></div>
      <div id="askLevels"></div>
      <div class="book-spread" id="bookSpread">—</div>
      <div id="bidLevels"></div>
      <small id="bookAge">等待报价</small>
    </div>
    <aside id="terminalControls"></aside>
  </div>`;
}

export async function loadCandles() {
  const id = ++generation;
  set("chartStatus", "加载中");
  try {
    const data = await getChart(selected);
    if (id !== generation) return;
    const unique = new Map();
    for (const row of data.candles || []) {
      if (![row.open, row.high, row.low, row.close].every(v => Number.isFinite(Number(v)))) continue;
      unique.set(Number(row.open_time) / 1000, {
        time: Number(row.open_time) / 1000,
        open: +row.open, high: +row.high, low: +row.low, close: +row.close,
      });
    }
    const bars = [...unique.values()].sort((a, b) => a.time - b.time);
    series?.setData(bars);
    chart?.timeScale().fitContent();
    set("chartStatus", bars.length
      ? bars.length + " 根 · " + new Date(bars.at(-1).time * 1000).toLocaleString()
      : "暂无 K 线");
  } catch (error) {
    set("chartStatus", "K 线获取失败：" + error.message);
  }
}

export function initTerminal() {
  const target = document.getElementById("candleChart");
  if (window.LightweightCharts) {
    chart = LightweightCharts.createChart(target, {
      autoSize: true,
      layout: {background: {color: "#0b0e11"}, textColor: "#929baa"},
      grid: {vertLines: {color: "#1c2027"}, horzLines: {color: "#1c2027"}},
      timeScale: {timeVisible: true, secondsVisible: false},
      rightPriceScale: {borderColor: "#2b3139"},
    });
    series = chart.addCandlestickSeries({
      upColor: "#0ecb81", downColor: "#f6465d",
      wickUpColor: "#0ecb81", wickDownColor: "#f6465d", borderVisible: false,
    });
  } else {
    set("chartStatus", "图表组件未加载");
  }
  document.getElementById("terminalControls").append(document.querySelector(".session-bar"));
  document.getElementById("contractSelect").onchange = event => {
    selected = event.target.value;
    renderTerminal(lastState);
    loadCandles();
  };
  document.getElementById("fitChart").onclick = () => chart?.timeScale().fitContent();
  loadCandles();
  setInterval(() => {
    if (!document.hidden && document.getElementById("trade").classList.contains("active")) loadCandles();
  }, 15000);
}

function renderBook(id, rows, color) {
  const node = document.getElementById(id);
  node.replaceChildren();
  if (!rows?.length) {
    node.textContent = "暂无报价";
    return;
  }
  for (const [price, qty] of rows.slice(0, 10)) {
    const row = document.createElement("div");
    row.className = "book-level " + color;
    const p = document.createElement("span"), q = document.createElement("span");
    p.textContent = fmt(price);
    q.textContent = fmt(qty);
    row.append(p, q);
    node.append(row);
  }
}

export function renderTerminal(state) {
  lastState = state;
  const select = document.getElementById("contractSelect");
  if (!select) return;

  const names = new Set([...Array.from(select.options).map(o => o.value), ...(state.all || []).map(x => x.symbol)]);
  if (names.size !== select.options.length) {
    select.replaceChildren(...[...names].sort().map(name => new Option(name, name)));
    select.value = selected;
  }

  const item = {
    ...(state.all || []).find(x => x.symbol === selected),
    ...(state.symbols || []).find(x => x.symbol === selected),
  };
  const fields = {
    contractPrice: {value: item.price, percent: false},
    contractMark: {value: item.mark_price, percent: false},
    contractChange: {value: item.change, percent: true},
    contractFunding: {value: item.funding_rate == null ? null : item.funding_rate * 100, percent: true},
  };
  for (const [id, field] of Object.entries(fields)) {
    set(id, fmt(field.value) + (field.percent ? "%" : ""));
  }

  const asks = item.asks || (item.ask ? [[item.ask, item.ask_qty]] : []);
  const bids = item.bids || (item.bid ? [[item.bid, item.bid_qty]] : []);
  renderBook("askLevels", asks, "down");
  renderBook("bidLevels", bids, "up");
  set("bookSpread", "价差 " + fmt(item.ask && item.bid ? item.ask - item.bid : null));
  set("bookAge", item.book_time
    ? "报价时间 " + new Date(item.book_time).toLocaleTimeString()
    : "报价时间未提供");
}
