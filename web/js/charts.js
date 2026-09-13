// Canvas chart helpers.

function setup(canvas) {
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(320, Math.round(rect.width));
  const h = Math.max(180, Math.round(rect.height));
  if (canvas.width !== w * ratio || canvas.height !== h * ratio) {
    canvas.width = w * ratio;
    canvas.height = h * ratio;
  }
  const c = canvas.getContext("2d");
  c.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {c, w, h};
}

function drawGrid(c, w, h, pad) {
  c.strokeStyle = "#2b3139";
  c.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = pad + (i * (h - pad * 2)) / 4;
    c.beginPath();
    c.moveTo(pad, y);
    c.lineTo(w - pad, y);
    c.stroke();
  }
}

function pointsFor(values, w, h, pad, min, span) {
  return values.map((v, i) => [
    pad + (w - pad * 2) * (values.length === 1 ? .5 : i / (values.length - 1)),
    h - pad - ((v - min) / span) * (h - pad * 2),
  ]);
}

function fillUnderCurve(c, points, color, h, pad) {
  const gradient = c.createLinearGradient(0, pad, 0, h);
  gradient.addColorStop(0, color + "38");
  gradient.addColorStop(1, color + "00");
  c.beginPath();
  c.moveTo(points[0][0], h - pad);
  points.forEach(p => c.lineTo(p[0], p[1]));
  c.lineTo(points.at(-1)[0], h - pad);
  c.closePath();
  c.fillStyle = gradient;
  c.fill();
}

export function drawSeries(canvas, values, color = "#f0b90b", fill = false) {
  if (!canvas) return;
  const {c, w, h} = setup(canvas);
  const pad = 18;
  c.clearRect(0, 0, w, h);
  drawGrid(c, w, h, pad);

  if (!values.length) {
    c.fillStyle = "#848e9c";
    c.font = "12px Arial";
    c.textAlign = "center";
    c.fillText("等待权益数据", w / 2, h / 2);
    return;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = pointsFor(values, w, h, pad, min, span);
  if (fill) fillUnderCurve(c, points, color, h, pad);

  c.beginPath();
  points.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1]));
  c.strokeStyle = color;
  c.lineWidth = 2;
  c.stroke();

  c.fillStyle = "#848e9c";
  c.font = "10px Arial";
  c.textAlign = "left";
  c.fillText(max.toLocaleString(undefined, {maximumFractionDigits: 2}), pad, 12);
  c.fillText(min.toLocaleString(undefined, {maximumFractionDigits: 2}), pad, h - 3);
}
