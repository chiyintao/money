// Thin DOM helpers so views never repeat guard-and-set logic.

export function set(id, value, cls) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = value;
  if (cls) node.className = cls;
}

export function html(id, markup) {
  const node = document.getElementById(id);
  if (node) node.innerHTML = markup;
}

export function width(id, percent) {
  const node = document.getElementById(id);
  if (node) node.style.width = Math.min(100, Number(percent || 0)) + "%";
}

export function disabled(id, value) {
  const node = document.getElementById(id);
  if (node) node.disabled = !!value;
}

export function value(id, fallback = "") {
  return document.getElementById(id)?.value || fallback;
}

export function checked(id) {
  return !!document.getElementById(id)?.checked;
}

export function toggleClass(id, name, on) {
  document.getElementById(id)?.classList.toggle(name, !!on);
}
