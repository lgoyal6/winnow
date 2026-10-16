export function showSources(data, repo) {
  const list = document.querySelector("#sources");
  for (const source of data.sources) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = "https://github.com/" + repo + "/blob/" + data.source_commit + "/" + source.path;
    link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = source.path;
    const hash = document.createElement("code");
    hash.className = "sourcehash"; hash.textContent = "SHA-256 " + source.sha256;
    item.append(link, hash); list.append(item);
  }
  document.querySelector("#source-commit").textContent = data.source_commit;
}
export function options(id, values, selected, label = value => value) {
  const element = document.querySelector(id);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value; option.textContent = label(value); element.append(option);
  }
  element.value = selected;
}
export const number = (value, digits = 2) => Number(value).toLocaleString("en-US", {minimumFractionDigits:digits, maximumFractionDigits:digits});
export function text(id, value) { document.querySelector(id).textContent = value; }
export function table(id, rows, selected) {
  const body = document.querySelector(id); body.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row[0] === selected) tr.className = "selected";
    for (const value of row) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
    body.append(tr);
  }
}
export function bars(id, rows, unit, digits = 1, ceiling = null) {
  const list = document.querySelector(id); list.replaceChildren();
  const max = ceiling ?? Math.max(...rows.map(row => row.value), 1);
  for (const row of rows) {
    const div = document.createElement("div"); div.className = "bar";
    const measure = document.createElement("div"); measure.className = "measure";
    const name = document.createElement("span"); name.textContent = row.label;
    const value = document.createElement("strong"); value.textContent = number(row.value, digits) + " " + unit;
    const progress = document.createElement("progress"); progress.max = max; progress.value = row.value;
    progress.setAttribute("aria-label", row.label + ": " + value.textContent);
    measure.append(name, value); div.append(measure, progress); list.append(div);
  }
}
