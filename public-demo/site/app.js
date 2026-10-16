import {showSources, options, number, text, table, bars} from "./shared.js";
let data;
const armName = arm => ({"fp16":"fp16 baseline","tq4-current":"Original 4-bit cache","tq4-packed":"Packed 4-bit","tq3-packed":"Packed 3-bit","tq3.5-packed":"Packed 3.5-bit"})[arm] || arm;
function memory() {
  const context = Number(document.querySelector("#context").value);
  const arm = document.querySelector("#memory-arm").value;
  const rows = ["fp16","tq4-current",arm].map(name => data.memory.find(row => row.ctx === context && row.arm === name));
  const [base, current, packed] = rows;
  bars("#memory-bars", rows.map(row => ({label:armName(row.arm), value:row.resident_kv_mb})), "MiB");
  table("#memory-table", rows.map(row => [armName(row.arm),number(row.resident_kv_mb),number(row.persistent_kv_mb),number(row.decode_tok_s),number(row.peak_mb)]),armName(arm));
  text("#memory-insight",number(base.resident_kv_mb / packed.resident_kv_mb) + "× lower resident cache memory");
  text("#memory-explanation", "The original holds " + number(current.resident_kv_mb) + " MiB. The packed arm holds " + number(packed.resident_kv_mb) + " MiB at this context; " + number(packed.persistent_kv_mb) + " MiB is persistent cache storage.");
  text("#memory-speed", "The packed PyTorch arm decodes at " + number(packed.decode_tok_s) + " tokens/s versus " + number(base.decode_tok_s) + " for fp16. Lower is worse here.");
  text("#logical","Original reported logical storage: " + number(current.logical_kv_mb) + " MiB. This is not its " + number(current.resident_kv_mb) + " MiB resident measurement.");
}
function quality() {
  const bits = Number(document.querySelector("#bits").value);
  const bucket = document.querySelector("#bucket").value, task = document.querySelector("#task").value;
  const rows = data.quality.filter(row => row.bucket === bucket && row.task === task);
  const baseline = rows.find(row => row.arm === "fp16"), chosen = rows.find(row => row.arm === "tq" + bits);
  const sweep = data.bitwidth.find(row => row.bits === bits);
  bars("#quality-bars", rows.map(row => ({label:row.arm, value:row.score})), "/ 100",2,100);
  table("#quality-table", rows.map(row => [row.arm,number(row.score),number(row.score-baseline.score) + " points",row.n]),chosen.arm);
  const delta = chosen.score - baseline.score;
  text("#quality-insight", number(delta) + " score points versus fp16");
  text("#quality-explanation", bits + "-bit vector storage is " + number(sweep.compression_vs_fp16) + "× smaller than fp16. The selected scored subset changes from " + number(baseline.score) + " to " + number(chosen.score) + ".");
  text("#proxy", "Earlier proxy token match: " + number(sweep.greedy_token_match_vs_fp16*100,0) + "%. A proxy result does not replace the scored benchmark.");
  text("#quality-scope",chosen.n + " scored samples in this subset. Model: " + data.quality_model + ". Ratios include per-vector metadata.");
}
function decode() {
  const context = document.querySelector("#decode-context").value;
  const rows = data.decode[context], names = ["fp16_baseline","triton_tf32","native_cuda_fp32","triton_fp32"];
  const labels = {fp16_baseline:"Unquantized baseline",triton_tf32:"Triton TF32",native_cuda_fp32:"Native CUDA fp32",triton_fp32:"Triton fp32"};
  bars("#decode-bars",names.map(name => ({label:labels[name],value:rows[name].ms_per_token})),"ms/token");
  const ratio = rows.native_cuda_fp32.ms_per_token / rows.triton_tf32.ms_per_token;
  text("#decode-insight","Native CUDA is " + number(ratio >= 1 ? ratio : 1/ratio) + "× " + (ratio >= 1 ? "slower" : "faster") + " than TF32 at decode");
  text("#decode-explanation","Unquantized baseline: " + number(rows.fp16_baseline.ms_per_token) + " ms/token. It is faster than every quantized arm in this recorded batch-1 study.");
  const arms = data.kernel.arms;
  text("#kernel","At the separate B=16, H=4, L=16,384, D=128, 6-bit kernel shape, native CUDA takes " + number(arms.native_cuda_fp32.us/1000) + " ms versus TF32's " + number(arms.triton_tf32.us/1000) + " ms. Maximum absolute error: native " + arms.native_cuda_fp32.max_abs_err + ", TF32 " + arms.triton_tf32.max_abs_err + ". This kernel ratio is not end-to-end speedup.");
}
function reset() {
  for (const [id,value] of [["#context","16384"],["#memory-arm","tq4-packed"],["#bits","6"],["#bucket","all"],["#task","all"],["#decode-context","16384"]]) document.querySelector(id).value=value;
  memory(); quality(); decode();
}
try {
  const response = await fetch("./data.json"); if (!response.ok) throw new Error("recorded data unavailable");
  data = await response.json();
  options("#context",[2048,8192,16384],16384);
  options("#memory-arm",["tq4-packed","tq3.5-packed","tq3-packed"],"tq4-packed",armName);
  options("#bits",[4,5,6,8],6,value => value + " bits");
  options("#bucket",["all","0-4k","4-8k","8k+"],"all");
  options("#task",["all",...data.quality_tasks],"all");
  options("#decode-context",[2048,8192,16384],16384);
  for (const id of ["#context","#memory-arm"]) document.querySelector(id).addEventListener("change",memory);
  for (const id of ["#bits","#bucket","#task"]) document.querySelector(id).addEventListener("change",quality);
  document.querySelector("#decode-context").addEventListener("change",decode);
  document.querySelector("#reset").addEventListener("click",reset);
  showSources(data,"lgoyal6/winnow"); reset(); document.querySelector("#app").hidden=false;
} catch (error) {
  const box=document.querySelector("#error"); box.hidden=false;
  box.textContent="Recorded measurements could not load. Reload the page, or open data.json to inspect the retained data.";
}
