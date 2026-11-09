"""Local explorer for the schedule search: legality, dispatch decisions, recorded runs.

    python tune_explorer.py [--registry REG] [--port 8766]
    open http://127.0.0.1:8766

Use case: see which launch schedules are legal for a kernel shape on a target,
why the others are rejected, and what the runtime would select from a promotion
registry (or why it falls back), then inspect a saved tuning run.

Live mode evaluates the real legality model and the real dispatcher against the
real registry. It never shows timings it did not measure: on a host without
CUDA there are no live timings, and the page says so. Recorded mode loads
tune/summary.json or heldout/heldout.json from a run and is labelled RECORDED.
Inputs are bounded: batch <= 256, seq_len <= 32768, rows <= 2^24, fixed targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tuned_dispatch as td
from tsc import schedule as sch

FAMILY = {"hq": 28, "hkv": 4, "head_dim": 128, "page_size": 16, "bit_width": 8}


def legality(q: dict) -> dict:
    target = sch.TARGETS[q.get("target", "sm80")]
    if q.get("op") == "dequant":
        rows = max(1, min(int(q.get("rows", 4096)), 1 << 24))
        p = sch.DequantProblem(rows, 128, int(q.get("bw", 8)))
        legal, bad = sch.enumerate_legal(sch.DEQUANT_SPACE, p, target, sch.dequant_legal)
        default = sch.HEURISTIC_DEQUANT.key()
    else:
        b = max(1, min(int(q.get("batch", 16)), 256))
        L = max(1, min(int(q.get("seq_len", 8192)), 32768))
        p = sch.AttnProblem(b, L, dense=q.get("op") == "dense", **FAMILY)
        legal, bad = sch.enumerate_legal(sch.ATTN_SPACE, p, target, sch.attn_legal)
        default = sch.HEURISTIC_ATTN.key()
    return {"problem": {k: v for k, v in p.__dict__.items()}, "target": target.name,
            "default": default, "legal": [s.key() for s in legal],
            "rejected": [{"schedule": s.key(), "reason": r} for s, r in bad]}


def decision(q: dict, disp: td.TunedDispatch) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = max(1, min(int(q.get("batch", 16)), 256))
    L = max(1, min(int(q.get("seq_len", 8192)), 32768))
    if q.get("op") == "dequant":
        d = disp.dequant_decision(max(1, min(int(q.get("rows", 4096)), 1 << 24)), 8, 128, device)
    else:
        p = sch.AttnProblem(b, L, dense=q.get("op") == "dense", **FAMILY)
        d = disp.attention_schedule(p.family(), b, L, device)
    return {"device": device, "path": d.path, "reason": d.reason, "schedule": d.schedule,
            "registry": str(disp.registry.root) if disp.registry else None}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedule Explorer</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;--ok:#2f6f5e;--bad:#a2412b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#9a9992;--card:#1e1e1c;--line:#33332f;--ok:#6cbfa6;--bad:#e08a72}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.mut{color:var(--mut)}.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;background:var(--ok);color:var(--bg)}
.rec{background:var(--bad)}label{margin-right:12px}input,select{font:inherit}
button{background:var(--fg);color:var(--bg);border:0;border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.chips span{display:inline-block;margin:2px;padding:1px 6px;border-radius:6px;border:1px solid var(--line);font:12px ui-monospace,monospace}
.chips span.def{border-color:var(--ok);color:var(--ok)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
.scroll{max-height:320px;overflow:auto}
</style></head><body><main>
<h1 style="font-size:20px;margin:0">Kernel schedule explorer</h1>
<div class="mut">Which launch schedules are legal for a shape, why the rest are rejected, and what the runtime selects.</div>
<div class="card"><span class="badge">LIVE: legality model and dispatcher evaluated now</span> <span class="mut" id="dev"></span>
<div style="margin-top:8px"><label>kernel <select id="op"><option value="packed">packed attention</option><option value="dense">dense attention</option><option value="dequant">tsc dequantizer</option></select></label>
<label>target <select id="target"><option>sm80</option><option>sm86</option><option>sm89</option><option>sm90</option></select></label>
<label>batch <input id="batch" type="number" min="1" max="256" value="16" style="width:64px"></label>
<label>seq_len <input id="seq_len" type="number" min="1" max="32768" value="8192" style="width:80px"></label>
<label>rows <input id="rows" type="number" min="1" max="16777216" value="32768" style="width:96px"></label>
<button id="go">Evaluate</button></div></div>
<div class="grid"><div class="card"><b>Runtime decision</b><div id="dec" class="mut"></div></div>
<div class="card"><b>Legal schedules</b> <span class="mut" id="nlegal"></span><div class="chips scroll" id="legal"></div></div></div>
<div class="card"><b>Rejected before any compile</b> <span class="mut" id="nbad"></span><div class="scroll"><table id="bad"></table></div></div>
<div class="card"><span class="badge rec">RECORDED RUN: saved results, nothing executes</span>
<p><input type="file" id="file" accept=".json"> Load <code>tune/summary.json</code> or <code>heldout/heldout.json</code>.</p><div id="rec" class="scroll mut">No file loaded.</div></div>
</main><script>
const $=id=>document.getElementById(id);const fx=(v,n)=>v==null?'-':(+v).toFixed(n);
async function go(){const q=new URLSearchParams({op:$('op').value,target:$('target').value,batch:$('batch').value,seq_len:$('seq_len').value,rows:$('rows').value});
 const r=await (await fetch('/api/eval?'+q)).json();
 $('dev').textContent=`device: ${r.decision.device}`+(r.decision.device==='cuda'?'':' (no live timings on this host)');
 $('dec').innerHTML=`path <b>${r.decision.path}</b><br>${r.decision.reason}`+(r.decision.schedule?`<br><code>${JSON.stringify(r.decision.schedule)}</code>`:'')+`<br>registry: ${r.decision.registry||'none configured'}`;
 $('nlegal').textContent=`${r.legality.legal.length} legal, default ${r.legality.default}`;
 $('legal').innerHTML=r.legality.legal.map(k=>`<span class="${k===r.legality.default?'def':''}">${k}</span>`).join('');
 $('nbad').textContent=`${r.legality.rejected.length}`;
 $('bad').innerHTML='<tr><th>schedule</th><th>reason</th></tr>'+r.legality.rejected.map(x=>`<tr><td><code>${x.schedule}</code></td><td>${x.reason}</td></tr>`).join('')}
$('go').onclick=go;go();
$('file').onchange=async e=>{let j;try{j=JSON.parse(await e.target.files[0].text())}catch{$('rec').textContent='Not JSON';return}
 let h='';if(j.attention){h='<table><tr><th>shape</th><th>legal</th><th>default us</th><th>exhaustive best</th><th>model-guided pick</th><th>pick us</th></tr>'+j.attention.map(r=>`<tr><td>${r.tag}</td><td>${r.legal}</td><td>${fx(r.heuristic_us,0)}</td><td>${r.exhaustive_best} ${fx(r.exhaustive_best_us,0)}</td><td>${r.model_guided_selected?JSON.stringify(r.model_guided_selected):'-'}</td><td>${fx(r.model_guided_us,0)}</td></tr>`).join('')+'</table>';
  h+='<p>'+(j.promoted||[]).map(x=>'promoted: '+x).join('<br>')+'<br>'+(j.not_promoted||[]).map(x=>'not promoted: '+x).join('<br>')+'</p>'}
 else if(j.rows){h='<table><tr><th>arm</th><th>batch</th><th>seq_len</th><th>decision</th><th>selected</th><th>us</th><th>default us</th><th>best us</th></tr>'+j.rows.map(r=>`<tr><td>${r.arm}</td><td>${r.batch}</td><td>${r.seq_len}</td><td>${r.decision}: ${r.reason}</td><td>${r.selected}</td><td>${fx(r.selected_us,0)}</td><td>${fx(r.default_us,0)}</td><td>${fx(r.best_us,0)}</td></tr>`).join('')+'</table>'}
 else h='<pre>'+JSON.stringify(j,null,1).slice(0,5000)+'</pre>';$('rec').innerHTML=h};
</script></body></html>"""


def handler(disp):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/eval":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if q.get("target") not in sch.TARGETS or q.get("op") not in ("packed", "dense", "dequant"):
                    return self._send(b'{"error":"bad request"}', "application/json", 400)
                try:
                    body = {"legality": legality(q), "decision": decision(q, disp)}
                except ValueError:
                    return self._send(b'{"error":"bad request"}', "application/json", 400)
                return self._send(json.dumps(body).encode(), "application/json")
            self._send(b'{"error":"not found"}', "application/json", 404)
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(td.TunedDispatch(a.registry)))
    print(f"explorer on http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
