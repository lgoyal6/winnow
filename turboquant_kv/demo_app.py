"""Local interactive demo: the real engine, packed vs dense, side by side.

    python demo_app.py --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --port 8765
    open http://127.0.0.1:8765

Use case: see what packed KV pages buy under a fixed KV memory budget, and
watch the lifecycle (streaming, cancellation, refusal, page reclamation).

The process loads one model and starts two gRPC servers on loopback, one over
8-bit packed pages and one over unquantized pages, each with its own page pool
sized from the same KV byte budget. The browser talks to this app; this app
talks to those servers over real gRPC. Every number shown in Live mode comes
from those engines as it happens. Recorded mode loads a saved study JSON and
says so on screen.

Bounds: loopback only; prompt <= 2000 characters; max_new_tokens <= 128;
parallel copies <= 6; one-minute request lifetime; budget chosen from a fixed
list. Nothing here loads another model or runs arbitrary code.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import grpc_server as gs
import packed_backend as pb
import winnow_serving_pb2 as pb2
import winnow_serving_pb2_grpc as pb2_grpc
from engine import EngineConfig

MAX_CHARS, MAX_NEW, MAX_COPIES, LIFETIME_S = 2000, 128, 6, 60.0
BUDGETS_MIB = (2, 4, 8, 16)


class Arm:
    def __init__(self, name, model, tok, kv_mode, budget_mib, revision, device):
        cfg = EngineConfig(max_batch=8, max_queue=8, prefill_chunk=256, max_prompt_tokens=1024,
                           max_new_tokens_cap=MAX_NEW)
        self.engine = gs.build_engine(model, kv_mode=kv_mode, kv_budget_bytes=budget_mib << 20,
                                      page_size=16, decode_backend="auto", cfg=cfg,
                                      eos_ids=tok.eos_ids)
        self.run = gs.start(self.engine, model_revision=revision, kv_mode=kv_mode,
                            tokenizer=tok)
        self.stub = pb2_grpc.GenerationStub(grpc.insecure_channel(f"127.0.0.1:{self.run.port}"))
        self.name, self.kv_mode, self.budget_mib = name, kv_mode, budget_mib


class App:
    def __init__(self, model_id, device):
        from model_guard import guarded_from_pretrained, pinned_revision
        from transformers import AutoModelForCausalLM, AutoTokenizer
        pb.register()
        self.model_id, self.device = model_id, device
        self.revision = pinned_revision(model_id)
        self.tok = guarded_from_pretrained(AutoTokenizer, model_id)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = guarded_from_pretrained(AutoModelForCausalLM, model_id, dtype=dtype,
                                             attn_implementation="sdpa").to(device).eval()
        pb.check_model(self.model.config, 8)
        eos = {self.tok.eos_token_id, *(self.model.generation_config.eos_token_id or [])} - {None}
        self.tok.eos_ids = eos
        self.lock = threading.Lock()
        self.arms = {}
        self.set_budget(BUDGETS_MIB[1])

    def set_budget(self, mib):
        if mib not in BUDGETS_MIB:
            raise ValueError("budget not offered")
        with self.lock:
            for a in self.arms.values():
                a.run.stop(1)
            self.arms = {
                "packed8": Arm("packed8", self.model, self.tok, "packed8", mib, self.revision, self.device),
                "dense": Arm("dense", self.model, self.tok, "dense", mib, self.revision, self.device),
            }
            self.budget = mib

    def state(self):
        out = {"model": self.model_id, "revision": self.revision, "device": self.device,
               "budget_mib": self.budget, "budgets": BUDGETS_MIB, "arms": {}}
        for k, a in self.arms.items():
            h = a.stub.Health(pb2.HealthRequest(), timeout=2)
            out["arms"][k] = {"kv_mode": h.kv_mode, "total_pages": h.total_pages,
                              "free": h.free_pages, "owned": h.owned_pages, "pending": h.pending_pages,
                              "queued": h.queued, "active": h.active,
                              "bytes_per_token": h.kv_bytes_per_token,
                              "capacity_tokens": h.total_pages * a.engine.pool.page_size}
        out["backend_counts"] = dict(pb.BACKEND_COUNTS)
        return out


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Winnow Packed Serving</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;--acc:#2f6f5e;--warn:#a2412b;--dense:#4a5a8a}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#9a9992;--card:#1e1e1c;--line:#33332f;--acc:#6cbfa6;--warn:#e08a72;--dense:#93a5df}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
.tabs button{margin-right:6px}.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600}
.live{background:var(--acc);color:var(--bg)}.rec{background:var(--warn);color:var(--bg)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
textarea{width:100%;min-height:90px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit}
label{margin-right:12px}button{background:var(--fg);color:var(--bg);border:0;border-radius:8px;padding:7px 14px;font:inherit;cursor:pointer}
button.sec{background:transparent;color:var(--fg);border:1px solid var(--line)}
.out{white-space:pre-wrap;font:13px/1.45 ui-monospace,monospace;background:var(--bg);border-radius:8px;padding:8px;min-height:60px;max-height:220px;overflow:auto;margin:6px 0}
.bar{height:14px;background:var(--line);border-radius:4px;overflow:hidden;display:flex}.bar i{display:block;height:100%}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
.k{font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1>Winnow packed KV serving</h1>
<div class="mut">Same model, same KV memory budget: 8-bit packed pages with compressed-domain decode attention vs unquantized pages.</div>
<div class="card tabs"><button id="tLive">Live engines</button><button class="sec" id="tRec">Recorded run</button>
 <span id="mode" class="badge live">LIVE: real engines on this machine</span></div>
<section id="live">
<div class="card"><div id="model" class="mut"></div>
<textarea id="prompt" maxlength="2000">Explain in three sentences why storing the KV cache in fewer bits lets a server hold more conversations at once.</textarea>
<div style="margin-top:8px"><label>max new tokens <input id="maxnew" type="number" min="1" max="128" value="48" style="width:64px"></label>
<label>parallel copies <input id="copies" type="number" min="1" max="6" value="3" style="width:52px"></label>
<label>KV budget <select id="budget"></select> MiB</label>
<button id="go">Run both</button> <button class="sec" id="stop">Cancel all</button></div></div>
<div class="grid"><div class="card" id="c-packed8"></div><div class="card" id="c-dense"></div></div>
<div class="card"><b>Why these numbers</b><div class="mut" id="why"></div></div>
</section>
<section id="rec" hidden><div class="card"><span class="badge rec">RECORDED RUN: saved results, nothing executes</span>
<p><input type="file" id="file" accept=".json"> Load a <code>report.json</code>, <code>model_contract.json</code> or <code>summary.json</code> from a study directory.</p>
<div id="recout" class="out">No file loaded.</div></div></section>
</main><script>
const $=id=>document.getElementById(id);let state=null,live=[];
const fx=(v,n)=>v==null?'-':v.toFixed(n);  // missing data is '-', never 0
function arm(name,s){const a=s.arms[name];const tot=a.total_pages||1;
 return `<b style="color:var(${name=='dense'?'--dense':'--acc'})">${name=='dense'?'Dense pages (unquantized, model dtype)':'Packed 8-bit pages'}</b>
 <div class="mut k">${a.total_pages} pages = ${a.capacity_tokens.toLocaleString()} tokens of KV &middot; ${a.bytes_per_token.toLocaleString()} B/token</div>
 <div class="bar" title="owned / pending fence / free"><i style="width:${100*a.owned/tot}%;background:var(${name=='dense'?'--dense':'--acc'})"></i><i style="width:${100*a.pending/tot}%;background:var(--warn)"></i></div>
 <div class="mut k">owned ${a.owned} &middot; awaiting fence ${a.pending} &middot; free ${a.free} &middot; active ${a.active} &middot; queued ${a.queued}</div>
 <div id="o-${name}"></div>`}
async function refresh(){state=await (await fetch('/api/state')).json();
 $('model').textContent=`${state.model}@${state.revision.slice(0,8)} on ${state.device}`;
 if(!$('budget').options.length){state.budgets.forEach(b=>{const o=new Option(b,b);$('budget').add(o)});$('budget').value=state.budget_mib}
 for(const n of ['packed8','dense']){const keep=$('o-'+n)?.innerHTML||'';$('c-'+n).innerHTML=arm(n,state);$('o-'+n).innerHTML=keep}
 const p=state.arms.packed8,d=state.arms.dense;
 $('why').innerHTML=`At ${state.budget_mib} MiB the packed pool holds ${(p.capacity_tokens/d.capacity_tokens).toFixed(2)}x the tokens of the dense pool (${p.bytes_per_token} vs ${d.bytes_per_token} bytes per token across all layers). A request is admitted only if its prompt plus max_new_tokens fits in free pages, so the dense arm refuses or queues sooner. Decode attention calls so far: ${Object.entries(state.backend_counts).map(([k,v])=>k+' '+v).join(', ')||'none'}.`
 +(state.backend_counts.tiled_reference?` <b>Speed note:</b> on ${state.device} packed decode runs the FP32 torch reference (tiled_reference), not the GPU Triton kernel, so its latency here says nothing about the optimized path.`:'')
 +(state.device!=='cuda'?` Dense pages use the model dtype (fp32 on CPU), so the capacity ratio here is larger than the ~1.9x it would be against bf16 pages on a GPU.`:'')}
function run(){const copies=Math.min(6,+$('copies').value||1);const body={prompt:$('prompt').value.slice(0,2000),max_new_tokens:Math.min(128,+$('maxnew').value||1)};
 for(const n of ['packed8','dense']){$('o-'+n).innerHTML='';for(let i=0;i<copies;i++){const id=`${n}-${i}`;const div=document.createElement('div');div.innerHTML=`<div class="mut k" id="m-${id}">copy ${i+1}: submitted</div><div class="out" id="t-${id}"></div>`;$('o-'+n).appendChild(div);
  const es=new EventSource(`/api/generate?arm=${n}&q=${encodeURIComponent(JSON.stringify(body))}`);live.push(es);let t0=performance.now(),k=0;
  es.onmessage=e=>{const m=JSON.parse(e.data);if(m.token!==undefined){k++;$('t-'+id).textContent+=m.text;$('m-'+id).textContent=`copy ${i+1}: ${k} tokens, server ${m.t.toFixed(2)} s`}
   else{es.close();$('m-'+id).textContent=`copy ${i+1}: ${m.status}${m.detail?' ('+m.detail+')':''}; ${m.output_tokens} tokens; TTFT ${(m.ttft_s||0).toFixed(2)} s; total ${(m.total_s||0).toFixed(2)} s`}}}}}
$('go').onclick=run;$('stop').onclick=async()=>{live.forEach(e=>e.close());live=[];await fetch('/api/cancel_all',{method:'POST'})};
$('budget').onchange=async()=>{await fetch('/api/budget',{method:'POST',body:JSON.stringify({mib:+$('budget').value})});refresh()};
$('tLive').onclick=()=>{$('live').hidden=false;$('rec').hidden=true;$('mode').className='badge live';$('mode').textContent='LIVE: real engines on this machine'};
$('tRec').onclick=()=>{$('live').hidden=true;$('rec').hidden=false;$('mode').className='badge rec';$('mode').textContent='RECORDED RUN: saved results'};
$('file').onchange=async e=>{const t=await e.target.files[0].text();let j;try{j=JSON.parse(t)}catch{ $('recout').textContent='Not JSON';return}
 let h='';if(j.runs){h='<table><tr><th>arm</th><th>load</th><th>offered</th><th>SLO met</th><th>goodput req/s</th><th>TTFT p95</th><th>ITL p95</th></tr>'+j.runs.map(r=>`<tr><td>${r.arm}</td><td>${r.load}</td><td>${r.offered}</td><td>${r.slo_met}</td><td>${r.goodput_req_s.toFixed(3)}</td><td>${fx(r.ttft_s.p95,3)}</td><td>${fx(r.itl_s.p95,4)}</td></tr>`).join('')+'</table>';
  if(j.comparisons)h+='<p>'+j.comparisons.map(c=>`${c.arm} vs ${c.vs} @ ${c.load}: ${(100*c.relative_change).toFixed(1)}% [${c.ci95.map(x=>(100*x).toFixed(1)+'%').join(', ')}]`).join('<br>')+'</p>'}
 else if(j.packed){h=`<p>${j.model}@${(j.revision||'').slice(0,8)} on ${j.device} (${j.dtype}); ${j.v4?'registered v4':'reported'} contract PASS=${j.v4?j.v4.pass:j.pass}</p>`+['packed','dense'].map(k=>`<p><b>${k}</b>: ${j.v4?`${j.v4[k+'_ok']}/${j[k].requests.length} accepted by v4; `:''}${j[k].requests.filter(r=>r.within_contract).length}/${j[k].requests.length} pass per-request strict checks</p>`).join('')}
 else h='<pre>'+JSON.stringify(j,null,1).slice(0,6000)+'</pre>';$('recout').innerHTML=h};
refresh();setInterval(refresh,1000);
</script></body></html>"""


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path)
            if u.path == "/":
                b = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            if u.path == "/api/state":
                return self._json(app.state())
            if u.path == "/api/generate":
                qs = parse_qs(u.query)
                arm = app.arms.get(qs.get("arm", [""])[0])
                try:
                    body = json.loads(qs.get("q", ["{}"])[0])
                    prompt = str(body["prompt"])[:MAX_CHARS]
                    n = max(1, min(MAX_NEW, int(body["max_new_tokens"])))
                except Exception:                                   # noqa: BLE001
                    return self._json({"error": "bad request"}, 400)
                if arm is None or not prompt.strip():
                    return self._json({"error": "bad request"}, 400)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                rid = f"demo-{uuid.uuid4().hex[:10]}"
                req = pb2.GenerateRequest(model_revision=app.revision, max_new_tokens=n,
                                          request_id=rid,
                                          messages=[pb2.Message(role="user", content=prompt)])
                stream = arm.stub.Generate(req, timeout=LIFETIME_S)
                try:
                    for ev in stream:
                        if ev.HasField("token"):
                            msg = {"token": ev.token.token_id, "text": ev.token.text,
                                   "t": ev.token.server_time_s}
                        else:
                            f = ev.final
                            msg = {"status": f.status, "detail": f.detail, "output_tokens": f.output_tokens,
                                   "ttft_s": f.ttft_s, "total_s": f.total_s}
                        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                        self.wfile.flush()
                except grpc.RpcError as e:
                    msg = {"status": "refused" if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED else "error",
                           "detail": e.details(), "output_tokens": 0}
                    try:
                        self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                    except OSError:
                        pass
                except OSError:                                     # browser went away
                    stream.cancel()
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = min(int(self.headers.get("Content-Length", 0)), 4096)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/budget":
                try:
                    app.set_budget(int(body.get("mib")))
                except (TypeError, ValueError) as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"ok": True})
            if self.path == "/api/cancel_all":
                k = 0
                for a in app.arms.values():
                    with a.engine._lock:
                        ids = [r.request_id for r in list(a.engine._queue) + a.engine._active]
                    for rid in ids:
                        k += a.engine.cancel(rid)
                return self._json({"cancelled": k})
            self._json({"error": "not found"}, 404)
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local packed-vs-dense serving demo.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    app = App(args.model, args.device)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"demo on http://127.0.0.1:{args.port}  ({args.model} on {args.device})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for a in app.arms.values():
            a.run.stop(1)


if __name__ == "__main__":
    main()
