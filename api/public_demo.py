from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from public_lab.run import run
FILES = ['public_lab/run.py', 'turboquant_kv/dispatch.py', 'turboquant_kv/results/phaseC_kernel.json', 'turboquant_kv/results/phaseC_kernel_tf32.json', 'turboquant_kv/results/phaseC_kernel_bw6.json', 'experiments/c27/out/matched_N262144.json']
class handler(BaseHTTPRequestHandler):
    def reply(self,status,value):
        body=json.dumps(value,allow_nan=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        self.reply(405,{"error":"Use POST"})
    def do_POST(self):
        try:
            size=int(self.headers.get("Content-Length","0"))
            if not 0<size<=16384: raise ValueError("Request exceeds 16 KiB")
            x=json.loads(self.rfile.read(size))
            if not isinstance(x,dict): raise ValueError("Use an object")
            result=run(x)
            self.reply(200,{"result":result,"source_hashes":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES}})
        except (ValueError,TypeError,KeyError,OverflowError) as e:
            self.reply(400,{"error":str(e)})
