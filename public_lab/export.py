"""Build an isolated deployment with only the bounded component and its inputs."""
from pathlib import Path
import shutil,json,sys,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1]
FILES=['public_lab/run.py', 'turboquant_kv/dispatch.py', 'turboquant_kv/results/phaseC_kernel.json', 'turboquant_kv/results/phaseC_kernel_tf32.json', 'turboquant_kv/results/phaseC_kernel_bw6.json', 'experiments/c27/out/matched_N262144.json', 'api/public_demo.py']
out=Path(sys.argv[1]);out.mkdir()
for name in FILES:
    target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
shutil.copytree(ROOT/'public-demo/site',out/'site')
(out/'requirements.txt').write_text('')
(out/'vercel.json').write_text(json.dumps({'framework':None,'buildCommand':'','outputDirectory':'site','functions':{'api/public_demo.py':{'maxDuration':5}}},indent=2))
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}}
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
