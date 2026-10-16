# Winnow recorded KV-cache explorer

This static demo is for `turboquant_kv`, not the meeting-transcript frontend.
It reads committed A6000 benchmark artifacts without importing torch or running inference.

From the repository root:

```sh
python3 public-demo/export.py
python3 public-demo/export.py --check
python3 -m http.server 8080 --directory public-demo/site --bind 127.0.0.1
```

Select contexts, packing widths, benchmark subsets and decode contexts. Reset restores the recording's default view. Quality and memory use Qwen2.5-7B; C27 decode uses Qwen3-0.6B. MiB values are binary units. The public export intentionally omits benchmark prompts and answers.

Vercel project root: `public-demo`; static output: `site`. No build command, functions, credentials or environment variables are needed. Free hosting only for this demo lane. Review the exact source packet before release. The source revision is the latest commit touching the retained source files, so adding demo commits does not relabel the measured artifacts.
