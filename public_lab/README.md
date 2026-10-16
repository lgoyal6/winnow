# Public component lab

The lab runs the project component imported in `run.py`, on bounded visitor inputs. It does not run inference, training, a cluster or a code sandbox. The original study is preserved as `recorded.html` alongside the live lab.

Run the focused checks with `python -m pytest public_lab/test_run.py`. Export a deployment to a new directory outside the repository with `python public_lab/export.py /absolute/new/stage`. Deploy that directory with the existing Vercel project. It contains only the endpoint, owning component, required recorded inputs and static site; no training dependencies or credentials are packaged. The export records the source head and SHA-256 of every packaged input in `source-manifest.json`.

The endpoint accepts POST JSON up to 16 KiB, validates component bounds before execution, disables caching, and reports the hashes of the actual component sources. The page discards old results when inputs change or a request fails. The serverless function has a five-second limit.
