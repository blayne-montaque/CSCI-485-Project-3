# Local environment setup

This checkpoint contains environment configuration only. Project planning and
implementation are deferred.

## Windows / NVIDIA installation

Run from the repository root in PowerShell. Use an installed 64-bit Python 3.13
interpreter. On the inspected machine, the command below selects Python 3.13.2;
adjust the interpreter path on other machines.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The PyTorch index is essential: `requirements.txt` specifies the release but does
not choose a CUDA build. Install the CUDA wheel first. Direct dependencies are
pinned to the versions checked during setup; transitive dependencies are resolved
by pip, so this is not a full environment lockfile. Downloads during the initial
setup used `--no-cache-dir` to avoid retaining a second copy of large wheels.

The official CUDA 13.0 wheel includes the runtime needed by PyTorch. The inspected
NVIDIA driver (616.92) supports it. `nvidia-smi` reports CUDA UMD 13.4, which is
driver information and does not need to equal `torch.version.cuda` (13.0).
No separate CUDA Toolkit or `nvcc` was found or needed for this wheel.
See the [official PyTorch installer](https://pytorch.org/get-started/locally/),
[CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/), and
[NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Select `.venv\Scripts\python.exe` with **Python: Select Interpreter** in VS Code.
The explicit interpreter commands above work without shell activation or changing
PowerShell execution policy. Optional activation: `.\.venv\Scripts\Activate.ps1`.
The machine's `python` on PATH is 3.10.0 and `py -0p` did not discover the installed
interpreters; use the explicit path to avoid selecting the wrong Python.

## Local configuration

`.env.example` contains only an empty `GROQ_API_KEY` placeholder. Add a private
`.env` later when API access is needed; no real `.env` or API key was created.
Virtual environments, secrets, caches, and generated model weights are ignored.
`data/` and `artifacts/` contain only `.gitkeep` placeholders. Required CSV, JSON,
PNG, source, and Markdown deliverables remain eligible for Git tracking.

## Verification

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -c "import transformers, datasets, accelerate, sklearn, pandas, numpy, matplotlib, scipy, groq, dotenv, streamlit"
.\.venv\Scripts\python.exe -m pip check
```

Installed direct library versions are recorded in `requirements.txt`. These checks
do not require model weights, datasets, or API requests.
