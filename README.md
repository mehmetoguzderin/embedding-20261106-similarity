# EmbeddingGemma 2 × history-aware edit fidelity

This lab compares exact preservation checks with real EmbeddingGemma 2
representations across a six-turn image and audio editing session. It generates
its media fixtures locally and includes a small, pre-extracted video; no video
codec, external dataset, paid service, or API key is needed.

## Set up and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a
native Python 3.12 interpreter. On Apple Silicon, use an arm64 Python rather
than a Rosetta interpreter. From this directory, run:

```sh
python3.12 run_lab.py
```

On Windows, use the Python launcher instead:

```powershell
py -3.12 run_lab.py
```

The runner uses uv 0.13.7 without modifying the system Python or globally
installed packages. It creates `.venv` for the notebook and Jupyter tools and
`.venv-model` for the isolated inference worker. No environment activation is
required. The notebook is the source of the core, Jupyter, and model
dependency declarations; the runner reads the core and Jupyter declarations
without executing notebook code during setup. `.python-version` specifies the
Python version used by the runner. On Windows and Linux, uv chooses a suitable
PyTorch backend; macOS uses the normal PyTorch distribution with MPS support
when available.

Add `--setup-only` to create both environments and install the notebook
dependencies without running inference. After a complete run, add
`--offline-check` to replay the validated cache in a temporary directory
without changing the delivered notebook or results.

The first run needs internet access and several GB of free space for Python
packages and the model checkpoint. The public Hugging Face checkpoint is
resolved to an immutable revision and recorded alongside the results. Later
matching runs can reuse validated local vectors.

The runner executes the notebook in a fresh kernel, checks model coverage, and
regenerates the notebook and HTML preview after successful completion. A failed
execution leaves a diagnostic notebook and traceback in `edit_fidelity_outputs/`.
For an interactive session, run `.venv/bin/python -m jupyter lab` on macOS or
Linux, or `.venv\Scripts\python.exe -m jupyter lab` in Windows PowerShell;
open the notebook and run all cells from this directory.

## Experiments and resource use

- A six-turn image/audio edit session with property-specific reference
  revisions, overrides, reversions, and a protected background.
- Exact masked recoloring, encoded-sRGB contrast, protected waveform samples,
  localized bell insertion, and soundtrack reversion.
- Correct editing, protected-background drift, forgotten history, and a no-op
  baseline, evaluated with MSE, PSNR, valid-window SSIM, and exact equality.
- Real image, text, audio, and native-video embeddings, including whole-image
  versus crop evidence, trajectories, correlations, and PCA.
- An eight-frame video and audio-onset relation, separate from the order of
  editing turns. The pinned video processor represents timestamps as integer
  `mm:ss` strings; exact timing checks use the original arrays instead.
- Typed trajectory differences with explicit evidence and unknown states;
  there is no paid Jev/LLM invocation.

The default model profile uses batch size one, float32, 70 image soft tokens,
two-second audio, and separate sequential image/text, audio, and video workers.
The audio worker omits the vision encoder; image and video workers omit the
audio encoder. `DEVICE = "auto"` attempts an available accelerator and falls
back to CPU if it cannot complete the model pass. CPU, MPS, and CUDA can also
be selected explicitly. EmbeddingGemma 2 does not support float16 inference.

Memory guards and bounded worker runs can stop inference under pressure. If this
happens, inspect the worker logs, close other memory-intensive applications,
and retry. GPU availability alone does not establish model-level compatibility,
and process RSS does not fully account for accelerator memory. Increase
`IMAGE_SOFT_TOKENS` only as a deliberate experiment: it changes processing and
can increase memory use.

## Modes and evidence integrity

- `auto` validates and reuses matching cached vectors, then computes missing
  inputs when possible.
- `live` recomputes genuine model embeddings without reusing vector results.
- `offline` uses matching local vectors and does not install model packages or
  contact the model server.
- `STRICT_MODEL = True` requires every requested model input; otherwise the run
  fails with diagnostics. `ENABLE_NATIVE_VIDEO = True` includes the video input.

No proxy features, random vectors, CLIP outputs, caption embeddings, or
invented scores are labeled as EmbeddingGemma 2. Missing model evidence is
explicitly marked **NOT MEASURED**. Exact metric and audio/video predicate
checks remain separate from embedding similarity. Reused identical media is
encoded once; this is not a cross-platform determinism test.

The 15 perturbations share one synthetic source. Their correlations are not a
human-judgment study, an independent-sample significance test, or a universal
measure of edit fidelity. The typed checker is deterministic, not a Jev or LLM
call, and cosine similarity is not a calibrated probability.

## Local validation (November 6, 2026)

A clean live run on this 16 GiB Apple Silicon Mac used native Python 3.12.14
and MPS float32 for all 38 model inputs: 32 image/text, five audio, and one
native video. All 32 notebook assertions passed. The public checkpoint was
resolved to `f9972ac22e696a105c482202646979e33e4b5a1e`.

The original live notebook execution took 58.18 seconds after installation and
model download. Peak worker process-tree RSS was 1.58 GiB for image/text,
2.30 GiB for audio, and 2.87 GiB for video. These are sampled process RSS values,
not total GPU memory; sampled MPS driver allocation reached 3.02 GiB in the
audio report. A separate real CPU text embedding also succeeded. Results and
memory use are specific to this machine and run.

The completed cache passed an offline replay. Dependency resolution was checked
for Python 3.12 on Windows x86-64 and Linux x86-64, but the notebook was not
executed on those platforms. The exact local package lists are in
`edit_fidelity_outputs/kernel_packages_freeze.txt` and
`edit_fidelity_outputs/model_packages_freeze.txt`.

## Repository and local outputs

The tracked files are the notebook, `run_lab.py`, this README,
`.python-version`, and `.gitignore`. The notebook retains its embedded results;
saving or rerunning it can therefore change the tracked notebook. The root
`.gitignore` allows only those five files, so adding another source file
requires explicitly allowing it there.

The generated HTML preview, `edit_fidelity_outputs/`, environments, model
weights, caches, logs, and temporary files remain on your machine and are
ignored by Git. The preview stays beside the notebook so its links to output
files work. These local files can be regenerated by the runner.

`edit_fidelity_outputs/` contains figures, CSV measurements, generated inputs,
the trajectory diff, model status, logs, and a run manifest. The manifest and
model reports record the checkpoint, package versions, preprocessing, input
hashes, backend, tests, and observed timings where available. Consult these
files and the executed notebook for the results and limitations of a particular
run. The HTML preview contains the same narrative and executed outputs.

## Troubleshooting

If installation or inference fails, inspect `model_status.json` and the
referenced installation or worker log. Model-stack versions are pinned to
Sentence Transformers 6.2.0 and Transformers 5.19.0. Record a version change
as a new experiment, and never substitute another checkpoint while retaining the
EmbeddingGemma 2 label. Model weights, credentials, and paid-service
dependencies are not bundled.
