# Inference Lab

One local Marimo notebook that teaches Bayesian inference and optimization by
prediction, simulation, inspection, coding, and explanation.

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).
All commands below use it.

## Try it in the cloud

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ElArkk/learn-inference/blob/main/inference_lab.py)

The badge opens the course on [molab](https://docs.marimo.io/guides/molab/),
marimo's free cloud environment. No installation is needed, and PyMC and Torch
run there. To enable the tutor, paste your OpenRouter key into the **Session API
key** field in the course sidebar. The key stays in session memory only. Do not
create a `.env` file on molab: forks of shared notebooks carry files along.

## Start the learner notebook

```bash
./scripts/run-local.sh
```

The launcher installs the locked environment. On first use, it copies the clean
course into the ignored local file `my_lab.py`. It then opens `my_lab.py` at
`http://localhost:2722`. Press `Ctrl+C` in the launcher terminal to stop it.

On each fresh kernel start, select the triangular **Run all stale cells** control
in the Marimo toolbar once. Wait until no cell is running. This loads the shared
course code, plots, tests, and tutor controls. Marimo then reruns dependent cells
when you edit code or use a control.

Always complete exercises in `my_lab.py`. The tracked `inference_lab.py` is the
clean course that GitHub users receive. It keeps TODO exercise cells and must not
contain personal solutions.

## Receive course updates

First stop the Marimo learner process. Then update the repository and run:

```bash
./scripts/sync-my-lab.sh
```

The sync uses the previous clean course as a three-way merge base. It applies
new course content while it keeps learner exercise edits. It validates a merge
before it replaces `my_lab.py`, writes `my_lab.py.backup`, and advances the merge
base only after success.

If course and learner edits overlap, the command leaves `my_lab.py` and
`.my_lab.base` unchanged. It writes a separate
`my_lab.py.merge-conflict-<timestamp>.py` file for manual repair.

The coding tasks are real Marimo cells. They support package completion, hover
documentation, go-to-definition, and function signatures. If these features do
not appear after the first start, reload the page once. In Marimo settings,
keep editor auto-completion and signature hints enabled.

After you edit an exercise, select **Run tests** below it. The test control runs
the exact code saved in the exercise cell in an isolated test namespace. Thus,
the test does not use an older kernel function if Marimo marks the cell as
stale. To also update the live kernel function, press **Cmd+Enter** on macOS or
**Ctrl+Enter** on Windows and Linux.

## Optional tutor feedback

The simulations and code tests work without a model or API key. To check written
answers and use the tutor in the notebook side panel, add an OpenRouter key with
one of these methods:

1. Open **Settings → AI → AI Providers → OpenRouter** in Marimo. Paste your own
   key in the OpenRouter API-key field and save the setting. Run all stale cells
   again. If the status does not change, restart the local notebook once.
2. Copy `.env.example` to `.env`, set `OPENROUTER_API_KEY=your-key`, and restart
   the notebook.

Never put an API key in `my_lab.py` or `inference_lab.py`. The `.env` file is
ignored by Git.

The default is `qwen/qwen3-235b-a22b-2507`. To select another OpenRouter model,
set `OPENROUTER_MODEL` in `.env`.

## Course maintenance

Course authors can inspect the clean `inference_lab.py` directly with:

```bash
uv run marimo edit inference_lab.py
```

Do not use this command for normal learning. The learner launcher intentionally
opens only `my_lab.py`.

## Verify

```bash
uv run marimo check --strict inference_lab.py
uv run marimo check --strict my_lab.py
uv run pytest -q
```

The notebook uses fixed random seeds. Heavy PyMC and Torch work starts only from
the related lab controls.
