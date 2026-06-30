# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**EHRI NER Cataloguing & Labelling Tool** (repo short name: EHRI-CATE) — a **human-in-the-loop, LLM-powered AI microservice** from EHRI / NIOD-KNAW that assists researchers and archivists in enriching EHRI Portal records with:

1. Subject indexing against the **EHRI Terms** SKOS vocabulary (hierarchical — broader/narrower relations matter).
2. Entity linking for **Camps** and **Ghettos**, against the EHRI ontology.
3. **Temporal metadata / date extraction.**

Suggestions are reviewed by a human before being committed — this is an assistant, not an autonomous tagger. Design choices should favour **precision over recall** and make suggestion provenance/override easy.

It depends on three external GRAPHIA components — **LLM4SSH**, **Federated Gateway**, **Automatic Tagger** — that are *not* built here; this repo integrates with them.

Prior art: Dermentzi, Bryant, Rovigo & García-González (2025), *Multilingual Automated Subject Indexing*, [hal-05142136](https://hal.science/hal-05142136). Mike is a co-author. The genuinely new contribution of EHRI-CATE relative to that paper is the **hierarchy-aware scoring**, which the paper flagged as future work but did not implement.

## Current state — MVP thin slice (subject indexing)

A working CLI evaluation harness for **subject indexing** is in place, comparing zero-shot LLMs against the five supervised Annif baselines. Entity linking and date extraction (the other two pilot tracks) are not started yet.

```
src/ehri_cate/
├── corpus.py             # 2,179-row labelled corpus loader, 4-field input builder
├── vocab.py              # SKOS Turtle loader → 913-concept DAG with ancestors/distance
├── scoring.py            # flat F1@5 (paper-aligned) + Kiritchenko hierarchical F1
├── split.py              # 70/30 stratified train/test split (iterative-stratification)
├── rate_limit.py         # optional sliding-window RPM limiter (off by default; LLM4SSH cap lifted for our account)
├── backends/
│   ├── llm4ssh.py        # zero-shot LLM backend over the LiteLLM proxy
│   └── annif_backend.py  # 5 supervised Annif baselines (train via CLI, suggest in-process)
└── cli.py                # `cate evaluate` (LLM + --annif), `cate train-baselines`, `cate label` (ad-hoc stdin), `cate models` (live list)
scripts/                  # build_omikuji_macos_arm64.sh — provenance for the vendored wheel
wheels/                   # vendored omikuji macOS-arm64 wheel (no working PyPI build exists)
tests/                    # scorer + rate-limiter + split tests; guarded Annif integration test
data/                     # gitignored: corpus JSON + ehri_terms.ttl; data/annif/ = trained baselines
results/                  # gitignored: per-run JSON dumps with per-doc predictions and scores
```

## Locked design decisions

Before changing any of these, surface the change to the user.

| Decision | Value | Why |
|---|---|---|
| Input fields | `name` + `biographicalHistory` + `archivalHistory` + `scopeAndContent`, concatenated with `\n\n`, omit missing | Matches the paper's ISAD(G) 3.1.2/3.2.2/3.2.3/3.3.1 concat for cross-paper comparability |
| Row filter | None — all 2,179 rows kept | Mirrors paper (paper applied no content-presence filter on top of "has labels") |
| Train/test split | 70/30 iterative multi-label stratification (Sechidis 2011, `iterative-stratification`), seed 0 | Supervised Annif baselines need training data; mirrors the paper's split. Baselines train on train; LLMs + baselines both scored on the **test** sample. `cate evaluate` samples `-n` from the test split |
| Candidate label set | 550 in-use labels (default `--candidates in-use`); 913 full vocab available via `--candidates full`. **LLM-only** — Annif's candidate set is whatever it saw in training | Paper used 554; 550 in-use are all SKOS-resolvable |
| LLM language strategy | **Prompt in English, feed input in source language** | Rely on model multilingual ability; corpus is 38% non-English |
| Primary metric | F1 at top-5 suggestions (Annif default), reported alongside micro-F1 and weighted-macro-F1 | Paper-aligned |
| Hierarchy metric | Kiritchenko ancestor-expansion hF1 — not distance-decay | One clear semantics, no tunable knobs to defend |
| LLM4SSH model name | Pass the bare model name; the backend prefixes `litellm_proxy/` automatically | LiteLLM SDK routing convention |
| Default LLM | `DeepSeek-V3.1-vLLM` (`DEFAULT_MODEL`) when neither `--model` nor `--annif` is given; validated against the live `/v1/models` first. No implicit multi-model run | Best single model in our eval (not lightest). The old 3-LLM default was dropped; repeat `--model` to run several. `cate models` lists the live set |
| Rate limit | **Off by default** (`--rpm 0`); the limiter is shared across **all** models in a run when enabled | LLM4SSH's 30 RPM/key cap was lifted for our account (2026-06). Set `--rpm 30` if using a still-capped key |

## Toolchain

- Python **3.12** (`.python-version` + `pyproject.toml`, pinned `>=3.12,<3.13`). Annif allows `<3.14`, but **Omikuji 0.5.x has no working Python-3.13 build** (old pyo3/`time` crate), so we pin 3.12. Don't bump to 3.13.
- Managed by **`uv`**. The supervised baselines live in the `baselines` dependency group, auto-included via `[tool.uv] default-groups`. `uv sync --no-group baselines` gives a lightweight LLM-only env.
- **Omikuji on macOS-arm64:** no usable PyPI wheel (the cp312 "universal2" wheel is x86_64-only; the sdist fails to compile on Rust ≥1.80). A locally-built arm64 wheel is vendored in `wheels/` and referenced via `[tool.uv.sources]` (macOS-arm64 marker only; Linux/Windows fall back to PyPI). Rebuild with `scripts/build_omikuji_macos_arm64.sh`.

```bash
uv sync                         # create/update .venv (includes Annif baselines)
uv run pytest                   # scorer + rate-limiter + split tests; Annif integration test if data present
uv run cate evaluate --help     # CLI help
```

## Running the evaluator

Set `LLM4SSH_API_KEY` (and optionally `LLM4SSH_API_BASE`) in `.env` — gitignored.

```bash
# Default: 50 test-split rows × the default LLM (DeepSeek-V3.1-vLLM), validated live
uv run cate evaluate

# List the models the proxy currently serves (default marked)
uv run cate models

# Single LLM, larger sample, custom output
uv run cate evaluate -n 167 --model DeepSeek-V3.1-vLLM --output results/deepseek-167.json

# Supervised Annif baselines only (no API key); trains on first use, caches in data/annif/
uv run cate evaluate --annif all -n 100

# Head-to-head: 5 baselines + an LLM on the same test sample
uv run cate evaluate --annif all --model DeepSeek-V3.1-vLLM -n 50

# Train baselines explicitly (e.g. before a big eval); --retrain forces a rebuild
uv run cate train-baselines --annif all
```

The LLM4SSH RPM cap has been lifted for our account, so `--rpm` defaults to 0 (no pacing) and LLM throughput scales with `--workers`. If you run against a still-capped key, pass `--rpm 30`, which imposes a hard floor of `(sample_size × n_models) / 30` minutes regardless of `--workers`. Annif backends run sequentially and are fast at suggest time; training the NN Ensemble (TensorFlow) is the slow part of a cold run.

Runs are **quiet by default**: `evaluate`/`label`/`models` take `--debug`, which surfaces the prompt-size report, the full rendered prompt, and litellm internals (litellm's import-time warnings are otherwise suppressed). The prompt-size `[debug]` lines are opt-in via this flag.

## Things deliberately not done yet

- **Annif hyper-parameter tuning** — the five baselines (TF-IDF, MLLM, fastText, Omikuji, NN Ensemble) are wired and trainable via `--annif`, but with sensible default params and the language-neutral `simple` analyzer; the paper's exact hyper-params weren't fully published. Tuning + a multilingual analyzer strategy are open.
- **Prompt iteration / few-shot / archival-context enrichment** — a single zero-shot prompt ships as the default, but it can now be swapped per-run via `cate evaluate --prompt-template <file.toml>` (`system`/`user` keys; `user` must keep `{text}`/`{candidates}` and the numbered-JSON output contract). The default is unchanged, so comparability holds; few-shot and archival-context enrichment are still not built.
- **Entity linking (Camps/Ghettos), date extraction** — pilot tracks #2 and #3, untouched.
- **Web service + mock Portal UI** — roadmap steps 4 and 5.

## Open scoping questions still on the table

- Hierarchy weighting: ancestor-expansion is in place but no per-distance tuning — defer until we have a reason to.
- Deployment target: "EHRI's servers" tentative.
- Implementation roadmap spreadsheet referenced as `#TODO` in the pilot description; not yet linked.
