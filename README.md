# EHRI-CATE

**EHRI NER Cataloguing & Labelling Tool** is an evaluation harness and (eventually) a human-in-the-loop AI microservice that assists researchers and archivists in enriching [EHRI Portal](https://portal.ehri-project.eu/) records with subject indexing, entity linking, and temporal metadata. Pilot from EHRI / NIOD-KNAW.

This repository currently contains the **subject-indexing evaluation harness**: a CLI that runs zero-shot LLMs against a corpus of 2,179 manually-labelled EHRI Portal descriptions and scores them against the [EHRI Terms](https://portal.ehri-project.eu/vocabularies/ehri_terms) SKOS vocabulary, with both standard F1 and a hierarchy-aware metric.

It builds on (and extends) Dermentzi, Bryant, Rovigo & García-González (2025), *Multilingual Automated Subject Indexing: a comparative study of LLMs vs alternative approaches in the context of the EHRI project*, [HAL hal-05142136](https://hal.science/hal-05142136). The genuine new contribution here is the **hierarchy-aware scoring** that the paper flagged as future work.

## Requirements

- Python **3.12** (pinned: the Omikuji baseline has no working Python-3.13 build; Annif's upper bound is `<3.14`).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- An API key for the GRAPHIA LLM4SSH proxy at `https://llm.graphia-ssh.eu` (only needed for the LLM backends).

## Setup

```bash
uv sync                 # installs everything, including the Annif baselines (TensorFlow etc.)
cp .env.example .env
# edit .env and paste your LLM4SSH_API_KEY
```

`uv sync` installs the supervised-baseline dependencies (the `baselines` dependency group) by
default. For a lightweight, LLM-only environment without TensorFlow / fastText / Omikuji:

```bash
uv sync --no-group baselines
```

> **macOS (Apple Silicon) note:** Omikuji ships no working arm64 wheel, so a prebuilt one lives in
> `wheels/` and is referenced from `pyproject.toml`. If you need to rebuild it (e.g. version bump),
> run `scripts/build_omikuji_macos_arm64.sh` (requires `brew install rust`). On Linux/Windows, uv
> pulls Omikuji from PyPI normally.

You also need the corpus and the SKOS vocabulary in `data/` (both gitignored):

- `data/manual-descriptions-labelled-with-ehri-terms.json` — labelled corpus (2,179 EHRI Portal descriptions).
- `data/ehri_terms.ttl` — EHRI Terms SKOS, fetch with:

  ```bash
  curl -sL -o data/ehri_terms.ttl \
    'https://portal.ehri-project.eu/vocabularies/ehri_terms/export?format=TTL'
  ```

## Usage

```bash
# Default: 50 test-split rows × 3 LLMs (Mistral-Small, DeepSeek-V3.1, MiniMax-M2.5)
uv run cate evaluate

# Single LLM, larger sample
uv run cate evaluate -n 167 --model DeepSeek-V3.1-vLLM --output results/deepseek-167.json

# Supervised Annif baselines only (no API key needed); trains on first use
uv run cate evaluate --annif all -n 100

# Head-to-head: all 5 baselines + an LLM, scored on the same test sample
uv run cate evaluate --annif all --model DeepSeek-V3.1-vLLM -n 50

# Train the baselines explicitly (otherwise `evaluate` trains them on first use)
uv run cate train-baselines --annif all

# All available CLI options
uv run cate evaluate --help
```

Per-document predictions (with gold labels, top-5 URIs, latencies, errors) are written to a JSON file in `results/`. A tab-separated summary (tagged `llm` / `annif`) is printed to stdout for piping into scripts or spreadsheets.

### Baselines and the train/test split

The five **native Annif backends** the prior paper benchmarked — **TF-IDF, MLLM (Maui-like Lexical
Matching), fastText, Omikuji/Parabel, and an NN Ensemble** of the other four — are available via
`--annif`. Unlike the zero-shot LLMs, these are **supervised**, so the corpus is split into a
**70/30 stratified train/test split** (iterative multi-label stratification, Sechidis et al. 2011,
via [`iterative-stratification`](https://github.com/trent-b/iterative-stratification); seed-stable).
Baselines train on the train split; **both** LLMs and baselines are scored on a sample of the
held-out test split, so the comparison is apples-to-apples. Trained models are cached under
`data/annif/` (gitignored) and reused until `--retrain`.

Two honest caveats when reading the baseline-vs-LLM table:

- **Multilingual handling.** Annif has no native heterogeneous-multilingual support, so the
  baselines use the language-neutral `simple` analyzer over the 38%-non-English corpus (the paper
  noted the same limitation). The LLMs are prompted in English over source-language input.
- **Candidate sets differ by construction.** A supervised model can only predict labels it saw in
  training; the LLM is handed an explicit candidate menu. The scorer is identical (gold URIs vs.
  top-5 predicted URIs), so the numbers are comparable, but this is an inherent
  supervised-vs-zero-shot difference, not a bug.

### Rate limiting

LLM4SSH enforces **30 requests/minute per API key**, shared across all models. The CLI's `--rpm` flag (default 30) paces calls accordingly, and the backend retries 429s once with backoff parsed from the response. At the default, runtime has a hard floor of `(sample_size × n_models) / 30` minutes.

## How it works

The evaluator follows the prior paper's setup for direct comparability, with one deliberate addition:

| Aspect | Choice |
|---|---|
| Input text per record | ISAD(G) 3.1.2 Title + 3.2.2 Admin/Biographical History + 3.2.3 Archival History + 3.3.1 Scope and Content, concatenated (missing fields omitted) |
| Candidate labels (LLM menu) | 550 in-use EHRI Terms (default) or all 913 (`--candidates full`) |
| LLM prompting | English instructions, multilingual input — relies on model's cross-lingual capability rather than translating |
| Train/test split (for baselines) | 70/30 iterative multi-label stratification (Sechidis 2011); baselines train on train, all backends scored on the test sample |
| Primary metric | F1 at top-5 (paper-aligned, matches Annif's default), plus micro-F1 and weighted-macro F1 |
| **New: hierarchy-aware metric** | Kiritchenko-style: expand each gold and predicted URI with its SKOS ancestors, then compute set-F1. Gives partial credit for predicting a broader or narrower concept than the gold |

### Hierarchical F1 (ancestor-expansion)

Flat F1 treats the EHRI Terms vocabulary as a flat list: predicting a parent or child of
the correct concept scores exactly the same as predicting something unrelated — zero. That
penalises a model for being *topically right but at the wrong level of granularity*, which is
common for zero-shot LLMs.

We use the **ancestor-expansion hierarchical F1 of Kiritchenko et al.** Before scoring, every
label is replaced by *itself plus all of its ancestors* along the SKOS `broader` relation, and
a standard set-based precision/recall/F1 is then computed over the expanded sets:

- Ĝ = ⋃ over gold concepts *c* of `{c} ∪ ancestors(c)`
- P̂ = ⋃ over predicted concepts *c* of `{c} ∪ ancestors(c)`
- **hP** = |Ĝ ∩ P̂| / |P̂|,  **hR** = |Ĝ ∩ P̂| / |Ĝ|,  **hF1** = 2·hP·hR / (hP + hR)

A predicted concept that is a parent or child of the gold concept shares part of its
ancestor path, so it earns partial credit proportional to that overlap; two concepts with no
common ancestor get zero overlap, exactly as in flat F1. We deliberately use the plain
ancestor-overlap form with **no per-distance weighting** (as opposed to distance-decay
schemes), so there are no tunable knobs to justify. Out-of-vocabulary URIs are dropped during
expansion. See `scoring.py` (`hierarchical_scores`).

References:

- Kiritchenko, S., Matwin, S., & Famili, A. F. (2005). *Functional Annotation of Genes Using
  Hierarchical Text Categorization.* Proc. BioLINK SIG, ISMB.
- Kiritchenko, S., Matwin, S., Nock, R., & Famili, A. F. (2006). *Learning and Evaluation in
  the Presence of Class Hierarchies: Application to Text Categorization.* Advances in
  Artificial Intelligence (Canadian AI 2006), LNCS 4013, pp. 395–406.

## Project structure

```
src/ehri_cate/
├── corpus.py             # Corpus loader, 4-field input builder
├── vocab.py              # SKOS Turtle → 913-concept DAG (ancestors, descendants, distance)
├── scoring.py            # flat F1 + Kiritchenko hierarchical F1, with three aggregations
├── split.py              # stratified 70/30 train/test split (iterative-stratification)
├── rate_limit.py         # Sliding-window limiter for the 30 RPM cap
├── backends/
│   ├── llm4ssh.py        # Zero-shot LLM backend over the LiteLLM proxy
│   └── annif_backend.py  # Supervised Annif baselines (train via CLI, suggest in-process)
└── cli.py                # `cate evaluate` + `cate train-baselines`
scripts/                  # build_omikuji_macos_arm64.sh (vendored-wheel provenance)
tests/                    # unit tests (scorer, rate limiter, split) + guarded Annif integration test
```

## Results so far

Baselines vs. LLM on a shared 50-doc test sample (`--annif all --model DeepSeek-V3.1-vLLM -n 50`,
70/30 stratified split, seed 0). **Sorted by flat doc-avg F1:**

| Backend | Kind | Doc-avg F1 | Micro F1 | Weighted-macro F1 | Hierarchical F1 |
|---|---|---:|---:|---:|---:|
| **NN Ensemble** | annif | **0.302** | **0.310** | **0.339** | 0.475 |
| Omikuji / Parabel | annif | 0.279 | 0.296 | 0.327 | 0.475 |
| DeepSeek-V3.1-vLLM | llm | 0.240 | 0.236 | 0.277 | **0.511** |
| MLLM | annif | 0.179 | 0.214 | 0.173 | 0.336 |
| TF-IDF | annif | 0.164 | 0.177 | 0.215 | 0.381 |
| fastText | annif | 0.090 | 0.087 | 0.037 | 0.349 |

Two things stand out, and both echo the prior paper:

1. **On flat F1 the supervised NN Ensemble and Omikuji/Parabel lead** — the same ranking the paper
   found among "alternative approaches". (fastText lags here because it is data-hungry and our
   ~1,500-doc train split is far smaller than the paper's 36k; with the language-neutral analyzer
   on a 38%-multilingual corpus it has little to work with.)
2. **On the hierarchy-aware F1 the zero-shot LLM is best of all (0.51)**, despite a middling flat
   F1. This is exactly the "topically right, wrong granularity" effect the hierarchical metric was
   built to expose — the LLM proposes broader/narrower relatives of the gold term that flat scoring
   throws away. This is the gap the paper flagged as future work, now measurable.

Numbers are from a single 50-doc sample with lightly-tuned Annif defaults and should not be quoted
as published results.

## Not yet implemented

- Annif hyper-parameter tuning and a proper multilingual analyzer strategy (currently `simple` + defaults).
- Few-shot prompting or archival-context enrichment (parent/sibling descriptions).
- Entity linking (Camps/Ghettos) and date extraction — the other two pilot tracks.
- Web service API and mock Portal admin UI — roadmap steps 4 and 5.

## License

EHRI-3 project work, EHRI / NIOD-KNAW. License TBD.
