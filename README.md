# EHRI-CATE

**EHRI NER Cataloguing & Labelling Tool** suggests [EHRI Terms](https://portal.ehri-project.eu/vocabularies/ehri_terms) subject labels for [EHRI Portal](https://portal.ehri-project.eu/) archival descriptions using LLMs, for review by a human cataloguer. It consists of a CLI (labelling + evaluation against a labelled corpus) and a JSON web API.

## Requirements

- Python **3.12** and [`uv`](https://docs.astral.sh/uv/)
- An API key for LiteLLM proxy (e.g. `https://llm.graphia-ssh.eu` for GRAPHIA LLM4SSH, or `https://openrouter.ai/api/v1`); only needed for LLM backends

## Setup

```bash
uv sync                          # full install, incl. Annif baselines (TensorFlow etc.)
uv sync --no-group baselines     # or: lightweight, LLM-only
cp .env.example .env             # then set LLM4SSH_API_KEY
```

For `cate evaluate`, put the corpus and vocabulary in `data/`:

- `data/manual-descriptions-labelled-with-ehri-terms.json` — the labelled corpus
- `data/ehri_terms.ttl`:

  ```bash
  curl -sL -o data/ehri_terms.ttl \
    'https://portal.ehri-project.eu/vocabularies/ehri_terms/export?format=TTL'
  ```

## Labelling text

```bash
cat description.txt | uv run cate label              # rank, score, label, URI table
uv run cate label --json --model GLM-5.1 description.txt
uv run cate models                                    # models available on the proxy
```

Useful options: `--candidates full` (all 913 terms instead of the 550 in use), `--prompt-template <file.toml>` (see `prompts/example.toml`), `--no-thinking` (disable reasoning on models that support it), `--debug` (show the prompt).

## Evaluating

```bash
uv run cate evaluate                                  # 50 test docs, default LLM
uv run cate evaluate -n 167 --model DeepSeek-V3.1-vLLM --output results/deepseek.json
uv run cate evaluate --annif all -n 100               # supervised Annif baselines (no API key)
uv run cate evaluate --annif all --model DeepSeek-V3.1-vLLM -n 50   # head-to-head
uv run cate train-baselines --annif all               # pre-train baselines (cached in data/annif/)
```

Repeat `--model` to compare several LLMs. Each run prints a tab-separated summary and writes per-document predictions to `results/`. Scores reported: F1@5 (doc-average, micro, weighted-macro) and **hierarchical F1**, which gives partial credit for predicting a broader or narrower term than the gold one.

If your API key is rate-limited, pass `--rpm 30`. Run `uv run cate evaluate --help` for all options.

## Web API

```bash
uv run python main.py                                 # dev server on 127.0.0.1:8000
uv run gunicorn wsgi:application                      # production

curl localhost:8000/models
curl -X POST localhost:8000/suggest -H 'Content-Type: application/json' \
  -d '{"text": "Letters from the Theresienstadt ghetto, 1943"}'
```

`POST /suggest` takes `text` (required) and optional `model`, `top_k` (1–50, default 5), `candidates` (`in-use` / `full`) and `thinking` (default `true`). It returns ranked `{rank, uri, label, score}` suggestions. Errors are `{"detail": ...}` with status 400 (bad input), 502 (upstream failure) or 503 (no API key).

## Reference

Builds on Dermentzi et al. (2025), *Multilingual Automated Subject Indexing*, [hal-05142136](https://hal.science/hal-05142136).
