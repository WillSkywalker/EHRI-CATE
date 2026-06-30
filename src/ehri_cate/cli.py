"""EHRI-CATE CLI.

Evaluates subject-indexing backends against the EHRI Terms vocabulary and reports
flat F1 + hierarchical F1. Two backend kinds, scored by the identical scorer:

  * zero-shot LLMs over the LLM4SSH proxy (`--model`), and
  * supervised Annif baselines (`--annif`): TF-IDF, MLLM, fastText, Omikuji, NN Ensemble.

Because the Annif baselines are supervised, the corpus is split (stratified, 70/30 by
default) into train/test; baselines train on the train split and BOTH kinds are scored
on a sample of the held-out test split, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

import click
from dotenv import load_dotenv

from .backends.annif_backend import ALL_BACKENDS, AnnifWorkspace
from .backends.llm4ssh import (
    PromptTemplate,
    load_backend_from_env,
    load_prompt_template,
    measure_messages,
)
from .corpus import LabelledDoc, load_corpus
from .rate_limit import NullRateLimiter, RateLimiter
from .scoring import AggregateScores, aggregate, flat_f1_at_k, hierarchical_f1_at_k
from .split import stratified_split
from .vocab import Vocab


DEFAULT_MODELS = (
    "Mistral-Small-3.2-24B-Instruct-2506",
    "DeepSeek-V3.1-vLLM",
    "MiniMax-M2.5",
)


class SuggestBackend(Protocol):
    """Anything that maps a document text to ranked (uri, score) predictions."""

    def suggest(self, text: str) -> list[tuple[str, float]]: ...


def _gold_set(d: LabelledDoc) -> set[str]:
    return set(d.gold_uris)


def _run_one_backend(
    label: str,
    backend: SuggestBackend,
    docs: list[LabelledDoc],
    vocab: Vocab,
    k: int,
    max_workers: int,
) -> tuple[AggregateScores, list[dict]]:
    """Run a backend over every doc and score it. Any per-doc rate limiting lives
    inside the backend (the LLM backend self-throttles); here we just parallelize."""
    per_doc_records: list[dict] = [None] * len(docs)  # type: ignore[list-item]

    def _call(idx: int, doc: LabelledDoc) -> tuple[int, dict]:
        t0 = time.monotonic()
        try:
            preds = backend.suggest(doc.text)
            err = None
        except Exception as e:  # noqa: BLE001 — one bad call shouldn't kill the run
            preds = []
            err = repr(e)
        elapsed = time.monotonic() - t0

        topk_uris = [u for u, _ in preds[:k]]
        flat = flat_f1_at_k(doc.gold_uris, preds, k=k)
        hier = hierarchical_f1_at_k(doc.gold_uris, preds, vocab, k=k)
        return idx, {
            "doc_id": doc.doc_id,
            "lang": doc.language_code,
            "gold": list(doc.gold_uris),
            "predicted_topk": topk_uris,
            "predicted_all": [{"uri": u, "score": s} for u, s in preds],
            "flat_f1": flat.f1,
            "flat_precision": flat.precision,
            "flat_recall": flat.recall,
            "hier_f1": hier.f1,
            "hier_precision": hier.precision,
            "hier_recall": hier.recall,
            "latency_s": elapsed,
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, i, d) for i, d in enumerate(docs)]
        done = 0
        for fut in as_completed(futures):
            idx, rec = fut.result()
            per_doc_records[idx] = rec
            done += 1
            if done % 25 == 0 or done == len(docs):
                click.echo(f"  [{label}] {done}/{len(docs)} done", err=True)

    per_doc_gold = [_gold_set(d) for d in docs]
    per_doc_topk = [rec["predicted_topk"] for rec in per_doc_records]
    agg = aggregate(per_doc_gold, per_doc_topk, vocab)
    return agg, per_doc_records


def _expand_annif(names: tuple[str, ...]) -> list[str]:
    """Expand the `all` shortcut and de-dupe while preserving order."""
    out: list[str] = []
    for n in names:
        for x in (ALL_BACKENDS if n == "all" else (n,)):
            if x not in out:
                out.append(x)
    return out


@click.group()
def main() -> None:
    """EHRI-CATE: evaluate subject-indexing backends against the EHRI Terms vocab."""
    load_dotenv()


# -- shared options ------------------------------------------------------------


def _corpus_option(f):
    return click.option(
        "--corpus", "corpus_path",
        default="data/manual-descriptions-labelled-with-ehri-terms.json",
        show_default=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )(f)


def _vocab_option(f):
    return click.option(
        "--vocab", "vocab_path",
        default="data/ehri_terms.ttl", show_default=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )(f)


def _split_options(f):
    f = click.option("--seed", default=0, show_default=True, type=int,
                     help="Split + sampling seed.")(f)
    f = click.option("--test-size", default=0.30, show_default=True, type=float,
                     help="Held-out test fraction for the stratified split.")(f)
    return f


# -- train-baselines -----------------------------------------------------------


@main.command(name="train-baselines")
@_corpus_option
@_vocab_option
@_split_options
@click.option("--annif", "annif_names", multiple=True,
              type=click.Choice([*ALL_BACKENDS, "all"]), default=("all",),
              show_default=True, help="Which baselines to train. Repeat or use 'all'.")
@click.option("--annif-workspace", default="data/annif", show_default=True,
              type=click.Path(file_okay=False, path_type=Path))
def train_baselines(
    corpus_path: Path, vocab_path: Path, seed: int, test_size: float,
    annif_names: tuple[str, ...], annif_workspace: Path,
) -> None:
    """Train the supervised Annif baselines on the train split (held-out test untouched)."""
    docs = load_corpus(corpus_path)
    train, test = stratified_split(docs, test_size=test_size, seed=seed)
    click.echo(f"Corpus {len(docs)} → train {len(train)} / test {len(test)} "
               f"(stratified, seed={seed})", err=True)

    names = _expand_annif(annif_names)
    ws = AnnifWorkspace(annif_workspace, vocab_path)
    click.echo(f"Training {names} into {annif_workspace} ...", err=True)
    summary = ws.train(names, train, echo=lambda m: click.echo(m, err=True))
    click.echo(f"Done. Trained {summary['trained']} on {summary['n_train_docs']} docs.", err=True)


# -- evaluate ------------------------------------------------------------------


@main.command()
@_corpus_option
@_vocab_option
@_split_options
@click.option("--model", "models", multiple=True,
              help="LLM4SSH model name. Repeat for multiple. If neither --model nor "
                   "--annif is given, defaults to the three LLMs.")
@click.option("--annif", "annif_names", multiple=True,
              type=click.Choice([*ALL_BACKENDS, "all"]),
              help="Annif baseline(s) to evaluate. Repeat or use 'all'.")
@click.option("--annif-workspace", default="data/annif", show_default=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--retrain", is_flag=True, default=False,
              help="Force retraining the requested Annif baselines even if a model exists.")
@click.option("-n", "--sample-size", default=50, show_default=True, type=int,
              help="Number of test-split docs to evaluate on (sampled, seeded).")
@click.option("-k", "--top-k", default=5, show_default=True, type=int,
              help="Top-k predictions to keep when scoring (paper-aligned default: 5).")
@click.option("--candidates", type=click.Choice(["in-use", "full"]), default="in-use",
              show_default=True,
              help="LLM candidate menu: 'in-use' = labels seen in the corpus; 'full' = all vocab.")
@click.option("--prompt-template", "prompt_template_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="TOML file with 'system' and/or 'user' keys overriding the LLM prompt. "
                   "The 'user' template must contain {text} and {candidates}. "
                   "Affects LLM backends only; Annif baselines ignore it.")
@click.option("--workers", default=4, show_default=True, type=int,
              help="Concurrent LLM API calls. Also bounded by --rpm when that is set. "
                   "(Annif runs sequentially.)")
@click.option("--rpm", default=0, show_default=True, type=int,
              help="Global LLM requests-per-minute cap, shared across all models. "
                   "0 (default) disables pacing; set e.g. 30 if your LLM4SSH key is rate-limited.")
@click.option("--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              help="Where to write per-doc JSON results (default: results/<timestamp>.json).")
def evaluate(
    corpus_path: Path, vocab_path: Path, seed: int, test_size: float,
    models: tuple[str, ...], annif_names: tuple[str, ...], annif_workspace: Path,
    retrain: bool, sample_size: int, top_k: int, candidates: str,
    prompt_template_path: Path | None, workers: int, rpm: int, output_path: Path | None,
) -> None:
    """Evaluate LLM and/or Annif backends on a shared held-out test sample."""
    annif_backends = _expand_annif(annif_names)
    # Default to the LLM trio only when nothing at all was requested.
    if not models and not annif_backends:
        models = DEFAULT_MODELS

    # Resolve the LLM prompt (default unless a template file is given) and fingerprint
    # it, so two runs of the same model with different prompts are distinguishable in
    # the results JSON.
    prompt = load_prompt_template(prompt_template_path) if prompt_template_path else PromptTemplate()
    prompt_hash = hashlib.sha256(
        (prompt.system + "\x00" + prompt.user).encode("utf-8")
    ).hexdigest()[:12]
    if prompt_template_path:
        click.echo(f"Using prompt template {prompt_template_path} (hash {prompt_hash})", err=True)

    click.echo(f"Loading corpus from {corpus_path}...", err=True)
    docs = load_corpus(corpus_path)
    click.echo(f"  {len(docs)} docs", err=True)

    click.echo(f"Loading vocab from {vocab_path}...", err=True)
    vocab = Vocab.from_turtle(vocab_path)
    click.echo(f"  {len(vocab)} concepts", err=True)

    # Stratified train/test split — supervised baselines train on train, everyone
    # is scored on a sample of test.
    train, test = stratified_split(docs, test_size=test_size, seed=seed)
    click.echo(f"Split: train {len(train)} / test {len(test)} (stratified, seed={seed})", err=True)

    in_use_uris = sorted({uri for d in docs for uri in d.gold_uris})
    candidate_uris = in_use_uris if candidates == "in-use" else sorted(vocab.concepts.keys())

    rng = random.Random(seed)
    sample = rng.sample(test, min(sample_size, len(test)))
    click.echo(f"Sampled {len(sample)} test docs (seed={seed})", err=True)

    rate_limiter: RateLimiter | NullRateLimiter = (
        RateLimiter(max_calls=rpm, period_s=60.0) if rpm > 0 else NullRateLimiter()
    )

    # Prepare Annif workspace + train if needed.
    ws: AnnifWorkspace | None = None
    if annif_backends:
        ws = AnnifWorkspace(annif_workspace, vocab_path)
        need_train = retrain or any(not ws.is_trained(n) for n in annif_backends)
        if need_train:
            click.echo(f"Training Annif baselines {annif_backends} (train split: "
                       f"{len(train)} docs)...", err=True)
            ws.train(annif_backends, train, echo=lambda m: click.echo(m, err=True))
        else:
            click.echo(f"Reusing trained Annif models in {annif_workspace}", err=True)

    if output_path is None:
        output_path = Path("results") / f"eval-{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict = {
        "config": {
            "corpus": str(corpus_path),
            "vocab": str(vocab_path),
            "split": "stratified",
            "test_size": test_size,
            "n_train": len(train),
            "n_test": len(test),
            "sample_size": len(sample),
            "seed": seed,
            "top_k": top_k,
            "candidates": candidates,
            "n_candidate_labels": len(candidate_uris),
            "models": list(models),
            "annif": annif_backends,
            "prompt_template": str(prompt_template_path) if prompt_template_path else None,
            "prompt_hash": prompt_hash,
        },
        "per_backend": {},
    }

    # Build the (label, kind, backend, workers) run plan.
    run_plan: list[tuple[str, str, SuggestBackend, int]] = []
    for model in models:
        be = load_backend_from_env(
            model=model, vocab=vocab, candidate_uris=candidate_uris, rate_limiter=rate_limiter,
            prompt=prompt,
        )
        run_plan.append((model, "llm", be, workers))
    if ws is not None:
        for name in annif_backends:
            run_plan.append((name, "annif", ws.backend(name), 1))  # Annif: sequential

    # Debug: how big is the LLM prompt before any doc text is added? The candidate
    # menu dominates, so this is the floor every request pays — useful for spotting
    # a prompt that's brushing against a model's context window.
    llm_backends = [(lbl, be) for (lbl, kind, be, _) in run_plan if kind == "llm"]
    if llm_backends:
        lbl0, be0 = llm_backends[0]
        size = measure_messages(lbl0, be0.build_messages(""))
        click.echo(
            f"[debug] LLM prompt static overhead (system + {len(candidate_uris)}-term menu, "
            f"no doc text): ~{size.tokens} tokens / {size.chars} chars "
            f"(tokens via {size.token_method}); each doc's text adds on top.",
            err=True,
        )

    for label, kind, backend, be_workers in run_plan:
        click.echo(f"\n=== Running {label} ({kind}) ===", err=True)
        t0 = time.monotonic()
        agg, per_doc = _run_one_backend(label, backend, sample, vocab, top_k, be_workers)
        elapsed = time.monotonic() - t0

        errors = [r for r in per_doc if r.get("error")]
        all_results["per_backend"][label] = {
            "kind": kind,
            "wall_time_s": elapsed,
            "n_errors": len(errors),
            "scores": {
                "doc_avg_f1": agg.doc_avg_f1,
                "micro_f1": agg.micro_f1,
                "weighted_macro_f1": agg.weighted_macro_f1,
                "hierarchical_doc_avg_f1": agg.hierarchical_doc_avg_f1,
            },
            "per_doc": per_doc,
        }
        click.echo(f"  wall {elapsed:.1f}s, errors {len(errors)}", err=True)
        click.echo(f"  F1 doc-avg@{top_k}: {agg.doc_avg_f1:.4f}  micro: {agg.micro_f1:.4f}  "
                   f"w-macro: {agg.weighted_macro_f1:.4f}  HIER: {agg.hierarchical_doc_avg_f1:.4f}",
                   err=True)

    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    click.echo(f"\nWrote per-doc results to {output_path}", err=True)

    # Compact, pipeable summary table on stdout.
    click.echo("\nbackend\tkind\tdoc_avg_F1\tmicro_F1\tweighted_macro_F1\thier_F1\twall_s")
    for label, res in all_results["per_backend"].items():
        s = res["scores"]
        click.echo(
            f"{label}\t{res['kind']}\t{s['doc_avg_f1']:.4f}\t{s['micro_f1']:.4f}\t"
            f"{s['weighted_macro_f1']:.4f}\t{s['hierarchical_doc_avg_f1']:.4f}\t"
            f"{res['wall_time_s']:.1f}"
        )


# -- label ---------------------------------------------------------------------


@main.command()
@_corpus_option
@_vocab_option
@click.option("--model", required=True, help="LLM4SSH model name (single model).")
@click.option("--candidates", type=click.Choice(["in-use", "full"]), default="in-use",
              show_default=True,
              help="Candidate menu: 'in-use' = labels seen in the corpus (needs --corpus); "
                   "'full' = all vocab.")
@click.option("--prompt-template", "prompt_template_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="TOML file with 'system' and/or 'user' keys overriding the LLM prompt. "
                   "The 'user' template must contain {text} and {candidates}.")
@click.option("-k", "--top-k", default=5, show_default=True, type=int,
              help="How many suggestions to print.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit JSON instead of a tab-separated table.")
@click.argument("input_file", type=click.File("r"), default="-")
def label(
    corpus_path: Path, vocab_path: Path, model: str, candidates: str,
    prompt_template_path: Path | None, top_k: int, as_json: bool, input_file,
) -> None:
    """Suggest EHRI Terms for an ad-hoc text read from stdin (or a file).

    Examples:
        cat notes.txt | cate label --model DeepSeek-V3.1-vLLM
        cate label --model DeepSeek-V3.1-vLLM --json description.txt

    Unlike `evaluate`, the input is fed to the model as a single blob; it is NOT
    parsed into the ISAD(G) fields the labelled corpus uses. No scoring is done.
    """
    text = input_file.read()
    if not text.strip():
        raise click.ClickException("No input text (stdin/file was empty).")

    vocab = Vocab.from_turtle(vocab_path)
    if candidates == "in-use":
        docs = load_corpus(corpus_path)
        candidate_uris = sorted({uri for d in docs for uri in d.gold_uris})
    else:
        candidate_uris = sorted(vocab.concepts.keys())

    prompt = load_prompt_template(prompt_template_path) if prompt_template_path else PromptTemplate()
    backend = load_backend_from_env(
        model=model, vocab=vocab, candidate_uris=candidate_uris, prompt=prompt,
    )

    # Debug: prompt size so truncation is visible. Report the full derived prompt
    # and, separately, the candidate-menu portion (the dominant, fixed part).
    messages = backend.build_messages(text)
    full = measure_messages(model, messages)
    menu = measure_messages(model, [{"role": "user", "content": backend.candidate_menu}])
    click.echo(
        f"[debug] candidates={candidates} ({len(candidate_uris)} terms)  "
        f"menu ~{menu.tokens} tok / {menu.chars} chars  "
        f"full prompt ~{full.tokens} tok / {full.chars} chars "
        f"(tokens via {full.token_method}; input text {len(text)} chars)",
        err=True,
    )

    preds = backend.suggest(text)
    labels = vocab.candidate_labels("en")
    top = preds[:top_k]

    if as_json:
        out = [
            {"rank": i, "uri": u, "label": labels.get(u, u), "score": round(s, 4)}
            for i, (u, s) in enumerate(top, start=1)
        ]
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
    elif not top:
        click.echo("(no suggestions)", err=True)
    else:
        for i, (u, s) in enumerate(top, start=1):
            click.echo(f"{i}\t{s:.3f}\t{labels.get(u, u)}\t{u}")


if __name__ == "__main__":
    main()
