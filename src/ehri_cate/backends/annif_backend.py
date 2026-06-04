"""Annif supervised baselines.

Wraps the five native Annif backends the prior paper (Dermentzi et al. 2025) used
as comparison points — TF-IDF, MLLM (Maui-like Lexical Matching), fastText,
Omikuji/Parabel, and an NN Ensemble of the other four — behind the same
``.suggest(text) -> list[(uri, score)]`` interface as the LLM4SSH backend, so the
evaluation harness can score them with the identical flat + hierarchical scorer.

Unlike the zero-shot LLM backend these are *supervised*: they must be trained on a
labelled corpus first (see ``AnnifWorkspace.train``). Training is driven through the
Annif CLI (the documented path, and it handles the nn_ensemble→sources ordering),
and prediction goes through Annif's in-process Python API so we load each model once
and reuse it across the test set.

Two honest caveats vs. the zero-shot LLMs:
  * Annif has no native heterogeneous-multilingual support; we use the language-
    neutral ``simple`` analyzer over the multilingual corpus (the paper noted the
    same limitation).
  * A supervised model's candidate set is whatever it saw in training, not a fixed
    menu. The scorer is identical (gold URIs vs. top-k predicted URIs), so the
    comparison stays valid.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# short CLI name -> Annif project id
PROJECT_IDS: dict[str, str] = {
    "tfidf": "ehri-tfidf",
    "mllm": "ehri-mllm",
    "fasttext": "ehri-fasttext",
    "omikuji": "ehri-omikuji",
    "nn_ensemble": "ehri-nn-ensemble",
}
ALL_BACKENDS: tuple[str, ...] = tuple(PROJECT_IDS.keys())

# The four base projects the NN ensemble combines (and which must be trained first).
_ENSEMBLE_SOURCES: tuple[str, ...] = ("tfidf", "fasttext", "omikuji", "mllm")

VOCAB_ID = "ehri"

# A predicted (uri, score), matching the LLM4SSH backend's shape.
Prediction = tuple[str, float]


def _annif_executable() -> str:
    """Path to the `annif` console script in the active venv."""
    exe = Path(sys.executable).parent / "annif"
    return str(exe) if exe.exists() else "annif"


def _projects_toml(language: str, limit: int) -> str:
    """Generate an Annif projects.toml for the five baselines, all sharing one vocab.

    Params are sensible, lightly-tuned defaults (not the paper's exact hyper-params,
    which it didn't fully publish); they are easy to revisit later.
    """
    src = ",".join(f"{PROJECT_IDS[s]}:1" for s in _ENSEMBLE_SOURCES)
    return f"""\
[ehri-tfidf]
name = "EHRI TF-IDF"
language = "{language}"
backend = "tfidf"
analyzer = "simple"
vocab = "{VOCAB_ID}"
limit = {limit}

[ehri-mllm]
name = "EHRI MLLM"
language = "{language}"
backend = "mllm"
analyzer = "simple"
vocab = "{VOCAB_ID}"
limit = {limit}

[ehri-fasttext]
name = "EHRI fastText"
language = "{language}"
backend = "fasttext"
analyzer = "simple"
vocab = "{VOCAB_ID}"
limit = {limit}
dim = 100
lr = 0.25
epoch = 10
loss = "hs"

[ehri-omikuji]
name = "EHRI Omikuji Parabel"
language = "{language}"
backend = "omikuji"
analyzer = "simple"
vocab = "{VOCAB_ID}"
limit = {limit}

[ehri-nn-ensemble]
name = "EHRI NN Ensemble"
language = "{language}"
backend = "nn_ensemble"
vocab = "{VOCAB_ID}"
limit = {limit}
sources = "{src}"
nodes = 100
dropout_rate = 0.2
epochs = 10
"""


def _clean_text(text: str) -> str:
    """Collapse whitespace so a document fits on a single TSV line."""
    return " ".join(text.split())


class AnnifWorkspace:
    """A self-contained Annif project directory (config + datadir + train corpus)."""

    def __init__(
        self,
        workspace_dir: str | Path,
        vocab_ttl: str | Path,
        language: str = "en",
        limit: int = 100,
    ):
        self.dir = Path(workspace_dir)
        self.vocab_ttl = Path(vocab_ttl)
        self.language = language
        self.limit = limit
        self.datadir = self.dir / "data"
        self.projects_config = self.dir / "projects.toml"
        self.train_tsv = self.dir / "train.tsv"
        self._registry = None

    # -- setup ---------------------------------------------------------------

    def write_config(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.datadir.mkdir(parents=True, exist_ok=True)
        self.projects_config.write_text(_projects_toml(self.language, self.limit))

    def write_train_corpus(self, train_docs) -> int:
        """Write the train split as an Annif full-text TSV: `text<TAB><uri> <uri>`.

        Skips docs with empty text or no gold labels (nothing to learn from).
        Returns the number of lines written.
        """
        n = 0
        with self.train_tsv.open("w", encoding="utf-8") as f:
            for doc in train_docs:
                text = _clean_text(doc.text)
                if not text or not doc.gold_uris:
                    continue
                uris = " ".join(f"<{u}>" for u in doc.gold_uris)
                f.write(f"{text}\t{uris}\n")
                n += 1
        return n

    # -- training (via the Annif CLI) ----------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["ANNIF_DATADIR"] = str(self.datadir)
        return env

    def _run(self, args: list[str], echo) -> None:
        cmd = [_annif_executable(), *args, "--projects", str(self.projects_config)]
        echo(f"  $ annif {' '.join(args)}")
        proc = subprocess.run(
            cmd, env=self._env(), capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"annif {args[0]} failed (exit {proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )

    def load_vocab(self, echo=lambda _: None) -> None:
        # load-vocab takes the vocab id (shared across all projects); --force
        # makes it idempotent across re-runs.
        self._run(["load-vocab", "--force", VOCAB_ID, str(self.vocab_ttl)], echo)

    def train(self, backends: list[str], train_docs, echo=lambda _: None) -> dict:
        """Train the requested baselines (training base projects first if the NN
        ensemble is requested). Returns a small summary dict."""
        self.write_config()
        n_train = self.write_train_corpus(train_docs)
        self.load_vocab(echo)

        # Expand: nn_ensemble needs its source projects trained first.
        to_train: list[str] = []
        for name in backends:
            if name == "nn_ensemble":
                for src in _ENSEMBLE_SOURCES:
                    if src not in to_train:
                        to_train.append(src)
            elif name not in to_train:
                to_train.append(name)
        # nn_ensemble last, if requested
        if "nn_ensemble" in backends:
            to_train.append("nn_ensemble")

        for name in to_train:
            echo(f"Training {name} ...")
            self._run(["train", PROJECT_IDS[name], str(self.train_tsv)], echo)

        # Fresh registry so the just-trained models are picked up.
        self._registry = None
        return {"n_train_docs": n_train, "trained": to_train}

    # -- prediction (in-process) ---------------------------------------------

    def registry(self):
        if self._registry is None:
            from annif.registry import AnnifRegistry

            self._registry = AnnifRegistry(
                str(self.projects_config), str(self.datadir), init_projects=False
            )
        return self._registry

    def backend(self, name: str) -> "AnnifBackend":
        if name not in PROJECT_IDS:
            raise ValueError(f"unknown Annif backend {name!r}; choose from {ALL_BACKENDS}")
        project = self.registry().get_project(PROJECT_IDS[name])
        return AnnifBackend(name=name, project=project, limit=self.limit)

    def is_trained(self, name: str) -> bool:
        """Best-effort check that a project's model artifacts exist on disk."""
        proj_dir = self.datadir / "projects" / PROJECT_IDS[name]
        return proj_dir.is_dir() and any(proj_dir.iterdir())


class AnnifBackend:
    """Prediction wrapper for one trained Annif project."""

    def __init__(self, name: str, project, limit: int = 100):
        self.name = name
        self.project = project
        self.limit = limit
        project.initialize()
        self._subjects = project.subjects

    def suggest(self, text: str) -> list[Prediction]:
        from annif.corpus import Document

        text = _clean_text(text)
        if not text:
            return []
        batch = self.project.suggest([Document(text=text)])
        result = batch[0]  # one document in, one SuggestionResult out
        preds: list[Prediction] = []
        for suggestion in result:
            uri = self._subjects[suggestion.subject_id].uri
            preds.append((uri, float(suggestion.score)))
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds
