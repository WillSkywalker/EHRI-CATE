"""Regenerate src/ehri_cate/resources/in_use_terms.txt from the labelled corpus.

The web API ships without data/ (gitignored), so it reads its "in-use" candidate
menu from this committed snapshot instead of the corpus. The order must stay the
sorted one `cate evaluate`/`cate label` use, because the LLM sees the menu as a
numbered list: same order, same prompt as the evaluated one.

Usage: uv run python scripts/export_in_use_terms.py [corpus.json]
Re-run (and re-copy data/ehri_terms.ttl into resources/) when the corpus or the
vocab snapshot changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ehri_cate.corpus import load_corpus

DEFAULT_CORPUS = Path("data/manual-descriptions-labelled-with-ehri-terms.json")
OUT = Path("src/ehri_cate/resources/in_use_terms.txt")


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    uris = sorted({uri for d in load_corpus(corpus) for uri in d.gold_uris})
    OUT.write_text("\n".join(uris) + "\n", encoding="utf-8")
    print(f"Wrote {len(uris)} URIs to {OUT}")


if __name__ == "__main__":
    main()
