from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import SKOS


@dataclass(frozen=True)
class Concept:
    uri: str
    pref_labels: dict[str, str]   # lang -> label
    alt_labels: dict[str, list[str]] = field(default_factory=dict)

    def label(self, lang: str = "en") -> str:
        return (
            self.pref_labels.get(lang)
            or self.pref_labels.get("en")
            or next(iter(self.pref_labels.values()), self.uri)
        )


class Vocab:
    """In-memory SKOS vocab: concepts + broader/narrower DAG.

    Hierarchy note: the EHRI Terms TTL has asymmetric edges (898 broader vs 363
    narrower triples). We treat skos:broader as authoritative and derive
    narrower from its inverse, so the graph is symmetric here even when the
    source file isn't.
    """

    def __init__(self, concepts: dict[str, Concept], broader: dict[str, set[str]]):
        self.concepts = concepts
        self._broader = {uri: frozenset(parents) for uri, parents in broader.items()}
        narrower: dict[str, set[str]] = defaultdict(set)
        for child, parents in broader.items():
            for parent in parents:
                narrower[parent].add(child)
        self._narrower = {uri: frozenset(kids) for uri, kids in narrower.items()}

    @classmethod
    def from_turtle(cls, path: str | Path) -> "Vocab":
        g = Graph()
        g.parse(str(path), format="turtle")

        concept_uris = {str(s) for s in g.subjects(predicate=None, object=SKOS.Concept)}
        for s, _, _ in g.triples((None, None, SKOS.Concept)):
            concept_uris.add(str(s))

        pref: dict[str, dict[str, str]] = defaultdict(dict)
        alt: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        broader: dict[str, set[str]] = defaultdict(set)

        for s, _, o in g.triples((None, SKOS.prefLabel, None)):
            pref[str(s)][o.language or "und"] = str(o)
        for s, _, o in g.triples((None, SKOS.altLabel, None)):
            alt[str(s)][o.language or "und"].append(str(o))
        for s, _, o in g.triples((None, SKOS.broader, None)):
            broader[str(s)].add(str(o))
        # Treat any narrower triple as evidence of a broader edge in reverse,
        # so we don't lose hierarchy when the source file is asymmetric.
        for s, _, o in g.triples((None, SKOS.narrower, None)):
            broader[str(o)].add(str(s))

        concepts: dict[str, Concept] = {}
        for uri in concept_uris:
            concepts[uri] = Concept(
                uri=uri,
                pref_labels=dict(pref.get(uri, {})),
                alt_labels={k: list(v) for k, v in alt.get(uri, {}).items()},
            )
        return cls(concepts=concepts, broader=broader)

    # -- accessors -----------------------------------------------------------

    def __contains__(self, uri: str) -> bool:
        return uri in self.concepts

    def __len__(self) -> int:
        return len(self.concepts)

    def broader_of(self, uri: str) -> frozenset[str]:
        return self._broader.get(uri, frozenset())

    def narrower_of(self, uri: str) -> frozenset[str]:
        return self._narrower.get(uri, frozenset())

    def ancestors(self, uri: str) -> set[str]:
        """All transitive broader concepts of uri (uri itself excluded)."""
        return self._walk(uri, self._broader)

    def descendants(self, uri: str) -> set[str]:
        return self._walk(uri, self._narrower)

    def _walk(self, start: str, edges: dict[str, frozenset[str]]) -> set[str]:
        seen: set[str] = set()
        queue: deque[str] = deque(edges.get(start, frozenset()))
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for nb in edges.get(node, frozenset()):
                if nb not in seen:
                    queue.append(nb)
        return seen

    def distance(self, a: str, b: str) -> int | None:
        """Shortest undirected hop count between two concepts; None if disconnected.

        Treats the SKOS graph as undirected for distance — a broader and a narrower
        hop both count as 1. Useful for hierarchy-aware scoring.
        """
        if a == b:
            return 0
        if a not in self.concepts or b not in self.concepts:
            return None
        seen = {a}
        frontier: deque[tuple[str, int]] = deque([(a, 0)])
        while frontier:
            node, d = frontier.popleft()
            neighbours = self._broader.get(node, frozenset()) | self._narrower.get(node, frozenset())
            for nb in neighbours:
                if nb == b:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, d + 1))
        return None

    def candidate_labels(self, lang: str = "en") -> dict[str, str]:
        """URI -> preferred label in the given language (falling back to en, then any)."""
        return {uri: c.label(lang) for uri, c in self.concepts.items()}
