"""Graph traversal: Personalized PageRank over the built graph (graph.py).

For a question:
  1. seeds   the top retrieved text units (by retrieval rank) and the concept nodes whose
             names or aliases appear in the question
  2. spread  Personalized PageRank from the seeds: x = alpha * seeds + (1 - alpha) * P^T x,
             where P moves along edges in proportion to their weight (mentions and adjacent
             edges 1, related edges `related_w`). Every node splits its mass over its edges,
             so a hub linked to 240 units passes little to each one: hubs are damped by the
             walk itself rather than removed from the graph.
  3. rank    text units by their score; a reranker can then reorder the top ones
"""

from collections import defaultdict
import re

import numpy as np

from ragsplit.merge import normalize

EDGE_W = {"mentions": 1.0, "adjacent": 1.0}
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "was", "what", "which",
        "who", "when", "where", "how", "why", "did", "does", "do", "many", "much"}


class GraphTraverser:
    def __init__(self, graph: dict, unit_ids: list[str], retriever, alpha: float = 0.5,
                 related_w: float = 0.3, concept_w: float = 0.5, n_seeds: int = 10,
                 n_iter: int = 30, max_alias_words: int = 6):
        from scipy.sparse import csr_matrix

        self.retriever = retriever
        self.alpha, self.concept_w, self.n_seeds, self.n_iter = alpha, concept_w, n_seeds, n_iter
        self.max_alias_words = max_alias_words
        index = {uid: i for i, uid in enumerate(unit_ids)}
        missing = [t["id"] for t in graph["text_nodes"] if t["id"] not in index]
        if missing or len(graph["text_nodes"]) != len(unit_ids):
            raise ValueError(f"graph text nodes do not match the units ({len(missing)} missing); "
                             "rebuild the graph with the same questions and k")
        self.n_text = len(unit_ids)
        for c in graph["concepts"]:
            index[c["id"]] = len(index)
        n = len(index)
        rows, cols, vals = [], [], []
        for e in graph["edges"]:
            w = related_w if e["type"] == "related" else EDGE_W[e["type"]]
            if w <= 0:
                continue
            a, b = index[e["a"]], index[e["b"]]
            rows += [a, b]
            cols += [b, a]
            vals += [w, w]
        W = csr_matrix((vals, (rows, cols)), shape=(n, n))
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        self.PT = (csr_matrix(W.multiply(1.0 / deg[:, None]))).T.tocsr()  # transpose of row-stochastic P
        self.alias = defaultdict(set)
        for c in graph["concepts"]:
            for name in c["aliases"]:
                n_ = normalize(name)
                if n_ and n_ not in STOP and len(n_) > 1 and len(n_.split()) <= max_alias_words:
                    self.alias[n_].add(index[c["id"]])
        self.n_nodes = n

    def question_concepts(self, question: str) -> set[int]:
        words = normalize(question).split()
        found = set()
        for size in range(1, self.max_alias_words + 1):
            for i in range(len(words) - size + 1):
                found |= self.alias.get(" ".join(words[i:i + size]), set())
        return found

    def scores(self, question: str, ranked: list[int]) -> np.ndarray:
        seeds = np.zeros(self.n_nodes)
        top = ranked[: self.n_seeds]
        for r, j in enumerate(top):
            seeds[j] += 1.0 / (r + 1)
        if seeds.sum() > 0:
            seeds /= seeds.sum()
        concepts = self.question_concepts(question)
        if concepts and self.concept_w > 0:
            c = np.zeros(self.n_nodes)
            c[list(concepts)] = 1.0 / len(concepts)
            seeds = (1 - self.concept_w) * seeds + self.concept_w * c
        x = seeds.copy()
        for _ in range(self.n_iter):
            x = self.alpha * seeds + (1 - self.alpha) * (self.PT @ x)
        return x[: self.n_text]

    def search(self, queries: list[str], prior=None, **kw) -> list[list[int]]:
        if prior is not None:
            raise ValueError("graph traversal does not support a prior")
        out = []
        for q, ranked in zip(queries, self.retriever.search(queries)):
            s = self.scores(q, ranked)
            out.append([int(j) for j in np.argsort(-s)[: self.retriever.depth] if s[j] > 0])
        return out

    def encoder(self):
        return self.retriever.encoder()

    @property
    def depth(self) -> int:
        return self.retriever.depth


class HopExpander:
    """Two-hop retrieval over the graph: hop 1 = reranked retrieval; hop 2 = units in other
    paragraphs that share a concept with a hop-1 seed, scored by the reranker against the
    question plus the seed's text (a second hop shares the bridge entity with the seed, not
    words with the question).

    Concept links are weighted 1 / log(2 + paragraphs linked) and concepts linking more than
    `max_paragraphs` paragraphs are skipped, so hubs carry little or no weight.
    mode "merge": one list of hop-1 and hop-2 units ordered by reranker score.
    mode "interleave": seed 1, its best hop-2 unit, seed 2, its best hop-2 unit, ...
    """

    def __init__(self, graph: dict, units: list, retriever, reranker, n_seeds: int = 3,
                 n_candidates: int = 20, mode: str = "merge", max_paragraphs: int = 50):
        self.retriever, self.reranker = retriever, reranker
        self.units, self.n_seeds, self.n_candidates, self.mode = units, n_seeds, n_candidates, mode
        index = {u.id: i for i, u in enumerate(units)}
        if len(graph["text_nodes"]) != len(units) or any(t["id"] not in index for t in graph["text_nodes"]):
            raise ValueError("graph text nodes do not match the units; rebuild the graph")
        self.unit_concepts = defaultdict(set)
        concept_units = defaultdict(set)
        for e in graph["edges"]:
            if e["type"] == "mentions":
                self.unit_concepts[index[e["a"]]].add(e["b"])
                concept_units[e["b"]].add(index[e["a"]])
        self.concept_units = {}
        self.weight = {}
        for c, us in concept_units.items():
            n_par = len({units[u].paragraph_id for u in us})
            if n_par <= max_paragraphs:
                self.concept_units[c] = us
                self.weight[c] = 1.0 / np.log(2 + n_par)

    def hop2(self, seed: int, exclude_pids: set) -> list[int]:
        score = defaultdict(float)
        for c in self.unit_concepts.get(seed, ()):
            for u in self.concept_units.get(c, ()):
                if self.units[u].paragraph_id not in exclude_pids:
                    score[u] += self.weight[c]
        return sorted(score, key=lambda u: -score[u])[: self.n_candidates]

    def search_one(self, question: str, ranked: list[int]) -> list[int]:
        head = ranked[: self.reranker.top_n]
        s1 = self.reranker.model.predict([(question, self.units[j].search_text) for j in head],
                                         batch_size=64, show_progress_bar=False)
        order = sorted(range(len(head)), key=lambda i: -float(s1[i]))
        hop1 = [(float(s1[i]), head[i]) for i in order]
        seeds = [j for _, j in hop1[: self.n_seeds]]
        seed_pids = {self.units[j].paragraph_id for j in seeds}
        hop2_best = []
        for seed in seeds:
            cands = self.hop2(seed, seed_pids)
            if not cands:
                hop2_best.append([])
                continue
            context = question + " " + self.units[seed].text
            s2 = self.reranker.model.predict([(context, self.units[u].search_text) for u in cands],
                                             batch_size=64, show_progress_bar=False)
            hop2_best.append(sorted(((float(s), u) for s, u in zip(s2, cands)), reverse=True))
        if self.mode == "interleave":
            out = []
            for seed, h in zip(seeds, hop2_best):
                out.append(seed)
                if h:
                    out.append(h[0][1])
            out += [j for _, j in hop1[self.n_seeds:]]
        else:  # merge by reranker score
            pool = hop1 + [x for h in hop2_best for x in h[:2]]
            out = [j for _, j in sorted(pool, reverse=True)]
        seen, final = set(), []
        for j in out + ranked:
            if j not in seen:
                seen.add(j)
                final.append(j)
        return final[: self.retriever.depth]

    def search(self, queries: list[str], prior=None, **kw) -> list[list[int]]:
        if prior is not None:
            raise ValueError("hop expansion does not support a prior")
        return [self.search_one(q, r) for q, r in zip(queries, self.retriever.search(queries))]

    def encoder(self):
        return self.retriever.encoder()

    @property
    def depth(self) -> int:
        return self.retriever.depth


class BeamTraverser:
    """Beam search over paths of units, for breadth and depth.

    A path starts at one of the top `beam` reranked units. Each step extends a path's last
    unit by one move:
      adjacent  the previous / next unit of the same text (reassembles context that cracking
                separated; counts toward depth like any other step)
      concept   a unit in another passage sharing a concept with the last unit (graph
                `mentions` edges; weight 1 / log(2 + paragraphs linked), hubs above
                `max_paragraphs` skipped)
    Candidates are scored by the reranker against the question plus the path's text, and the
    best `beam` paths survive each step, up to `depth` steps. A path stops growing when its
    best next step scores below `min_step` (reranker logit). Output: units of the final paths
    in path-score order, each path in its own order, then the rest of the reranked list, so a
    tight budget keeps the start of the best chains and a generous one lets them run deeper.
    With `stop="flat"`, a step is only taken if it scores at least as high as the best reranked
    unit that is not yet in a path (what plain retrieval would add next): traversal then spends
    budget only where it beats the reranker's next pick. `self.step_log` counts moves by type.
    """

    def __init__(self, units: list, retriever, reranker, graph: dict | None = None, beam: int = 3,
                 depth: int = 2, n_candidates: int = 10, min_step: float = 0.0,
                 max_paragraphs: int = 50, cross_paragraph_adjacency: bool = False,
                 path_chars: int = 1200, stop: str = "fixed"):
        from ragsplit.run import doc_of

        self.units, self.retriever, self.reranker = units, retriever, reranker
        self.beam, self.max_depth, self.n_candidates = beam, depth, n_candidates
        self.min_step, self.path_chars, self.stop = min_step, path_chars, stop
        self.step_log = defaultdict(int)
        # adjacency: units in text order within a paragraph (and across consecutive paragraphs of
        # one document when the source is continuous text, e.g. syllabus chunks)
        order = sorted(range(len(units)), key=lambda i: (units[i].paragraph_id,
                                                         (units[i].span or (units[i].sent_lo, 0))[0]))
        self.prev, self.next = {}, {}
        for a, b in zip(order, order[1:]):
            pa, pb = units[a].paragraph_id, units[b].paragraph_id
            if pa == pb or (cross_paragraph_adjacency and doc_of(pa) == doc_of(pb)):
                self.next[a], self.prev[b] = b, a
        self.unit_concepts, self.concept_units, self.weight = defaultdict(set), {}, {}
        if graph is not None:
            index = {u.id: i for i, u in enumerate(units)}
            matched = sum(t["id"] in index for t in graph["text_nodes"])
            if matched < 0.9 * len(graph["text_nodes"]):
                raise ValueError(f"graph text nodes match only {matched}/{len(graph['text_nodes'])} units; "
                                 "rebuild the graph for these units")
            cu = defaultdict(set)
            for e in graph["edges"]:
                if e["type"] == "mentions" and e["a"] in index:
                    self.unit_concepts[index[e["a"]]].add(e["b"])
                    cu[e["b"]].add(index[e["a"]])
            for c, us in cu.items():
                n_par = len({units[u].paragraph_id for u in us})
                if n_par <= max_paragraphs:
                    self.concept_units[c] = us
                    self.weight[c] = 1.0 / np.log(2 + n_par)

    def moves(self, last: int, path: list[int]) -> list[tuple[int, str]]:
        out = [(n, "adjacent") for n in (self.prev.get(last), self.next.get(last)) if n is not None]
        score = defaultdict(float)
        pids = {self.units[j].paragraph_id for j in path}
        for c in self.unit_concepts.get(last, ()):
            for u in self.concept_units.get(c, ()):
                if self.units[u].paragraph_id not in pids:
                    score[u] += self.weight[c]
        out += [(u, "concept") for u in sorted(score, key=lambda u: -score[u])[: self.n_candidates]]
        return [(u, kind) for u, kind in out if u not in path]

    def path_text(self, question: str, path: list[int]) -> str:
        text = " ".join(self.units[j].text for j in path)
        return f"{question} {text[-self.path_chars:]}"

    def search_one(self, question: str, ranked: list[int]) -> list[int]:
        head = ranked[: self.reranker.top_n]
        s0 = self.reranker.model.predict([(question, self.units[j].search_text) for j in head],
                                         batch_size=64, show_progress_bar=False)
        hop0 = sorted(zip((float(x) for x in s0), head), reverse=True)
        beam = [(s, [j], False) for s, j in hop0[: self.beam]]   # (score, path, finished)
        for _ in range(self.max_depth):
            grown, pairs, owners = [], [], []
            for score, path, done in beam:
                if done:
                    grown.append((score, path, True))
                    continue
                cands = self.moves(path[-1], path)
                if not cands:
                    grown.append((score, path, True))
                    continue
                q = self.path_text(question, path)
                for u, kind in cands:
                    pairs.append((q, self.units[u].search_text))
                    owners.append((score, path, u, kind))
            if pairs:
                if self.stop == "flat":  # the best reranked unit not yet in any path
                    in_paths = {j for _, path, _ in beam for j in path}
                    floor = next((s for s, j in hop0 if j not in in_paths), float("-inf"))
                else:
                    floor = self.min_step
                sc = self.reranker.model.predict(pairs, batch_size=64, show_progress_bar=False)
                best = {}
                for (score, path, u, kind), s in zip(owners, sc):
                    key = tuple(path)
                    best.setdefault(key, []).append((float(s), u, kind, score))
                for key, opts in best.items():
                    opts.sort(reverse=True)
                    s_best = opts[0][0]
                    if s_best < floor:  # nothing worth adding: the path is complete
                        grown.append((opts[0][3], list(key), True))
                        continue
                    for s, u, kind, score in opts[: self.beam]:
                        if s >= floor:
                            grown.append((score + s, list(key) + [u], False, kind))
            beam = []
            for g in sorted(grown, key=lambda g: -g[0])[: self.beam]:
                if len(g) == 4:
                    self.step_log[g[3]] += 1
                beam.append(g[:3])
            if all(done for _, _, done in beam):
                break
        seen, final = set(), []
        for _, path, _ in sorted(beam, key=lambda b: -b[0]):
            for j in path:
                if j not in seen:
                    seen.add(j)
                    final.append(j)
        for _, j in hop0:
            if j not in seen:
                seen.add(j)
                final.append(j)
        for j in ranked:
            if j not in seen:
                seen.add(j)
                final.append(j)
        return final[: self.retriever.depth]

    def search(self, queries: list[str], prior=None, **kw) -> list[list[int]]:
        if prior is not None:
            raise ValueError("beam traversal does not support a prior")
        return [self.search_one(q, r) for q, r in zip(queries, self.retriever.search(queries))]

    def encoder(self):
        return self.retriever.encoder()

    @property
    def depth(self) -> int:  # retrieval depth, as the wrapped retriever's (not the beam depth)
        return self.retriever.depth
