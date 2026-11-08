"""Serve GNN link predictions with numpy only (the app does not need torch).

The embeddings and decoder weights come from gnn.export_predictor.
"""

import numpy as np


class Predictor:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.chem_ids = list(z["chem_ids"])
        self.prot_ids = list(z["prot_ids"])
        self.ci = {c: i for i, c in enumerate(self.chem_ids)}
        self.pi = {p: i for i, p in enumerate(self.prot_ids)}
        self.hc, self.hp = z["h_chem"].astype(np.float32), z["h_prot"].astype(np.float32)
        self.w1, self.b1, self.w2, self.b2 = (z[k].astype(np.float32) for k in ("w1", "b1", "w2", "b2"))

    def _decode(self, a, b):
        x = np.concatenate([a * b, np.abs(a - b)], axis=-1)
        h = np.maximum(x @ self.w1.T + self.b1, 0)
        logit = (h @ self.w2.T + self.b2)[..., 0]
        return 1 / (1 + np.exp(-logit))

    def score(self, chem, prot):
        if chem not in self.ci or prot not in self.pi:
            return None
        return float(self._decode(self.hc[self.ci[chem]][None], self.hp[self.pi[prot]][None])[0])

    def targets_for(self, chem, candidates, exclude=(), k=25):
        """Rank candidate proteins for one compound."""
        if chem not in self.ci:
            return []
        cand = [p for p in candidates if p in self.pi and p not in exclude]
        if not cand:
            return []
        s = self._decode(self.hc[self.ci[chem]][None], self.hp[[self.pi[p] for p in cand]])
        order = np.argsort(-s)[:k]
        return [(cand[i], float(s[i])) for i in order]

    def ligands_for(self, prot, candidates=None, exclude=(), k=25, batch=20000):
        """Rank compounds for one protein (all compounds by default)."""
        if prot not in self.pi:
            return []
        ids = self.chem_ids if candidates is None else [c for c in candidates if c in self.ci]
        ex = set(exclude)
        hp = self.hp[self.pi[prot]][None]
        scores = []
        for i in range(0, len(ids), batch):
            part = ids[i:i + batch]
            scores.append(self._decode(self.hc[[self.ci[c] for c in part]], hp))
        s = np.concatenate(scores) if scores else np.zeros(0)
        order = np.argsort(-s)
        out = []
        for i in order:
            if ids[i] not in ex:
                out.append((ids[i], float(s[i])))
                if len(out) == k:
                    break
        return out
