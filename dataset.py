# -*- coding: utf-8 -*-
"""
Lightweight dataset & in-BM splitter for KGC with canonical relations
- No per-entity type_id; use relation -> (head_type, tail_type) dict
- Vectorized canonical relation ids (Categorical)
- BM-internal ID-level stratified split (8:1:1) + entity coverage cleanup
- Typed negative sampling via type offsets
"""

from __future__ import annotations
import os, pickle, hashlib, json, time
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_ETA = 0.0
DEFAULT_PI = 1e-5
REL_SEP = ':'  # canonical relation string: 'htype:rel:ttype'


# -----------------------------
# BucketManager
# -----------------------------
class BucketManager:
    """
    Manage entity/relation mappings with type-contiguous IDs and canonical relations.

    Attributes
    ----------
    num_e : int               # number of entities
    num_r : int               # number of relations (canonical keys)
    num_t : int               # number of entity types

    type2id / id2type : dict  # type name ↔ type id
    entkey2gid : dict[(type_id,str), int]  # (type id, entity name) → global entity id
    type_local_ents : list[list[str]]      # names per type ordered by local id

    rel2id / id2rel : dict    # canonical relation key (string) ↔ relation id
    rel_types : dict[int, Tuple[int,int]]  # r_id -> (head_type_id, tail_type_id)

    offsets : torch.LongTensor[num_t+1]    # type-contiguous ranges

    _df_ids : pd.DataFrame     # full graph mapped to ids (h_id, r_id, t_id)

    pi / eta : dict[int,float] # optional priors per relation id (from TRAIN only)
    """
    def __init__(self,
                 df_full: pd.DataFrame,
                 priors_train: Optional[pd.DataFrame] = None,
                 relation_mode: str = 'original',
                 cache_dir: Optional[str] = None,
                 force_rebuild: bool = False):
        assert {'htype','ttype','h','r','t'}.issubset(df_full.columns)
        relation_mode = relation_mode.lower(); assert relation_mode in {'original','simplified'}

        # ---- cache ----
        self._cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # faster fingerprint than CSV: hash selected columns
            fp = pd.util.hash_pandas_object(
                df_full[['htype','ttype','h','r','t']].astype(str), index=False, categorize=True
            ).values.tobytes()
            m = hashlib.md5(); m.update(fp); m.update(json.dumps({'relation_mode': relation_mode}, sort_keys=True).encode('utf-8'))
            finger = m.hexdigest()
            self._cache_path = os.path.join(cache_dir, f"bm-{finger}.pkl")
            if os.path.exists(self._cache_path) and not force_rebuild:
                print(f"Loading cache {self._cache_path}")
                with open(self._cache_path,'rb') as f:
                    state = pickle.load(f)
                self.__dict__.update(state)
                print('Done')
                return

        # --- build types & contiguous entity IDs ---
        print("Preparing data")
        types = pd.unique(pd.concat([df_full['htype'], df_full['ttype']], ignore_index=True))
        self.type2id = {t:i for i,t in enumerate(types)}
        self.id2type = {i:t for t,i in self.type2id.items()}
        self.num_t = len(self.type2id)

        # names per type (unique over heads/tails)
        per_type_names: Dict[str, np.ndarray] = {}
        for tname in types:
            nh = df_full.loc[df_full['htype']==tname, 'h'].astype(str).unique()
            nt = df_full.loc[df_full['ttype']==tname, 't'].astype(str).unique()
            per_type_names[tname] = np.unique(np.concatenate([nh, nt], axis=0))
        counts = np.array([len(per_type_names[self.id2type[i]]) for i in range(self.num_t)], dtype=np.int64)
        self.offsets = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), dtype=torch.long)
        self.num_e = int(self.offsets[-1].item())

        self.type_local_ents: List[List[str]] = []
        self.entkey2gid: Dict[Tuple[int,str], int] = {}
        for tid in range(self.num_t):
            tname = self.id2type[tid]
            names = per_type_names[tname].tolist()
            self.type_local_ents.append(names)
            base = int(self.offsets[tid].item())
            for lid, name in enumerate(names):
                self.entkey2gid[(tid, name)] = base + lid

        # --- canonical relations (vectorized) ---
        df = df_full[['htype','r','ttype']].astype(str).copy()
        h = df['htype']; t = df['ttype']
        if relation_mode == 'simplified':
            hl, tl = h.str.lower(), t.str.lower()
            mid = np.where((hl=='compound')&(tl=='protein'), 'interaction',
                  np.where((hl=='protein')&(tl=='compound'), 'metabolite',
                           'association'))  # other pairs collapse to 'association'
            mid = pd.Series(mid, index=df.index)
        else:
            mid = df_full['r'].astype(str)

        sep = REL_SEP
        rel_key_str = (h + sep + mid + sep + t).astype('category')
        cats = rel_key_str.cat.categories                       # unique canonical keys
        self.rel2id = {k:i for i,k in enumerate(cats)}
        self.id2rel = {i:k for i,k in enumerate(cats)}
        r_ids = rel_key_str.cat.codes.to_numpy(np.int64)
        self.num_r = len(cats)

        # relation -> (head_type_id, tail_type_id)
        cats_series = pd.Series(cats, dtype="object")
        parts = cats_series.str.split(sep, n=2, expand=True)    # 0=htype, 1=rel, 2=ttype
        head_type_ids = parts[0].map(self.type2id).to_numpy(np.int64)
        tail_type_ids = parts[2].map(self.type2id).to_numpy(np.int64)
        self.rel_types: Dict[int, Tuple[int,int]] = {i: (int(ht), int(tt))
                                                     for i,(ht,tt) in enumerate(zip(head_type_ids, tail_type_ids))}

        # --- map full triples to IDs ---
        # heads/tails via entkey2gid (typed)
        h_keys = list(zip(df_full['htype'].map(self.type2id), df_full['h'].astype(str)))
        t_keys = list(zip(df_full['ttype'].map(self.type2id), df_full['t'].astype(str)))
        h_ids = np.fromiter((self.entkey2gid[k] for k in h_keys), count=len(h_keys), dtype=np.int64)
        t_ids = np.fromiter((self.entkey2gid[k] for k in t_keys), count=len(t_keys), dtype=np.int64)
        self._df_ids = pd.DataFrame({'h_id': h_ids, 'r_id': r_ids, 't_id': t_ids})

        # --- priors per relation id (from TRAIN only) ---
        self.pi: Dict[int, float] = {}
        self.eta: Dict[int, float] = {}
        if priors_train is not None and len(priors_train):
            for _, row in priors_train.iterrows():
                if relation_mode == 'simplified':
                    htype, rname, ttype = str(row['htype']), str(row['r']), str(row['ttype'])
                    if htype.lower()=='compound' and ttype.lower()=='protein':
                        mid_s = 'interaction'
                    elif htype.lower()=='protein' and ttype.lower()=='compound':
                        mid_s = 'metabolite'
                    else:
                        mid_s = 'association'
                    key_s = f"{htype}{sep}{mid_s}{sep}{ttype}"
                else:
                    key_s = f"{row['htype']}{sep}{row['r']}{sep}{row['ttype']}"
                rid = self.rel2id.get(key_s)
                if rid is not None:
                    self.pi[rid] = float(row.get('pi', DEFAULT_PI))
                    self.eta[rid] = float(row.get('eta', DEFAULT_ETA))

        # ---- save cache ----
        if self._cache_path:
            state = self.__dict__.copy()
            with open(self._cache_path,'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done')

    # ---- helpers ----
    def sample_from_type(self, t_id: int, k: int) -> torch.LongTensor:
        start = int(self.offsets[t_id].item()); end = int(self.offsets[t_id+1].item())
        if end <= start:
            return torch.empty(0, dtype=torch.long)
        return torch.randint(low=start, high=end, size=(k,))

    # ---- in-BM splitter returning **ID-level** DataFrames ----
    def split_ids(self, valid_size: float = 0.1, test_size: float = 0.1,
                  min_bucket: int = 5, random_state: int = 42,
                  ensure_entity_coverage: bool = True, verbose: bool = True):
        """Stratified split by canonical relation id (r_id) with optional entity-coverage cleanup.
        Returns: (train_ids, valid_ids, test_ids) with columns [h_id, r_id, t_id].
        """
        t0 = time.time()
        if verbose: print('[split] start')
        N = len(self._df_ids)
        df = self._df_ids

        # group row indices by r_id
        g = df.groupby('r_id', sort=False).indices  # dict[rid] -> ndarray of row idx
        rng = np.random.default_rng(random_state)
        train_ix, valid_ix, test_ix = [], [], []
        for rid, idx in g.items():
            idx = np.asarray(idx, dtype=np.int64)
            n = idx.size
            if n < min_bucket:
                train_ix.append(idx); continue
            rng.shuffle(idx)
            n_test  = int(n * test_size)
            n_valid = int(n * valid_size)
            n_train = n - n_test - n_valid
            if n_train <= 0:
                n_train = 1; rem = n - 1
                n_valid = int(rem * (valid_size / (valid_size + test_size + 1e-12)))
                n_test  = rem - n_valid
            train_ix.append(idx[:n_train])
            valid_ix.append(idx[n_train:n_train+n_valid])
            test_ix.append(idx[n_train+n_valid:])
        cat = lambda parts: (np.concatenate(parts) if len(parts) else np.empty(0, dtype=np.int64))
        tri, vi, ti = cat(train_ix), cat(valid_ix), cat(test_ix)
        train = df.iloc[tri].copy(); valid = df.iloc[vi].copy(); test = df.iloc[ti].copy()
        if verbose:
            print(f"[stratify] train={len(train)} valid={len(valid)} test={len(test)} in {time.time()-t0:.2f}s")

        if ensure_entity_coverage:
            # coverage cleanup: ensure every entity id appears in TRAIN at least once
            def cleanup_once(src: pd.DataFrame):
                nonlocal train
                if src.empty: return src, 0
                seen = np.zeros(self.num_e, dtype=bool)
                seen[train['h_id'].to_numpy()] = True
                seen[train['t_id'].to_numpy()] = True
                s_h = src['h_id'].to_numpy(); s_t = src['t_id'].to_numpy()
                need = (~seen[s_h]) | (~seen[s_t])
                if not need.any(): return src, 0
                move_rows = src.loc[need]
                train = pd.concat([train, move_rows], ignore_index=True)
                kept = src.loc[~need]
                return kept.reset_index(drop=True), int(need.sum())
            t1 = time.time()
            valid, mv1 = cleanup_once(valid)
            test,  mv2 = cleanup_once(test)
            if verbose:
                print(f"[cleanup] moved={mv1+mv2} rows in {time.time()-t1:.2f}s")

        if verbose:
            print(f"[done] total={N} in {time.time()-t0:.2f}s")
        return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


# -----------------------------
# KGCDataset
# -----------------------------
class KGCDataset(Dataset):
    """A minimal dataset for KGC training or scoring.

    - Holds only its own ID-mapped triples (does not touch bm._df_ids).
    - Optional filtered negative sampling is provided but off by default.
    """
    def __init__(self, bm: BucketManager, ids_df: pd.DataFrame, num_ng: int = 0, filtered: bool = False):
        super().__init__()
        assert {'h_id','r_id','t_id'}.issubset(ids_df.columns)
        self.bm = bm
        self.h = torch.from_numpy(ids_df['h_id'].to_numpy(dtype=np.int64))
        self.r = torch.from_numpy(ids_df['r_id'].to_numpy(dtype=np.int64))
        self.t = torch.from_numpy(ids_df['t_id'].to_numpy(dtype=np.int64))
        self.num_ng = int(num_ng)
        self.filtered = bool(filtered)
        if self.filtered and self.num_ng > 0:
            # lightweight ban sets for this dataset only
            self.ban_tail: Dict[Tuple[int,int], set] = {}
            self.ban_head: Dict[Tuple[int,int], set] = {}
            for h_id, r_id, t_id in zip(self.h.tolist(), self.r.tolist(), self.t.tolist()):
                self.ban_tail.setdefault((h_id, r_id), set()).add(t_id)
                self.ban_head.setdefault((t_id, r_id), set()).add(h_id)

    def __len__(self) -> int:
        return self.h.numel()

    def _sample_from_type(self, t_id: int, k: int) -> torch.LongTensor:
        return self.bm.sample_from_type(t_id, k)

    def _rel_types_fast(self, r_id: int) -> Tuple[int,int]:
        # O(1) dictionary lookup: r_id -> (head_type_id, tail_type_id)
        return self.bm.rel_types[int(r_id)]

    def _sample_tails(self, h: int, r: int, k: int) -> torch.LongTensor:
        _, ttid = self._rel_types_fast(r)
        out = self._sample_from_type(ttid, k)
        if self.filtered and self.num_ng > 0:
            ban = self.ban_tail.get((h, r), None)
            if ban:
                cap = 10
                for i in range(k):
                    tries = 0
                    while out[i].item() in ban and tries < cap:
                        out[i] = self._sample_from_type(ttid, 1)[0]
                        tries += 1
        return out

    def _sample_heads(self, t: int, r: int, k: int) -> torch.LongTensor:
        htid, _ = self._rel_types_fast(r)
        out = self._sample_from_type(htid, k)
        if self.filtered and self.num_ng > 0:
            ban = self.ban_head.get((t, r), None)
            if ban:
                cap = 10
                for i in range(k):
                    tries = 0
                    while out[i].item() in ban and tries < cap:
                        out[i] = self._sample_from_type(htid, 1)[0]
                        tries += 1
        return out

    def __getitem__(self, idx: int) -> torch.Tensor:
        h = int(self.h[idx].item()); r = int(self.r[idx].item()); t = int(self.t[idx].item())
        if self.num_ng <= 0:
            return torch.tensor([[h, r, t]], dtype=torch.long)
        k = max(1, self.num_ng // 2)
        neg_t = self._sample_tails(h, r, k)
        neg_h = self._sample_heads(t, r, k)
        pos = torch.tensor([[h, r, t]], dtype=torch.long)
        neg_t_pack = torch.stack([torch.full((k,), h, dtype=torch.long),
                                  torch.full((k,), r, dtype=torch.long),
                                  neg_t], dim=1)
        neg_h_pack = torch.stack([neg_h,
                                  torch.full((k,), r, dtype=torch.long),
                                  torch.full((k,), t, dtype=torch.long)], dim=1)
        return torch.cat([pos, neg_t_pack, neg_h_pack], dim=0)


def collate_stack(batch):
    return torch.stack(batch)