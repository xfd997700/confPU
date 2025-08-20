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
    def split_ids(self, valid_size: float = 0.1, test_size: float = 0.1, calib_size: float = 0.0,
                min_bucket: int = 5, random_state: int = 42,
                ensure_entity_coverage: bool = True, verbose: bool = True):
        """Stratified split by canonical relation id (r_id) with optional entity-coverage cleanup.
        Returns:
        - if calib_size <= 0: (train_ids, valid_ids, test_ids)
        - else:               (train_ids, calib_ids, valid_ids, test_ids)
        All DataFrames have columns [h_id, r_id, t_id].
        """
        t0 = time.time()
        if verbose: print('[split] start')
        N = len(self._df_ids)
        df = self._df_ids

        # group row indices by r_id
        g = df.groupby('r_id', sort=False).indices  # dict[rid] -> ndarray of row idx
        rng = np.random.default_rng(random_state)
        train_ix, valid_ix, test_ix, calib_ix = [], [], [], []
        use_calib = calib_size is not None and calib_size > 0.0

        for rid, idx in g.items():
            idx = np.asarray(idx, dtype=np.int64)
            n = idx.size
            if n < min_bucket:
                # 小桶保持稳健：全部进 train
                train_ix.append(idx)
                continue

            rng.shuffle(idx)

            if use_calib:
                n_calib = int(n * calib_size)
                n_test  = int(n * test_size)
                n_valid = int(n * valid_size)
                n_train = n - n_calib - n_test - n_valid
                if n_train <= 0:
                    # 至少留 1 个在 train，然后按比例分剩余
                    n_train = 1
                    rem = n - 1
                    denom = (valid_size + test_size + calib_size) + 1e-12
                    n_valid = int(rem * (valid_size / denom))
                    n_test  = int(rem * (test_size  / denom))
                    n_calib = rem - n_valid - n_test
                p0 = 0
                p1 = p0 + n_train
                p2 = p1 + n_valid
                p3 = p2 + n_test
                p4 = p3 + n_calib
                train_ix.append(idx[p0:p1])
                valid_ix.append(idx[p1:p2])
                test_ix.append(idx[p2:p3])
                calib_ix.append(idx[p3:p4])
            else:
                n_test  = int(n * test_size)
                n_valid = int(n * valid_size)
                n_train = n - n_test - n_valid
                if n_train <= 0:
                    n_train = 1
                    rem = n - 1
                    n_valid = int(rem * (valid_size / (valid_size + test_size + 1e-12)))
                    n_test  = rem - n_valid
                train_ix.append(idx[:n_train])
                valid_ix.append(idx[n_train:n_train+n_valid])
                test_ix.append(idx[n_train+n_valid:])

        # concat 索引
        cat = lambda parts: (np.concatenate(parts) if len(parts) else np.empty(0, dtype=np.int64))
        tri, vi, ti = cat(train_ix), cat(valid_ix), cat(test_ix)
        train = df.iloc[tri].copy(); valid = df.iloc[vi].copy(); test = df.iloc[ti].copy()
        if use_calib:
            ci = cat(calib_ix)
            calib = df.iloc[ci].copy()
        else:
            calib = pd.DataFrame(columns=df.columns).astype(df.dtypes.to_dict())  # 类型保持一致的空表

        if verbose:
            if use_calib:
                print(f"[stratify] train={len(train)} valid={len(valid)} test={len(test)} calib={len(calib)} in {time.time()-t0:.2f}s")
            else:
                print(f"[stratify] train={len(train)} valid={len(valid)} test={len(test)} in {time.time()-t0:.2f}s")

        # 覆盖清理：确保每个实体至少在 TRAIN 出现一次
        if ensure_entity_coverage:
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
            mv_calib = mv_valid = mv_test = 0
            if use_calib:
                calib, mv_calib = cleanup_once(calib)
            valid, mv_valid = cleanup_once(valid)
            test,  mv_test  = cleanup_once(test)
            if verbose:
                if use_calib:
                    print(f"[cleanup] moved={mv_calib+mv_valid+mv_test} rows (calib={mv_calib}, valid={mv_valid}, test={mv_test}) in {time.time()-t1:.2f}s")
                else:
                    print(f"[cleanup] moved={mv_valid+mv_test} rows (valid={mv_valid}, test={mv_test}) in {time.time()-t1:.2f}s")

        if verbose:
            print(f"[done] total={N} in {time.time()-t0:.2f}s")

        # 返回形式兼容旧代码
        if use_calib:
            return (train.reset_index(drop=True),
                    calib.reset_index(drop=True),
                    valid.reset_index(drop=True),
                    test.reset_index(drop=True))
        else:
            return (train.reset_index(drop=True),
                    valid.reset_index(drop=True),
                    test.reset_index(drop=True))

    
    # ---- kfold ----
    def kfold_ids_for_oof(self, n_splits: int = 5, random_state: int = 42,
                          ensure_entity_coverage: bool = True,
                          support_policy: str = 'allow_support',
                          min_bucket: int = 5,
                          verbose: bool = True):
        """
        Stratified K-fold by r_id for OOF.
        - Only edge-level split; no entity split.
        - If ensure_entity_coverage=True, each fold's TRAIN will cover all entities.
        For degree-1 entities that would be missing, we mark their unique edges as
        'support' (added into train AND removed from test) when support_policy='allow_support'.
        Returns: list of dicts with keys: 'train', 'test', 'support' (all ID-level DataFrames).
        """
        assert n_splits >= 2
        assert support_policy in {'allow_support', 'move_min'}

        df = self._df_ids
        # --- group by relation id for stratification ---
        groups = df.groupby('r_id', sort=False).indices  # dict[rid] -> np.ndarray row indices
        rng = np.random.default_rng(random_state)

        # --- per-rid shuffle & split into K almost-equal chunks ---
        # store per-rid list of index-chunks
        per_rid_chunks = {}
        for rid, idx in groups.items():
            idx = np.asarray(idx, dtype=np.int64)
            n = idx.size
            if n < min_bucket:
                # small buckets: put all into a single training pool by distributing minimally
                # but still create K chunks (some empty)
                rng.shuffle(idx)
                chunks = [idx[i::n_splits] for i in range(n_splits)]  # round-robin
            else:
                rng.shuffle(idx)
                # split into K contiguous chunks
                splits = np.linspace(0, n, n_splits + 1, dtype=int)
                chunks = [idx[splits[i]:splits[i+1]] for i in range(n_splits)]
            per_rid_chunks[rid] = chunks

        # --- build K folds' TEST index sets (disjoint) ---
        fold_test_indices = []
        for k in range(n_splits):
            parts = [per_rid_chunks[rid][k] for rid in per_rid_chunks]
            if len(parts):
                fold_test_indices.append(np.concatenate(parts))
            else:
                fold_test_indices.append(np.empty(0, dtype=np.int64))

        # --- assemble folds, add coverage cleanup if requested ---
        folds = []
        all_idx = np.arange(len(df), dtype=np.int64)
        for k in range(n_splits):
            test_ix = fold_test_indices[k]
            # TRAIN = all - TEST
            train_ix = np.setdiff1d(all_idx, test_ix, assume_unique=False)

            train = df.iloc[train_ix].copy()
            test  = df.iloc[test_ix].copy()
            support = pd.DataFrame(columns=df.columns).astype(df.dtypes.to_dict())

            if ensure_entity_coverage and len(test):
                # seen entities in train
                seen = np.zeros(self.num_e, dtype=bool)
                seen[train['h_id'].to_numpy()] = True
                seen[train['t_id'].to_numpy()] = True

                # locate rows in TEST that touch unseen entities
                t_h = test['h_id'].to_numpy(); t_t = test['t_id'].to_numpy()
                need = (~seen[t_h]) | (~seen[t_t])
                if need.any():
                    need_rows = test.loc[need]
                    if support_policy == 'allow_support':
                        # promote to support: add to train; remove from test
                        support = need_rows.copy()
                        train = pd.concat([train, need_rows], ignore_index=True)
                        test = test.loc[~need].reset_index(drop=True)
                    else:
                        # 'move_min'：与 allow_support 行为等价于把 need_rows 移到 train；
                        # 为清晰起见也按同样处理
                        train = pd.concat([train, need_rows], ignore_index=True)
                        test = test.loc[~need].reset_index(drop=True)

            folds.append({
                'train': train.reset_index(drop=True),
                'test':  test.reset_index(drop=True),
                'support': support.reset_index(drop=True),
            })

            if verbose:
                print(f"[kfold {k+1}/{n_splits}] train={len(folds[-1]['train'])} "
                    f"test={len(folds[-1]['test'])} support={len(folds[-1]['support'])}")

        return folds



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