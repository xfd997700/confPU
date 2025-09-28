# -*- coding: utf-8 -*-
import os
import json
import math
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import networkx as nx
from community import community_louvain  # python-louvain


# ============================== Evaluator ==============================

class KGPEvaluator:
    """
    评估思路（全图，无向投影；证据只用 links）：
      1) HS2（两跳异构支持度）：
         - 对任意预测边 (u,v)：在 links 中为所有“预测端点”构建按邻居类型分桶的索引
           neighbors_by_type[u][typeX] = {neighbors of typeX in links}
         - HS2(u,v) = sum_X |Nu_X ∩ Nv_X| / sqrt(|Nu_X|*|Nv_X|) ，X 为 u,v 的共同邻居类型集合。
           若 INCLUDE_ENDPOINT_TYPES_AS_MEDIATORS=False，则 X 排除 type(u), type(v)。
         - Z_HS：与“类型&度分箱匹配”的 null 非边对照比较（仅用 links），可整体也可按关系类型统计。
      2) 社区/模块度（全图）：
         - 用 links 的全体无向投影图跑 Louvain 得到 partition。
         - SCR_obs：links 边中同社群比例（整体 & 按关系类型(按端点类型、无向)）。
         - SCR_pred：预测边同社群比例（整体 & 按关系类型）。
         - ΔQ：固定 partition，比较 Q(G_links∪E_pred) 与 Q(G_links)（整体 & 按关系类型）。
    关系类型定义：只看端点实体类型，正反合并，格式 'TypeA|TypeB'（字典序）。
    """

    def __init__(self, links_csv, pred_csv, h_col='h', t_col='t', r_col='r',
                 htype_col='htype', ttype_col='ttype', seed=42):
        self.links_csv = links_csv
        self.pred_csv = pred_csv
        self.h_col = h_col
        self.t_col = t_col
        self.r_col = r_col
        self.htype_col = htype_col
        self.ttype_col = ttype_col

        self.rng = random.Random(seed)
        np.random.seed(seed)

        # DataFrames
        self.df_links = None
        self.df_pred = None

        # Node/type/indexes
        self.pred_pairs = []                    # list[(u,v)]
        self.pred_nodes = set()                 # nodes appearing in pred endpoints
        self.node_type = dict()                 # node -> main type (from links majority, pred supplements)
        self.nodes_by_type = defaultdict(list)  # type -> [nodes]
        self.node_degree = dict()               # node -> degree in links undirected projection
        self.neighbors_by_type = defaultdict(lambda: defaultdict(set))  # node -> {typeX: set(nei)}

        # Edge sets to exclude in null sampling
        self.links_undirected_pairs = set()
        self.pred_undirected_pairs = set()

        # Full graph from links (undirected projection)
        self.G_links = None
        self.partition = None                   # node -> community id
        self.communities = None                 # list[list[nodes]]
        self.Q_links = None                     # base modularity

        # For relation type grouping on pred
        self.pred_reltype = []                  # relation type label for each pred pair 'A|B'

    # ---------------------- Loading ----------------------

    def load_data(self):
        # 全量载入（用户说明内存足够）
        usecols = [self.h_col, self.r_col, self.t_col, self.htype_col, self.ttype_col]
        self.df_links = pd.read_csv(self.links_csv, dtype=str, usecols=usecols)
        self.df_pred = pd.read_csv(self.pred_csv, dtype=str, usecols=usecols)

        # 仅保留预测
        self.df_pred = self.df_pred[self.df_pred[self.r_col].astype(str).str.startswith('PREDICT_')].copy()

        # 预测 pair 去重
        self.df_pred['pair'] = list(zip(self.df_pred[self.h_col], self.df_pred[self.t_col]))
        self.df_pred.drop_duplicates(subset=['pair'], inplace=True)
        self.pred_pairs = self.df_pred['pair'].tolist()
        self.pred_nodes = set(self.df_pred[self.h_col].unique()) | set(self.df_pred[self.t_col].unique())

        # 构建预测关系类型（无向：按类型名排序后用 'A|B'）
        rel_labels = []
        for ht, tt in self.df_pred[[self.htype_col, self.ttype_col]].itertuples(index=False):
            a, b = sorted([ht, tt])
            rel_labels.append(f"{a}|{b}")
        self.pred_reltype = rel_labels

    # ---------------------- Full graph & indexes ----------------------

    def build_full_graph_and_indexes(self):
        """
        用 links 构建：
          - G_links（无向）
          - node_type（取 links 众数，pred 补全）
          - nodes_by_type
          - node_degree（links 无向投影度）
          - neighbors_by_type：仅为 pred 端点构建按类型分桶的邻接
          - links/pred 的无向 pair 集合（用于 null 排除）
        """
        dfL = self.df_links

        # node_type：links 众数
        type_cnt = defaultdict(Counter)
        for h, ht, t, tt in tqdm(dfL[[self.h_col, self.htype_col, self.t_col, self.ttype_col]].itertuples(index=False),
                                 total=len(dfL), desc="Scan links for node types"):
            type_cnt[h][ht] += 1
            type_cnt[t][tt] += 1
        for n, cnt in type_cnt.items():
            self.node_type[n] = cnt.most_common(1)[0][0]
        # pred 补全
        for n, tp in self.df_pred[[self.h_col, self.htype_col]].itertuples(index=False):
            if n not in self.node_type:
                self.node_type[n] = tp
        for n, tp in self.df_pred[[self.t_col, self.ttype_col]].itertuples(index=False):
            if n not in self.node_type:
                self.node_type[n] = tp

        # nodes_by_type
        for n, tp in self.node_type.items():
            self.nodes_by_type[tp].append(n)

        # Build G_links undirected projection
        G = nx.Graph()
        # 先加所有出现过的节点，避免后面 modularity 因节点缺失报警
        all_nodes = set(dfL[self.h_col].unique()).union(set(dfL[self.t_col].unique()))
        G.add_nodes_from(all_nodes)
        # 加边（无向；Graph 会去重）
        G.add_edges_from(dfL[[self.h_col, self.t_col]].itertuples(index=False, name=None))
        self.G_links = G

        # node_degree（links 无向投影度）
        deg = dict(G.degree())
        self.node_degree = deg

        # neighbors_by_type：仅为“预测端点”构建
        pred_nodes = self.pred_nodes
        # h -> t
        mask_h_in = dfL[self.h_col].isin(pred_nodes)
        for h, ht, t, tt in tqdm(dfL.loc[mask_h_in, [self.h_col, self.htype_col, self.t_col, self.ttype_col]].itertuples(index=False),
                                 total=int(mask_h_in.sum()), desc="Index neighbors_by_type (h in pred)"):
            self.neighbors_by_type[h][tt].add(t)
        # t -> h
        mask_t_in = dfL[self.t_col].isin(pred_nodes)
        for h, ht, t, tt in tqdm(dfL.loc[mask_t_in, [self.h_col, self.htype_col, self.t_col, self.ttype_col]].itertuples(index=False),
                                 total=int(mask_t_in.sum()), desc="Index neighbors_by_type (t in pred)"):
            self.neighbors_by_type[t][ht].add(h)

        # links/pred 无向 pair 集（用于 null 排除）
        self.links_undirected_pairs = set()
        for h, t in tqdm(dfL[[self.h_col, self.t_col]].itertuples(index=False),
                         total=len(dfL), desc="Materialize links undirected pair set"):
            a, b = (h, t) if h <= t else (t, h)
            self.links_undirected_pairs.add((a, b))

        self.pred_undirected_pairs = set()
        for u, v in self.pred_pairs:
            a, b = (u, v) if u <= v else (v, u)
            self.pred_undirected_pairs.add((a, b))

    # ---------------------- Louvain partition on full graph ----------------------

    def compute_partition_and_Qlinks(self):
        print("Louvain on full links graph ...")
        self.partition = community_louvain.best_partition(self.G_links)  # node -> comm
        self.communities = self._partition_to_communities(self.partition)
        self.Q_links = nx.algorithms.community.quality.modularity(self.G_links, self.communities)

    @staticmethod
    def _partition_to_communities(partition_dict):
        comm2nodes = defaultdict(list)
        for n, c in partition_dict.items():
            comm2nodes[c].append(n)
        return list(comm2nodes.values())

    # ---------------------- HS2 ----------------------

    @staticmethod
    def _reltype_label(tu, tv):
        a, b = sorted([tu, tv])
        return f"{a}|{b}"

    @staticmethod
    def _hs2_for_pair(u, v, neighbors_by_type, node_type, include_endpoint_types_as_mediators=True):
        map_u = neighbors_by_type.get(u, {})
        map_v = neighbors_by_type.get(v, {})
        if not map_u or not map_v:
            return 0.0, False
        common_types = set(map_u.keys()).intersection(set(map_v.keys()))
        if not include_endpoint_types_as_mediators:
            tu = node_type.get(u)
            tv = node_type.get(v)
            common_types.discard(tu)
            common_types.discard(tv)
        if not common_types:
            return 0.0, False
        total = 0.0
        covered = False
        for X in common_types:
            sU = map_u.get(X)
            sV = map_v.get(X)
            if not sU or not sV:
                continue
            inter = len(sU & sV)
            if inter == 0:
                continue
            covered = True
            denom = math.sqrt(len(sU) * len(sV))
            if denom > 0:
                total += inter / denom
        return total, covered

    def compute_hs2_pred_scores(self, include_endpoint_types_as_mediators=True,
                                save_edge_scores_path=None):
        """
        计算每条预测的 HS2 与 covered，并返回 numpy 数组；可选保存 parquet。
        同时返回每条预测的关系类型标签数组（与 self.pred_reltype 一致）。
        """
        n = len(self.pred_pairs)
        hs_scores = np.empty(n, dtype=np.float64)
        covered_flags = np.empty(n, dtype=np.int8)

        it = tqdm(range(n), desc="HS2 on predictions")
        for i in it:
            u, v = self.pred_pairs[i]
            s, cov = self._hs2_for_pair(u, v, self.neighbors_by_type, self.node_type,
                                        include_endpoint_types_as_mediators)
            hs_scores[i] = s
            covered_flags[i] = 1 if cov else 0

        if save_edge_scores_path:
            df = pd.DataFrame({
                'u': [x[0] for x in self.pred_pairs],
                'v': [x[1] for x in self.pred_pairs],
                'tu': [self.node_type.get(x[0], None) for x in self.pred_pairs],
                'tv': [self.node_type.get(x[1], None) for x in self.pred_pairs],
                'reltype': self.pred_reltype,
                'HS2': hs_scores,
                'covered': covered_flags
            })
            df.to_parquet(save_edge_scores_path, index=False)

        return hs_scores, covered_flags, self.pred_reltype

    def _hs2_null_sample_for_pairs(self, pairs, null_n, deg_bins, include_endpoint_types_as_mediators):
        """
        针对一组给定的预测 pair（通常是某个关系类型的子集），
        做“类型+度分箱”匹配的 null 抽样并计算 HS2。
        仅用 links；排除 links 已有边与整份预测对。
        """
        if null_n <= 0 or len(pairs) == 0:
            return np.zeros(0, dtype=np.float64)

        def to_bin(val):
            lo, hi = 0, len(deg_bins) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if val >= deg_bins[mid]:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        # 统计该组 pair 的 (type, deg_bin) 组合频次
        combo_counter = Counter()
        for u, v in pairs:
            tu = self.node_type.get(u, None)
            tv = self.node_type.get(v, None)
            du = self.node_degree.get(u, 0)
            dv = self.node_degree.get(v, 0)
            combo = (tu, to_bin(du), tv, to_bin(dv))
            combo_counter[combo] += 1

        # 构造类型+度分箱到节点的索引
        deg_bin_index = defaultdict(list)   # (type, bin) -> [nodes]
        for tp, nodes in self.nodes_by_type.items():
            for n in nodes:
                deg_bin_index[(tp, to_bin(self.node_degree.get(n, 0)))].append(n)

        scores = []
        total_pairs = max(1, len(pairs))
        with tqdm(total=null_n, desc="HS2 null (per-type)") as pbar:
            for (tu, bu, tv, bv), cnt in combo_counter.items():
                if tu is None or tv is None:
                    continue
                target = max(1, int(cnt / total_pairs * null_n))
                pool_u = deg_bin_index.get((tu, bu), self.nodes_by_type.get(tu, []))
                pool_v = deg_bin_index.get((tv, bv), self.nodes_by_type.get(tv, []))
                if not pool_u or not pool_v:
                    continue
                tries = 0
                while target > 0 and tries < target * 50:
                    u = self.rng.choice(pool_u)
                    v = self.rng.choice(pool_v)
                    if u == v:
                        tries += 1
                        continue
                    a, b = (u, v) if u <= v else (v, u)
                    if (a, b) in self.links_undirected_pairs or (a, b) in self.pred_undirected_pairs:
                        tries += 1
                        continue
                    s, _ = self._hs2_for_pair(u, v, self.neighbors_by_type, self.node_type,
                                              include_endpoint_types_as_mediators)
                    scores.append(s)
                    target -= 1
                    tries += 1
                    pbar.update(1)

        # 若不足，回退到随机构造（从预测端点全集）
        pred_nodes_list = list(self.pred_nodes)
        while len(scores) < null_n and pred_nodes_list:
            u = self.rng.choice(pred_nodes_list)
            v = self.rng.choice(pred_nodes_list)
            if u == v:
                continue
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) in self.links_undirected_pairs or (a, b) in self.pred_undirected_pairs:
                continue
            s, _ = self._hs2_for_pair(u, v, self.neighbors_by_type, self.node_type,
                                      include_endpoint_types_as_mediators)
            scores.append(s)

        return np.array(scores, dtype=np.float64)

    def compute_hs2_summary(self, null_sample_size=200000, deg_bins=(0,1,2,4,8,16,32,64,128,256,512,1<<30),
                             include_endpoint_types_as_mediators=True,
                             save_edge_scores_path=None, save_null_scores_path=None):
        """
        输出：
          overall: {mean_HS2_pred, coverage_HS2, mean_HS2_null, std_HS2_null, Z_HS}
          per_type: reltype -> 同样结构
        """
        # 预测边逐条 HS2
        hs_scores, covered_flags, reltypes = self.compute_hs2_pred_scores(
            include_endpoint_types_as_mediators=include_endpoint_types_as_mediators,
            save_edge_scores_path=save_edge_scores_path
        )
        n = len(hs_scores)

        # per-type 聚合（预测分布）
        df_pred_scores = pd.DataFrame({'reltype': reltypes, 'HS2': hs_scores, 'covered': covered_flags})
        pred_group = df_pred_scores.groupby('reltype', sort=False)
        mean_pred_by_type = pred_group['HS2'].mean().to_dict()
        coverage_by_type = pred_group['covered'].mean().to_dict()
        count_by_type = pred_group.size().to_dict()

        # overall 预测均值与覆盖
        mean_hs_pred_overall = float(df_pred_scores['HS2'].mean()) if n else 0.0
        coverage_overall = float(df_pred_scores['covered'].mean()) if n else 0.0

        # null 抽样：按关系类型分配份额做 per-type 的 null
        null_scores_records = []  # (reltype, score)
        overall_quota_left = null_sample_size
        types = list(count_by_type.keys())
        total_count = sum(count_by_type.values()) if count_by_type else 0

        for rt in tqdm(types, desc="Allocating HS2 null quotas by reltype"):
            cnt = count_by_type[rt]
            quota = max(1, int(cnt / max(1, total_count) * null_sample_size))
            overall_quota_left -= quota

            # 取该类型的子集 pair
            mask = (df_pred_scores['reltype'] == rt).to_numpy()
            pairs_rt = [self.pred_pairs[i] for i, m in enumerate(mask) if m]

            null_rt = self._hs2_null_sample_for_pairs(
                pairs_rt, quota, deg_bins, include_endpoint_types_as_mediators
            )
            null_scores_records.extend([(rt, x) for x in null_rt])

        # 如果配额之和不等于 null_sample_size，做少量补齐（随便挑一种类型补）
        while overall_quota_left > 0 and types:
            rt = types[0]
            null_scores_records.append((rt, 0.0))
            overall_quota_left -= 1

        df_null = pd.DataFrame(null_scores_records, columns=['reltype', 'HS2_null'])
        if save_null_scores_path:
            df_null.to_parquet(save_null_scores_path, index=False)

        # per-type null 统计
        null_group = df_null.groupby('reltype', sort=False)
        mean_null_by_type = null_group['HS2_null'].mean().to_dict()
        std_null_by_type = null_group['HS2_null'].std(ddof=1).fillna(1.0).to_dict()

        # per-type Z_HS
        z_by_type = {}
        for rt in types:
            mu_p = mean_pred_by_type.get(rt, 0.0)
            mu_n = mean_null_by_type.get(rt, 0.0)
            sd_n = std_null_by_type.get(rt, 1.0)
            z_by_type[rt] = (mu_p - mu_n) / sd_n if sd_n > 0 else 0.0

        # overall null/ Z
        mean_null_overall = float(df_null['HS2_null'].mean()) if len(df_null) else 0.0
        std_null_overall = float(df_null['HS2_null'].std(ddof=1)) if len(df_null) > 1 else 1.0
        z_overall = (mean_hs_pred_overall - mean_null_overall) / std_null_overall if std_null_overall > 0 else 0.0

        # 汇总
        overall = {
            'mean_HS2_pred': mean_hs_pred_overall,
            'coverage_HS2': coverage_overall,
            'mean_HS2_null': mean_null_overall,
            'std_HS2_null': std_null_overall,
            'Z_HS': z_overall
        }
        per_type = {
            rt: {
                'mean_HS2_pred': float(mean_pred_by_type.get(rt, 0.0)),
                'coverage_HS2': float(coverage_by_type.get(rt, 0.0)),
                'mean_HS2_null': float(mean_null_by_type.get(rt, 0.0)),
                'std_HS2_null': float(std_null_by_type.get(rt, 1.0)),
                'Z_HS': float(z_by_type.get(rt, 0.0)),
                'count_pred_edges': int(count_by_type.get(rt, 0))
            }
            for rt in types
        }
        return overall, per_type

    # ---------------------- Community / Modularity (full graph) ----------------------

    def community_and_modularity_full(self, save_partition_map_path=None):
        """
        计算社区/模块度指标，返回：
          overall: {SCR_obs, SCR_pred, SCR_Lift, Q_links, Q_all_with_pred, Delta_Q}
          per_type: reltype -> {SCR_obs_type, SCR_pred_type, SCR_Lift_type, Delta_Q_type, count_pred_edges, count_links_edges}
        """
        part = self.partition
        comms = self.communities
        Q_links = self.Q_links

        # === links 侧：同社群比例（整体 & 按关系类型） ===
        dfL = self.df_links.copy()
        dfL['h_comm'] = dfL[self.h_col].map(part)
        dfL['t_comm'] = dfL[self.t_col].map(part)
        dfL['same_comm'] = (dfL['h_comm'] == dfL['t_comm']).astype(np.int8)
        # 关系类型标签（无向）
        relL = []
        for ht, tt in dfL[[self.htype_col, self.ttype_col]].itertuples(index=False):
            a, b = sorted([ht, tt])
            relL.append(f"{a}|{b}")
        dfL['reltype'] = relL

        SCR_obs_overall = float(dfL['same_comm'].mean()) if len(dfL) else float('nan')
        groupL = dfL.groupby('reltype', sort=False)
        SCR_obs_by_type = groupL['same_comm'].mean().to_dict()
        links_count_by_type = groupL.size().to_dict()

        # === pred 侧：同社群比例（整体 & 按关系类型） ===
        dfP = pd.DataFrame({
            'u': [p[0] for p in self.pred_pairs],
            'v': [p[1] for p in self.pred_pairs],
            'reltype': self.pred_reltype
        })
        dfP['u_comm'] = dfP['u'].map(part)
        dfP['v_comm'] = dfP['v'].map(part)
        dfP['same_comm'] = (dfP['u_comm'] == dfP['v_comm']).astype(np.int8)

        SCR_pred_overall = float(dfP['same_comm'].mean()) if len(dfP) else float('nan')
        groupP = dfP.groupby('reltype', sort=False)
        SCR_pred_by_type = groupP['same_comm'].mean().to_dict()
        pred_count_by_type = groupP.size().to_dict()

        # === Lift（整体 & 按类型） ===
        SCR_Lift_overall = (SCR_pred_overall / SCR_obs_overall) if (SCR_obs_overall and SCR_obs_overall > 0) else float('nan')
        SCR_Lift_by_type = {}
        for rt in SCR_pred_by_type.keys():
            obs = SCR_obs_by_type.get(rt, np.nan)
            pred = SCR_pred_by_type.get(rt, np.nan)
            SCR_Lift_by_type[rt] = (pred / obs) if (obs and obs > 0) else float('nan')

        # === ΔQ（整体 & 按类型）：固定 partition ===
        # 整体
        G_all = self.G_links.copy()
        G_all.add_edges_from(self.pred_pairs)
        Q_all = nx.algorithms.community.quality.modularity(G_all, comms)
        Delta_Q_overall = Q_all - Q_links

        # 按类型
        Delta_Q_by_type = {}
        types = list(pred_count_by_type.keys())
        for rt in tqdm(types, desc="Compute Delta_Q by reltype"):
            # 该类型的预测边集合
            mask_rt = (dfP['reltype'] == rt).to_numpy()
            edges_rt = [(dfP['u'].iat[i], dfP['v'].iat[i]) for i, m in enumerate(mask_rt) if m]
            G_rt = self.G_links.copy()
            G_rt.add_edges_from(edges_rt)
            Q_rt = nx.algorithms.community.quality.modularity(G_rt, comms)
            Delta_Q_by_type[rt] = Q_rt - Q_links

        overall = {
            'SCR_obs': SCR_obs_overall,
            'SCR_pred': SCR_pred_overall,
            'SCR_Lift': SCR_Lift_overall,
            'Q_links': Q_links,
            'Q_all_with_pred': Q_all,
            'Delta_Q': Delta_Q_overall
        }
        per_type = {
            rt: {
                'SCR_obs': float(SCR_obs_by_type.get(rt, np.nan)),
                'SCR_pred': float(SCR_pred_by_type.get(rt, np.nan)),
                'SCR_Lift': float(SCR_Lift_by_type.get(rt, np.nan)),
                'Delta_Q': float(Delta_Q_by_type.get(rt, 0.0)),
                'count_pred_edges': int(pred_count_by_type.get(rt, 0)),
                'count_links_edges': int(links_count_by_type.get(rt, 0))
            }
            for rt in types
        }

        if save_partition_map_path:
            # 可选保存 node->community 映射，便于外部复用
            with open(save_partition_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.partition, f, ensure_ascii=False)

        return overall, per_type


# ============================== Plotter（可选） ==============================

class KGPlotter:
    def __init__(self, out_dir):
        import matplotlib.pyplot as plt
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def plot_hs2_hist(self, pred_scores_parquet, null_scores_parquet, bins=100):
        import matplotlib.pyplot as plt
        dfp = pd.read_parquet(pred_scores_parquet)
        dfn = pd.read_parquet(null_scores_parquet)
        plt.figure()
        plt.hist(dfp['HS2'].values, bins=bins, alpha=0.6, density=True, label='Pred HS2')
        plt.hist(dfn['HS2_null'].values, bins=bins, alpha=0.6, density=True, label='Null HS2')
        plt.xlabel('HS2 score')
        plt.ylabel('Density')
        plt.title('HS2: Predictions vs Type/Degree-matched Null')
        plt.legend()
        out = os.path.join(self.out_dir, 'hs2_hist.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[Saved] {out}")

    def plot_scr_bar(self, scr_obs, scr_pred):
        import matplotlib.pyplot as plt
        plt.figure()
        labels = ['Observed (links)', 'Predicted']
        vals = [scr_obs, scr_pred]
        plt.bar(labels, vals)
        plt.ylabel('Same-Community Ratio')
        plt.title('Same-Community Ratio: Overall')
        out = os.path.join(self.out_dir, 'scr_bar_overall.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[Saved] {out}")


# ============================== 应用过程 ==============================

if __name__ == '__main__':
    # 路径配置
    LINKS_CSV = 'screen_cache/unibiomap.links.csv'
    PRED_CSV  = 'screen_cache/unibiomap.pred.csv'

    # 输出
    OUT_DIR = 'screen_cache/graph_eval_out_full'
    os.makedirs(OUT_DIR, exist_ok=True)

    # HS2 配置
    NULL_SAMPLE_SIZE = 200_000                 # 总的 null 抽样规模（会按类型分配）
    INCLUDE_ENDPOINT_TYPES_AS_MEDIATORS = True # 是否允许端点类型也作为中介类型
    SAVE_PRED_EDGE_SCORES = True
    PRED_EDGE_SCORES_PATH = os.path.join(OUT_DIR, 'pred_hs2.parquet')
    NULL_SCORES_PATH = os.path.join(OUT_DIR, 'null_hs2.parquet')

    # 1) 加载数据
    evaler = KGPEvaluator(LINKS_CSV, PRED_CSV, seed=42)
    print("Loading data ...")
    evaler.load_data()

    # 2) 构建全图与索引（仅用 links）
    print("Building full graph & indexes from links ...")
    evaler.build_full_graph_and_indexes()

    # 3) Louvain 社区与基线模块度
    evaler.compute_partition_and_Qlinks()

    # 4) HS2（整体 & 按关系类型）
    hs_overall, hs_by_type = evaler.compute_hs2_summary(
        null_sample_size=NULL_SAMPLE_SIZE,
        include_endpoint_types_as_mediators=INCLUDE_ENDPOINT_TYPES_AS_MEDIATORS,
        save_edge_scores_path=PRED_EDGE_SCORES_PATH if SAVE_PRED_EDGE_SCORES else None,
        save_null_scores_path=NULL_SCORES_PATH
    )
    print("[HS2 Overall]")
    print(json.dumps(hs_overall, indent=2))
    print("[HS2 By Relation Type]")
    print(json.dumps(hs_by_type, indent=2)[:2000] + ("...\n" if len(json.dumps(hs_by_type))>2000 else ""))  # 防止过长

    # 5) 社区/模块度（整体 & 按关系类型）
    comm_overall, comm_by_type = evaler.community_and_modularity_full()
    print("[Community/Modularity Overall]")
    print(json.dumps(comm_overall, indent=2))
    print("[Community/Modularity By Relation Type]")
    print(json.dumps(comm_by_type, indent=2)[:2000] + ("...\n" if len(json.dumps(comm_by_type))>2000 else ""))

    # 6) 汇总保存
    SUMMARY_JSON = os.path.join(OUT_DIR, 'summary_full.json')
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'HS2_overall': hs_overall,
            'HS2_by_reltype': hs_by_type,
            'Community_overall': comm_overall,
            'Community_by_reltype': comm_by_type
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {SUMMARY_JSON}")

    # 7) 可视化（可选）
    PLOT = True
    if PLOT:
        plotter = KGPlotter(OUT_DIR)
        if SAVE_PRED_EDGE_SCORES:
            plotter.plot_hs2_hist(PRED_EDGE_SCORES_PATH, NULL_SCORES_PATH, bins=100)
        plotter.plot_scr_bar(scr_obs=comm_overall['SCR_obs'], scr_pred=comm_overall['SCR_pred'])
