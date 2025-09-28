# kg_pred_evaluator.py
import os
import math
import gc
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class KGPredEvaluator:
    """
    面向大规模KG预测评估的无先验指标计算器（高可扩，节省内存）.

    输入文件（均为列：htype, ttype, h, r, t, conf）：
      - links: 原始已知三元组（多关系、异构），规模可达数千万行
      - pred : 模型新预测三元组（关系名以'PREDICT_'前缀），节点是links子集，主要集中在蛋白/化合物

    指标（均不依赖手工先验）：
      * CN（Common Neighbors）
      * RA（Resource Allocation）
      * AA（Adamic-Adar）
      * Jaccard
      * CN_prob = CN / (|Nh| * |Nt|)   —— 抑制hub的重叠概率视角
      * Overlap = CN / min(|Nh|, |Nt|) —— Szymkiewicz–Simpson overlap coefficient
      * （可选）Path3Approx：统计 E(N(h), N(t)) 的边数（受限展开的3跳近似）
      * （可选）Rule Support：对接 AnyBURL/SAFRAN 规则输出

    设计要点：
      - 单次构建“预测端点的一阶邻居表 + 全图度数(去向无关，按无向计数)”
      - 所有集合默认去重；度数对预测端点用唯一邻居数，交叉邻居的度数优先用唯一邻居数（可回退）
      - 逐步/分块读取，进度条可见；指标计算对每条预测按小邻居优先交集遍历，降低时间复杂度
      - 可选二次扫描构建“邻居的邻居”以支持3跳近似（默认关闭，避免爆内存）
    """

    def __init__(self,
                 links_path: str,
                 pred_path: str,
                 sep: str = ",",
                 chunksize: int = 2_000_000,
                 store_neighbors_for_neighbors: bool = False,
                 path3_max_expand_per_h: int = 50,
                 use_pyarrow: bool = True,
                 node_cap_notice: int = 50_000_000):
        self.links_path = links_path
        self.pred_path = pred_path
        self.sep = sep
        self.chunksize = chunksize
        self.store_neighbors_for_neighbors = store_neighbors_for_neighbors
        self.path3_max_expand_per_h = path3_max_expand_per_h
        self.use_pyarrow = use_pyarrow and self._check_pyarrow()
        self.node_cap_notice = node_cap_notice  # 节流保护提示阈值

        # 基础容器（字符串 -> int 索引）
        self.node2id = dict()
        self.id2node = list()

        # 预测对（内部用id）
        self.pred_pairs = []   # list of (hid, rid(optional), tid, conf_float, r_str)
        self.pred_nodes = set()   # 所有预测端点节点id

        # 图索引
        self.neigh_pred = defaultdict(set)   # 仅为预测端点保存一阶邻居（去重）
        self.deg_total = defaultdict(int)    # 全图无向度数（计数，可能含多关系重复，但对邻接我们去重）
        self.num_edges_count = 0             # 计的边数（无向，按三元组计数，可能包含重复关系）
        self.unique_neighbor_counts_ready = False

        # 若开启3跳近似/或需邻居节点的唯一度数：
        self.neigh_neighbors = None          # dict: node_id -> set(neighbors) for nodes in neighbor-of-pred

        # 列名假设
        self.col_h = "h"; self.col_t = "t"; self.col_r = "r"; self.col_conf = "conf"
        self.usecols = ["htype","ttype","h","r","t","conf"]

        # 可视化输出目录自动创建
        self._ensure_dir(os.path.dirname(links_path))
        self._ensure_dir(os.path.dirname(pred_path))

    # -------------------- 工具方法 --------------------
    def _check_pyarrow(self):
        try:
            import pyarrow  # noqa
            return True
        except Exception:
            return False

    def _ensure_dir(self, d):
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _nid(self, node):
        """字符串节点映射为连续int id"""
        if node in self.node2id:
            return self.node2id[node]
        nid = len(self.id2node)
        self.node2id[node] = nid
        self.id2node.append(node)
        if nid > self.node_cap_notice and nid % 5_000_000 == 0:
            print(f"[warn] node id reached {nid:,}. Ensure you have enough RAM.")
        return nid

    # -------------------- 预处理流程 --------------------
    def preprocess(self):
        """
        1) 先扫描pred，收集预测端点集合、预测三元组列表（全部转id）
        2) 扫描links，构建：预测端点一阶邻居(neigh_pred) + 全图度数(deg_total)
           —— 为节省内存，不保存全图邻接，仅对预测端点保存邻接
        3) 如需3跳近似/邻居唯一度数：二次扫描，构建“邻居节点”的邻接（neigh_neighbors）
        """
        print("\n[Step 1/2] Reading PRED to collect target nodes & triples...")
        self._scan_pred_build_targets()

        print("\n[Step 2/2] Scanning LINKS to build neighbor index (for pred nodes) & degrees...")
        self._scan_links_build_index()

        if self.store_neighbors_for_neighbors:
            print("\n[Optional] Second pass to build neighbors-of-neighbors (for Path-3 approx / unique deg)...")
            self._scan_links_build_neighbors_of_neighbors()

        print("\n[Done] Preprocess complete.")
        gc.collect()

    def _scan_pred_build_targets(self):
        reader = pd.read_csv(
            self.pred_path,
            sep=self.sep,
            usecols=self.usecols,
            dtype={self.col_conf: "float32"},
            chunksize=self.chunksize,
            engine="c",
            low_memory=False
        )
        for chunk in tqdm(reader, desc="Reading pred.csv", unit="chunk"):
            # 清理列类型
            chunk[self.col_h] = chunk[self.col_h].astype(str)
            chunk[self.col_t] = chunk[self.col_t].astype(str)
            chunk[self.col_r] = chunk[self.col_r].astype(str)
            # conf 可能有缺失
            if self.col_conf not in chunk.columns:
                chunk[self.col_conf] = np.nan
            # 建索引
            for h, r, t, conf in zip(chunk[self.col_h].values,
                                     chunk[self.col_r].values,
                                     chunk[self.col_t].values,
                                     chunk[self.col_conf].values):
                hid = self._nid(h)
                tid = self._nid(t)
                self.pred_nodes.add(hid); self.pred_nodes.add(tid)
                self.pred_pairs.append((hid, None, tid, float(conf) if pd.notna(conf) else np.nan, r))
        print(f"  - Pred nodes: {len(self.pred_nodes):,}")
        print(f"  - Pred triples: {len(self.pred_pairs):,}")

    def _scan_links_build_index(self):
        # 仅统计与图结构相关的字段，加快读取
        reader = pd.read_csv(
            self.links_path,
            sep=self.sep,
            usecols=[self.col_h, self.col_r, self.col_t],
            chunksize=self.chunksize,
            engine="c",
            low_memory=False
        )
        for chunk in tqdm(reader, desc="Scanning links.csv", unit="chunk"):
            chunk[self.col_h] = chunk[self.col_h].astype(str)
            chunk[self.col_t] = chunk[self.col_t].astype(str)
            # 无向处理：度数按无向+1
            for h, t in zip(chunk[self.col_h].values, chunk[self.col_t].values):
                hid = self._nid(h); tid = self._nid(t)
                # 全图度数（按三元组计，可能含多关系重复；对邻接我们会去重）
                self.deg_total[hid] += 1
                self.deg_total[tid] += 1
                self.num_edges_count += 1  # 以三元组为单位（无向贡献一次）
                # 若是预测端点，则保存唯一邻居（避免多关系重复）
                if hid in self.pred_nodes:
                    self.neigh_pred[hid].add(tid)
                if tid in self.pred_nodes:
                    self.neigh_pred[tid].add(hid)
        # 统计预测端点的唯一度（用于Jaccard/Overlap等）
        self.pred_unique_deg = {nid: len(self.neigh_pred.get(nid, ())) for nid in self.pred_nodes}
        self.unique_neighbor_counts_ready = True

        print(f"  - total node ids seen: {len(self.id2node):,}")
        print(f"  - counted triple-edges (undirected): {self.num_edges_count:,}")
        print(f"  - pred endpoints with neighbors indexed: {len(self.neigh_pred):,}")

    def _scan_links_build_neighbors_of_neighbors(self):
        """
        为“预测端点的一阶邻居集合”中的所有节点（称M）构建其唯一邻接。
        这样可：
         - 计算更准确的 RA/AA（用唯一度数而非多关系计数）
         - 支持 Path-3 近似（统计 E(N(h), N(t)) 的边数）
        注：可能较耗内存，请谨慎开启。
        """
        # M = 所有预测端点的邻居并集
        M = set()
        for nid in self.pred_nodes:
            M.update(self.neigh_pred.get(nid, ()))
        print(f"  - neighbor-of-pred node count: {len(M):,}")

        self.neigh_neighbors = defaultdict(set)

        reader = pd.read_csv(
            self.links_path,
            sep=self.sep,
            usecols=[self.col_h, self.col_r, self.col_t],
            chunksize=self.chunksize,
            engine="c",
            low_memory=False
        )
        for chunk in tqdm(reader, desc="2nd pass for neighbors-of-neighbors", unit="chunk"):
            chunk[self.col_h] = chunk[self.col_h].astype(str)
            chunk[self.col_t] = chunk[self.col_t].astype(str)
            for h, t in zip(chunk[self.col_h].values, chunk[self.col_t].values):
                hid = self._nid(h); tid = self._nid(t)
                if hid in M:
                    self.neigh_neighbors[hid].add(tid)
                if tid in M:
                    self.neigh_neighbors[tid].add(hid)

        print(f"  - built neighbor-of-neighbor adjacency for {len(self.neigh_neighbors):,} nodes.")

    # -------------------- 指标计算 --------------------
    def compute_similarity_metrics(self, save_to=None):
        """
        计算局部相似性相关指标（全部无先验）：
          CN, RA, AA, Jaccard, CN_prob, Overlap, deg_h, deg_t
        返回 DataFrame（含原始 r, conf）
        """
        assert self.unique_neighbor_counts_ready, "Call preprocess() first."

        results = {
            "h": [], "t": [], "r": [], "conf": [],
            "deg_h": [], "deg_t": [],
            "CN": [], "RA": [], "AA": [], "Jaccard": [],
            "CN_prob": [], "Overlap": []
        }

        # 为加速，提前把需要的邻居集引用提取出来
        # 注意：neigh_pred 仅保证预测端点有邻居集
        getN = self.neigh_pred.get
        deg_uniq = self.pred_unique_deg  # 预测端点唯一度

        # 如果构建了neigh_neighbors，则交叉邻居的度数可用唯一度替代；否则回退用deg_total
        use_neighbor_unique_degree = self.neigh_neighbors is not None
        if use_neighbor_unique_degree:
            # 预构一个快速取唯一度的函数
            def uniq_deg(u):
                nb = self.neigh_neighbors.get(u)
                return len(nb) if nb is not None else self.deg_total.get(u, 0)
        else:
            def uniq_deg(u):
                return self.deg_total.get(u, 0)

        # 主循环
        for (hid, _, tid, conf, rstr) in tqdm(self.pred_pairs, desc="Computing similarity metrics", unit="triple"):
            Nh = getN(hid, None)
            Nt = getN(tid, None)

            if not Nh or not Nt:
                # 任一端在links中无邻居，所有基于邻域的指标为0
                cn = 0
                ra = 0.0
                aa = 0.0
                jac = 0.0
                cn_prob = 0.0
                overlap = 0.0
                deg_h = len(Nh) if Nh else 0
                deg_t = len(Nt) if Nt else 0
            else:
                # 交集按小集合遍历，避免大集合遍历开销
                if len(Nh) <= len(Nt):
                    small, big = Nh, Nt
                else:
                    small, big = Nt, Nh

                cn = 0
                ra = 0.0
                aa = 0.0
                # intersection &贡献
                for u in small:
                    if u in big:
                        cn += 1
                        du = max(uniq_deg(u), 1)  # 防0
                        # RA：sum(1/deg(u))
                        ra += 1.0 / du
                        # AA：sum(1/log(deg(u)+e))
                        # 常用是 log(deg(u)), 避免0/1取log问题，用log1p或加epsilon
                        aa += 1.0 / max(math.log(du), 1e-9)

                deg_h = deg_uniq.get(hid, 0)
                deg_t = deg_uniq.get(tid, 0)

                union = deg_h + deg_t - cn
                jac = (cn / union) if union > 0 else 0.0

                # CN概率视角：CN / (|Nh| * |Nt|)
                denom = deg_h * deg_t
                cn_prob = (cn / denom) if denom > 0 else 0.0

                # Overlap：CN / min(|Nh|, |Nt|)
                denom2 = min(deg_h, deg_t)
                overlap = (cn / denom2) if denom2 > 0 else 0.0

            results["h"].append(self.id2node[hid])
            results["t"].append(self.id2node[tid])
            results["r"].append(rstr)
            results["conf"].append(conf)
            results["deg_h"].append(deg_h)
            results["deg_t"].append(deg_t)
            results["CN"].append(cn)
            results["RA"].append(ra)
            results["AA"].append(aa)
            results["Jaccard"].append(jac)
            results["CN_prob"].append(cn_prob)
            results["Overlap"].append(overlap)

        df = pd.DataFrame(results)
        if save_to:
            self._ensure_dir(os.path.dirname(save_to))
            if save_to.endswith(".parquet"):
                if self.use_pyarrow:
                    df.to_parquet(save_to, index=False)
                else:
                    # 无pyarrow也支持，pandas会fallback到fastparquet（若安装）或报错
                    df.to_parquet(save_to, index=False)
            else:
                df.to_csv(save_to, index=False)
            print(f"[saved] metrics -> {save_to}")
        return df

    def compute_path3_scores(self, df_metrics: pd.DataFrame, col_name="Path3Approx", save_to=None):
        """
        （可选）3跳路径近似：统计 E(N(h), N(t)) 的边数
        思路：对每个(h,t)，遍历N(h)中最多K个邻居u，累加 |N(u) ∩ N(t)|
        需要 neigh_neighbors 已构建（即 store_neighbors_for_neighbors=True）
        """
        assert self.neigh_neighbors is not None, "Enable store_neighbors_for_neighbors=True and preprocess() again."

        getNh = self.neigh_pred.get
        getNu = self.neigh_neighbors.get

        vals = []
        for h, t in tqdm(zip(df_metrics["h"].values, df_metrics["t"].values),
                         total=len(df_metrics), desc="Computing Path3 approx", unit="triple"):
            hid = self.node2id[h]; tid = self.node2id[t]
            Nh = getNh(hid, None); Nt = getNh(tid, None)

            if not Nh or not Nt:
                vals.append(0)
                continue

            # 选择N(h)中前K个邻居（可按其唯一度从小到大或大到小，这里按度小优先避免高hub爆炸）
            # 取唯一度（若无则回退deg_total）
            cand = []
            for u in Nh:
                Nu = getNu(u, None)
                du = len(Nu) if Nu is not None else self.deg_total.get(u, 0)
                cand.append((du, u))
            cand.sort(key=lambda x: x[0])  # 度小优先
            expand = [u for _, u in cand[:self.path3_max_expand_per_h]]

            cnt = 0
            Nt_set = Nt  # set
            for u in expand:
                Nu = getNu(u, None)
                if not Nu:
                    continue
                # 交集计数（遍历小集合）
                if len(Nu) <= len(Nt_set):
                    small, big = Nu, Nt_set
                else:
                    small, big = Nt_set, Nu
                for v in small:
                    if v in big:
                        cnt += 1
            vals.append(cnt)

        df_metrics[col_name] = vals
        if save_to:
            self._ensure_dir(os.path.dirname(save_to))
            if save_to.endswith(".parquet"):
                df_metrics.to_parquet(save_to, index=False)
            else:
                df_metrics.to_csv(save_to, index=False)
            print(f"[saved] metrics+Path3 -> {save_to}")
        return df_metrics

    def merge_rule_support(self, rule_file: str, df_metrics: pd.DataFrame,
                           key_cols=("h","r","t"), rule_score_col="rule_support",
                           sep=","):
        """
        （可选）将规则系统（AnyBURL/SAFRAN等）输出的 (h,r,t,score) 与df_metrics对齐。
        规则文件应含列：h,r,t,rule_support（或你在 rule_score_col 指定的列名）。
        未命中者记为0。
        """
        if not os.path.exists(rule_file):
            print(f"[warn] rule file not found: {rule_file}")
            df_metrics[rule_score_col] = 0.0
            return df_metrics

        df_rule = pd.read_csv(rule_file, sep=sep)
        if rule_score_col not in df_rule.columns:
            # 尝试自动识别（如 "confidence" 或 "score"）
            for c in ["confidence", "score", "conf"]:
                if c in df_rule.columns:
                    df_rule = df_rule.rename(columns={c: rule_score_col})
                    break
        if rule_score_col not in df_rule.columns:
            raise ValueError("rule file must have a rule_support/confidence/score column.")

        df_rule_small = df_rule[list(key_cols) + [rule_score_col]].copy()
        df = df_metrics.merge(df_rule_small, on=list(key_cols), how="left")
        df[rule_score_col] = df[rule_score_col].fillna(0.0)
        return df

    # -------------------- 可视化 --------------------
    def plot_histograms(self, df_metrics: pd.DataFrame, columns=None, bins=100, out_prefix=None):
        """
        生成若干指标的直方图（png）
        """
        import matplotlib.pyplot as plt

        if columns is None:
            columns = ["conf", "CN", "RA", "AA", "Jaccard", "CN_prob", "Overlap"]
            if "Path3Approx" in df_metrics.columns:
                columns.append("Path3Approx")
            if "rule_support" in df_metrics.columns:
                columns.append("rule_support")

        for col in columns:
            if col not in df_metrics.columns:
                continue
            data = df_metrics[col].values
            # 过滤NaN/inf
            data = data[np.isfinite(data)]
            plt.figure()
            plt.hist(data, bins=bins)
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"Histogram of {col} (n={len(data):,})")
            if out_prefix:
                self._ensure_dir(os.path.dirname(out_prefix))
                out_path = f"{out_prefix}_{col}.png"
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

    def plot_conf_scatter(self, df_metrics: pd.DataFrame, x="conf", y="RA", out_path=None, sample=200_000):
        """
        画一个散点：模型conf vs 某指标（抽样，避免过多点）
        """
        import matplotlib.pyplot as plt
        df = df_metrics[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) > sample:
            df = df.sample(sample, random_state=42)
        plt.figure()
        plt.scatter(df[x].values, df[y].values, s=2, alpha=0.3)
        plt.xlabel(x); plt.ylabel(y)
        plt.title(f"{x} vs {y} (n={len(df):,})")
        if out_path:
            self._ensure_dir(os.path.dirname(out_path))
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

# -------------------- 直接运行支持 --------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # ap.add_argument("--links", required=True)
    # ap.add_argument("--pred", required=True)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--use_pyarrow", action="store_true")
    ap.add_argument("--store_neighbors_for_neighbors", action="store_true")
    ap.add_argument("--path3_max_expand_per_h", type=int, default=50)
    ap.add_argument("--save_to", default='screen_cache/unibiomap.pred.support.parquet')
    args = ap.parse_args()

    e = KGPredEvaluator(
        links_path="screen_cache/unibiomap.links.csv",
        pred_path="screen_cache/unibiomap.pred.csv",
        sep=",",
        chunksize=2_000_000,          # 按内存情况调
        store_neighbors_for_neighbors=False,  # 若要计算3跳近似，可设为True（更耗内存）
        path3_max_expand_per_h=50,     # 3跳近似时每个h最多展开多少一阶邻居
        use_pyarrow=True               # 装了pyarrow就开
    )
    e.preprocess()
    df = e.compute_similarity_metrics(save_to=args.save_to)
    
    df = e.compute_path3_scores(df, save_to=args.save_to)


