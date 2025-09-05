# -*- coding: utf-8 -*-
"""
model.py — NoisyPUDA (pair-wise Noisy-PU loss, PUDA-style training)

完全对齐 PUDA 的训练管线：
- G-step: 用判别器的 real 得分 + 生成器产生的假负得分拼接为 pred，最小化 -PUR（这里等价为最大化带噪 pair-wise 难度）；只更新 G
- D-step: 重新前向计算 real；重新采一批假负（与 G-step 不同批），拼接 pred，最小化带噪 PUR；只更新判别器（嵌入+打分器），并加 L2 正则

参考对齐点：
1) 两步式循环与 pred = cat([pred_real, pred_fake]) 再喂 PUR（G: 取负；D: 取正）  —— 见 PUDA 训练主循环与 pur_loss 用法
2) get_gen_neg(flag='gen'/'dis')：G-step 不让判别器图参与反传；D-step 不让生成器图参与反传 —— 见 PUDA.get_gen_neg
3) 判别器在 D-step 叠加 L2 正则 —— 见 PUDA.DistMult 返回 l2_reg 并加入 loss_dis

【注】本版本按当前方案：D-step 不把伪负并入 U；G-step 额外加入“feature matching”正则以对齐生成向量分布到真实实体流形；PU 风险支持可选样本级 τ（默认退化为常量 1-η，与原式一致）。
"""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.retrieval as rt

# =============================
# 常量（关系先验）
# =============================
DEFAULT_PI = 1e-4   # P(y=1)
# DEFAULT_ETA = 2e-3   # 噪声率
DEFAULT_ETA = 2e-3   # 噪声率

# =============================
# 基础组件：FiLM + TypedEmbedding
# =============================
class TypeFiLM(nn.Module):
    """按类型做 FiLM 调制：x -> x * (1 + tanh(gamma_t)) + beta_t"""
    def __init__(self, num_t: int, dim: int):
        super().__init__()
        self.dim = dim
        self.type_emb = nn.Embedding(num_t, dim * 2)
        nn.init.xavier_uniform_(self.type_emb.weight)

    def forward(self, x: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        tb = self.type_emb(t_ids)                      # [N, 2D]
        gamma, beta = tb[:, : self.dim], tb[:, self.dim :]
        return x * (1.0 + torch.tanh(gamma)) + beta


class TypedEmbedding(nn.Module):
    """实体/关系嵌入 + 由关系决定的(头,尾)类型，再用 FiLM 调制实体向量。"""
    def __init__(self, num_e: int, num_r: int, num_t: int, dim: int, rel_types: dict):
        super().__init__()
        self.emb_e = nn.Embedding(num_e, dim)
        self.emb_r = nn.Embedding(num_r, dim)
        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)
        self.film = TypeFiLM(num_t, dim)

        # 注册 r_id -> (head_type, tail_type)
        head_t = torch.empty(num_r, dtype=torch.long)
        tail_t = torch.empty(num_r, dtype=torch.long)
        for r_id, (ht, tt) in rel_types.items():
            head_t[r_id] = int(ht)
            tail_t[r_id] = int(tt)
        self.register_buffer("rel_head_type", head_t, persistent=False)
        self.register_buffer("rel_tail_type", tail_t, persistent=False)

    def forward(self, triples: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # triples: [N,3] -> (h_id, r_id, t_id)
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        eh = self.emb_e(h)                         # [N,D]
        er = self.emb_r(r)                         # [N,D]
        et = self.emb_e(t)                         # [N,D]

        # 由 r_id 查类型，再做 FiLM 调制（仅对实体）
        th = self.rel_head_type[r]                 # [N]
        tt = self.rel_tail_type[r]                 # [N]
        eh = self.film(eh, th)
        et = self.film(et, tt)
        return eh, er, et


# =============================
# 判别器：DistMult（返回 logits 与 per-batch L2 正则）
# =============================
class KGEDistMult(nn.Module):
    def forward(self, eh: torch.Tensor, er: torch.Tensor, et: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = (eh * er * et).sum(dim=-1)  # logits
        l2_reg = (eh.pow(2).mean() + er.pow(2).mean() + et.pow(2).mean())
        return score, l2_reg


# =============================
# 生成器：从噪声 z 生成“假头/假尾”的实体向量（与 PUDA 一致的接口）
# =============================
class NoiseGenerator(nn.Module):
    """
    输入: z ∈ R^{B,K,L}，输出: 生成的实体嵌入 ∈ R^{B,K,D}
    这里用简单 MLP + Tanh 有界（与 PUDA 的有界生成一致性）
    """
    def __init__(self, emb_dim: int, latent_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim or max(8, emb_dim // 8)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, emb_dim * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),   # 有界输出，避免无界拉差
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, K, L = z.shape
        out = self.net(z.view(B * K, L))
        return out.view(B, K, self.emb_dim)


# =============================
# 带噪 PU 风险（支持 pair-wise U 部分 & 自对抗权重 & 可选样本级 τ）
# =============================
class NoisyNNPULoss(nn.Module):
    """
    输入:
    - s_pos: [B]（正例 logits）
    - s_unl: [B, U]（未标注 logits）
    - pi, eta: 可为 [B]（按关系）或标量
    形式：pi(1-eta)R_p^+ + [R_u^- - pi(1-eta)R_p^-]_+ + pi*eta*R_p^-
    surrogate 使用 logistic：softplus

    额外开关：
    - pairwise: True 时，把 U 风险按 pair-wise 计算：softplus(s_unl - s_pos)
                False 时，按 point-wise 计算：softplus(s_unl)
    - self_adv: True 时，对 U 的每组样本用 softmax(alpha * score) 加权（自对抗加权）
    - tau_pos:  样本级置信度 τ(x) ∈ [0,1]（只作用在三处 P 项）。若不传，则 τ≡(1-eta)，与原式完全一致。
    """
    def __init__(self, pairwise_default: bool = True, self_adv_default: bool = False, self_adv_alpha: float = 10.0):
        super().__init__()
        self.pairwise_default = pairwise_default
        self.self_adv_default = self_adv_default
        self.self_adv_alpha = self_adv_alpha

    def forward(self, s_pos: torch.Tensor, s_unl: torch.Tensor,
                pi: torch.Tensor, eta: torch.Tensor,
                pairwise: bool = None, self_adv: bool = None, self_adv_alpha: float = None,
                tau_pos: Optional[Union[torch.Tensor, float]] = None) -> torch.Tensor:
        if s_unl.dim() == 1:
            s_unl = s_unl.view(s_pos.size(0), -1)

        pairwise = self.pairwise_default if pairwise is None else pairwise
        self_adv = self.self_adv_default if self_adv is None else self_adv
        alpha = self.self_adv_alpha if self_adv_alpha is None else float(self_adv_alpha)

        # logistic surrogate
        l_pos  = F.softplus(-s_pos)                # R_p^+
        l_pneg = F.softplus(s_pos)                 # R_p^-

        if pairwise:
            # U 风险按 pair-wise：对每个未标注与该组正例做 margin
            margin = s_unl - s_pos.view(-1, 1)
            base_uneg = F.softplus(margin)
            weight_scores = margin
        else:
            # U 风险按 point-wise：直接把未标注当负类 surrogate
            base_uneg = F.softplus(s_unl)
            weight_scores = s_unl

        if self_adv:
            w = F.softmax(alpha * weight_scores, dim=1)           # [B,U]
            l_uneg = (w * base_uneg).sum(dim=1)
        else:
            l_uneg = base_uneg.mean(dim=1)

        # ===== 样本级 τ(x)：只作用在三处 P 项；不传时退化为 τ=1-eta（与原式一致） =====
        if tau_pos is None:
            tau = 1.0 - eta
        elif isinstance(tau_pos, float):
            tau = torch.full_like(s_pos, float(tau_pos))
        else:
            # 传入张量时，允许 [B] 或 [B,1] 形状
            tau = tau_pos.view(-1).to(s_pos.dtype)

        # 三处 P 项
        pos_term   = (pi * tau) * l_pos
        deduct_pn  = (pi * tau) * l_pneg
        noise_term = (pi * (1.0 - tau)) * l_pneg

        neg_risk = torch.clamp_min(l_uneg - deduct_pn, 0.0)
        return (pos_term + neg_risk + noise_term).mean()


# =============================
# LightningModule
# =============================
class NoisyPUDA(pl.LightningModule):
    def __init__(self,
                 bm,
                 emb_dim: int = 256,
                 num_ng_gen: int = 8,
                 lr_d: float = 1e-3,
                 lr_g: float = 1e-3,
                 reg: float = 0.0,
                 gen_std: float = 0.1,
                 gen_dropout: float = 0.1,
                 use_adversarial: bool = True,

                 # === 新增：U 风险 pair-wise 开关 & 自对抗加权 ===
                 pairwise_u: bool = True,
                 self_adv: bool = False,
                 self_adv_alpha: float = 10.0,
                 # === 新增：G 的目标/对抗强度 ===
                 g_obj: str = "linear",            # {"linear", "ns"}: 线性 -E[s_fake]；非饱和 softplus(-s_fake)
                 beta_fake: float = 1.0,           # D-step 的额外“拉低伪负”权重（未并入 U）
                 # === 新增：Feature Matching 正则权重（G-step）===
                 fm_gamma: float = 0.5,
                 fm2_gamma: float = 0.0,
                 
                 default_pi: float = DEFAULT_PI,
                 default_eta: float = DEFAULT_ETA,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["bm"])
        self.bm = bm

        # 模型组件
        self.emb = TypedEmbedding(bm.num_e, bm.num_r, bm.num_t, emb_dim, bm.rel_types)
        self.dis = KGEDistMult()
        self.gen = NoiseGenerator(emb_dim, dropout=gen_dropout)

        # 带开关的带噪 PU 风险（支持样本级 τ）
        self.loss_fn = NoisyNNPULoss(pairwise_default=pairwise_u,
                                     self_adv_default=self_adv,
                                     self_adv_alpha=self_adv_alpha)

        self.emb_dim = emb_dim
        self.num_ng_gen = int(num_ng_gen)          # 生成器每个正例的伪负个数（总数）；内部各一半用于“假尾/假头”
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.reg = reg
        self.gen_std = gen_std

        # === 控制开关 ===
        self.use_adversarial = bool(use_adversarial)
        self.pairwise_u = bool(pairwise_u)
        self.self_adv = bool(self_adv)
        self.self_adv_alpha = float(self_adv_alpha)

        # 生成器目标 & 伪负额外项权重
        assert g_obj in {"linear", "ns"}
        self.g_obj = g_obj
        self.beta_fake = float(beta_fake)

        # Feature Matching 正则
        self.fm_gamma = float(fm_gamma)
        self.fm2_gamma = float(fm2_gamma)

        self.automatic_optimization = False

        self.default_pi = float(default_pi)
        self.default_eta = float(default_eta)
        # === Validation/Test 检索指标 ===
        self.top_k = (3, 10)
        self.val_metrics = nn.ModuleDict({
            "MRR": rt.RetrievalMRR(),
            **{f"Hits@{k}": rt.RetrievalHitRate(top_k=k) for k in self.top_k},
            **{f"nDCG@{k}": rt.RetrievalNormalizedDCG(top_k=k) for k in self.top_k},
            # **{f"Recall@{k}\": rt.RetrievalRecall(top_k=k) for k in self.top_k},
            # **{f"Precision@{k}\": rt.RetrievalPrecision(top_k=k) for k in self.top_k},
            # **{f"MAP@{k}\": rt.RetrievalMAP(top_k=k) for k in self.top_k},
            "AUROC": rt.RetrievalAUROC(),
        })
        self._val_losses = []


    # ---------- 工具 ----------
    def _pi_eta_for_rel_ids(self, r_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从 bm 取每个关系的先验（若不存在则回落到默认值）
        device = r_ids.device
        pi = torch.full_like(r_ids, float(self.default_pi), dtype=torch.float32, device=device)
        eta = torch.full_like(r_ids, float(self.default_eta), dtype=torch.float32, device=device)
        if hasattr(self.bm, "pi"):
            for rid, val in self.bm.pi.items():
                pi[r_ids == int(rid)] = float(val)
        if hasattr(self.bm, "eta"):
            for rid, val in self.bm.eta.items():
                eta[r_ids == int(rid)] = float(val)
        return pi, eta

    @staticmethod
    def _view_group(x: torch.Tensor, bs: int, d: int) -> torch.Tensor:
        """把 [B*(1+K), D] 还原为 [B, 1+K, D]，以便取每组第 0 个（正例）。"""
        return x.view(bs, -1, d)

    def _get_gen_neg(self,
                     h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor,
                     bs: int, flag: str, r_pos_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        与 PUDA.get_gen_neg 完全一致的接口/逻辑（新增：传入 r_pos_ids 用于对生成向量做 FiLM，使其风格与真实一致）：
        - 采样两份噪声 z_head, z_tail ∈ R^{B, K, L}
        - flag='gen'：生成器前向可训练；(h,r,t) 全部 detach（阻断到判别器/嵌入的梯度）
        - flag='dis'：生成器输出 detach；(h,r,t) 可训练（用于更新判别器/嵌入）
        - 输出拼成 2K 条：假尾（用 neg_tail 替换 t）与假头（用 neg_head 替换 h）
        """
        K = max(1, self.num_ng_gen // 2)
        D = h_emb.size(-1)
        L = max(8, D // 8)

        z_tail = torch.normal(mean=0.0, std=self.gen_std, size=(bs, K, L), device=h_emb.device)
        z_head = torch.normal(mean=0.0, std=self.gen_std, size=(bs, K, L), device=h_emb.device)

        if flag == 'gen':
            neg_gen_tail = self.gen(z_tail)              # [B,K,D]（对 G 可导）
            neg_gen_head = self.gen(z_head)
            # 判别器侧嵌入全部脱图
            h_emb = h_emb.detach()
            r_emb = r_emb.detach()
            t_emb = t_emb.detach()
        elif flag == 'dis':
            neg_gen_tail = self.gen(z_tail).detach()     # 对 D 不回传到 G
            neg_gen_head = self.gen(z_head).detach()
        else:
            raise ValueError("flag must be 'gen' or 'dis'")

        # —— 新增：让生成的 head/tail 经过与真实一致的类型 FiLM ——
        # 取每组正例的关系 r，用于查 (head_type, tail_type)
        th = self.emb.rel_head_type[r_pos_ids]    # [B]
        tt = self.emb.rel_tail_type[r_pos_ids]    # [B]
        K = neg_gen_head.size(1)
        D = neg_gen_head.size(2)
        th_expand = th.view(bs, 1).expand(-1, K).reshape(-1)
        tt_expand = tt.view(bs, 1).expand(-1, K).reshape(-1)
        # 展平做 FiLM，再还原形状 [B,K,D]
        neg_gen_head = self.emb.film(neg_gen_head.reshape(-1, D), th_expand).view(bs, K, D)
        neg_gen_tail = self.emb.film(neg_gen_tail.reshape(-1, D), tt_expand).view(bs, K, D)

        # 取每组第 0 条（正例）并扩展到 [B,K,D]
        h0 = self._view_group(h_emb, bs, D)[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
        r0 = self._view_group(r_emb, bs, D)[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
        t0 = self._view_group(t_emb, bs, D)[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)

        # 组装 2K 条：（假尾：替换 t），（假头：替换 h）
        h_out = torch.cat([h0,          neg_gen_head], dim=1).reshape(-1, D)  # [B,2K,D] -> [B*2K,D]
        r_out = torch.cat([r0,          r0         ], dim=1).reshape(-1, D)
        t_out = torch.cat([neg_gen_tail, t0        ], dim=1).reshape(-1, D)

        return h_out, r_out, t_out

    # ---------- Lightning API ----------
    def configure_optimizers(self):
        opt_d = torch.optim.Adam(list(self.emb.parameters()) + list(self.dis.parameters()), lr=self.lr_d)
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g)
        return [opt_d, opt_g]

    def forward(self, triples: torch.LongTensor) -> torch.Tensor:
        """
        接收 [N,3] 或 [B,C,3] 的三元组张量，返回对应 logits（一维展平）。
        用于推理/评分；不返回正则项。
        """
        if triples.dim() == 3:
            triples = triples.reshape(-1, triples.size(-1))  # [B*C,3]
        eh, er, et = self.emb(triples)                       # [*,D]
        scores, _ = self.dis(eh, er, et)                     # [*]
        return scores.view(-1)

    # 兼容你外面的 score_triples/forward_triples 调用习惯（可选）
    @torch.no_grad()
    def score_triples(self, triples: torch.LongTensor) -> torch.Tensor:
        return self.forward(triples)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        PUDA-style 两步（完全对齐原版）：
        1) G-step: 用当前判别器的 real 得分 + G 生成的假负得分拼接 pred，最小化 -PUR（这里用带噪版本）；只更新 G
        2) D-step: 重新前向 real；用 flag='dis' 重新采一批假负并拼接 pred，最小化带噪 PUR + L2；只更新判别器
        """
        opt_d, opt_g = self.optimizers()
        B, C, _ = batch.size()

        # 展平为 [B*(1+K_data), 3]
        X = batch.reshape(-1, 3)

        # ==========
        # Train G
        # ==========
        if self.use_adversarial:
            # 判别器前向（real/data-unlabeled）
            eh, er, et = self.emb(X)                    # [B*(1+K_data), D]
            Ddim = eh.size(-1)

            # 取每组正例的实体向量（作为真实参考用于 FM）
            eh_g = self._view_group(eh, B, Ddim)[:, 0, :]   # [B,D]
            et_g = self._view_group(et, B, Ddim)[:, 0, :]   # [B,D]

            r_ids = batch[:, 0, 1]                      # 正例的关系 id: [B]
            h_gen_all, r_gen_all, t_gen_all = self._get_gen_neg(eh, er, et, bs=B, flag='gen', r_pos_ids=r_ids)
            s_fake, _ = self.dis(h_gen_all, r_gen_all, t_gen_all)   # [B*2K]

            # —— G 的主目标：难负（线性 / 非饱和） ——
            if self.g_obj == "linear":
                loss_g_main = - s_fake.mean()
            else:  # "ns"
                loss_g_main = F.softplus(-s_fake).mean()

            # —— Feature Matching 正则（只对齐一阶；可选二阶） ——
            fm_loss = torch.tensor(0.0, device=X.device)
            if self.fm_gamma > 0.0 or self.fm2_gamma > 0.0:
                K = max(1, self.num_ng_gen // 2)
                # 从拼接的输出中切分出“生成的 head/ tail”部分
                gen_head = h_gen_all[B*K:, :]       # 后半段是生成的 head
                gen_tail = t_gen_all[:B*K, :]       # 前半段是生成的 tail

                mean_h_real = eh_g.mean(dim=0)
                mean_t_real = et_g.mean(dim=0)
                mean_h_gen  = gen_head.mean(dim=0)
                mean_t_gen  = gen_tail.mean(dim=0)

                fm1 = (mean_h_gen - mean_h_real).pow(2).mean() + (mean_t_gen - mean_t_real).pow(2).mean()
                fm_loss = self.fm_gamma * fm1

                if self.fm2_gamma > 0.0:
                    m2_h_real = (eh_g * eh_g).mean(dim=0)
                    m2_t_real = (et_g * et_g).mean(dim=0)
                    m2_h_gen  = (gen_head * gen_head).mean(dim=0)
                    m2_t_gen  = (gen_tail * gen_tail).mean(dim=0)
                    fm2 = (m2_h_gen - m2_h_real).pow(2).mean() + (m2_t_gen - m2_t_real).pow(2).mean()
                    fm_loss = fm_loss + self.fm2_gamma * fm2

            loss_g = loss_g_main + fm_loss

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()
            self.untoggle_optimizer(opt_g)  # <== 关 G
        else:
            loss_g = torch.tensor(0.0, device=X.device)
            loss_g_main = torch.tensor(0.0, device=X.device)
            fm_loss = torch.tensor(0.0, device=X.device)

        self.log("loss_gen", loss_g, prog_bar=True, on_step=True, on_epoch=False)
        self.log("loss_g_main", loss_g_main, prog_bar=False, on_step=True, on_epoch=False)
        self.log("loss_fm", fm_loss, prog_bar=False, on_step=True, on_epoch=False)

        # ==========
        # Train D
        # ==========
        # 重新前向 real（注意：前一步更新了 G，这里不复用上面的张量）
        eh, er, et = self.emb(X)
        pred_real, reg_real = self.dis(eh, er, et)
        pred_real = pred_real.view(B, -1)

        # 生成器产生假负（不并入 U，直接单列额外项）
        r_ids = batch[:, 0, 1]
        pred = pred_real
        reg_all = reg_real
        s_fake_extra = None
        if self.use_adversarial and self.beta_fake > 0:
            h_gen, r_gen, t_gen = self._get_gen_neg(eh, er, et, bs=B, flag='dis', r_pos_ids=r_ids)
            s_fake_extra, _ = self.dis(h_gen, r_gen, t_gen)  # [B*2K]

        s_pos = pred[:, 0]
        s_unl = pred[:, 1:]

        pi_b, eta_b = self._pi_eta_for_rel_ids(r_ids)

        # —— D 的主损失：带噪 nnPU（可选 pair-wise U + 自对抗加权 + 可选 τ） ——
        loss_d = self.loss_fn(s_pos, s_unl, pi_b, eta_b,
                              pairwise=self.pairwise_u, self_adv=self.self_adv, self_adv_alpha=self.self_adv_alpha,
                              tau_pos=None)  # 第一步训练：τ 接口打通，但默认退化为 τ=1-eta

        # —— 可选的“伪负拉低”额外项（未并入 U 时启用） ——
        if s_fake_extra is not None and self.beta_fake > 0:
            loss_d = loss_d + self.beta_fake * F.softplus(s_fake_extra).mean()

        # —— 判别器正则（保持与原逻辑一致） ——
        loss_d = loss_d + self.reg * reg_all

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
        self.untoggle_optimizer(opt_d)  # <== 关 D

        # ---- Logging ----
        self.log("loss_dis", loss_d, prog_bar=True, on_step=True, on_epoch=False)

        # 返回一个代表性 loss（用于进度条）；两步均已手动优化
        return {"loss": loss_d.detach()}

    # ====== Validation hooks ======
    def on_validation_epoch_start(self):
        # 跟 gnn.py 一样：准备 loss 容器并 reset 指标
        self._val_losses = []
        for m in self.val_metrics.values():
            m.reset()

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # batch: [B, 1+neg, 3]，第 0 列为正，其余为未标注候选
        B, C, _ = batch.size()
        pos = batch[:, 0, :]                 # [B,3]
        unl = batch[:, 1:, :].reshape(-1, 3) # [B*(C-1),3]

        # 判别器打分（一次前向；只用 logits）
        # eh, er, et = self.emb(pos)
        # s_pos, _ = self.dis(eh, er, et)                          # [B]
        # eh_u, er_u, et_u = self.emb(unl)
        # s_unl, _ = self.dis(eh_u, er_u, et_u)                    # [B*(C-1)]
        # s_unl = s_unl.view(B, C - 1)

        s_pos = self.forward(pos)                      # [B]
        s_unl = self.forward(unl).view(B, C - 1)      # [B, C-1]

        # 验证损失（Noisy-PU 风险；与训练保持同一开关设置；τ 默认退化为 1-eta）
        r_ids = pos[:, 1]
        pi_b, eta_b = self._pi_eta_for_rel_ids(r_ids)
        val_loss = self.loss_fn(s_pos, s_unl, pi_b, eta_b,
                                pairwise=self.pairwise_u, self_adv=self.self_adv, self_adv_alpha=self.self_adv_alpha,
                                tau_pos=None)
        self._val_losses.append(val_loss.detach())
        # === 检索评估 ===
        # 不再额外前向 triples_all；直接用已算好的 s_pos/s_unl 拼完整候选得分
        scores_all = torch.cat([s_pos.view(B, 1), s_unl], dim=1).reshape(-1)  # [B*C]
        # 注意：排序指标对单调变换不敏感，不需要 sigmoid

        # 构造标签与分组键：(h,r) 作为 group，与现逻辑一致
        targets = torch.zeros(B, C, device=scores_all.device)
        targets[:, 0] = 1.0
        targets = targets.reshape(-1)  # [B*C]

        R = self.bm.num_r
        group_idx = (pos[:, 0].long() * R + pos[:, 1].long()).repeat_interleave(C)  # [B*C]

        # —— 关键：迁到 CPU 再 update，和 gnn.py 完全一致 ——
        scores_cpu  = scores_all.detach().cpu()
        targets_cpu = targets.detach().cpu()
        groups_cpu  = group_idx.detach().cpu()

        for name, metric in self.val_metrics.items():
            metric.update(scores_cpu, targets_cpu, groups_cpu)

    def on_validation_epoch_end(self):
        # 跟 gnn.py：先聚合 val_loss，再 compute 每个指标
        if len(self._val_losses) > 0:
            mean_loss = torch.stack(self._val_losses).mean()
            self.log("val_loss", mean_loss, prog_bar=True, logger=True)

        for name, metric in self.val_metrics.items():
            val_value = metric.compute()
            # 你可以只给核心指标上进度条
            self.log(f"val_{name}", val_value, prog_bar=(name in ["MRR", "Hits@10"]), logger=True)
            metric.reset()

    # ====== Test hooks ======
    def on_test_epoch_start(self):
        for m in self.val_metrics.values():
            m.reset()
        self._test_losses = []

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        B, C, _ = batch.size()
        pos = batch[:, 0, :]
        unl = batch[:, 1:, :].reshape(-1, 3)

        # 一次前向（与 valid 相同）
        # eh, er, et = self.emb(pos)
        # s_pos, _ = self.dis(eh, er, et)
        # eh_u, er_u, et_u = self.emb(unl)
        # s_unl, _ = self.dis(eh_u, er_u, et_u)
        # s_unl = s_unl.view(B, C - 1)

        s_pos = self.forward(pos)
        s_unl = self.forward(unl).view(B, C - 1)

        r_ids = pos[:, 1]
        pi_b, eta_b = self._pi_eta_for_rel_ids(r_ids)
        test_loss = self.loss_fn(s_pos, s_unl, pi_b, eta_b,
                                 pairwise=self.pairwise_u, self_adv=self.self_adv, self_adv_alpha=self.self_adv_alpha,
                                 tau_pos=None)
        self._test_losses.append(test_loss.detach())

        # 检索评估（CPU 上 update）
        scores_all = torch.cat([s_pos.view(B, 1), s_unl], dim=1).reshape(-1)
        targets = torch.zeros(B, C, device=scores_all.device); targets[:, 0] = 1.0
        targets = targets.reshape(-1)
        R = self.bm.num_r
        group_idx = (pos[:, 0].long() * R + pos[:, 1].long()).repeat_interleave(C)

        scores_cpu  = scores_all.detach().cpu()
        targets_cpu = targets.detach().cpu()
        groups_cpu  = group_idx.detach().cpu()
        for name, metric in self.val_metrics.items():
            metric.update(scores_cpu, targets_cpu, groups_cpu)

    def on_test_epoch_end(self):
        if len(self._test_losses) > 0:
            self.log("test_loss", torch.stack(self._test_losses).mean(), prog_bar=True, logger=True)
        for name, metric in self.val_metrics.items():
            self.log(f"test_{name}", metric.compute(), prog_bar=(name in ["MRR", "Hits@10"]), logger=True)
            metric.reset()
