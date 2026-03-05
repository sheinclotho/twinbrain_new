"""
TwinBrain — 扰动分析器 (Perturbation Analyzer)
===============================================

基于时间递推的持续扰动仿真 + 响应矩阵分析。

设计哲学
--------
本模块将大脑建模为自主动力系统：

    X(t+1) = f(X(t))

"刺激"被视为对**状态**的扰动，而非额外输入：

    X(t) → X(t) + δ → f → trajectory

这在动力系统理论中对应的是**扰动分析（perturbation analysis）**：
    * 研究内在动力学的响应性质（intrinsic response）
    * 测量局部稳定性 / 敏感性 / 轨迹散度
    * 与 TMS 实验的 Green's function 框架对齐（Deco et al. 2013）

响应矩阵 R[i, j, k] 定义：
    * i = 刺激脑区索引
    * j = 被测量脑区索引
    * k = 时间步（刺激后第 k 步）
    * R[i, j, k] = X_stim_j(k) - X_base_j(k)

支持两种扰动模式：
    * impulse   — 仅在第 0 步注入扰动，随后系统自由演化（脉冲响应）
    * sustained — 每步持续注入扰动，共 K 步（持续刺激，如 rTMS）

科学背景：与 NPI（Nature Methods 2025）的关系
    * NPI 通过反向传播学习有效连通性（EC），是"因果作用"框架
    * 本模块通过模型前向推断直接计算，是"扰动分析"框架
    * 两者均有科学价值；本框架更适合"虚拟刺激-响应"仿真场景
    * 对接 Unity 实时仿真时，本框架的前向计算更适合实时推断

与 compute_effective_connectivity 的区别：
    * EC：N×N 矩阵，每列是"区域 i 影响哪些区域"（时间平均后的静态摘要）
    * R 矩阵：N×N×K 三维，保留时间分辨率（空间传播轨迹）
    * EC 更适合论文级网络分析；R 矩阵更适合实时交互仿真
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class PerturbationAnalyzer:
    """基于时间递推的持续扰动仿真器。

    将大脑视为动力系统 X(t+1) = f(X(t))，在时间序列中注入小幅扰动，
    通过比较"有扰动"和"无扰动"两条轨迹，量化空间传播响应。

    Args:
        model: 已训练的 :class:`~models.graph_native_system.GraphNativeBrainModel`。
        modality: 扰动和测量的节点类型（'fmri' 或 'eeg'）。
        device: 推断设备。None = 使用模型所在设备。
    """

    def __init__(
        self,
        model,
        modality: str = 'fmri',
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.modality = modality
        self.device = device or next(model.parameters()).device

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_response_matrix(
        self,
        data: HeteroData,
        alpha: float = 0.3,
        num_steps: int = 15,
        mode: str = 'sustained',
        source_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """对每个刺激脑区计算时间分辨响应矩阵 R[i, j, k]。

        算法：
            对每个刺激脑区 i：
                1. 运行"无刺激"基线轨迹   X_base(0..K)
                2. 运行"有刺激"轨迹       X_stim(0..K)
                   * impulse 模式：仅在 t=0 注入扰动
                   * sustained 模式：每步 t=0..K-1 均注入扰动
                3. R[i, j, k] = X_stim_j(k) - X_base_j(k)

        扰动强度：
            delta_i = alpha × ts_std[i]

        其中 ts_std[i] 是区域 i 在数据窗口内的信号标准差。
        使用 z-score 为单位的小扰动（alpha=0.3 约对应 0.3σ），
        确保扰动在模型训练分布内（避免 OOD）。

        Args:
            data: 当前脑状态的 HeteroData 窗口。
            alpha: 扰动强度（以信号标准差为单位）。
                0.3 = 温和扰动（生理范围内）。
                1.0 = 1σ 中等扰动。
                2.0 = 2σ 超生理强度（类强 TMS）。
            num_steps: 刺激后要预测的时间步数 K。
            mode: 扰动模式。
                'impulse'   — 仅在 t=0 注入，随后自由演化（脉冲响应函数）。
                'sustained' — 每步持续注入（持续刺激，如 rTMS/tDCS）。
            source_indices: 要刺激的脑区索引列表。
                None = 刺激所有 N 个脑区。

        Returns:
            R: Float Tensor [N_src, N, K]。
                R[i, j, k] = 刺激区域 i 后，区域 j 在第 k 步的响应差（stimulated - baseline）。
                如果 source_indices 指定了子集，N_src = len(source_indices)；
                否则 N_src = N（全脑）。

        Raises:
            ValueError: 若 modality 不在 data 中。
            ValueError: 若 mode 不是 'impulse' 或 'sustained'。
        """
        if mode not in ('impulse', 'sustained'):
            raise ValueError(f"mode 必须是 'impulse' 或 'sustained'，得到: {mode!r}")

        data = data.to(self.device)
        self.model.eval()

        if self.modality not in data.node_types:
            raise ValueError(
                f"modality='{self.modality}' 不在数据中。"
                f" 可用节点类型: {list(data.node_types)}"
            )

        node_feat = data[self.modality].x  # [N, T, C]
        N, T, C = node_feat.shape

        # 计算每个节点的信号标准差，作为扰动强度参考
        # ts_std[i] = 区域 i 在 T 个时间点上的信号标准差（标量）
        ts_std = node_feat.float().std(dim=(1, 2)).clamp(min=1e-6)  # [N] — std across T and C for each node

        if source_indices is None:
            source_indices = list(range(N))

        N_src = len(source_indices)

        # 预计算基线轨迹（一次，所有刺激区域共享）
        baseline_traj = self._run_trajectory(data, perturbation=None, num_steps=num_steps)
        # baseline_traj: [N, K, C]

        R = torch.zeros(N_src, N, num_steps, device=self.device)

        for idx, i in enumerate(source_indices):
            delta_i = float(alpha) * ts_std[i].item()  # 标量扰动幅度

            # 构建扰动（单个区域，沿平均方向）
            h_mean_dir = self._get_node_direction(data, i)  # [H] in latent space
            delta_latent = h_mean_dir * delta_i

            stim_traj = self._run_trajectory(
                data,
                perturbation=(i, delta_latent, mode),
                num_steps=num_steps,
            )  # [N, K, C]

            R[idx] = stim_traj - baseline_traj

        return R  # [N_src, N, K]

    @staticmethod
    def analyze_response_matrix(
        R: torch.Tensor,
        modality: str = 'fmri',
    ) -> Dict[str, object]:
        """分析响应矩阵 R 的结构属性。

        输入：
            R: [N_src, N, K] — 响应矩阵（来自 compute_response_matrix）

        计算四类结构指标：
            1. 空间传播率（spatial_spread_ratio）— 有多少脑区被影响
            2. 时间衰减斜率（decay_slope）— 响应随时间的衰减速度
            3. 传播延迟峰值步（peak_delay_mean）— 平均峰值响应时间步
            4. 离轴/对角比（offdiag_diag_ratio）— 传播是否超出刺激区域

        布尔指标：
            has_spatial_spread — 空间传播率 > 0.1（10% 脑区受影响）
            has_decay          — 衰减斜率 < -0.01（响应随时间减弱）
            has_delay          — 平均峰值步 > 1（响应有传播延迟）

        Args:
            R: [N_src, N, K] 响应矩阵。
            modality: 节点类型名（用于日志输出）。

        Returns:
            包含以下键的字典：
                'spatial_spread_ratio': float  — 受影响脑区比例
                'decay_slope': float           — 时间衰减斜率
                'peak_delay_mean': float       — 平均峰值时间步（0-indexed）
                'offdiag_diag_ratio': float    — 离轴/对角比
                'has_spatial_spread': bool
                'has_decay': bool
                'has_delay': bool
                'summary': str                 — 中文可读性摘要
        """
        R_np = R.float().cpu().numpy()  # [N_src, N, K]
        N_src, N, K = R_np.shape

        # ── 1. 空间传播率 ──────────────────────────────────────────────────
        # 对每个刺激区域，计算在 K 步内有响应（|R| > threshold）的脑区比例
        threshold = 0.01  # |signal| > 1% 视为有效响应
        max_response = np.abs(R_np).max(axis=2)  # [N_src, N]
        has_response = max_response > threshold   # bool [N_src, N]
        spatial_spread_ratio = float(has_response.mean())

        # ── 2. 时间衰减斜率 ────────────────────────────────────────────────
        # 对全局响应的绝对值取均值 → 时间曲线 [K]
        global_curve = np.abs(R_np).mean(axis=(0, 1))  # [K]
        if K >= 3:
            # 线性拟合 decay_slope（负值 = 衰减）
            # 使用解析最小二乘：slope = cov(t,y) / var(t)，数值更稳定
            time_axis = np.arange(K, dtype=float)
            t_mean = time_axis.mean()
            y_mean = global_curve.mean()
            cov_ty = float(((time_axis - t_mean) * (global_curve - y_mean)).mean())
            var_t = float(((time_axis - t_mean) ** 2).mean())
            decay_slope = cov_ty / var_t if var_t > 1e-10 else 0.0
        else:
            decay_slope = 0.0

        # ── 3. 传播延迟（峰值时间步）──────────────────────────────────────
        # 每个 (刺激区域, 目标区域) 对的峰值时间步
        peak_steps = np.abs(R_np).argmax(axis=2)  # [N_src, N]
        peak_delay_mean = float(peak_steps.mean())

        # ── 4. 离轴/对角比 ────────────────────────────────────────────────
        # 仅在 N_src == N（全脑刺激）时有意义
        if N_src == N:
            max_over_time = np.abs(R_np).max(axis=2)  # [N, N]
            diag_vals = np.diag(max_over_time)         # [N]
            offdiag_mask = ~np.eye(N, dtype=bool)
            offdiag_mean = float(max_over_time[offdiag_mask].mean()) if N > 1 else 0.0
            diag_mean = float(diag_vals.mean()) + 1e-8
            offdiag_diag_ratio = offdiag_mean / diag_mean
        else:
            offdiag_diag_ratio = float('nan')

        # ── 布尔指标 ──────────────────────────────────────────────────────
        has_spatial_spread = spatial_spread_ratio > 0.10
        has_decay = decay_slope < -0.01
        has_delay = peak_delay_mean > 1.0

        # ── 摘要文字 ──────────────────────────────────────────────────────
        parts = []
        if has_spatial_spread:
            parts.append(f"✅ 空间传播：{spatial_spread_ratio:.1%} 脑区有响应")
        else:
            parts.append(f"⚠️ 空间传播有限：仅 {spatial_spread_ratio:.1%} 脑区有响应")

        if has_decay:
            parts.append(f"✅ 时间衰减：斜率 {decay_slope:.4f}（响应随时间减弱）")
        else:
            parts.append(f"ℹ️ 无明显时间衰减（斜率 {decay_slope:.4f}）")

        if has_delay:
            parts.append(f"✅ 传播延迟：平均峰值步 {peak_delay_mean:.1f}（信息经由网络传播）")
        else:
            parts.append(f"ℹ️ 峰值出现较早（平均步 {peak_delay_mean:.1f}，可能为直接激活）")

        if not np.isnan(offdiag_diag_ratio):
            parts.append(f"离轴/对角比 {offdiag_diag_ratio:.2f}")

        summary = ' | '.join(parts)

        return {
            'spatial_spread_ratio': spatial_spread_ratio,
            'decay_slope': decay_slope,
            'peak_delay_mean': peak_delay_mean,
            'offdiag_diag_ratio': offdiag_diag_ratio,
            'has_spatial_spread': has_spatial_spread,
            'has_decay': has_decay,
            'has_delay': has_delay,
            'summary': summary,
        }

    @torch.no_grad()
    def validate_response_matrix(
        self,
        data_windows: List[HeteroData],
        alpha: float = 0.3,
        num_steps: int = 15,
        mode: str = 'sustained',
        source_indices: Optional[List[int]] = None,
    ) -> Dict[str, object]:
        """从多个初始状态分别计算 R，通过 Pearson r 检验自一致性。

        自一致性高（consistency_r ≥ 0.5）表明响应矩阵反映的是模型的
        稳定内在动力学，而非对特定初始状态的偶然响应。

        算法：
            对每个窗口 w 计算 R_w（展平为向量）
            计算所有窗口对之间的 Pearson r
            consistency_r = 平均 Pearson r

        Args:
            data_windows: 多个 HeteroData 窗口（来自不同时间段或被试）。
                建议 ≥ 3 个窗口，以使统计结果有意义。
            alpha: 扰动强度。
            num_steps: 预测步数。
            mode: 扰动模式。
            source_indices: 刺激区域子集（None = 全脑）。

        Returns:
            Dict 包含：
                'consistency_r': float  — 平均 Pearson r（0-1 越高越稳定）
                'per_window_norms': list[float]  — 每个窗口的 R 矩阵 Frobenius 范数
                'interpretation': str  — 中文解读
                'is_reliable': bool  — consistency_r ≥ 0.5
        """
        if len(data_windows) < 2:
            return {
                'consistency_r': float('nan'),
                'per_window_norms': [],
                'interpretation': "⚠️ 至少需要 2 个窗口才能计算一致性。",
                'is_reliable': False,
            }

        R_vecs: List[np.ndarray] = []
        norms: List[float] = []

        for i, window in enumerate(data_windows):
            try:
                R = self.compute_response_matrix(
                    window,
                    alpha=alpha,
                    num_steps=num_steps,
                    mode=mode,
                    source_indices=source_indices,
                )
                r_vec = R.float().cpu().numpy().ravel()
                R_vecs.append(r_vec)
                norms.append(float(np.linalg.norm(r_vec)))
            except Exception as e:
                logger.warning(f"窗口 {i} 的 R 矩阵计算失败: {e}，跳过")

        if len(R_vecs) < 2:
            return {
                'consistency_r': float('nan'),
                'per_window_norms': norms,
                'interpretation': "⚠️ 成功计算的窗口数不足 2，无法评估一致性。",
                'is_reliable': False,
            }

        # 计算所有窗口对之间的 Pearson r
        r_values: List[float] = []
        for a_idx in range(len(R_vecs)):
            for b_idx in range(a_idx + 1, len(R_vecs)):
                v_a, v_b = R_vecs[a_idx], R_vecs[b_idx]
                std_a, std_b = v_a.std(), v_b.std()
                if std_a < 1e-10 or std_b < 1e-10:
                    # 其中一个是全零向量（无扰动响应），特殊处理
                    r = 0.0
                else:
                    r = float(np.corrcoef(v_a, v_b)[0, 1])
                r_values.append(r)

        consistency_r = float(np.mean(r_values))
        is_reliable = consistency_r >= 0.5

        # 解读
        if consistency_r >= 0.8:
            interp = (
                f"✅ 高度一致（r={consistency_r:.3f}）：响应矩阵非常稳定，"
                f" 反映了模型习得的内在动力学，适合用于神经仿真和虚拟刺激。"
            )
        elif consistency_r >= 0.5:
            interp = (
                f"✅ 中等一致（r={consistency_r:.3f}）：响应矩阵基本稳定，"
                f" 结果具有参考价值，但建议在更多窗口上验证。"
            )
        elif consistency_r >= 0.3:
            interp = (
                f"⚠️ 弱一致（r={consistency_r:.3f}）：不同初始状态下响应有差异，"
                f" 可能反映动力学的初始状态依赖性。建议检查数据质量或增加窗口数。"
            )
        else:
            interp = (
                f"⛔ 一致性低（r={consistency_r:.3f}）：响应矩阵随初始状态波动，"
                f" 当前结果不可靠。可能原因：窗口数过少、数据质量差、模型欠拟合。"
            )

        return {
            'consistency_r': consistency_r,
            'per_window_norms': norms,
            'interpretation': interp,
            'is_reliable': is_reliable,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _run_trajectory(
        self,
        data: HeteroData,
        perturbation: Optional[Tuple],
        num_steps: int,
    ) -> torch.Tensor:
        """运行一条轨迹，返回 [N, K, C] 信号预测。

        Args:
            data: 脑状态 HeteroData 窗口。
            perturbation: None = 基线轨迹；
                (node_idx, delta_latent, mode) = 扰动轨迹：
                    * node_idx: int，被扰动的节点索引
                    * delta_latent: Tensor[H]，潜空间扰动向量
                    * mode: 'impulse' | 'sustained'
            num_steps: 预测时间步数。

        Returns:
            sig: [N, K, C] 信号预测张量。
        """
        combined_embed = self.model._get_combined_embed(data, device=self.device)

        if perturbation is None:
            # 基线：直接调用 simulate_intervention（空扰动 = 基线）
            result = self.model.simulate_intervention(
                data=data,
                interventions={},  # 空字典 = 无扰动
                num_steps=num_steps,
            )
            sig = result['baseline']
            return sig.get(self.modality, torch.zeros(
                data[self.modality].x.shape[0], num_steps, 1,
                device=self.device
            ))

        node_idx, delta_latent, mode = perturbation

        if mode == 'impulse':
            # 脉冲模式：仅注入一次扰动，然后自由演化
            result = self.model.simulate_intervention(
                data=data,
                interventions={self.modality: ([node_idx], delta_latent)},
                num_steps=num_steps,
            )
            sig_stim = result['perturbed'].get(self.modality, torch.zeros(
                data[self.modality].x.shape[0], num_steps, 1,
                device=self.device
            ))
            return sig_stim

        else:
            # sustained 模式：每步持续注入扰动
            # 逐步递推：每步将扰动叠加在上一步的预测结果上
            # 实现方式：在时间递推中循环注入扰动（K步）
            return self._run_sustained_trajectory(
                data, node_idx, delta_latent, combined_embed, num_steps
            )

    def _run_sustained_trajectory(
        self,
        data: HeteroData,
        node_idx: int,
        delta_latent: torch.Tensor,
        combined_embed,
        num_steps: int,
    ) -> torch.Tensor:
        """Sustained 模式：每步持续注入扰动的轨迹。

        将 K 步预测拆分为 K 次单步预测，每次预测前都重新注入扰动。
        这是对 rTMS/tDCS 等持续刺激范式的建模。

        注意：此方法比 impulse 模式计算成本高 K 倍，
        但对于持续刺激场景是更忠实的建模。
        """
        # 对 simulate_intervention 的每一步循环调用
        # 简化实现：将 sustained 建模为 impulse，但在每步开始前重新克隆数据并注入扰动
        # 这等价于"每步都施加一次脉冲"的叠加
        all_steps: List[torch.Tensor] = []
        current_data = data  # 第一步使用原始数据；后续通过 _advance_data_one_step 更新（不修改原始 data）

        for step in range(num_steps):
            # 注入扰动 → 预测 1 步
            result = self.model.simulate_intervention(
                data=current_data,
                interventions={self.modality: ([node_idx], delta_latent)},
                num_steps=1,
            )
            # 取第 step 步的信号（仅第 1 步）
            step_sig = result['perturbed'].get(
                self.modality,
                torch.zeros(data[self.modality].x.shape[0], 1, 1, device=self.device)
            )  # [N, 1, C]
            all_steps.append(step_sig)

            # 更新 current_data：用预测结果替换节点特征（滚动窗口）
            # 注意：此处进行浅拷贝，不修改原始 data
            current_data = self._advance_data_one_step(current_data, result['perturbed'])

        return torch.cat(all_steps, dim=1)  # [N, K, C]

    def _advance_data_one_step(
        self,
        data: HeteroData,
        predicted: Dict[str, torch.Tensor],
    ) -> HeteroData:
        """将 HeteroData 窗口滚动前进一步：追加预测帧，移除最早一帧。

        用于 sustained 模式的时间递推仿真。
        """
        from torch_geometric.data import HeteroData as HD
        new_data = HD()

        # 复制所有图结构属性
        for key in data.node_types:
            new_data[key].x = data[key].x.clone()
            # 滚动：丢弃最早一帧，追加预测帧
            pred = predicted.get(key)
            if pred is not None and pred.shape[1] >= 1:
                # data[key].x: [N, T, C]；pred: [N, 1, C]
                new_data[key].x = torch.cat(
                    [data[key].x[:, 1:, :], pred[:, :1, :]], dim=1
                )

        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            edge_store = data[src, rel, dst]
            new_edge_store = new_data[src, rel, dst]
            new_edge_store.edge_index = edge_store.edge_index
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                new_edge_store.edge_attr = edge_store.edge_attr

        # 保留元数据
        for attr in ('subject_idx', 'run_idx', 'task_id', 'subject_id_str'):
            if hasattr(data, attr):
                setattr(new_data, attr, getattr(data, attr))

        return new_data

    def _get_node_direction(
        self,
        data: HeteroData,
        node_idx: int,
    ) -> torch.Tensor:
        """获取节点 node_idx 在潜空间中的扰动方向向量 [H]。

        通过编码基线数据，取该节点在 T 时间步上的均值方向（归一化）作为
        扰动方向。这确保扰动沿节点的"自然激活方向"注入，符合生理意义。

        如果均值方向范数过小，退化为均匀方向向量。
        """
        combined_embed = self.model._get_combined_embed(data, device=self.device)
        encoded_data = self.model.encoder(data, subject_embed=combined_embed)

        if self.modality in encoded_data.node_types:
            h = encoded_data[self.modality].x  # [N, T, H]
            h_node = h[node_idx]               # [T, H]
            mean_dir = h_node.mean(dim=0)      # [H]
            norm = mean_dir.norm()
            H = mean_dir.shape[0]
            if norm > 1e-6:
                return (mean_dir / norm).detach()
            else:
                return (torch.ones(H, device=self.device) / (H ** 0.5)).detach()  # unit-norm uniform direction
        else:
            # 退化：使用节点信号特征作为近似方向
            sig = data[self.modality].x[node_idx]  # [T, C]
            mean_dir = sig.mean(dim=0).float()
            norm = mean_dir.norm()
            if norm > 1e-6:
                return (mean_dir / norm).detach()
            else:
                C = mean_dir.shape[0]
                return (torch.ones(C, device=self.device) / (C ** 0.5)).detach()  # unit-norm uniform direction
