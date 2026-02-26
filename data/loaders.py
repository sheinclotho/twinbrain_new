"""
数据加载器 - Data Loaders
=======================

负责加载和预处理EEG和fMRI数据，以及可选的DTI结构连通性矩阵。
"""

import re
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import mne
import nibabel as nib

from .eeg_preprocessor import EEGPreprocessor
from .fmri_preprocessor import FMRI_Preprocessor

logger = logging.getLogger(__name__)


class BrainDataLoader:
    """
    统一的脑数据加载器
    
    支持:
    - EEG数据加载和预处理
    - fMRI数据加载和预处理
    - 自动检测BIDS格式
    - 1:N EEG→fMRI 任务对齐（通过 fmri_task_mapping 显式配置）
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        modalities: List[str] = ['eeg', 'fmri'],
        fmri_task_mapping: Optional[Dict[str, Optional[str]]] = None,
    ):
        """
        初始化数据加载器
        
        Args:
            data_root: 数据根目录
            modalities: 要加载的模态 ['eeg', 'fmri']
            fmri_task_mapping: EEG 任务名 → fMRI 任务名的显式映射。
                用于「1 fMRI 对应多个 EEG 条件」（1:N）的场景。
                示例（GRADON/GRADOFF 共享同一 CB fMRI）：
                    {"GRADON": "CB", "GRADOFF": "CB"}
                None（默认）= 不使用映射，按 EEG 任务名直接查找同名 fMRI，
                找不到则回退到该被试下任意 bold 文件。
        """
        self.data_root = Path(data_root)
        self.modalities = modalities
        # None → 空字典，方便后续 .get() 调用
        self.fmri_task_mapping: Dict[str, Optional[str]] = fmri_task_mapping or {}
        
        # 初始化预处理器
        if 'eeg' in modalities:
            self.eeg_preprocessor = EEGPreprocessor()
        
        if 'fmri' in modalities:
            self.fmri_preprocessor = FMRI_Preprocessor()
        
        if self.fmri_task_mapping:
            logger.info(
                f"fMRI 任务映射已配置: {self.fmri_task_mapping}"
                " — 将按映射查找 fMRI 文件（1:N EEG→fMRI 对齐模式）"
            )
        logger.info(f"初始化数据加载器: {data_root}, 模态: {modalities}")
    
    def load_subject(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        加载单个被试数据
        
        Args:
            subject_id: 被试ID (e.g., 'sub-01')
            task: 任务名称 (e.g., 'rest', 'task')
            
        Returns:
            包含各模态数据的字典
        """
        data = {'subject_id': subject_id}
        
        # 加载EEG
        if 'eeg' in self.modalities:
            eeg_data = self._load_eeg(subject_id, task)
            if eeg_data is not None:
                data['eeg'] = eeg_data
        
        # 加载fMRI
        if 'fmri' in self.modalities:
            fmri_data = self._load_fmri(subject_id, task)
            if fmri_data is not None:
                data['fmri'] = fmri_data
        
        # 加载DTI结构连通性（可选）
        # 不依赖 modalities 列表：只要找到 DTI 连接矩阵文件就自动加载。
        # 调用方（build_graphs）在配置了 dti_structural_edges: true 时才将其
        # 转为图中的 ('fmri','structural','fmri') 边，缺失时安静跳过。
        dti_data = self._load_dti(subject_id, task)
        if dti_data is not None:
            data['dti'] = dti_data

        return data
    
    def _load_eeg(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载EEG数据"""
        try:
            # 查找EEG文件
            eeg_pattern = f"{subject_id}*eeg.set"
            if task:
                eeg_pattern = f"{subject_id}*task-{task}*eeg.set"
            
            eeg_files = list(self.data_root.glob(f"**/{eeg_pattern}"))
            
            if not eeg_files:
                logger.warning(f"未找到EEG文件: {subject_id}")
                return None
            
            eeg_file = eeg_files[0]
            logger.info(f"加载EEG: {eeg_file.name}")
            
            # 加载原始EEG
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True)
            
            # 预处理
            raw = self.eeg_preprocessor.preprocess(raw)
            
            # 提取数据
            data, times = raw.get_data(return_times=True)
            
            # 提取电极坐标（由 EEGPreprocessor.preprocess 通过 set_montage("standard_1020")
            # 设置）。转换单位：MNE 内部坐标以米为单位，转为毫米供空间距离计算使用。
            # 若 montage 未设置，loc 向量全零；使用 1e-3 m (= 1 mm) 作为"有效位置"门限。
            ch_pos = None
            try:
                locs = np.array([ch['loc'][:3] for ch in raw.info['chs']])  # [N, 3] 单位：米
                if np.any(np.abs(locs) > 1e-3):  # montage 已设置（至少有电极距原点 > 1 mm）
                    ch_pos = locs * 1000.0  # 米 → 毫米
            except Exception:
                pass
            
            return {
                'data': data,  # [n_channels, n_times]
                'times': times,
                'ch_names': raw.ch_names,
                'sfreq': raw.info['sfreq'],
                'ch_pos': ch_pos,  # [n_channels, 3] 单位 mm，或 None（montage 未设置时）
            }
            
        except Exception as e:
            logger.error(f"加载EEG失败 {subject_id}: {e}")
            return None
    
    def _load_fmri(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载fMRI数据。

        查找顺序（优先级从高到低）：
        1. 若配置了 ``fmri_task_mapping``，用映射后的 fMRI 任务名查找；
        2. 用 EEG 任务名直接查找同名 fMRI 文件；
        2.5 若任务名以 ``ON``/``OFF`` 结尾，自动剥离后缀重试（CBON→CB，CBOFF→CB，
            ECON→EC，ECOFF→EC，GRADON→GRAD，GRADOFF→GRAD 等），无需任何配置；
        3. 回退到该被试目录下任意 bold.nii* 文件。

        这支持三种场景：
        - 1:1（standard）：每个 EEG 任务有同名 fMRI（步骤 2 命中）。
        - 1:N ON/OFF（自动）：EEG 任务名以 ON/OFF 结尾，fMRI 无后缀（步骤 2.5 命中）。
        - 1:N 任意（显式）：多个 EEG 条件共用一个 fMRI run，显式配置映射（步骤 1 命中）。
        """
        try:
            # 查找fMRI文件
            fmri_pattern = f"{subject_id}*bold.nii*"
            if task:
                # 步骤 1：检查是否有显式映射
                mapped_fmri_task = self.fmri_task_mapping.get(task)
                if mapped_fmri_task is not None:
                    # 使用映射后的 fMRI 任务名（1:N 场景的显式配置）
                    task_pattern = f"{subject_id}*task-{mapped_fmri_task}*bold.nii*"
                    fmri_files = list(self.data_root.glob(f"**/{task_pattern}"))
                    if fmri_files:
                        logger.debug(
                            f"fMRI 任务映射命中: EEG task-{task} → fMRI task-{mapped_fmri_task}"
                            f" ({fmri_files[0].name})"
                        )
                    else:
                        logger.warning(
                            f"fMRI 任务映射配置了 {task}→{mapped_fmri_task}，"
                            f"但未找到对应文件 (task-{mapped_fmri_task})，"
                            f"回退到任意 bold 文件"
                        )
                        fmri_files = list(self.data_root.glob(f"**/{fmri_pattern}"))
                        if fmri_files:
                            logger.warning(f"  回退到: {fmri_files[0].name}")
                else:
                    # 步骤 2：按 EEG 任务名直接查找同名 fMRI
                    task_pattern = f"{subject_id}*task-{task}*bold.nii*"
                    fmri_files = list(self.data_root.glob(f"**/{task_pattern}"))

                    if not fmri_files:
                        # 步骤 2.5：ON/OFF 后缀自动检测
                        # CBON→CB, CBOFF→CB, ECON→EC, ECOFF→EC, GRADON→GRAD 等
                        auto_base: Optional[str] = None
                        for suffix in ('ON', 'OFF'):
                            if task.upper().endswith(suffix):
                                auto_base = task[: -len(suffix)]
                                break
                        if auto_base:
                            auto_pattern = f"{subject_id}*task-{auto_base}*bold.nii*"
                            fmri_files = list(self.data_root.glob(f"**/{auto_pattern}"))
                            if fmri_files:
                                logger.debug(
                                    f"ON/OFF 自动检测命中: EEG task-{task} → fMRI task-{auto_base}"
                                    f" ({fmri_files[0].name})"
                                )

                    if not fmri_files:
                        # 步骤 3：回退到该被试目录下搜索任意 bold.nii* 文件
                        fmri_files = list(self.data_root.glob(f"**/{fmri_pattern}"))
                        if fmri_files:
                            logger.warning(
                                f"未找到任务特异性fMRI文件 (task-{task})，"
                                f"回退到: {fmri_files[0].name}"
                                f"（提示：可配置 fmri_task_mapping 显式指定此对应关系）"
                            )
            else:
                fmri_files = list(self.data_root.glob(f"**/{fmri_pattern}"))
            
            if not fmri_files:
                logger.warning(f"未找到fMRI文件: {subject_id}")
                return None
            
            fmri_file = fmri_files[0]
            logger.info(f"加载fMRI: {fmri_file.name}")
            
            # 加载和预处理
            img = nib.load(str(fmri_file))
            img = self.fmri_preprocessor.preprocess(img)
            
            return {
                'img': img,
                'data': img.get_fdata(),
                'affine': img.affine,
            }
            
        except Exception as e:
            logger.error(f"加载fMRI失败 {subject_id}: {e}")
            return None

    def _load_dti(
        self,
        subject_id: str,
        task: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """加载预计算的DTI结构连通性矩阵（可选）。

        DTI（弥散张量成像）预处理（纤维跟踪、网络矩阵提取）需要专用工具链
        （FSL、MRtrix3、Dipy 等），本方法仅加载其输出：与 fMRI 图谱 ROI 对齐的
        连通性矩阵 ``[N_rois, N_rois]``。

        若找到矩阵，其将在 ``build_graphs()`` 中通过
        ``mapper.add_dti_structural_edges()`` 转化为异质图中的
        ``('fmri', 'structural', 'fmri')`` 边，为 fMRI 节点同时提供
        功能连通性（来自 fMRI 时序相关）和结构连通性（来自 DTI 白质纤维）两套边。

        支持的文件格式（按优先级搜索）：
        - ``sub-XX_*connmat*.npy`` — NumPy 二进制，存储 [N, N] 矩阵
        - ``sub-XX_*connectivity*.npy``
        - ``sub-XX_*connmat*.csv`` — 逗号分隔（无表头）
        - ``sub-XX_*connmat*.tsv`` — 制表符分隔（无表头）
        - ``sub-XX_*connectivity*.csv``

        Args:
            subject_id: 被试ID（如 'sub-01'）
            task: 任务名（当前未用于 DTI 文件搜索，保留供未来按任务区分 DTI 使用）

        Returns:
            ``{'connectivity': ndarray [N_rois, N_rois]}`` 或 ``None``（未找到时静默返回）
        """
        try:
            patterns = [
                f"{subject_id}*connmat*.npy",
                f"{subject_id}*connectivity*.npy",
                f"{subject_id}*connmat*.csv",
                f"{subject_id}*connmat*.tsv",
                f"{subject_id}*connectivity*.csv",
            ]
            for pattern in patterns:
                files = list(self.data_root.glob(f"**/{pattern}"))
                if files:
                    if len(files) > 1:
                        logger.warning(
                            f"发现 {len(files)} 个匹配 DTI 矩阵文件 (pattern={pattern})，"
                            f"将使用第一个: {files[0].name}。"
                            f"其他文件: {[f.name for f in files[1:]]}。"
                            f"若需指定文件，请将其重命名为唯一匹配的文件名。"
                        )
                    f = files[0]
                    if f.suffix == '.npy':
                        mat = np.load(str(f))
                    else:
                        delimiter = ',' if f.suffix == '.csv' else '\t'
                        mat = np.loadtxt(str(f), delimiter=delimiter)
                    mat = mat.astype(np.float32)
                    logger.info(f"加载DTI连通性矩阵: {f.name} shape={mat.shape}")
                    return {'connectivity': mat}
            # 未找到时静默返回（DTI 是可选模态）
            return None
        except Exception as e:
            logger.warning(f"加载DTI矩阵失败 {subject_id}: {e}")
            return None

    def _discover_tasks(self, subject_id: str) -> List[Optional[str]]:
        """自动发现该被试下所有可用的 BIDS 任务名。

        **任务发现策略**：

        - 若 EEG 在模态列表中：**只**扫描 EEG 文件名（无论是否配置了 fmri_task_mapping）。
          fMRI 任务的确定推迟到 ``_load_fmri()`` 时按优先级解析（显式映射 > 直接同名
          匹配 > ON/OFF 自动检测 > 任意文件回退），从而避免将「fMRI-only 任务」
          （如 task-CB）作为独立 run 加载——该 run 因无 EEG 配对而产生无跨模态边的
          单模态图，对联合训练毫无价值。

        - 若 EEG **不在**模态列表中（纯 fMRI 场景）：扫描 fMRI 文件名。

        返回去重排序后的任务名列表。若未发现任何任务标记，则返回 ``[None]``，
        表示不过滤任务（加载首个匹配文件）。
        """
        tasks: set = set()
        patterns: List[str] = []
        if 'eeg' in self.modalities:
            # EEG 存在时：仅依赖 EEG 文件名发现任务。
            # fMRI 任务由 _load_fmri() 在加载时自动确定（支持同名/ON-OFF/显式映射）。
            patterns.append(f"{subject_id}*task-*eeg.set")
        elif 'fmri' in self.modalities:
            # 纯 fMRI 场景：从 fMRI 文件名发现任务。
            patterns.append(f"{subject_id}*task-*bold.nii*")
        for pat in patterns:
            for f in self.data_root.glob(f"**/{pat}"):
                m = re.search(r'task-([^_]+)', f.name)
                if m:
                    tasks.add(m.group(1))
        return sorted(tasks) if tasks else [None]

    def load_all_subjects(
        self,
        tasks: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """加载所有被试数据，每个被试可跨多个任务加载。

        Args:
            tasks: 要加载的任务名列表，例如 ``["rest", "wm"]``。
                   ``None``（默认）表示自动发现该被试下所有可用任务；
                   ``[]`` 表示不过滤任务（直接加载首个匹配文件）。
            max_subjects: 最大被试数（``0`` 或 ``None`` 表示不限制）。

        Returns:
            被试-任务数据列表，每项字典由 ``load_subject()`` 填充（含
            ``subject_id`` 字段），本方法额外追加 ``task`` 字段。
        """
        # 查找所有被试目录
        subject_dirs = sorted(self.data_root.glob("sub-*"))

        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        all_data = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name

            # 确定本被试要加载的任务列表
            if tasks is None:
                # 自动发现：扫描该被试的文件名
                subject_tasks: List[Optional[str]] = self._discover_tasks(subject_id)
            elif len(tasks) == 0:
                # 空列表 = 不过滤任务
                subject_tasks = [None]
            else:
                subject_tasks = list(tasks)

            for t in subject_tasks:
                data = self.load_subject(subject_id, t)
                if data:
                    data['task'] = t  # 记录来自哪个任务
                    all_data.append(data)

        logger.info(f"成功加载 {len(all_data)} 个被试-任务组合")
        return all_data
