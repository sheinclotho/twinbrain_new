"""
数据加载器 - Data Loaders
=======================

负责加载和预处理EEG和fMRI数据
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
        3. 回退到该被试目录下任意 bold.nii* 文件。

        这支持两种场景：
        - 1:1（standard）：每个 EEG 任务有同名 fMRI（步骤 2 命中）。
        - 1:N（共享 fMRI）：多个 EEG 条件共用一个 fMRI run（步骤 1 命中）。
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
    
    def _discover_tasks(self, subject_id: str) -> List[Optional[str]]:
        """自动发现该被试下所有可用的 BIDS 任务名。

        **任务发现策略**：

        - 若配置了 ``fmri_task_mapping``（1:N 模式），**只**扫描 EEG 文件名——
          fMRI 任务名由映射表提供，无需从文件名发现。这样可避免将"只有 fMRI
          没有 EEG"的任务（例如 ``task-CB``）作为独立 run 加载，防止生成
          无 EEG 的单模态图（对跨模态训练毫无价值）。

        - 若未配置映射，同时扫描 EEG 和 fMRI 文件名（旧行为，向后兼容）。

        返回去重排序后的任务名列表。若未发现任何任务标记，则返回 ``[None]``，
        表示不过滤任务（加载首个匹配文件）。
        """
        tasks: set = set()
        patterns: List[str] = []
        if 'eeg' in self.modalities:
            patterns.append(f"{subject_id}*task-*eeg.set")
        # 只有在未配置显式映射时，才从 fMRI 文件名发现任务。
        # 配置了映射意味着 EEG 任务 → fMRI 任务已由用户显式指定，
        # 从 fMRI 文件名额外发现任务只会引入无 EEG 配对的单模态 run。
        if 'fmri' in self.modalities and not self.fmri_task_mapping:
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
