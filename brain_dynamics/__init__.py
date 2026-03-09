"""
TwinBrain — 脑动力学分析流水线 (brain_dynamics)
================================================

该模块提供基于 TwinBrain 训练输出（或直接 fMRI 时序数据）的
大脑功能动力学分析工具，包括：

  Phase 1 (structural):
    - 功能连接（FC）矩阵的特征值谱分析
    - 响应矩阵（可配置刺激节点数量）

  Advanced:
    - 传递熵（Transfer Entropy）有向信息流分析
    - Granger 因果分析（可选）

主要入口：
  python brain_dynamics/run_pipeline.py --config brain_dynamics/config/dynamics.yaml
"""

__version__ = "1.0.0"
