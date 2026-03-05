"""
TwinBrain — Unity 实时集成包
=============================

提供两个核心模块：

* :mod:`unity_integration.perturbation_analyzer` —
  基于时间递推的持续扰动仿真 + 响应矩阵分析（R[i,j,k]）。

* :mod:`unity_integration.realtime_server` —
  WebSocket 推断服务器，供 Unity 或其他实时仿真环境调用。
"""

from .perturbation_analyzer import PerturbationAnalyzer

__all__ = ['PerturbationAnalyzer']
