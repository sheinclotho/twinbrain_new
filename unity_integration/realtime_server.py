"""
TwinBrain — 实时推断服务器 (Realtime Inference Server)
=======================================================

提供 WebSocket 接口，支持实时脑状态推断和虚拟刺激仿真。

设计目标
--------
1. 供 Unity 游戏引擎或其他实时仿真环境调用
2. 承载有效连通性（EC）推断 → 供神经仿真参数化
3. 承载响应矩阵计算（R）→ 供虚拟刺激响应仿真

通信协议
--------
所有消息均为 JSON 格式：

请求::

    {
      "type": "<request_type>",
      "request_id": "<optional_uuid>",
      ... 其他参数
    }

响应::

    {
      "type": "<request_type>_response",
      "request_id": "<echo>",
      "status": "ok" | "error",
      "data": { ... }  // 或 "error": "message"
    }

支持的请求类型
--------------
* "ping"                    — 心跳检测
* "compute_ec"              — 计算全脑有效连通性矩阵 N×N
* "simulate_intervention"   — 虚拟 TMS 刺激仿真
* "compute_response_matrix" — 计算时间分辨响应矩阵 N×N×K
* "validate_consistency"    — 跨窗口响应一致性验证

依赖
----
* websockets >= 11.0
* torch, torch_geometric（已在主环境安装）

启动服务器::

    python -m unity_integration.realtime_server \\
        --checkpoint outputs/best_model.pt \\
        --subject_to_idx outputs/.../subject_to_idx.json \\
        --cache_dir outputs/graph_cache \\
        --host 0.0.0.0 --port 8765

Unity 端连接示例::

    ws://localhost:8765

"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Server state
# ──────────────────────────────────────────────────────────────────────────────

class ServerState:
    """服务器全局状态（单例）。"""

    def __init__(self) -> None:
        self.twin = None            # TwinBrainDigitalTwin 实例（加载后设置）
        self.analyzer = None        # PerturbationAnalyzer 实例（加载后设置）
        self._ec_analyzer_cache: Optional[torch.Tensor] = None   # 最近一次的 EC 矩阵
        self._graph_cache: Dict[str, Any] = {}  # 已加载的图缓存 {filename: graph}
        self.modality: str = 'fmri'

    @property
    def is_ready(self) -> bool:
        return self.twin is not None


_state = ServerState()


# ──────────────────────────────────────────────────────────────────────────────
# Request handlers
# ──────────────────────────────────────────────────────────────────────────────

async def handle_ping(request: Dict, state: ServerState) -> Dict:
    """心跳检测。"""
    return {
        'status': 'ok',
        'message': 'pong',
        'server_ready': state.is_ready,
        'timestamp': time.time(),
    }


async def handle_compute_ec(request: Dict, state: ServerState) -> Dict:
    """计算全脑有效连通性矩阵。

    请求参数::

        {
          "cache_file": "sub-01_GRADON_a1b2c3d4.pt",  // 可选；None = 使用最近加载的
          "modality": "fmri",                           // 默认 fmri
          "perturbation_strength": 1.0,
          "signed": true,
          "normalize": true
        }

    响应数据::

        {
          "ec_matrix": [[...], ...],   // N×N float 列表
          "n_nodes": N,
          "modality": "fmri"
        }
    """
    if not state.is_ready:
        return {'status': 'error', 'error': '服务器尚未加载模型。请先调用 /load 或启动时指定 --checkpoint。'}

    data = _resolve_data(request, state)
    if data is None:
        return {'status': 'error', 'error': '未找到可用的图数据窗口。请在请求中指定 cache_file。'}

    modality = request.get('modality', state.modality)
    strength = float(request.get('perturbation_strength', 1.0))
    signed = bool(request.get('signed', True))
    normalize = bool(request.get('normalize', True))

    ec = state.twin.compute_effective_connectivity(
        data_windows=data,
        modality=modality,
        perturbation_strength=strength,
        signed=signed,
        normalize=normalize,
    )  # [N, N]

    state._ec_analyzer_cache = ec
    ec_list = ec.cpu().tolist()

    return {
        'status': 'ok',
        'data': {
            'ec_matrix': ec_list,
            'n_nodes': ec.shape[0],
            'modality': modality,
        },
    }


async def handle_simulate_intervention(request: Dict, state: ServerState) -> Dict:
    """虚拟 TMS 刺激仿真。

    请求参数::

        {
          "cache_file": "sub-01_GRADON_a1b2c3d4.pt",
          "modality": "fmri",
          "node_indices": [42, 43],
          "delta": 2.0,
          "num_steps": 15
        }

    响应数据::

        {
          "causal_effect": [[...], ...],   // N×steps float 列表（C=1 已压缩）
          "baseline": [[...], ...],
          "perturbed": [[...], ...],
          "n_nodes": N,
          "n_steps": K
        }
    """
    if not state.is_ready:
        return {'status': 'error', 'error': '服务器尚未加载模型。'}

    data = _resolve_data(request, state)
    if data is None:
        return {'status': 'error', 'error': '未找到可用的图数据窗口。'}

    modality = request.get('modality', state.modality)
    node_indices = [int(i) for i in request.get('node_indices', [0])]
    delta = float(request.get('delta', 1.0))
    num_steps = int(request.get('num_steps', 15))

    result = state.twin.simulate_intervention(
        baseline_data=data,
        interventions={modality: (node_indices, delta)},
        num_prediction_steps=num_steps,
    )

    def _to_list(t: Optional[torch.Tensor]) -> list:
        if t is None:
            return []
        # 压缩 C=1 维度：[N, steps, 1] → [N, steps]
        if t.dim() == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        return t.cpu().tolist()

    causal = result.get('causal_effect', {}).get(modality)
    baseline = result.get('baseline', {}).get(modality)
    perturbed = result.get('perturbed', {}).get(modality)

    return {
        'status': 'ok',
        'data': {
            'causal_effect': _to_list(causal),
            'baseline': _to_list(baseline),
            'perturbed': _to_list(perturbed),
            'n_nodes': causal.shape[0] if causal is not None else 0,
            'n_steps': causal.shape[1] if causal is not None else 0,
            'modality': modality,
        },
    }


async def handle_compute_response_matrix(request: Dict, state: ServerState) -> Dict:
    """计算时间分辨响应矩阵 R[i, j, k]。

    此端点要求在此之前已完成至少一次 EC 推断（_ec_analyzer_cache 非空），
    或在请求中提供 cache_file。

    请求参数::

        {
          "cache_file": "sub-01_GRADON_a1b2c3d4.pt",  // 图缓存文件名
          "modality": "fmri",
          "alpha": 0.3,           // 扰动强度（z-score 单位），默认 0.3
          "num_steps": 15,        // 预测步数 K
          "mode": "sustained",    // "impulse" | "sustained"
          "source_indices": [0, 1, 2],  // 要刺激的脑区，null = 全脑
          "validate": false,      // 是否同时计算自一致性（需要多窗口）
          "extra_cache_files": [] // validate=true 时的额外窗口文件
        }

    响应数据::

        {
          "R_mean": [[...], ...],      // N×N 归一化后的响应强度矩阵（max over K）
          "R_timecourse": [[...], ...],// N×K 全脑平均响应时间轮廓
          "analysis": {                // analyze_response_matrix 结果
            "spatial_spread_ratio": 0.25,
            "decay_slope": -0.02,
            "peak_delay_mean": 3.5,
            "offdiag_diag_ratio": 1.8,
            "has_spatial_spread": true,
            "has_decay": true,
            "has_delay": true,
            "summary": "✅ ..."
          },
          "consistency": { ... },      // validate=true 时才有
          "n_src": N_src,
          "n_nodes": N,
          "n_steps": K,
          "modality": "fmri"
        }

    建议（非强制）先调用 compute_ec：
        EC 推断缓存不影响 R 矩阵计算本身，但建议先执行 compute_ec
        以确认当前数据窗口的动力系统编码质量。若未调用，服务器会
        打印警告日志但继续执行（不阻塞）。
    """
    if not state.is_ready:
        return {'status': 'error', 'error': '服务器尚未加载模型。'}

    # 检查 EC 推断缓存（作为数据质量前置条件）
    if state._ec_analyzer_cache is None:
        logger.warning(
            "compute_response_matrix 调用时 EC 缓存为空。"
            " 建议先调用 compute_ec 以确认数据窗口的动力系统已被正确编码。"
            " 继续执行（非阻塞）。"
        )

    data = _resolve_data(request, state)
    if data is None:
        return {'status': 'error', 'error': '未找到可用的图数据窗口。'}

    if state.analyzer is None:
        return {'status': 'error', 'error': '扰动分析器未初始化，请检查服务器启动配置。'}

    modality = request.get('modality', state.modality)
    alpha = float(request.get('alpha', 0.3))
    num_steps = int(request.get('num_steps', 15))
    mode = str(request.get('mode', 'sustained'))
    source_raw = request.get('source_indices')
    source_indices = [int(i) for i in source_raw] if source_raw is not None else None

    # 切换 analyzer 的 modality（若请求中指定了不同 modality）
    original_modality = state.analyzer.modality
    state.analyzer.modality = modality

    try:
        R = state.analyzer.compute_response_matrix(
            data=data,
            alpha=alpha,
            num_steps=num_steps,
            mode=mode,
            source_indices=source_indices,
        )  # [N_src, N, K]

        analysis = state.analyzer.analyze_response_matrix(R, modality=modality)

        # R_mean: N_src × N — 每个源区域对所有目标区域的最大响应幅度（over K）
        R_mean_tensor = R.abs().max(dim=2).values  # [N_src, N]
        max_val = R_mean_tensor.max()
        if max_val > 1e-8:
            R_mean_tensor = R_mean_tensor / max_val
        R_mean = R_mean_tensor.cpu().tolist()

        # R_timecourse: N × K — 全脑平均响应时间轮廓
        R_timecourse_tensor = R.abs().mean(dim=0)  # [N, K]
        R_timecourse = R_timecourse_tensor.cpu().tolist()

        response_data: Dict[str, Any] = {
            'R_mean': R_mean,
            'R_timecourse': R_timecourse,
            'analysis': analysis,
            'n_src': R.shape[0],
            'n_nodes': R.shape[1],
            'n_steps': R.shape[2],
            'modality': modality,
        }

        # 可选：自一致性验证
        if bool(request.get('validate', False)):
            extra_files = request.get('extra_cache_files', [])
            extra_windows = [_load_cache_file(f, state) for f in extra_files]
            extra_windows = [w for w in extra_windows if w is not None]
            all_windows = [data] + extra_windows

            if len(all_windows) >= 2:
                consistency = state.analyzer.validate_response_matrix(
                    data_windows=all_windows,
                    alpha=alpha,
                    num_steps=num_steps,
                    mode=mode,
                    source_indices=source_indices,
                )
                response_data['consistency'] = consistency
            else:
                response_data['consistency'] = {
                    'consistency_r': float('nan'),
                    'interpretation': '⚠️ 需要至少 2 个窗口（extra_cache_files）进行一致性验证。',
                    'is_reliable': False,
                }

        return {'status': 'ok', 'data': response_data}

    finally:
        state.analyzer.modality = original_modality


async def handle_validate_consistency(request: Dict, state: ServerState) -> Dict:
    """跨窗口响应一致性验证（独立端点）。

    请求参数::

        {
          "cache_files": ["sub-01_GRADON_a.pt", "sub-01_GRADON_b.pt"],
          "modality": "fmri",
          "alpha": 0.3,
          "num_steps": 15,
          "mode": "sustained",
          "source_indices": null
        }
    """
    if not state.is_ready or state.analyzer is None:
        return {'status': 'error', 'error': '服务器尚未加载模型或扰动分析器。'}

    cache_files = request.get('cache_files', [])
    if len(cache_files) < 2:
        return {'status': 'error', 'error': '至少需要提供 2 个 cache_file。'}

    windows = [_load_cache_file(f, state) for f in cache_files]
    windows = [w for w in windows if w is not None]

    if len(windows) < 2:
        return {'status': 'error', 'error': '成功加载的图窗口不足 2 个，无法计算一致性。'}

    modality = request.get('modality', state.modality)
    alpha = float(request.get('alpha', 0.3))
    num_steps = int(request.get('num_steps', 15))
    mode = str(request.get('mode', 'sustained'))
    source_raw = request.get('source_indices')
    source_indices = [int(i) for i in source_raw] if source_raw is not None else None

    original_modality = state.analyzer.modality
    state.analyzer.modality = modality
    try:
        result = state.analyzer.validate_response_matrix(
            data_windows=windows,
            alpha=alpha,
            num_steps=num_steps,
            mode=mode,
            source_indices=source_indices,
        )
    finally:
        state.analyzer.modality = original_modality

    return {'status': 'ok', 'data': result}


# ──────────────────────────────────────────────────────────────────────────────
# Request router
# ──────────────────────────────────────────────────────────────────────────────

_HANDLERS = {
    'ping':                     handle_ping,
    'compute_ec':               handle_compute_ec,
    'simulate_intervention':    handle_simulate_intervention,
    'compute_response_matrix':  handle_compute_response_matrix,
    'validate_consistency':     handle_validate_consistency,
}


async def process_request(message: str, state: ServerState) -> str:
    """解析 JSON 请求，调度到对应 handler，返回 JSON 响应。"""
    try:
        request = json.loads(message)
    except json.JSONDecodeError as e:
        return json.dumps({'status': 'error', 'error': f'JSON 解析失败: {e}'})

    req_type = request.get('type', '')
    req_id = request.get('request_id', '')

    handler = _HANDLERS.get(req_type)
    if handler is None:
        resp = {
            'status': 'error',
            'error': f"未知请求类型: {req_type!r}。支持: {list(_HANDLERS.keys())}",
        }
    else:
        t0 = time.time()
        try:
            resp = await handler(request, state)
        except Exception as e:
            logger.exception(f"处理请求 {req_type!r} 时发生错误: {e}")
            resp = {'status': 'error', 'error': str(e)}
        elapsed_ms = (time.time() - t0) * 1000
        resp['elapsed_ms'] = round(elapsed_ms, 1)

    resp['type'] = f'{req_type}_response'
    if req_id:
        resp['request_id'] = req_id

    return json.dumps(resp, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket connection handler
# ──────────────────────────────────────────────────────────────────────────────

async def websocket_handler(websocket, state: ServerState) -> None:
    """处理单个 WebSocket 连接的所有请求。"""
    client_addr = 'unknown'
    try:
        client_addr = websocket.remote_address
    except Exception:
        pass
    logger.info(f"新连接: {client_addr}")
    try:
        async for message in websocket:
            response = await process_request(message, state)
            await websocket.send(response)
    except Exception as e:
        logger.info(f"连接 {client_addr} 断开: {e}")
    finally:
        logger.info(f"连接 {client_addr} 已关闭")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_data(request: Dict, state: ServerState):
    """从请求中解析图数据窗口。

    优先级：
        1. request['cache_file'] — 加载指定缓存文件
        2. _graph_cache 中最近加载的图
    """
    cache_file = request.get('cache_file')
    if cache_file:
        return _load_cache_file(cache_file, state)

    # 返回缓存中最后一个加载的图
    if state._graph_cache:
        last_key = list(state._graph_cache.keys())[-1]
        return state._graph_cache[last_key]

    return None


def _load_cache_file(filename: str, state: ServerState):
    """加载 .pt 图缓存文件，带内存缓存。"""
    if filename in state._graph_cache:
        return state._graph_cache[filename]

    cache_dir = getattr(state, '_cache_dir', None)
    if cache_dir is None:
        logger.warning(f"cache_dir 未设置，无法加载 {filename}")
        return None

    path = Path(cache_dir) / filename
    if not path.exists():
        logger.warning(f"缓存文件不存在: {path}")
        return None

    try:
        graph = torch.load(path, map_location='cpu', weights_only=False)
        state._graph_cache[filename] = graph
        logger.info(f"已加载缓存图: {filename}")
        return graph
    except Exception as e:
        logger.error(f"加载缓存文件 {filename} 失败: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Server startup
# ──────────────────────────────────────────────────────────────────────────────

def create_server_state(
    checkpoint_path: str,
    cache_dir: Optional[str] = None,
    subject_to_idx_path: Optional[str] = None,
    modality: str = 'fmri',
    device: str = 'cpu',
) -> ServerState:
    """初始化服务器状态：加载模型检查点和扰动分析器。

    Args:
        checkpoint_path: 训练检查点路径（best_model.pt 或 checkpoint_epoch_*.pt）。
        cache_dir: 图缓存目录路径（outputs/graph_cache）。
        subject_to_idx_path: subject_to_idx.json 路径（可选）。
        modality: 默认分析模态（'fmri' 或 'eeg'）。
        device: 推断设备（'cpu', 'cuda', 'cuda:0' 等）。

    Returns:
        初始化完成的 ServerState。
    """
    from models.digital_twin_inference import TwinBrainDigitalTwin
    from unity_integration.perturbation_analyzer import PerturbationAnalyzer

    state = ServerState()
    state.modality = modality
    state._cache_dir = cache_dir

    logger.info(f"正在从检查点加载模型: {checkpoint_path}")
    state.twin = TwinBrainDigitalTwin.from_checkpoint(
        checkpoint_path=checkpoint_path,
        subject_to_idx_path=subject_to_idx_path,
        device=device,
    )
    state.analyzer = PerturbationAnalyzer(
        model=state.twin.model,
        modality=modality,
        device=torch.device(device),
    )
    logger.info(f"✅ 模型加载完成，modality={modality}, device={device}")
    return state


async def run_server(
    state: ServerState,
    host: str = '0.0.0.0',
    port: int = 8765,
) -> None:
    """启动 WebSocket 服务器（asyncio 入口）。

    Args:
        state: 已初始化的 ServerState。
        host: 监听地址。
        port: 监听端口。
    """
    try:
        import websockets
    except ImportError:
        raise ImportError(
            "realtime_server 需要 websockets 库：pip install websockets>=11.0"
        )

    import functools
    handler = functools.partial(websocket_handler, state=state)

    logger.info(f"TwinBrain 实时推断服务器启动: ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.get_event_loop().create_future()  # run forever (never resolves)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """命令行入口：python -m unity_integration.realtime_server"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='TwinBrain 实时推断 WebSocket 服务器',
    )
    parser.add_argument('--checkpoint', required=True,
                        help='训练检查点路径（best_model.pt）')
    parser.add_argument('--cache_dir', default=None,
                        help='图缓存目录（outputs/graph_cache）')
    parser.add_argument('--subject_to_idx', default=None,
                        help='subject_to_idx.json 路径（可选）')
    parser.add_argument('--modality', default='fmri',
                        help='默认分析模态（fmri 或 eeg，默认 fmri）')
    parser.add_argument('--device', default='cpu',
                        help='推断设备（cpu / cuda / cuda:0，默认 cpu）')
    parser.add_argument('--host', default='0.0.0.0',
                        help='WebSocket 监听地址（默认 0.0.0.0）')
    parser.add_argument('--port', type=int, default=8765,
                        help='WebSocket 监听端口（默认 8765）')
    args = parser.parse_args()

    state = create_server_state(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        subject_to_idx_path=args.subject_to_idx,
        modality=args.modality,
        device=args.device,
    )

    asyncio.run(run_server(state, host=args.host, port=args.port))


if __name__ == '__main__':
    main()
