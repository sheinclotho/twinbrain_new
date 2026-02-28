"""
ds006040 数据集下载工具
=======================

从 OpenNeuro 下载 ds006040 数据集的指定被试数据。

**下载策略（优先级从高到低）**

1. **OpenNeuro GraphQL ``downloadFiles`` API**（主路径）：
   通过 OpenNeuro GraphQL API 查询 ``snapshot.downloadFiles``，该字段在服务端
   递归遍历全部文件树并做缓存（有效期 5 天），一次请求即可获得快照所有文件的
   平坦列表（含完整路径 + 下载 URL）。客户端按被试前缀过滤后直接下载，无需
   多次往返 API。

2. **OpenNeuro GraphQL ``files(tree: key)`` 逐层递归**（第二备用）：
   使用 ``snapshot(datasetId, tag).files`` 字段逐层遍历目录树。根目录调用时
   不传 ``tree`` 参数（等同于 null）；对于目录项，使用其 ``key``（git tree hash）
   作为下一层的 ``tree`` 参数，依此递归直到遍历完被试全部文件。
   此方式符合 OpenNeuro 官方 API 文档（https://docs.openneuro.org/api.html）规范。

3. **S3 公共存储桶**（第三备用）：
   公共数据集均已发布至 S3 存储桶 ``openneuro.org``，支持匿名读取。
   使用 S3 REST API（XML 响应）列出文件，天然支持分页，无条目数限制。
   每个文件直接通过 HTTPS 下载，无需 openneuro-py 介入。

4. **openneuro-py**（最终备用）：
   若上述方式均失败，回退到 openneuro-py 库。

**GraphQL API 说明（来源：https://docs.openneuro.org/api.html）**

OpenNeuro GraphQL schema 关键说明：

- 获取快照信息：``dataset(id: $id) { latestSnapshot { id } }``
  其中 ``id`` 格式为 ``"ds006040:1.0.1"``，从中解析出 tag ``"1.0.1"``。

- 获取全部文件（推荐）：``snapshot(datasetId: $id, tag: $tag) { downloadFiles { ... } }``
  ``downloadFiles`` 服务端完成递归遍历，返回平坦文件列表（含完整相对路径）。
  服务端对结果缓存约 5 天，重复调用无额外开销。

- 逐层获取文件（备用）：``snapshot(datasetId: $id, tag: $tag) { files { ... } }``
  不带 ``tree`` 参数时返回根目录内容；对目录项使用其 ``key`` 字段（git tree hash）
  作为 ``files(tree: "key_value")`` 的参数继续递归（官方文档明确说明）。

  注意：不使用 ``dataset.snapshot(tag)``（该字段不存在于真实 schema），
  也不使用 ``files(first: N, after: cursor)``（Relay Connection 分页，schema 不支持）。

**性能参数说明**

- ``max_concurrent_downloads``：同时下载的文件数，默认 3。
- ``per_file_connections``：每个大文件的并行 Range 连接数，默认 1（关闭分片）。
- 断点续传：每个文件分片先写到 ``.p<N>`` 临时文件；若中断重启，自动检测
  已有字节数并从断点续传；小文件或单连接模式则沿用 ``.part`` 临时文件。
- 大分块：``_CHUNK_SIZE`` 为 4 MiB，减少系统调用开销，提高吞吐量。
- 连接复用：每线程内使用长连接 ``httpx.Client``，避免重复 TCP 握手。
- 卡顿检测：流式下载使用拆分超时（``_CONNECT_TIMEOUT``/``_STREAM_STALL_TIMEOUT``），
  若 ``_STREAM_STALL_TIMEOUT``（默认 120 秒）内未收到任何字节，立即触发重试。
- GraphQL 重试：``_fetch_latest_tag``、``_list_subject_files_via_download_files``
  和 ``_list_subject_files_via_openneuro`` 均支持最多 ``_METADATA_MAX_RETRIES``
  （默认 3）次指数退避重试，应对
  ``RemoteProtocolError: Server disconnected without sending a response``
  等偶发服务端断线错误，避免直接回退到较慢的 S3 路径。

使用方法::

    # 下载单个被试
    python -m data.download_ds006040 029

    # 下载单个被试到指定目录
    python -m data.download_ds006040 029 /data/ds006040

    # 下载多个被试（逗号分隔，或 PowerShell 展开后的空格分隔）
    python -m data.download_ds006040 001,002,029
    python -m data.download_ds006040 001 002 029

    # 在 Python 代码中调用
    from data.download_ds006040 import download_subject
    download_subject("029", target_dir="/test_file3")
"""

from __future__ import annotations

import inspect
import logging
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import httpx
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

DATASET_ID = "ds006040"
_OPENNEURO_GQL_URL = "https://openneuro.org/crn/graphql"

# OpenNeuro open-access datasets are stored in this public S3 bucket.
_S3_BUCKET = "openneuro.org"
_S3_BASE_URL = f"https://s3.amazonaws.com/{_S3_BUCKET}"
_S3_XML_NS = "http://s3.amazonaws.com/doc/2006-03-01/"
_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB per read chunk
_MAX_BACKOFF_SECONDS = 60  # cap for exponential retry delay
# Files larger than this threshold are split into parallel Range-request parts.
_LARGE_FILE_THRESHOLD = 32 * 1024 * 1024  # 32 MiB
# Default number of parallel Range connections per large file.
# Set to 1 (disabled) by default: enabling multipart (e.g. 8 files × 4 parts = 32
# simultaneous S3 connections) causes S3 throttling that stalls large .fdt files at 0%.
_DEFAULT_PER_FILE_CONNECTIONS = 1
# Default max concurrent file downloads.  8 was the previous value but caused
# too many simultaneous connections when combined with multipart mode.
_DEFAULT_MAX_CONCURRENT = 3
# Number of retries for metadata (GraphQL tag / snapshot files) requests before giving up.
_METADATA_MAX_RETRIES = 3
# Stall timeout for streaming downloads: if no bytes arrive within this many seconds,
# raise ReadTimeout so the retry loop can re-open the connection.  Much shorter than
# the legacy flat 600s value, which caused stalled connections to go undetected.
_STREAM_STALL_TIMEOUT = 120.0  # seconds of silence before retrying
# Short timeout used for the initial TCP connection and TLS handshake.
_CONNECT_TIMEOUT = 15.0

# User-Agent header sent with all GraphQL requests (matches openneuro-py convention).
try:
    import openneuro as _on_pkg  # type: ignore[import-untyped]
    _OPENNEURO_PY_VERSION = getattr(_on_pkg, "__version__", "standalone")
except ImportError:
    _OPENNEURO_PY_VERSION = "standalone"
_GQL_HEADERS = {"user-agent": f"openneuro-py/{_OPENNEURO_PY_VERSION}"}

# Minimum timeout (seconds) for the downloadFiles query, which may return a large
# JSON payload covering all files in the entire snapshot.
_DOWNLOAD_FILES_TIMEOUT = 120.0

_TAG_QUERY = """
query GetLatestTag($datasetId: ID!) {
  dataset(id: $datasetId) {
    latestSnapshot {
      id
    }
  }
}
"""

# ── OpenNeuro GraphQL：downloadFiles 查询（主路径）────────────────────────────
# 官方文档（https://docs.openneuro.org/api.html）schema：
#   snapshot.downloadFiles 字段由服务端完成整棵文件树的递归遍历（使用 file.id），
#   返回平坦文件列表，包含完整相对路径和下载 URL。服务端对结果缓存约 5 天。
#   一次 GraphQL 请求即可获取快照的所有文件，客户端按被试前缀过滤即可。
_DOWNLOAD_FILES_QUERY = """
query GetDownloadFiles($datasetId: ID!, $tag: String!) {
  snapshot(datasetId: $datasetId, tag: $tag) {
    downloadFiles {
      id
      key
      filename
      size
      urls
      directory
      annexed
    }
  }
}
"""

# ── OpenNeuro GraphQL：快照文件列表查询（逐层递归，第二备用）──────────────────
# 官方文档规范（https://docs.openneuro.org/api.html）：
#   - 不带 tree 参数（传 null）时返回快照根目录内容
#   - 对目录项，使用其 key 字段（git tree hash）作为下一层 files(tree: "key") 的参数
#   - key 是用于 tree 参数的正确字段（官方文档明确："the directory key field can be
#     used to retrieve additional trees"）
#   - 同时请求 id 字段以便调试（id ≠ key；id 是 git blob/tree 对象 hash）
_SNAPSHOT_FILES_QUERY = """
query GetSnapshotFiles($datasetId: ID!, $tag: String!, $tree: String) {
  snapshot(datasetId: $datasetId, tag: $tag) {
    files(tree: $tree) {
      id
      key
      filename
      size
      urls
      directory
      annexed
    }
  }
}
"""


def _fetch_latest_tag(
    dataset_id: str,
    timeout: float = 30.0,
    max_retries: int = _METADATA_MAX_RETRIES,
) -> Optional[str]:
    """Fetch the latest snapshot tag from the OpenNeuro GraphQL API.

    Retries up to *max_retries* times with exponential back-off to handle
    transient ``RemoteProtocolError`` / server-disconnect errors.

    Returns the tag string (e.g. ``"1.0.1"``) or ``None`` on failure.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.post(
                _OPENNEURO_GQL_URL,
                json={"query": _TAG_QUERY, "variables": {"datasetId": dataset_id}},
                headers=_GQL_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            snapshot_id: str = data["data"]["dataset"]["latestSnapshot"]["id"]
            # snapshot_id 格式为 "ds006040:1.0.1"
            tag = snapshot_id.split(":", 1)[-1]
            logger.debug("数据集 %s 最新快照标签: %s", dataset_id, tag)
            return tag
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                wait = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
                logger.debug(
                    "获取快照标签失败（第 %d/%d 次），%d 秒后重试: %s: %s",
                    attempt, max_retries, wait, type(exc).__name__, exc,
                )
                time.sleep(wait)
    logger.warning(
        "获取快照标签失败（将尝试 S3 直接下载）: %s: %s",
        type(last_exc).__name__,
        last_exc,
    )
    return None


# ── OpenNeuro GraphQL downloadFiles API（主路径）──────────────────────────────


def _list_subject_files_via_download_files(
    dataset_id: str,
    tag: str,
    subject_prefix: str,
    timeout: float = _DOWNLOAD_FILES_TIMEOUT,
    max_retries: int = _METADATA_MAX_RETRIES,
) -> List[Tuple[str, int, str]]:
    """通过 OpenNeuro GraphQL ``downloadFiles`` 字段查询指定被试的全部文件。

    ``snapshot.downloadFiles`` 由服务端完成整棵文件树的递归遍历，返回包含完整
    相对路径的平坦文件列表，并在服务端缓存约 5 天。一次 GraphQL 请求即可获取
    快照所有文件；客户端按被试前缀（如 ``"sub-001/"``）过滤即可。

    相比逐层 ``files(tree: ...)`` 递归，此方式更可靠：
    - 避免客户端多轮递归请求（减少超时风险）
    - 服务端已按 ``file.id``（非 ``key``）完成递归，不受 ``key`` 为空的影响

    Parameters
    ----------
    dataset_id:
        OpenNeuro 数据集 ID（如 ``"ds006040"``）。
    tag:
        快照标签（如 ``"1.0.1"``）。
    subject_prefix:
        被试目录名（如 ``"sub-004"``）。用于过滤全量文件列表。
    timeout:
        单次 GraphQL 请求超时秒数（默认 ``_DOWNLOAD_FILES_TIMEOUT``，因全量文件列表可能较大）。
    max_retries:
        最大重试次数。

    Returns
    -------
    list of (relative_path, size_bytes, download_url)
        *relative_path*: 相对于数据集根目录的路径（如 ``"sub-004/eeg/..."``）。
        *size_bytes*: 文件字节数（0 表示未知）。
        *download_url*: 可直接下载的 URL（由 OpenNeuro 提供，通常指向 S3/CDN）。
    """
    variables: dict = {"datasetId": dataset_id, "tag": tag}
    body: dict = {}
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.post(
                _OPENNEURO_GQL_URL,
                json={"query": _DOWNLOAD_FILES_QUERY, "variables": variables},
                headers=_GQL_HEADERS,
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            break  # success
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                raise RuntimeError(
                    f"OpenNeuro downloadFiles 请求失败（已重试 {max_retries} 次）: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            wait = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
            logger.debug(
                "downloadFiles 请求失败（第 %d/%d 次），%d 秒后重试: %s: %s",
                attempt, max_retries, wait, type(exc).__name__, exc,
            )
            time.sleep(wait)

    if "errors" in body:
        raise RuntimeError(
            f"OpenNeuro downloadFiles GraphQL 返回错误: {body['errors']}"
        )

    try:
        all_files: list = body["data"]["snapshot"]["downloadFiles"] or []
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"OpenNeuro downloadFiles 响应格式不符合预期: {exc}\n响应: {body}"
        ) from exc

    # Filter to only this subject's files.
    # downloadFiles returns full paths like "sub-001/eeg/file.set".
    # We also check exact equality as a defensive guard in case the API ever
    # returns top-level entries without a trailing separator.
    prefix_slash = f"{subject_prefix}/"
    results: List[Tuple[str, int, str]] = []
    for node in all_files:
        filename: str = node.get("filename") or ""
        if not filename.startswith(prefix_slash):
            continue
        if node.get("directory"):
            # The server-side getFilesRecursive only adds leaf files, so directory
            # entries in downloadFiles are unexpected.  Log at warning to surface
            # any future API behaviour changes.
            logger.warning(
                "downloadFiles 返回了目录项（不符合预期），已跳过: %s", filename
            )
            continue
        urls: list = node.get("urls") or []
        url = urls[0] if urls else None
        if url:
            size = int(node.get("size") or 0)
            results.append((filename, size, url))
        else:
            logger.debug("文件无下载 URL（可能尚未导出至 S3），跳过: %s", filename)

    logger.debug(
        "downloadFiles 过滤结果: %s 共 %d 个文件（全量 %d 个）",
        subject_prefix, len(results), len(all_files),
    )
    return results


# ── OpenNeuro GraphQL files(tree) 逐层递归（第二备用）───────────────────────────


def _list_subject_files_via_openneuro(
    dataset_id: str,
    tag: str,
    subject_prefix: str,
    timeout: float = 60.0,
    max_retries: int = _METADATA_MAX_RETRIES,
) -> List[Tuple[str, int, str]]:
    """通过 OpenNeuro GraphQL ``files(tree: ...)`` 逐层递归查询指定被试的全部文件。

    使用官方 API 文档（https://docs.openneuro.org/api.html）规范的 schema：
    - 顶层 ``snapshot(datasetId, tag)`` 字段
    - 根目录调用 ``files``（不带 tree 参数，即 tree=null）返回根目录内容
    - 对目录项使用其 ``key`` 字段（git tree hash）作为 ``files(tree: "key_value")``
      的参数递归获取子目录内容（官方文档："the directory key field can be used to
      retrieve additional trees"）
    - 不使用 Relay Connection 分页（schema 不支持）

    每层请求发生网络错误时，自动重试最多 *max_retries* 次（指数退避）。

    .. note::
        此函数作为 ``_list_subject_files_via_download_files`` 的备用。
        ``downloadFiles`` 方式在服务端完成递归（使用 ``file.id``），对客户端更可靠。

    Parameters
    ----------
    dataset_id:
        OpenNeuro 数据集 ID（如 ``"ds006040"``）。
    tag:
        快照标签（如 ``"1.0.1"``）。
    subject_prefix:
        被试目录名（如 ``"sub-004"``）。
    timeout:
        单次 GraphQL 请求超时秒数。
    max_retries:
        每层请求最大重试次数。

    Returns
    -------
    list of (relative_path, size_bytes, download_url)
        *relative_path*: 相对于数据集根目录的路径（如 ``"sub-004/eeg/..."``）。
        *size_bytes*: 文件字节数（0 表示未知）。
        *download_url*: 可直接下载的 URL（由 OpenNeuro 提供，通常指向 S3/CDN）。
    """

    def _fetch_tree_level(tree: Optional[str]) -> List[dict]:
        """Fetch one directory level from the snapshot.

        Parameters
        ----------
        tree:
            git tree hash for the directory, or ``None`` for the root level.
            Per the official API docs, the ``key`` field of a directory node
            is the correct value to pass as the ``tree`` parameter.

        Returns a list of file/directory node dicts from the GraphQL response.
        """
        variables: dict = {"datasetId": dataset_id, "tag": tag, "tree": tree}
        body: dict = {}
        for attempt in range(1, max_retries + 1):
            try:
                resp = httpx.post(
                    _OPENNEURO_GQL_URL,
                    json={"query": _SNAPSHOT_FILES_QUERY, "variables": variables},
                    headers=_GQL_HEADERS,
                    timeout=timeout,
                )
                resp.raise_for_status()
                body = resp.json()
                break  # success
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries:
                    raise RuntimeError(
                        f"OpenNeuro GraphQL 请求失败（已重试 {max_retries} 次）: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                wait = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
                logger.debug(
                    "GraphQL 请求失败（第 %d/%d 次），%d 秒后重试: %s: %s",
                    attempt, max_retries, wait, type(exc).__name__, exc,
                )
                time.sleep(wait)

        if "errors" in body:
            raise RuntimeError(
                f"OpenNeuro GraphQL 返回错误: {body['errors']}"
            )

        try:
            nodes = body["data"]["snapshot"]["files"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"OpenNeuro GraphQL 响应格式不符合预期: {exc}\n响应: {body}"
            ) from exc

        return nodes or []

    results: List[Tuple[str, int, str]] = []

    def _collect_recursive(nodes: List[dict], path_prefix: str) -> None:
        """Recursively traverse directory nodes, appending file entries to ``results``.

        Per the official API docs, the ``key`` field of a directory node is the
        git tree hash to use for the ``files(tree: ...)`` call at the next level.
        """
        for node in nodes:
            name: str = node.get("filename") or ""
            full_path = f"{path_prefix}/{name}" if path_prefix else name
            if node.get("directory"):
                # Use 'key' as the tree parameter (official API docs confirmed).
                # 'key' is the git tree hash for directories.
                key: Optional[str] = node.get("key")
                if key:
                    sub_nodes = _fetch_tree_level(key)
                    _collect_recursive(sub_nodes, full_path)
                else:
                    logger.debug(
                        "目录节点缺少 key 字段，跳过: %s (id=%s)",
                        full_path, node.get("id"),
                    )
            else:
                urls: list = node.get("urls") or []
                url = urls[0] if urls else None
                if url:
                    size = int(node.get("size") or 0)
                    results.append((full_path, size, url))
                else:
                    logger.debug("文件无下载 URL（可能尚未导出至 S3），跳过: %s", full_path)

    # Step 1: get root-level entries
    root_nodes = _fetch_tree_level(None)

    # Step 2: find the subject directory (or file) at root level
    found = False
    for node in root_nodes:
        name: str = node.get("filename") or ""
        if name != subject_prefix:
            continue
        found = True
        if node.get("directory"):
            key = node.get("key")
            if key:
                sub_nodes = _fetch_tree_level(key)
                _collect_recursive(sub_nodes, subject_prefix)
            else:
                logger.debug(
                    "被试目录节点缺少 key 字段: %s (id=%s)",
                    subject_prefix, node.get("id"),
                )
        else:
            # subject is a top-level file (unlikely but handle gracefully)
            urls = node.get("urls") or []
            url = urls[0] if urls else None
            if url:
                results.append((name, int(node.get("size") or 0), url))
        break  # found the subject; no need to continue scanning

    if not found:
        logger.debug(
            "OpenNeuro API 根目录中未找到 %s（快照中可能不含该被试）",
            subject_prefix,
        )

    return results


def _download_subject_via_openneuro_api(
    subject_prefix: str,
    dataset_id: str,
    tag: str,
    target_dir: Path,
    max_retries: int = 10,
    timeout: float = 60.0,
    max_concurrent_downloads: int = _DEFAULT_MAX_CONCURRENT,
) -> int:
    """通过 OpenNeuro GraphQL API 下载指定被试的全部文件。

    查询策略（按优先级）：
    1. 使用 ``snapshot.downloadFiles`` 字段（官方推荐，服务端缓存全量平坦文件列表）。
    2. 若 ``downloadFiles`` 返回空或失败，回退到 ``snapshot.files(tree: key)``
       逐层递归遍历（符合官方 API 文档规范）。
    3. 多线程并发下载，支持断点续传。

    Returns
    -------
    int
        本次操作处理的文件总数（含已存在跳过的文件和新下载的文件）。
        若该被试在快照中不存在，返回 0。
    """
    logger.debug(
        "OpenNeuro API 文件列表请求: dataset=%s tag=%s subject=%s",
        dataset_id, tag, subject_prefix,
    )

    # ── 主路径：downloadFiles（服务端全量递归，更可靠）──────────────────────
    files: List[Tuple[str, int, str]] = []
    try:
        files = _list_subject_files_via_download_files(
            dataset_id=dataset_id,
            tag=tag,
            subject_prefix=subject_prefix,
            timeout=max(timeout, _DOWNLOAD_FILES_TIMEOUT),  # ensure generous timeout for full list
        )
        if files:
            logger.debug(
                "downloadFiles 路径成功获取 %s 文件列表: %d 个文件",
                subject_prefix, len(files),
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "downloadFiles 路径失败，将尝试 files(tree) 递归路径: %s: %s",
            type(exc).__name__, exc,
        )

    # ── 备用路径：files(tree: key) 逐层递归（官方 API 文档规范）──────────────
    if not files:
        logger.debug(
            "downloadFiles 未返回文件，改用 files(tree) 递归路径: %s",
            subject_prefix,
        )
        files = _list_subject_files_via_openneuro(
            dataset_id=dataset_id,
            tag=tag,
            subject_prefix=subject_prefix,
            timeout=timeout,
        )

    if not files:
        logger.debug(
            "OpenNeuro API 未返回 %s 的文件（可能快照中不存在该被试）",
            subject_prefix,
        )
        return 0

    logger.info("OpenNeuro API 文件列表: %s 共 %d 个文件", subject_prefix, len(files))

    # ── 筛选需要下载的文件 ──────────────────────────────────────────────────
    to_download: List[Tuple[str, int, str, Path]] = []  # (rel_path, size, url, dest)
    skipped = 0
    for rel_path, size, url in files:
        dest = target_dir / rel_path
        if dest.exists() and (size == 0 or dest.stat().st_size == size):
            logger.debug("跳过已存在文件: %s", rel_path)
            skipped += 1
        else:
            to_download.append((rel_path, size, url, dest))

    if not to_download:
        logger.info("全部文件已存在，无需下载")
        return len(files)

    logger.info(
        "待下载 %d 个文件，跳过 %d 个已完成文件，并发数: %d",
        len(to_download), skipped, max_concurrent_downloads,
    )

    # ── 多线程下载 ──────────────────────────────────────────────────────────
    bar_lock = threading.Lock()
    errors: List[str] = []

    def _worker_api(args: Tuple[str, int, str, Path]) -> str:
        rel_path, _size, url, dest = args
        _stream_timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT, read=_STREAM_STALL_TIMEOUT,
            write=None, pool=None,
        )
        with httpx.Client(timeout=_stream_timeout, follow_redirects=True) as cli:
            _s3_download_file(
                key="",  # not used when url is provided directly
                dest=dest,
                file_size=_size,
                timeout=_STREAM_STALL_TIMEOUT,
                max_retries=max_retries,
                file_label=rel_path,
                client=cli,
                per_file_connections=1,  # multipart disabled; avoid S3 throttling
                _override_url=url,
            )
        return rel_path

    with logging_redirect_tqdm():
        with tqdm(
            total=len(files),
            initial=skipped,
            unit="file",
            desc=subject_prefix,
            dynamic_ncols=True,
        ) as file_bar:
            with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as pool:
                futures = {
                    pool.submit(_worker_api, item): item for item in to_download
                }
                for future in as_completed(futures):
                    rel_path, _size, _url, _dest = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("下载失败 %s: %s", rel_path, exc)
                        errors.append(rel_path)
                    with bar_lock:
                        file_bar.set_postfix_str(rel_path, refresh=True)
                        file_bar.update(1)

    if errors:
        raise RuntimeError(
            f"以下 {len(errors)} 个文件下载失败（已用尽重试次数）:\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    return len(files)


def _s3_list_files(prefix: str, timeout: float = 60.0) -> Iterator[Tuple[str, int]]:
    """列出 S3 存储桶中指定前缀下的全部对象。

    使用 AWS S3 REST API（XML 响应）进行分页迭代，不受单页条目数限制。

    Yields
    ------
    (key, size) :
        S3 对象键（完整路径）和文件大小（字节数）。
    """
    continuation_token: Optional[str] = None

    while True:
        params: dict = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        response = httpx.get(_S3_BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        for content in root.findall(f"{{{_S3_XML_NS}}}Contents"):
            key_elem = content.find(f"{{{_S3_XML_NS}}}Key")
            size_elem = content.find(f"{{{_S3_XML_NS}}}Size")
            if key_elem is None or not key_elem.text or size_elem is None:
                continue  # skip malformed entries
            yield key_elem.text, int(size_elem.text or "0")

        is_truncated = root.find(f"{{{_S3_XML_NS}}}IsTruncated")
        if is_truncated is not None and is_truncated.text and is_truncated.text.lower() == "true":
            token_elem = root.find(f"{{{_S3_XML_NS}}}NextContinuationToken")
            continuation_token = token_elem.text if token_elem is not None else None
            if not continuation_token:
                break
        else:
            break


def _assemble_parts(dest: Path, part_paths: List[Path]) -> None:
    """Concatenate byte-range part files in order into *dest*, then delete them.

    Writes to a ``.part`` staging file first, then atomically renames to *dest*,
    so the final file is never visible in a partially-written state.
    """
    staging = dest.with_suffix(dest.suffix + ".part")
    with open(staging, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, out)
    staging.replace(dest)
    for p in part_paths:
        try:
            p.unlink()
        except OSError:
            pass


def _s3_download_file_multipart(
    key: str,
    dest: Path,
    file_size: int,
    n_parts: int = _DEFAULT_PER_FILE_CONNECTIONS,
    timeout: float = 600.0,
    max_retries: int = 10,
    file_label: str = "",
    client: Optional[httpx.Client] = None,
) -> None:
    """Download a large file by splitting it into *n_parts* parallel Range requests.

    Each part is written to ``<dest>.p<N>`` while in progress.  If a previous
    run was interrupted, any already-complete part files are detected by their
    exact size and skipped, providing per-part resume support.

    After all parts finish, :func:`_assemble_parts` concatenates them into
    *dest* and removes the temporary ``.p<N>`` files.

    Parameters
    ----------
    client:
        Optional shared ``httpx.Client`` instance.  httpx clients are
        thread-safe, so passing a single client lets all parts share its
        connection pool.  If ``None``, each part uses the module-level
        ``httpx.stream`` function.
    """
    url = f"{_S3_BASE_URL}/{key}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = file_label or dest.name

    # Divide the file into equal byte ranges.
    part_size = file_size // n_parts
    ranges: List[Tuple[int, int]] = []
    for i in range(n_parts):
        start = i * part_size
        end = (start + part_size - 1) if i < n_parts - 1 else (file_size - 1)
        ranges.append((start, end))

    part_paths = [dest.with_suffix(dest.suffix + f".p{i}") for i in range(n_parts)]

    # Compute already-downloaded bytes so the progress bar starts at the right offset.
    initial_bytes = sum(
        min(p.stat().st_size, end - start + 1)
        for p, (start, end) in zip(part_paths, ranges)
        if p.exists()
    )

    pbar_lock = threading.Lock()

    with tqdm(
        total=file_size,
        initial=initial_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=label,
        leave=False,
        dynamic_ncols=True,
    ) as pbar:

        def _download_part(part_idx: int) -> None:
            start, end = ranges[part_idx]
            part_path = part_paths[part_idx]
            expected = end - start + 1

            for attempt in range(1, max_retries + 1):
                existing = part_path.stat().st_size if part_path.exists() else 0
                if existing >= expected:
                    return  # already complete; bytes counted in initial_bytes

                resume_start = start + existing
                headers: Dict[str, str] = {"Range": f"bytes={resume_start}-{end}"}

                try:
                    _stream = client.stream if client is not None else httpx.stream
                    _req_timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=timeout, write=None, pool=None)
                    with _stream(  # type: ignore[operator]
                        "GET", url, headers=headers,
                        timeout=_req_timeout, follow_redirects=True,
                    ) as resp:
                        if resp.status_code == 200 and existing > 0:
                            # Server ignored Range → restart this part from scratch.
                            logger.debug("服务端不支持 Range，重新下载分片 %d", part_idx)
                            existing = 0
                        resp.raise_for_status()

                        open_mode = "ab" if existing > 0 else "wb"
                        with open(part_path, open_mode) as fh:
                            for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                                fh.write(chunk)
                                with pbar_lock:
                                    pbar.update(len(chunk))
                    return  # success
                except Exception as exc:  # noqa: BLE001
                    if attempt == max_retries:
                        raise
                    wait = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
                    logger.warning(
                        "文件分片 %s [part %d/%d, bytes %d-%d] 失败"
                        "（第 %d/%d 次），%d 秒后重试: %s",
                        label, part_idx + 1, n_parts, start, end,
                        attempt, max_retries, wait, exc,
                    )
                    time.sleep(wait)

        with ThreadPoolExecutor(max_workers=n_parts) as pool:
            futs = {pool.submit(_download_part, i): i for i in range(n_parts)}
            for fut in as_completed(futs):
                fut.result()  # re-raise any unrecoverable part failure

    _assemble_parts(dest, part_paths)


def _s3_download_file(
    key: str = "",
    dest: Path = Path(),
    file_size: int = 0,
    timeout: float = 600.0,
    max_retries: int = 10,
    file_label: str = "",
    client: Optional[httpx.Client] = None,
    per_file_connections: int = 1,
    _override_url: Optional[str] = None,
) -> None:
    """从 S3（或任意 URL）下载单个文件（流式传输，断点续传，含字节级进度条）。

    当 ``per_file_connections > 1`` 且文件大小超过 ``_LARGE_FILE_THRESHOLD``
    时，自动调用 :func:`_s3_download_file_multipart` 以并行 Range 请求加速。

    断点续传逻辑：下载期间写入 ``<dest>.part`` 临时文件，完成后原子重命名。
    若 ``.part`` 文件已存在（上次中断残留），自动发送
    ``Range: bytes=<offset>-`` 请求从断点继续，而非从头重下。

    Parameters
    ----------
    file_size:
        Known file size in bytes from the S3 listing (0 = unknown).
        Required for multipart splitting; if 0, falls back to single-stream.
    per_file_connections:
        Number of parallel Range connections to use for this file.
        Values > 1 trigger multipart mode for files larger than
        ``_LARGE_FILE_THRESHOLD``.
    client:
        可选的复用 ``httpx.Client`` 实例（多线程场景下每线程传入独立实例）。
        若为 ``None``，则为每次请求临时创建连接。
    _override_url:
        若指定，使用此 URL 替代由 ``key`` 推导的 S3 URL。
        供 OpenNeuro API 路径使用，直接传入 GraphQL 返回的 CDN URL。
        注意：``_override_url`` 与 ``per_file_connections > 1`` 不兼容——
        多分片模式目前仅支持 S3 URL 推导方式。若同时传入，自动降级为单连接。
    """
    # ── 大文件走并行分片路径 ────────────────────────────────────────────────
    # Multipart mode constructs its own URL from `key`; it does not support
    # _override_url.  If an override URL is provided, always use single-stream.
    if per_file_connections > 1 and not _override_url and file_size > _LARGE_FILE_THRESHOLD:
        _s3_download_file_multipart(
            key=key,
            dest=dest,
            file_size=file_size,
            n_parts=per_file_connections,
            timeout=timeout,
            max_retries=max_retries,
            file_label=file_label,
            client=client,
        )
        return
    if per_file_connections > 1 and not _override_url and file_size == 0:
        logger.debug(
            "文件大小未知（file_size=0），跳过分片模式，改用单连接下载: %s", key
        )

    url = _override_url if _override_url else f"{_S3_BASE_URL}/{key}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    label = file_label or dest.name

    for attempt in range(1, max_retries + 1):
        # ── 断点续传：检查已有的字节数 ────────────────────────────────────
        try:
            resume_offset = part.stat().st_size if part.exists() else 0
        except OSError:
            resume_offset = 0

        headers: Dict[str, str] = {}
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"
            logger.debug("断点续传 %s，从 %d 字节处继续", key, resume_offset)

        try:
            _get = client.stream if client is not None else httpx.stream
            # Use a split timeout: short connect timeout + per-read stall timeout.
            # A flat 600s timeout would not detect a stalled stream for up to 10 min.
            _req_timeout = httpx.Timeout(connect=_CONNECT_TIMEOUT, read=timeout, write=None, pool=None)
            with _get(  # type: ignore[operator]
                "GET", url, headers=headers, timeout=_req_timeout, follow_redirects=True
            ) as resp:
                # 206 Partial Content → resume OK；200 → server ignores Range, restart
                if resp.status_code == 200 and resume_offset > 0:
                    logger.debug("服务端不支持 Range，重新下载 %s", key)
                    resume_offset = 0
                resp.raise_for_status()

                # Determine open mode AFTER confirming server's Range support.
                # If resume_offset was reset to 0 (server returned 200), use "wb"
                # so the .part file is overwritten rather than appended to.
                open_mode = "ab" if resume_offset > 0 else "wb"

                content_length = int(resp.headers["content-length"]) if "content-length" in resp.headers else None
                total_bytes = (resume_offset + content_length) if content_length is not None else None

                with tqdm(
                    total=total_bytes,
                    initial=resume_offset,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                    leave=False,
                    dynamic_ncols=True,
                ) as bar:
                    with open(part, open_mode) as fh:
                        for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                            fh.write(chunk)
                            bar.update(len(chunk))

            # ── 成功：原子重命名 ──────────────────────────────────────────
            part.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise
            wait = min(2 ** attempt, _MAX_BACKOFF_SECONDS)
            logger.warning(
                "下载 %s 失败（第 %d/%d 次），%d 秒后重试: %s",
                key, attempt, max_retries, wait, exc,
            )
            time.sleep(wait)


def _download_subject_via_s3(
    subject_prefix: str,
    dataset_id: str,
    target_dir: Path,
    max_retries: int = 10,
    timeout: float = 60.0,
    max_concurrent_downloads: int = _DEFAULT_MAX_CONCURRENT,
    per_file_connections: int = _DEFAULT_PER_FILE_CONNECTIONS,
) -> int:
    """通过 S3 公共存储桶下载指定被试的全部文件。

    多线程并发下载（由 ``max_concurrent_downloads`` 控制），每线程独立维护
    ``httpx.Client`` 连接池，支持断点续传。超过 ``_LARGE_FILE_THRESHOLD``
    的大文件使用 ``per_file_connections`` 个并行 Range 请求加速。

    Returns
    -------
    int
        本次操作处理的文件总数（含已存在跳过的文件和新下载的文件）。
        若该被试在 S3 中不存在，返回 0。
    """
    s3_prefix = f"{dataset_id}/{subject_prefix}/"
    logger.debug("S3 列表请求: %s/%s", _S3_BASE_URL, s3_prefix)

    files = list(_s3_list_files(s3_prefix, timeout=timeout))

    if not files:
        logger.debug("S3 中未找到 %s（可能非公共数据集或被试不存在）", s3_prefix)
        return 0

    logger.info("S3 列表: %s 共 %d 个文件", subject_prefix, len(files))

    # dataset_id 部分已包含在 key 中，下载到 target_dir/{dataset_id}/... 下
    dataset_prefix = f"{dataset_id}/"

    # ── 筛选需要下载的文件 ──────────────────────────────────────────────────
    to_download: List[Tuple[str, int, Path]] = []  # (key, size, dest)
    skipped = 0
    for key, size in files:
        if not key.startswith(dataset_prefix):
            logger.warning("S3 返回的对象键不符合预期格式，已跳过: %s", key)
            skipped += 1
            continue
        rel = key[len(dataset_prefix):]
        dest = target_dir / rel
        if dest.exists() and dest.stat().st_size == size:
            logger.debug("跳过已存在文件: %s", rel)
            skipped += 1
        else:
            to_download.append((key, size, dest))

    if not to_download:
        logger.info("全部文件已存在，无需下载")
        return len(files)

    logger.info(
        "待下载 %d 个文件，跳过 %d 个已完成文件，并发数: %d，每大文件连接数: %d",
        len(to_download), skipped, max_concurrent_downloads, per_file_connections,
    )

    # ── 多线程下载 ──────────────────────────────────────────────────────────
    bar_lock = threading.Lock()
    errors: List[str] = []

    def _worker(args: Tuple[str, int, Path]) -> str:
        """单文件下载任务，返回相对路径（用于进度条）。"""
        key, _size, dest = args
        rel = key[len(dataset_prefix):]
        _stream_timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT, read=_STREAM_STALL_TIMEOUT,
            write=None, pool=None,
        )
        with httpx.Client(
            timeout=_stream_timeout,
            follow_redirects=True,
        ) as cli:
            _s3_download_file(
                key=key,
                dest=dest,
                file_size=_size,
                timeout=_STREAM_STALL_TIMEOUT,
                max_retries=max_retries,
                file_label=rel,
                client=cli,
                per_file_connections=per_file_connections,
            )
        return rel

    with logging_redirect_tqdm():
        with tqdm(
            total=len(files),
            initial=skipped,
            unit="file",
            desc=subject_prefix,
            dynamic_ncols=True,
        ) as file_bar:
            with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as pool:
                futures = {pool.submit(_worker, item): item for item in to_download}
                for future in as_completed(futures):
                    key, _size, dest = futures[future]
                    rel = key[len(dataset_prefix):]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("下载失败 %s: %s", rel, exc)
                        errors.append(rel)
                    with bar_lock:
                        file_bar.set_postfix_str(rel, refresh=True)
                        file_bar.update(1)

    if errors:
        raise RuntimeError(
            f"以下 {len(errors)} 个文件下载失败（已用尽重试次数）:\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    return len(files)


# ── openneuro-py 回退（备用路径）─────────────────────────────────────────────


def _download_subject_via_openneuro_py(
    subject_prefix: str,
    dataset_id: str,
    tag: Optional[str],
    target_dir: Path,
    max_retries: int,
    metadata_timeout: float,
    max_concurrent_downloads: int,
) -> None:
    """使用 openneuro-py 库下载（备用方案）。"""
    from openneuro import download  # type: ignore[import-untyped]

    _download_kwargs: dict = {
        "dataset": dataset_id,
        "tag": tag,
        "target_dir": str(target_dir),
        "include": [subject_prefix],
        "max_retries": max_retries,
        "max_concurrent_downloads": max_concurrent_downloads,
    }
    if "metadata_timeout" in inspect.signature(download).parameters:
        _download_kwargs["metadata_timeout"] = metadata_timeout

    download(**_download_kwargs)


# ── 公共 API ─────────────────────────────────────────────────────────────────


def download_subject(
    subject: str,
    target_dir: str | Path = "/test_file3",
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
    max_concurrent_downloads: int = _DEFAULT_MAX_CONCURRENT,
    per_file_connections: int = _DEFAULT_PER_FILE_CONNECTIONS,
) -> None:
    """Download all data files for *subject* from OpenNeuro.

    下载策略（按优先级顺序尝试）：

    1. **OpenNeuro GraphQL ``downloadFiles`` + ``files(tree)`` API**（主路径）：
       先用 ``snapshot.downloadFiles`` 获取快照全量文件列表（服务端缓存，一次请求）；
       若失败则回退到 ``snapshot.files(tree: key)`` 逐层递归（符合官方 API 文档）。
       两种方式均使用 OpenNeuro 返回的下载 URL（CDN/S3 均可）直接下载文件。
    2. **S3 公共存储桶**（第二备用）：列文件 + 直接下载，无分页问题，
       但大并发时 S3 可能限速（大文件 0% 卡顿）。
    3. **openneuro-py**（最终备用）：若前两种方式均失败。

    Parameters
    ----------
    subject:
        Subject ID without the ``"sub-"`` prefix. Zero-padding is applied
        automatically (e.g. ``"4"`` → ``sub-004``).
    target_dir:
        Destination directory. Defaults to ``/test_file3``.
    dataset_id:
        OpenNeuro dataset ID. Defaults to ``ds006040``.
    max_retries:
        Maximum retry attempts per file.
    metadata_timeout:
        Timeout in seconds for GraphQL metadata requests and S3 listing.
    max_concurrent_downloads:
        Number of files to download in parallel. Defaults to 3.
        (Reduced from 8 to avoid S3 throttling that stalls large .fdt files.)
    per_file_connections:
        Number of parallel Range-request connections per large file
        (files > ``_LARGE_FILE_THRESHOLD`` = 32 MiB). Defaults to 1 (disabled).
        (Multipart mode previously caused 32 simultaneous S3 connections
        which stalled large .fdt files at 0%.  Set to 4 only when you have
        confirmed your network and S3 can sustain the load.)
    """
    target_dir = Path(target_dir)
    subject_prefix = f"sub-{subject.zfill(3)}"

    logger.info(
        "开始下载 %s / %s → %s",
        dataset_id,
        subject_prefix,
        target_dir,
    )

    # ── 主路径：OpenNeuro GraphQL 直接 API ────────────────────────────────
    tag = _fetch_latest_tag(dataset_id, timeout=metadata_timeout)
    if tag is not None:
        try:
            count = _download_subject_via_openneuro_api(
                subject_prefix=subject_prefix,
                dataset_id=dataset_id,
                tag=tag,
                target_dir=target_dir,
                max_retries=max_retries,
                timeout=metadata_timeout,
                max_concurrent_downloads=max_concurrent_downloads,
            )
            if count > 0:
                logger.info(
                    "下载完成（OpenNeuro API）: %s / %s，共 %d 个文件",
                    dataset_id, subject_prefix, count,
                )
                return
            logger.warning(
                "OpenNeuro API 未返回 %s/%s 的文件，将尝试 S3 方式",
                dataset_id, subject_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OpenNeuro API 下载失败，将尝试 S3 方式: %s: %s",
                type(exc).__name__, exc,
            )
    else:
        logger.warning("无法获取快照标签，直接尝试 S3 方式")

    # ── 第二备用：S3 直接下载 ──────────────────────────────────────────────
    try:
        count = _download_subject_via_s3(
            subject_prefix=subject_prefix,
            dataset_id=dataset_id,
            target_dir=target_dir,
            max_retries=max_retries,
            timeout=metadata_timeout,
            max_concurrent_downloads=max_concurrent_downloads,
            per_file_connections=per_file_connections,
        )
        if count > 0:
            logger.info(
                "下载完成（S3）: %s / %s，共 %d 个文件",
                dataset_id, subject_prefix, count,
            )
            return
        logger.warning(
            "S3 中未找到 %s/%s，将尝试 openneuro-py 备用方式",
            dataset_id,
            subject_prefix,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "S3 下载失败，将尝试 openneuro-py 备用方式: %s: %s",
            type(exc).__name__,
            exc,
        )

    # ── 最终备用：openneuro-py ─────────────────────────────────────────────
    if tag is None:
        tag = _fetch_latest_tag(dataset_id, timeout=metadata_timeout)
    if tag is None:
        raise RuntimeError("无法从 OpenNeuro 获取数据集 tag，请检查网络连接。")

    _download_subject_via_openneuro_py(
        subject_prefix=subject_prefix,
        dataset_id=dataset_id,
        tag=tag,
        target_dir=target_dir,
        max_retries=max_retries,
        metadata_timeout=metadata_timeout,
        max_concurrent_downloads=max_concurrent_downloads,
    )
    logger.info("下载完成（openneuro-py）: %s / %s", dataset_id, subject_prefix)


def download_subjects(
    subjects: list[str],
    target_dir: str | Path = "/test_file3",
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
    per_file_connections: int = _DEFAULT_PER_FILE_CONNECTIONS,
) -> None:
    """Download data for multiple subjects sequentially.

    A failure for one subject does not abort downloads for the remaining subjects.

    Parameters
    ----------
    subjects:
        List of subject IDs without the ``"sub-"`` prefix.
    target_dir:
        Destination directory.
    dataset_id:
        OpenNeuro dataset ID.
    max_retries:
        Maximum retry attempts per file.
    metadata_timeout:
        Timeout in seconds for GraphQL metadata requests.
    per_file_connections:
        Number of parallel Range-request connections per large file.
        See :func:`download_subject` for details.
    """
    failed: List[str] = []
    with logging_redirect_tqdm():
        with tqdm(
            subjects,
            unit="subject",
            desc="被试下载进度",
            dynamic_ncols=True,
        ) as subject_bar:
            for subject in subject_bar:
                subject_bar.set_postfix_str(f"sub-{subject}", refresh=True)
                try:
                    download_subject(
                        subject=subject,
                        target_dir=target_dir,
                        dataset_id=dataset_id,
                        max_retries=max_retries,
                        metadata_timeout=metadata_timeout,
                        per_file_connections=per_file_connections,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("下载 sub-%s 失败 (%s): %s", subject, type(exc).__name__, exc)
                    failed.append(subject)

    if failed:
        logger.warning("以下被试下载失败: %s", failed)
    else:
        logger.info("全部 %d 个被试下载成功", len(subjects))


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 用法: python -m data.download_ds006040 [subjects] [target_dir]
    #   subjects: 逗号分隔（"001,002,029"）或空格分隔（"1" "2" "29"）的被试 ID
    #             零填充自动处理（"4" → "004"）
    #             在 Windows PowerShell 中 004,005,006 会被展开为独立参数 4 5 6，
    #             本 CLI 对两种形式均兼容。
    #   target_dir: 可选，默认 /test_file3；必须包含路径分隔符或以 / \ 开头
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Collect subject IDs from all positional args, handling both:
    #   "001,002,029"       (single comma-separated arg)
    #   "1" "2" "29"        (multiple args from PowerShell array expansion)
    # Once subjects are collected, any subsequent non-numeric arg is the
    # target_dir (e.g. "001 002 /data/ds006040" or "001 002 mydata").
    cli_target = Path("/test_file3")
    subject_list: list[str] = []

    for arg in sys.argv[1:]:
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        if parts and all(p.isdigit() for p in parts):
            subject_list.extend(parts)
        elif not subject_list:
            # No subjects collected yet → unrecognized leading arg
            print(
                f"错误: 无法识别的参数 {arg!r}。\n"
                f"期望被试 ID（纯数字）作为首个参数，例如: 004 或 001,002,029。",
                file=sys.stderr,
            )
            print(__doc__)
            sys.exit(1)
        else:
            # Subjects already collected → treat this arg as the target_dir
            cli_target = Path(arg)
            break

    if not subject_list:
        print(__doc__)
        sys.exit(1)

    if len(subject_list) == 1:
        download_subject(subject_list[0], target_dir=cli_target)
    else:
        download_subjects(subject_list, target_dir=cli_target)
