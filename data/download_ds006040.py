"""
ds006040 数据集下载工具
=======================

从 OpenNeuro 下载 ds006040 数据集的指定被试数据。

**问题根因与修复说明**

openneuro-py 的 ``download()`` 函数在遍历数据集根目录时，无论是否传入
``tag`` 参数，服务端 GraphQL ``files(tree: null)`` 查询均受分页限制，
每次最多返回约 9 条顶层条目。当数据集根目录条目数超过该限制时
（根文件 + 多个 sub-XXX 目录），靠后的被试（如 sub-004、sub-005）
不在第一页，导致 openneuro-py 报告 "Could not find path in the dataset"。

**重新设计**：改用 OpenNeuro S3 公共存储桶直接下载。
- 公共数据集均已发布至 S3 存储桶 ``openneuro.org``，支持匿名读取。
- 使用 S3 REST API（XML 响应）列出文件，天然支持分页，无条目数限制。
- 每个文件直接通过 HTTPS 下载，无需 openneuro-py 介入。
- 若 S3 方式失败（非公共数据集等情况），自动回退到 openneuro-py。
**设计说明**

直接调用 OpenNeuro GraphQL API 递归枚举被试目录下的所有文件，并通过 API
返回的下载 URL 逐一下载，完全绕过 openneuro-py 的 ``include`` 过滤器。

这样做的原因：openneuro-py 的 ``include`` 过滤器在实现中存在缺陷——当顶层
目录条目超出 API 单次返回上限时，无法找到位于后续页的被试目录，导致
"Could not find path in the dataset" 错误。本实现通过直接查询目标被试的
git tree key，再递归遍历子目录，彻底规避这一问题。

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

import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from pathlib import Path
from typing import Generator, Optional

import httpx
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

DATASET_ID = "ds006040"
_OPENNEURO_GQL_URL = "https://openneuro.org/crn/graphql"

# OpenNeuro open-access datasets are stored in this public S3 bucket.
_S3_BUCKET = "openneuro.org"
_S3_BASE_URL = f"https://{_S3_BUCKET}.s3.amazonaws.com"
_S3_XML_NS = "http://s3.amazonaws.com/doc/2006-03-01/"
_CHUNK_SIZE = 64 * 1024  # 64 KiB per read chunk
_MAX_BACKOFF_SECONDS = 60  # cap for exponential retry delay

_TAG_QUERY = """
query GetLatestTag($datasetId: ID!) {
  dataset(id: $datasetId) {
    latestSnapshot {
      id
    }
  }
}
"""

# GraphQL query to list files at a given tree node within a snapshot.
# tree: null  → top-level entries; tree: "<hash>" → contents of that directory.
_FILES_QUERY_TEMPLATE = """
query {{
  snapshot(datasetId: "{dataset_id}", tag: "{tag}") {{
    files(tree: {tree_arg}) {{
      filename
      urls
      size
      directory
      key
    }}
  }}
}}
"""


def _fetch_latest_tag(dataset_id: str, timeout: float = 30.0) -> Optional[str]:
    """Fetch the latest snapshot tag from the OpenNeuro GraphQL API.

    Returns the tag string (e.g. ``"1.0.1"``) or ``None`` on failure.
    """
    try:
        response = httpx.post(
            _OPENNEURO_GQL_URL,
            json={"query": _TAG_QUERY, "variables": {"datasetId": dataset_id}},
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
        logger.warning(
            "获取快照标签失败（将尝试 S3 直接下载）: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None


# ── S3 直接下载（主路径）──────────────────────────────────────────────────────


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


def _s3_download_file(
    key: str,
    dest: Path,
    timeout: float = 600.0,
    max_retries: int = 10,
    file_label: str = "",
) -> None:
    """从 S3 下载单个文件（流式传输，支持断点重试，含字节级进度条）。"""
    url = f"{_S3_BASE_URL}/{key}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                total_bytes = int(resp.headers["content-length"]) if "content-length" in resp.headers else None
                label = file_label or dest.name
                with tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                    leave=False,
                    dynamic_ncols=True,
                ) as bar:
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=_CHUNK_SIZE):
                            fh.write(chunk)
                            bar.update(len(chunk))
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
) -> int:
    """通过 S3 公共存储桶下载指定被试的全部文件。

    Returns
    -------
    int
        本次操作处理的文件总数（含已存在跳过的文件和新下载的文件）。
        若该被试在 S3 中不存在，返回 0。
        logger.debug("数据集 %s 最新 tag: %s", dataset_id, tag)
        return tag
    except Exception as exc:  # noqa: BLE001
        logger.warning("获取 tag 失败: %s: %s", type(exc).__name__, exc)
        return None


def _query_tree_files(
    dataset_id: str,
    tag: str,
    tree: Optional[str] = None,
    timeout: float = 30.0,
) -> list:
    """Query the files at a specific tree node in a snapshot.

    Parameters
    ----------
    dataset_id:
        OpenNeuro dataset ID.
    tag:
        Snapshot tag (e.g. ``"1.0.1"``).
    tree:
        Git tree hash for a specific directory, or ``None`` for the root.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    list of dict with keys: filename, urls, size, directory, key.
    """
    tree_arg = f'"{tree}"' if tree is not None else "null"
    query = _FILES_QUERY_TEMPLATE.format(
        dataset_id=dataset_id, tag=tag, tree_arg=tree_arg
    )
    response = httpx.post(
        _OPENNEURO_GQL_URL,
        json={"query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL error: {body['errors']}")
    return body["data"]["snapshot"]["files"]


def _iter_tree(
    dataset_id: str,
    tag: str,
    tree_key: str,
    path_prefix: str,
    timeout: float,
) -> Generator[tuple[str, str], None, None]:
    """Recursively yield (relative_path, download_url) for all files under tree_key."""
    entries = _query_tree_files(dataset_id, tag, tree=tree_key, timeout=timeout)
    for entry in entries:
        full_path = f"{path_prefix}/{entry['filename']}"
        if entry["directory"]:
            yield from _iter_tree(dataset_id, tag, entry["key"], full_path, timeout)
        else:
            urls = entry.get("urls") or []
            if urls:
                yield full_path, urls[0]
            else:
                logger.warning("文件无下载 URL，已跳过: %s", full_path)


def _iter_subject_files(
    dataset_id: str,
    tag: str,
    subject_prefix: str,
    timeout: float = 30.0,
) -> Generator[tuple[str, str], None, None]:
    """Yield (relative_path, download_url) for every file under subject_prefix.

    First queries the snapshot root to locate the subject's directory entry
    (and its git tree key), then recursively traverses all subdirectories.

    Raises
    ------
    RuntimeError
        S3 列表请求失败（网络错误、存储桶不可访问等）。
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
    downloaded = 0
    with logging_redirect_tqdm():
        with tqdm(
            total=len(files),
            unit="file",
            desc=subject_prefix,
            dynamic_ncols=True,
        ) as file_bar:
            for key, _size in files:
                if not key.startswith(dataset_prefix):
                    logger.warning("S3 返回的对象键不符合预期格式，已跳过: %s", key)
                    file_bar.update(1)
                    continue
                rel = key[len(dataset_prefix):]
                dest = target_dir / rel
                file_bar.set_postfix_str(rel, refresh=True)
                if dest.exists() and dest.stat().st_size == _size:
                    logger.debug("跳过已存在文件: %s", rel)
                    downloaded += 1
                    file_bar.update(1)
                    continue
                logger.debug("下载: %s", rel)
                _s3_download_file(key, dest, timeout=600.0, max_retries=max_retries, file_label=rel)
                downloaded += 1
                file_bar.update(1)

    return downloaded


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
        If the subject directory is not found in the snapshot root listing.
    """
    top_entries = _query_tree_files(dataset_id, tag, tree=None, timeout=timeout)
    subject_entry = next(
        (e for e in top_entries if e["filename"] == subject_prefix and e["directory"]),
        None,
    )
    if subject_entry is None:
        available = [e["filename"] for e in top_entries]
        raise RuntimeError(
            f"在快照 {tag!r} 中未找到被试目录 {subject_prefix!r}。"
            f"顶层条目: {available!r}"
        )
    yield from _iter_tree(
        dataset_id, tag, subject_entry["key"], subject_prefix, timeout
    )


def _download_file_direct(url: str, dest: Path, timeout: float = 300.0) -> None:
    """Stream-download a single file from *url* and save it to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=65536):
                fh.write(chunk)


def download_subject(
    subject: str,
    target_dir: str | Path = "/test_file3",
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
    max_concurrent_downloads: int = 5,  # kept for API compatibility; downloads are sequential
) -> None:
    """Download all data files for *subject* from OpenNeuro.

    先ず S3 公共存储桶方式（主路径，无分页限制）；若 S3 方式失败，
    自动回退到 openneuro-py 方式（备用路径，需先获取快照标签）。

    Parameters
    ----------
    subject:
        被試 ID（先頭の "sub-" なし）。例: ``"029"`` → ``sub-029`` をダウンロード。
        若传入的 ID 已有前导零（如 "004"）则直接使用；否则自动补零至 3 位。
        Subject ID without the ``"sub-"`` prefix. Zero-padding is applied
        automatically (e.g. ``"4"`` → ``sub-004``).
    target_dir:
        Destination directory. Defaults to ``/test_file3``.
    dataset_id:
        OpenNeuro dataset ID. Defaults to ``ds006040``.
    max_retries:
        Maximum retry attempts per file.
    metadata_timeout:
        Timeout in seconds for GraphQL metadata requests.
    max_concurrent_downloads:
        同時並行ダウンロード数の上限（openneuro-py 回退时使用）。
        Retained for API compatibility; currently downloads are sequential.
    """
    target_dir = Path(target_dir)
    subject_prefix = f"sub-{subject.zfill(3)}"

    tag = _fetch_latest_tag(dataset_id, timeout=metadata_timeout)
    if tag is None:
        raise RuntimeError("无法从 OpenNeuro 获取数据集 tag，请检查网络连接。")

    logger.info(
        "开始下载 %s / %s → %s (tag=%s)",
        dataset_id,
        subject_prefix,
        target_dir,
        tag,
    )

    # ── 主路径：S3 直接下载 ────────────────────────────────────────────────
    try:
        count = _download_subject_via_s3(
            subject_prefix=subject_prefix,
            dataset_id=dataset_id,
            target_dir=target_dir,
            max_retries=max_retries,
            timeout=metadata_timeout,
        )
        if count > 0:
            logger.info("下载完成（S3）: %s / %s，共 %d 个文件", dataset_id, subject_prefix, count)
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

    # ── 备用路径：openneuro-py ─────────────────────────────────────────────
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
    downloaded = 0
    skipped = 0
    for rel_path, url in _iter_subject_files(
        dataset_id, tag, subject_prefix, timeout=metadata_timeout
    ):
        dest = target_dir / rel_path
        if dest.exists():
            logger.debug("已存在，跳过: %s", rel_path)
            skipped += 1
            continue

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                logger.debug("下载: %s", rel_path)
                _download_file_direct(url, dest)
                downloaded += 1
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = min(2 ** attempt, 60)
                logger.debug(
                    "第 %d/%d 次重试 %s (等待 %ds): %s",
                    attempt + 1,
                    max_retries,
                    rel_path,
                    wait,
                    exc,
                )
                time.sleep(wait)
        if last_exc is not None:
            raise RuntimeError(
                f"下载 {rel_path} 失败（已重试 {max_retries} 次）: {last_exc}"
            )

    logger.info(
        "下载完成: %s / %s（新增 %d 个文件，跳过 %d 个）",
        dataset_id,
        subject_prefix,
        downloaded,
        skipped,
    )


def download_subjects(
    subjects: list[str],
    target_dir: str | Path = "/test_file3",
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
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
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("下载 sub-%s 失败 (%s): %s", subject, type(exc).__name__, exc)
                    failed.append(subject)
    failed: list[str] = []
    for subject in subjects:
        subject_padded = subject.zfill(3)
        try:
            download_subject(
                subject=subject_padded,
                target_dir=target_dir,
                dataset_id=dataset_id,
                max_retries=max_retries,
                metadata_timeout=metadata_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "下载 sub-%s 失败 (%s): %s", subject_padded, type(exc).__name__, exc
            )
            failed.append(subject_padded)

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
