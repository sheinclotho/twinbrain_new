"""
ds006040 数据集下载工具
=======================

从 OpenNeuro 下载 ds006040 数据集的指定被试数据。

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
from pathlib import Path
from typing import Generator, Optional

import httpx

logger = logging.getLogger(__name__)

DATASET_ID = "ds006040"
_OPENNEURO_GQL_URL = "https://openneuro.org/crn/graphql"

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
        Timeout in seconds for GraphQL metadata requests.
    max_concurrent_downloads:
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
