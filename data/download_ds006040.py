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

使用方法::

    # 下载单个被试
    python -m data.download_ds006040 029

    # 下载单个被试到指定目录
    python -m data.download_ds006040 029 /data/ds006040

    # 下载多个被试（逗号分隔）
    python -m data.download_ds006040 001,002,029

    # 在 Python 代码中调用
    from data.download_ds006040 import download_subject
    download_subject("029", target_dir="/test_file3")
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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


def _fetch_latest_tag(dataset_id: str, timeout: float = 30.0) -> Optional[str]:
    """最新スナップショットタグを取得する。

    OpenNeuro GraphQL API から ``dataset.latestSnapshot.id`` を取得し、
    ``"ds006040:1.0.0"`` 形式の文字列から ``"1.0.0"`` 部分を返す。
    ネットワークエラー時は ``None`` を返す。
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
        # snapshot_id は "ds006040:1.0.0" の形式
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


def download_subject(
    subject: str,
    target_dir: str | Path = "/test_file3",  # 与原始示例代码保持一致；按需修改为实际路径
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
    max_concurrent_downloads: int = 5,
) -> None:
    """指定被試のデータを OpenNeuro からダウンロードする。

    先ず S3 公共存储桶方式（主路径，无分页限制）；若 S3 方式失败，
    自动回退到 openneuro-py 方式（备用路径，需先获取快照标签）。

    Parameters
    ----------
    subject:
        被試 ID（先頭の "sub-" なし）。例: ``"029"`` → ``sub-029`` をダウンロード。
        若传入的 ID 已有前导零（如 "004"）则直接使用；否则自动补零至 3 位。
    target_dir:
        ダウンロード先ディレクトリ。デフォルトは ``/test_file3``。
    dataset_id:
        OpenNeuro データセット ID。デフォルトは ``ds006040``。
    max_retries:
        ファイルごとの最大リトライ回数。
    metadata_timeout:
        メタデータクエリのタイムアウト秒数。
    max_concurrent_downloads:
        同時並行ダウンロード数の上限（openneuro-py 回退时使用）。
    """
    target_dir = Path(target_dir)
    subject_prefix = f"sub-{subject.zfill(3)}"

    tag = _fetch_latest_tag(dataset_id, timeout=metadata_timeout)

    logger.info(
        "开始下载 %s / %s → %s (tag=%s)",
        dataset_id,
        subject_prefix,
        target_dir,
        tag or "latest(untagged)",
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


def download_subjects(
    subjects: List[str],
    target_dir: str | Path = "/test_file3",  # 与原始示例代码保持一致；按需修改为实际路径
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
) -> None:
    """批量下载多个被試数据。

    依次下载每个被試，单个失败不中断其余被試的下载。

    Parameters
    ----------
    subjects:
        被試 ID 列表（不含 ``"sub-"`` 前缀）。例如: ``["001", "002", "029"]``。
    target_dir:
        下载目标目录。
    dataset_id:
        OpenNeuro 数据集 ID。
    max_retries:
        每个文件的最大重试次数。
    metadata_timeout:
        元数据查询超时秒数。
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

    if failed:
        logger.warning("以下被試下载失败: %s", failed)
    else:
        logger.info("全部 %d 个被試下载成功", len(subjects))


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 用法: python -m data.download_ds006040 [subjects] [target_dir]
    #   subjects: 单个 ID（如 "029"）或逗号分隔列表（如 "001,002,029"）
    #   target_dir: 可选，默认 /test_file3
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    raw_subjects = sys.argv[1]
    subject_list = [s.strip() for s in raw_subjects.split(",") if s.strip()]

    cli_target = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/test_file3")

    if len(subject_list) == 1:
        download_subject(subject_list[0], target_dir=cli_target)
    else:
        download_subjects(subject_list, target_dir=cli_target)
