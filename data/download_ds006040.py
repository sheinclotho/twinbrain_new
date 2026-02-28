"""
ds006040 数据集下载工具
=======================

从 OpenNeuro 下载 ds006040 数据集的指定被试数据。

**已知问题及修复说明**

openneuro-py 在未提供 ``tag`` 参数时，根目录文件列表查询使用
``dataset.latestSnapshot.files`` GraphQL 端点。该端点受 OpenNeuro 服务端
分页限制，大型数据集（被试数 > 首页返回上限，约 25–30 条）只返回前 N 个
顶层目录条目。sub-001 和 sub-002 恰好位于第一页，而 sub-029 等后续被试
不在第一页，导致 openneuro-py 报告 "Could not find path in the dataset"。

修复方法：先查询数据集当前最新快照的 ``tag``，再将其显式传入 ``download()``。
这使根目录查询切换为 ``snapshot.files(tree: null)`` 端点，该端点返回
完整的顶层目录列表，不受首页分页限制。

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
from pathlib import Path
from typing import List, Optional

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
        logger.debug("データセット %s の最新タグ: %s", dataset_id, tag)
        return tag
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "タグ取得に失敗しました（ページネーション回避できない可能性あり）: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None


def download_subject(
    subject: str,
    target_dir: str | Path = "/test_file3",  # 与原始示例代码保持一致；按需修改为实际路径
    dataset_id: str = DATASET_ID,
    max_retries: int = 10,
    metadata_timeout: float = 30.0,
    max_concurrent_downloads: int = 5,
) -> None:
    """指定被試のデータを OpenNeuro からダウンロードする。

    Parameters
    ----------
    subject:
        被試 ID（先頭の "sub-" なし）。例: ``"029"`` → ``sub-029`` をダウンロード。
    target_dir:
        ダウンロード先ディレクトリ。デフォルトは ``/test_file3``。
    dataset_id:
        OpenNeuro データセット ID。デフォルトは ``ds006040``。
    max_retries:
        ファイルごとの最大リトライ回数。
    metadata_timeout:
        メタデータクエリのタイムアウト秒数。
    max_concurrent_downloads:
        同時並行ダウンロード数の上限。
    """
    from openneuro import download  # type: ignore[import-untyped]

    target_dir = Path(target_dir)
    subject_prefix = f"sub-{subject.zfill(3)}"

    # ── 修复核心 ──────────────────────────────────────────────────────────────
    # 先取得最新 tag，传入 download() 后根目录查询切换为
    # snapshot.files(tree: null)，返回完整被試列表，不受分页限制。
    # tag 为 None 时退化为原始行为（仅前 N 个被試可见，001/002 以外无法找到）。
    tag = _fetch_latest_tag(dataset_id, timeout=metadata_timeout)

    logger.info(
        "开始下载 %s / %s → %s (tag=%s)",
        dataset_id,
        subject_prefix,
        target_dir,
        tag or "latest(untagged)",
    )

    # 兼容旧版 openneuro-py（不支持 metadata_timeout 参数）
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
    else:
        logger.debug(
            "当前 openneuro-py 不支持 metadata_timeout 参数，已跳过（请升级到最新版本）"
        )

    download(**_download_kwargs)

    logger.info("下载完成: %s / %s", dataset_id, subject_prefix)


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
    for subject in subjects:
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
