"""
tests/test_val.py
=================

Tests for val.py (standalone validation script).

Divided into two groups:

* ``mne``-free tests (run in CI without heavy neuro deps):
    - ``TestInferEmbeddingSizes``   — pure torch
    - ``TestResolveHelpers``        — filesystem only

* ``mne``-required tests (skipped when mne is unavailable):
    - ``TestLoadGraphsFromCache``   — calls main.extract_windowed_samples
    - ``TestRunVal``                — full end-to-end, calls main.create_model

All tests run on CPU.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

# ── path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Skip marker: skip tests that import from main.py (→ mne dependency).
# ---------------------------------------------------------------------------
_MNE_AVAILABLE = importlib.util.find_spec("mne") is not None
needs_mne = pytest.mark.skipif(
    not _MNE_AVAILABLE,
    reason="mne not installed; skipping tests that require main.py imports",
)

# ---------------------------------------------------------------------------
# Imports that do NOT trigger main.py / mne
# ---------------------------------------------------------------------------
from val import (
    _CACHE_FILENAME_RE,
    _infer_embedding_sizes,
    _resolve_cache_dir,
    _resolve_config,
)

# These imports only execute their lazy `from main import …` on first call,
# so importing the symbols themselves is safe without mne.
from val import _load_graphs_from_cache, run_val

from models.graph_native_mapper import GraphNativeBrainMapper
from models.graph_native_system import GraphNativeBrainModel, GraphNativeTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "graph_cache"
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_minimal_model() -> GraphNativeBrainModel:
    return GraphNativeBrainModel(
        node_types=["eeg", "fmri"],
        edge_types=[
            ("eeg", "connects", "eeg"),
            ("fmri", "connects", "fmri"),
            ("eeg", "projects_to", "fmri"),
        ],
        in_channels_dict={"eeg": 1, "fmri": 1},
        hidden_channels=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        use_prediction=True,
        prediction_steps=3,
        dropout=0.0,
        use_dynamic_graph=False,
        use_spectral_loss=False,
        use_cross_modal_align=False,
    )


def _save_fake_checkpoint(
    path: Path, model: GraphNativeBrainModel, epoch: int = 5
) -> None:
    trainer = GraphNativeTrainer(
        model=model,
        node_types=["eeg", "fmri"],
        learning_rate=1e-3,
        use_adaptive_loss=False,
        use_eeg_enhancement=False,
        use_amp=False,
        use_torch_compile=False,
        device="cpu",
    )
    trainer.save_checkpoint(path, epoch=epoch)


# ===========================================================================
# 1. _infer_embedding_sizes  (no mne required)
# ===========================================================================


class TestInferEmbeddingSizes:
    def test_no_embeddings(self):
        sd = {"some_layer.weight": torch.zeros(8, 4)}
        assert _infer_embedding_sizes(sd) == {}

    def test_subject_embed(self):
        sd = {"subject_embed.weight": torch.zeros(7, 16)}
        assert _infer_embedding_sizes(sd) == {"num_subjects": 7}

    def test_run_embed(self):
        sd = {"run_embed.weight": torch.zeros(3, 16)}
        assert _infer_embedding_sizes(sd) == {"num_runs": 3}

    def test_both_embeds(self):
        sd = {
            "subject_embed.weight": torch.zeros(5, 16),
            "run_embed.weight": torch.zeros(4, 16),
        }
        assert _infer_embedding_sizes(sd) == {"num_subjects": 5, "num_runs": 4}


# ===========================================================================
# 2. _resolve_config / _resolve_cache_dir  (no mne required)
# ===========================================================================


class TestResolveHelpers:
    # ── _resolve_config ──────────────────────────────────────────────────

    @needs_mne
    def test_explicit_config_arg(self, tmp_path):
        import shutil
        dst = tmp_path / "my_config.yaml"
        shutil.copy(DEFAULT_CONFIG_PATH, dst)
        cfg = _resolve_config(str(dst), Path("/fake/checkpoint.pt"))
        assert "data" in cfg and "model" in cfg

    @needs_mne
    def test_sibling_config_yaml(self, tmp_path):
        import shutil
        ckpt = tmp_path / "best_model.pt"
        ckpt.touch()
        shutil.copy(DEFAULT_CONFIG_PATH, tmp_path / "config.yaml")
        cfg = _resolve_config(None, ckpt)
        assert "model" in cfg

    @needs_mne
    def test_fallback_to_default(self, tmp_path):
        ckpt = tmp_path / "best_model.pt"
        ckpt.touch()
        cfg = _resolve_config(None, ckpt)
        assert "training" in cfg

    def test_explicit_config_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_config(str(tmp_path / "nonexistent.yaml"), Path("/fake/ckpt.pt"))

    # ── _resolve_cache_dir ───────────────────────────────────────────────

    def test_explicit_arg(self):
        assert _resolve_cache_dir("/my/cache", {}) == Path("/my/cache")

    def test_from_config(self):
        cfg = {"data": {"cache": {"dir": "/config/cache"}}}
        assert _resolve_cache_dir(None, cfg) == Path("/config/cache")

    def test_default_fallback(self):
        result = _resolve_cache_dir(None, {})
        assert "graph_cache" in str(result)

    def test_relative_config_path_resolved(self):
        cfg = {"data": {"cache": {"dir": "outputs/graph_cache"}}}
        result = _resolve_cache_dir(None, cfg)
        assert result.is_absolute()

    # ── _CACHE_FILENAME_RE ───────────────────────────────────────────────

    def test_regex_valid_filename(self):
        m = _CACHE_FILENAME_RE.match("sub-029_EOEC_abcd1234.pt")
        assert m is not None
        assert m.group(1) == "sub-029"
        assert m.group(2) == "EOEC"
        assert m.group(3) == "abcd1234"

    def test_regex_rejects_non_pt(self):
        assert _CACHE_FILENAME_RE.match("sub-029_EOEC_abcd1234.txt") is None

    def test_regex_rejects_short_hash(self):
        assert _CACHE_FILENAME_RE.match("sub-029_EOEC_abc123.pt") is None


# ===========================================================================
# 3. _load_graphs_from_cache  (requires mne via main.extract_windowed_samples)
# ===========================================================================


@needs_mne
class TestLoadGraphsFromCache:
    @pytest.fixture(scope="class")
    def minimal_config(self):
        from main import load_config
        cfg = load_config(str(DEFAULT_CONFIG_PATH))
        cfg["windowed_sampling"]["enabled"] = False
        return cfg

    def test_returns_nonempty_list(self, minimal_config):
        mapper = GraphNativeBrainMapper(device="cpu")
        graphs = _load_graphs_from_cache(FIXTURE_DIR, minimal_config, mapper)
        assert len(graphs) == 3

    def test_each_graph_is_heterodata(self, minimal_config):
        mapper = GraphNativeBrainMapper(device="cpu")
        graphs = _load_graphs_from_cache(FIXTURE_DIR, minimal_config, mapper)
        for g in graphs:
            assert isinstance(g, HeteroData)

    def test_cross_modal_edges_added(self, minimal_config):
        mapper = GraphNativeBrainMapper(device="cpu")
        graphs = _load_graphs_from_cache(FIXTURE_DIR, minimal_config, mapper)
        for g in graphs:
            assert ("eeg", "projects_to", "fmri") in g.edge_types

    def test_metadata_attached(self, minimal_config):
        mapper = GraphNativeBrainMapper(device="cpu")
        graphs = _load_graphs_from_cache(FIXTURE_DIR, minimal_config, mapper)
        for g in graphs:
            assert hasattr(g, "subject_idx")
            assert hasattr(g, "run_idx")
            assert hasattr(g, "task_id")
            assert hasattr(g, "subject_id_str")

    def test_subject_filter(self, minimal_config):
        mapper = GraphNativeBrainMapper(device="cpu")
        graphs = _load_graphs_from_cache(
            FIXTURE_DIR, minimal_config, mapper, subject_filter="sub-test01"
        )
        assert len(graphs) == 2
        for g in graphs:
            assert g.subject_id_str == "sub-test01"

    def test_missing_dir_raises(self, minimal_config):
        mapper = GraphNativeBrainMapper(device="cpu")
        with pytest.raises(FileNotFoundError):
            _load_graphs_from_cache(
                Path("/nonexistent/cache/dir"), minimal_config, mapper
            )

    def test_empty_dir_raises(self, minimal_config, tmp_path):
        mapper = GraphNativeBrainMapper(device="cpu")
        with pytest.raises(FileNotFoundError):
            _load_graphs_from_cache(tmp_path, minimal_config, mapper)


# ===========================================================================
# 4. run_val end-to-end  (requires mne via main.create_model)
# ===========================================================================


@needs_mne
class TestRunVal:
    """Full integration smoke test.

    Creates a tiny checkpoint, then calls run_val() which loads the model,
    loads fixture graphs, and runs GraphNativeTrainer.validate().
    """

    @pytest.fixture(scope="class")
    def checkpoint_dir(self, tmp_path_factory):
        import shutil
        tmp = tmp_path_factory.mktemp("val_test_output")
        ckpt_path = tmp / "best_model.pt"
        shutil.copy(DEFAULT_CONFIG_PATH, tmp / "config.yaml")
        model = _make_minimal_model()
        _save_fake_checkpoint(ckpt_path, model, epoch=3)
        return tmp

    def test_run_val_returns_dict(self, checkpoint_dir):
        result = run_val(
            checkpoint=str(checkpoint_dir / "best_model.pt"),
            cache_dir=str(FIXTURE_DIR),
            use_all=True,
            device="cpu",
            log_level="WARNING",
        )
        assert isinstance(result, dict)
        assert "val_loss" in result
        assert "r2_dict" in result

    def test_val_loss_is_finite(self, checkpoint_dir):
        import math
        result = run_val(
            checkpoint=str(checkpoint_dir / "best_model.pt"),
            cache_dir=str(FIXTURE_DIR),
            use_all=True,
            device="cpu",
            log_level="WARNING",
        )
        assert math.isfinite(result["val_loss"])

    def test_r2_dict_contains_expected_keys(self, checkpoint_dir):
        result = run_val(
            checkpoint=str(checkpoint_dir / "best_model.pt"),
            cache_dir=str(FIXTURE_DIR),
            use_all=True,
            device="cpu",
            log_level="WARNING",
        )
        r2 = result["r2_dict"]
        assert "r2_eeg" in r2
        assert "r2_fmri" in r2
        assert "pred_r2_eeg" in r2
        assert "pred_r2_fmri" in r2

    def test_output_json_saved(self, checkpoint_dir, tmp_path):
        json_out = tmp_path / "val_output.json"
        run_val(
            checkpoint=str(checkpoint_dir / "best_model.pt"),
            cache_dir=str(FIXTURE_DIR),
            use_all=True,
            device="cpu",
            output_json=str(json_out),
            log_level="WARNING",
        )
        assert json_out.exists()
        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "val_loss" in data and "r2_dict" in data

    def test_saved_epoch_recorded(self, checkpoint_dir):
        result = run_val(
            checkpoint=str(checkpoint_dir / "best_model.pt"),
            cache_dir=str(FIXTURE_DIR),
            use_all=True,
            device="cpu",
            log_level="WARNING",
        )
        assert result["saved_epoch"] == 3

    def test_n_val_samples_positive(self, checkpoint_dir):
        result = run_val(
            checkpoint=str(checkpoint_dir / "best_model.pt"),
            cache_dir=str(FIXTURE_DIR),
            use_all=True,
            device="cpu",
            log_level="WARNING",
        )
        assert result["n_val_samples"] >= 1

    def test_checkpoint_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_val(
                checkpoint=str(tmp_path / "nonexistent.pt"),
                cache_dir=str(FIXTURE_DIR),
                use_all=True,
                device="cpu",
                log_level="WARNING",
            )

    def test_cache_dir_not_found_raises(self, checkpoint_dir):
        with pytest.raises(FileNotFoundError):
            run_val(
                checkpoint=str(checkpoint_dir / "best_model.pt"),
                cache_dir="/nonexistent/cache",
                use_all=True,
                device="cpu",
                log_level="WARNING",
            )
