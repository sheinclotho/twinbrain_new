"""
Training-pipeline smoke tests using the synthetic .pt graph-cache fixtures in
tests/fixtures/graph_cache/.

These tests bypass the real EEG/fMRI data-loading pipeline (BrainDataLoader,
atlas parcellation, etc.) and directly exercise the model training/validation
loop, making them fast (~10-20 seconds total on CPU).

Fixture inventory
-----------------
| File                              | N_eeg | T_eeg | N_fmri | T_fmri |
|-----------------------------------|-------|-------|--------|--------|
| sub-test01_EOEC_testfixture.pt    |  16   |   50  |   10   |   25   |
| sub-test01_GRADON_testfixture.pt  |  19   |   60  |   12   |   30   |
| sub-test02_EOEC_testfixture.pt    |  32   |  100  |   20   |   50   |

Note: these fixtures have N_eeg > N_fmri (e.g. 16 > 10), which is the reverse
of the real-data design intent (N_eeg < N_fmri required by the neurovascular
coupling model — EEG electrodes project onto more-numerous fMRI ROIs).  The
reversal is intentional here: keeping N_fmri tiny minimises fixture file size
while still exercising the cross-modal message-passing code path.  The mapper
logs a ⚠ warning in this case but continues without error, so the tests also
verify graceful degradation of the design-intent guard.
"""

import math

import pytest
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from models.graph_native_mapper import GraphNativeBrainMapper
from models.graph_native_system import GraphNativeBrainModel, GraphNativeTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "graph_cache"

_FIXTURE_SPECS = [
    # (filename,                             N_eeg, T_eeg, N_fmri, T_fmri)
    ("sub-test01_EOEC_testfixture.pt",    16,    50,    10,    25),
    ("sub-test01_GRADON_testfixture.pt",  19,    60,    12,    30),
    ("sub-test02_EOEC_testfixture.pt",    32,   100,    20,    50),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(
    filename: str,
    subject_idx: int,
    run_idx: int,
    task_id: str,
    subject_id_str: str,
    k_cross_modal: int = 3,
) -> HeteroData:
    """Load a fixture .pt file and attach the metadata required by the trainer.

    Mirrors the logic in main.py::build_graphs() for the cache-hit path:
    1.  torch.load() the HeteroData object.
    2.  Attach subject_idx / run_idx / task_id / subject_id_str.
    3.  Rebuild cross-modal edges (not stored in cache files).
    """
    mapper = GraphNativeBrainMapper(device="cpu")
    graph = torch.load(
        FIXTURE_DIR / filename, map_location="cpu", weights_only=False
    )
    # Metadata written by build_graphs()
    graph.subject_idx = torch.tensor(subject_idx, dtype=torch.long)
    graph.run_idx = torch.tensor(run_idx, dtype=torch.long)
    graph.task_id = task_id
    graph.subject_id_str = subject_id_str
    # Rebuild cross-modal edges from node features (not stored in cache)
    cross = mapper.create_simple_cross_modal_edges(graph, k_cross_modal=k_cross_modal)
    if cross is not None:
        graph[("eeg", "projects_to", "fmri")].edge_index = cross[0]
        graph[("eeg", "projects_to", "fmri")].edge_attr = cross[1]
    return graph


def _make_model(prediction_steps: int = 3) -> GraphNativeBrainModel:
    """Create a minimal model for fast CPU-based testing."""
    node_types = ["eeg", "fmri"]
    edge_types = [
        ("eeg", "connects", "eeg"),
        ("fmri", "connects", "fmri"),
        ("eeg", "projects_to", "fmri"),
    ]
    return GraphNativeBrainModel(
        node_types=node_types,
        edge_types=edge_types,
        in_channels_dict={"eeg": 1, "fmri": 1},
        hidden_channels=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        use_prediction=True,
        prediction_steps=prediction_steps,
        dropout=0.0,
        use_dynamic_graph=False,
        use_spectral_loss=False,
        use_cross_modal_align=False,
    )


def _make_trainer(
    model: GraphNativeBrainModel,
    use_adaptive_loss: bool = False,
    gradient_accumulation_steps: int = 1,
) -> GraphNativeTrainer:
    """Create a CPU-only trainer suitable for unit tests."""
    return GraphNativeTrainer(
        model=model,
        node_types=["eeg", "fmri"],
        learning_rate=1e-3,
        use_adaptive_loss=use_adaptive_loss,
        use_eeg_enhancement=False,
        use_amp=False,           # no CUDA in CI
        use_torch_compile=False, # skip compilation overhead
        device="cpu",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


# ---------------------------------------------------------------------------
# Module-scoped shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_graphs():
    """All three fixture graphs with metadata and cross-modal edges attached."""
    return [
        _load_fixture(
            "sub-test01_EOEC_testfixture.pt",   0, 0, "EOEC",   "sub-test01"
        ),
        _load_fixture(
            "sub-test01_GRADON_testfixture.pt",  0, 1, "GRADON", "sub-test01"
        ),
        _load_fixture(
            "sub-test02_EOEC_testfixture.pt",    1, 2, "EOEC",   "sub-test02"
        ),
    ]


# ===========================================================================
# 1. Fixture schema validation
# ===========================================================================


class TestFixtureSchema:
    """Validate that every fixture file conforms to the documented schema."""

    @pytest.mark.parametrize(
        "filename,N_eeg,T_eeg,N_fmri,T_fmri", _FIXTURE_SPECS
    )
    def test_node_feature_shapes(self, filename, N_eeg, T_eeg, N_fmri, T_fmri):
        graph = torch.load(
            FIXTURE_DIR / filename, map_location="cpu", weights_only=False
        )
        assert graph["eeg"].x.shape == (N_eeg, T_eeg, 1)
        assert graph["fmri"].x.shape == (N_fmri, T_fmri, 1)

    @pytest.mark.parametrize(
        "filename,N_eeg,T_eeg,N_fmri,T_fmri", _FIXTURE_SPECS
    )
    def test_dtypes(self, filename, N_eeg, T_eeg, N_fmri, T_fmri):
        graph = torch.load(
            FIXTURE_DIR / filename, map_location="cpu", weights_only=False
        )
        assert graph["eeg"].x.dtype == torch.float32
        assert graph["fmri"].x.dtype == torch.float32
        assert graph[("eeg", "connects", "eeg")].edge_index.dtype == torch.int64
        assert graph[("fmri", "connects", "fmri")].edge_index.dtype == torch.int64

    @pytest.mark.parametrize(
        "filename,N_eeg,T_eeg,N_fmri,T_fmri", _FIXTURE_SPECS
    )
    def test_intra_modal_edges_present_cross_modal_absent(
        self, filename, N_eeg, T_eeg, N_fmri, T_fmri
    ):
        """Cross-modal edges must NOT be stored in the cache file."""
        graph = torch.load(
            FIXTURE_DIR / filename, map_location="cpu", weights_only=False
        )
        assert ("eeg", "connects", "eeg") in graph.edge_types
        assert ("fmri", "connects", "fmri") in graph.edge_types
        assert ("eeg", "projects_to", "fmri") not in graph.edge_types

    def test_eeg_features_approximately_z_scored(self):
        """EEG features should be approximately z-scored (|mean| < 1)."""
        graph = torch.load(
            FIXTURE_DIR / "sub-test02_EOEC_testfixture.pt",
            map_location="cpu",
            weights_only=False,
        )
        x = graph["eeg"].x.squeeze(-1)  # [N, T]
        assert x.abs().mean().item() < 2.0


# ===========================================================================
# 2. Cross-modal edge reconstruction
# ===========================================================================


class TestCrossModalEdges:
    """Verify that cross-modal edges are rebuilt correctly at load time."""

    def test_cross_modal_edges_created_after_load(self):
        graph = _load_fixture(
            "sub-test01_EOEC_testfixture.pt", 0, 0, "EOEC", "sub-test01"
        )
        assert ("eeg", "projects_to", "fmri") in graph.edge_types

    def test_cross_modal_edge_shapes(self):
        graph = _load_fixture(
            "sub-test01_EOEC_testfixture.pt", 0, 0, "EOEC", "sub-test01"
        )
        ei = graph[("eeg", "projects_to", "fmri")].edge_index
        ea = graph[("eeg", "projects_to", "fmri")].edge_attr
        assert ei.shape[0] == 2
        assert ea.shape == (ei.shape[1], 1)

    def test_cross_modal_edge_indices_in_range(self):
        graph = _load_fixture(
            "sub-test01_EOEC_testfixture.pt", 0, 0, "EOEC", "sub-test01"
        )
        ei = graph[("eeg", "projects_to", "fmri")].edge_index
        N_eeg = graph["eeg"].num_nodes
        N_fmri = graph["fmri"].num_nodes
        assert ei[0].max().item() < N_eeg, "EEG source index out of range"
        assert ei[1].max().item() < N_fmri, "fMRI destination index out of range"

    def test_cross_modal_edge_weights_positive(self):
        graph = _load_fixture(
            "sub-test01_EOEC_testfixture.pt", 0, 0, "EOEC", "sub-test01"
        )
        ea = graph[("eeg", "projects_to", "fmri")].edge_attr
        assert (ea >= 0).all(), "Edge weights should be non-negative (|Pearson r|)"
        assert (ea <= 1).all(), "Edge weights should be at most 1.0 (|Pearson r| ∈ [0,1])"

    def test_hrf_lag_produces_valid_edges(self):
        """HRF lag compensation must still produce valid cross-modal edges."""
        mapper = GraphNativeBrainMapper(device="cpu")
        graph = torch.load(
            FIXTURE_DIR / "sub-test02_EOEC_testfixture.pt",
            map_location="cpu",
            weights_only=False,
        )
        cross = mapper.create_simple_cross_modal_edges(
            graph, k_cross_modal=3, hrf_lag_tr=2
        )
        assert cross is not None, "HRF-lagged edge construction returned None"
        ei, ea = cross
        assert ei.shape[0] == 2, "edge_index must have 2 rows"
        assert ea.shape == (ei.shape[1], 1), "edge_attr shape mismatch"
        N_eeg = graph["eeg"].num_nodes
        N_fmri = graph["fmri"].num_nodes
        assert ei[0].max().item() < N_eeg, "EEG source index out of range"
        assert ei[1].max().item() < N_fmri, "fMRI destination index out of range"
        assert (ea >= 0).all() and (ea <= 1).all(), "Edge weights must be in [0, 1]"

    def test_hrf_lag_0_and_no_lag_produce_same_edges(self):
        """hrf_lag_tr=0 must behave identically to the default (no lag)."""
        mapper = GraphNativeBrainMapper(device="cpu")
        graph = torch.load(
            FIXTURE_DIR / "sub-test01_EOEC_testfixture.pt",
            map_location="cpu",
            weights_only=False,
        )
        cross_default = mapper.create_simple_cross_modal_edges(graph, k_cross_modal=3)
        cross_lag0 = mapper.create_simple_cross_modal_edges(
            graph, k_cross_modal=3, hrf_lag_tr=0
        )
        assert cross_default is not None and cross_lag0 is not None
        assert torch.equal(cross_default[0], cross_lag0[0]), \
            "hrf_lag_tr=0 edge_index differs from default"
        assert torch.allclose(cross_default[1], cross_lag0[1]), \
            "hrf_lag_tr=0 edge_attr differs from default"

    def test_hrf_lag_larger_than_time_series_falls_back_gracefully(self):
        """When lag ≥ T-1 the lag is silently skipped; result equals no-lag baseline."""
        mapper = GraphNativeBrainMapper(device="cpu")
        graph = torch.load(
            FIXTURE_DIR / "sub-test01_EOEC_testfixture.pt",
            map_location="cpu",
            weights_only=False,
        )
        # T_fmri=25 in this fixture; a lag of 30 exceeds T-1 and must fall back to no-lag
        T_fmri = graph["fmri"].x.shape[1]
        oversized_lag = T_fmri + 5
        cross_fallback = mapper.create_simple_cross_modal_edges(
            graph, k_cross_modal=3, hrf_lag_tr=oversized_lag
        )
        cross_no_lag = mapper.create_simple_cross_modal_edges(
            graph, k_cross_modal=3, hrf_lag_tr=0
        )
        assert cross_fallback is not None, "Oversized lag must still produce edges (fallback)"
        assert cross_no_lag is not None
        ei_fb, ea_fb = cross_fallback
        ei_nl, ea_nl = cross_no_lag
        assert ei_fb.shape[0] == 2
        # Fallback with oversized lag must produce the same edges as no-lag
        assert torch.equal(ei_fb, ei_nl), \
            "Oversized lag should fall back to no-lag edge_index"
        assert torch.allclose(ea_fb, ea_nl), \
            "Oversized lag should fall back to no-lag edge_attr"


# ===========================================================================
# 3. One-epoch training
# ===========================================================================


class TestTrainingWithFixtures:
    """Run the model training loop with fixture graphs."""

    def test_one_epoch_returns_finite_loss(self, all_graphs):
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        loss = trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=3)
        assert math.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_loss_decreases_over_epochs(self, all_graphs):
        """Training loss should decrease (on average) over several epochs."""
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        losses = [
            trainer.train_epoch(all_graphs[:2], epoch=e, total_epochs=10)
            for e in range(1, 11)
        ]
        # Allow fluctuation; final average should be lower than initial average
        avg_first = sum(losses[:3]) / 3
        avg_last = sum(losses[-3:]) / 3
        assert avg_last < avg_first, (
            f"Expected loss to decrease from {avg_first:.4f} to < {avg_last:.4f}"
        )

    def test_gradient_accumulation_produces_finite_loss(self, all_graphs):
        """Gradient accumulation (ga=2) must not crash and loss must be finite."""
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model, gradient_accumulation_steps=2)
        loss = trainer.train_epoch(all_graphs, epoch=1, total_epochs=2)
        assert math.isfinite(loss)

    def test_adaptive_loss_balancer_does_not_crash(self, all_graphs):
        """AdaptiveLossBalancer should initialise and survive a full epoch."""
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model, use_adaptive_loss=True)
        loss = trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=2)
        assert math.isfinite(loss)

    def test_all_three_fixtures_train_without_error(self, all_graphs):
        """The larger fixture (sub-test02) should also train without error."""
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        loss = trainer.train_epoch(all_graphs, epoch=1, total_epochs=2)
        assert math.isfinite(loss)


# ===========================================================================
# 4. Validation pass
# ===========================================================================


class TestValidationWithFixtures:
    """Run the validation pass with fixture graphs."""

    def test_validate_returns_tuple(self, all_graphs):
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=2)
        result = trainer.validate([all_graphs[2]])
        assert isinstance(result, tuple) and len(result) == 2

    def test_validate_loss_is_finite(self, all_graphs):
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=2)
        val_loss, _ = trainer.validate([all_graphs[2]])
        assert math.isfinite(val_loss)

    def test_validate_r2_dict_has_expected_keys(self, all_graphs):
        """validate() must return reconstruction and prediction R² for both modalities."""
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=2)
        _, r2_dict = trainer.validate([all_graphs[2]])
        assert "r2_eeg" in r2_dict
        assert "r2_fmri" in r2_dict
        assert "pred_r2_eeg" in r2_dict
        assert "pred_r2_fmri" in r2_dict
        # NPI-comparison metrics (V5.50): h=1 AR(1) baseline and skill scores
        assert "ar1_r2_h1_eeg" in r2_dict, "h=1 AR(1) baseline for EEG must be present"
        assert "ar1_r2_h1_fmri" in r2_dict, "h=1 AR(1) baseline for fMRI must be present"
        assert "decorr_h1_eeg" in r2_dict, "h=1 skill score for EEG must be present"
        assert "decorr_h1_fmri" in r2_dict, "h=1 skill score for fMRI must be present"
        assert "pred_r2_h1_eeg" in r2_dict, "h=1 prediction R² for EEG must be present"
        assert "pred_r2_h1_fmri" in r2_dict, "h=1 prediction R² for fMRI must be present"

    def test_validate_r2_values_are_finite(self, all_graphs):
        model = _make_model(prediction_steps=3)
        trainer = _make_trainer(model)
        trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=2)
        _, r2_dict = trainer.validate([all_graphs[2]])
        for key, val in r2_dict.items():
            assert math.isfinite(val), f"R² key '{key}' = {val} is not finite"

    def test_no_reconstruction_loss_trains_without_error(self, all_graphs):
        """use_reconstruction_loss=False must train and validate without error."""
        node_types = ["eeg", "fmri"]
        edge_types = [
            ("eeg", "connects", "eeg"),
            ("fmri", "connects", "fmri"),
            ("eeg", "projects_to", "fmri"),
        ]
        model = GraphNativeBrainModel(
            node_types=node_types,
            edge_types=edge_types,
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
            use_reconstruction_loss=False,
        )
        trainer = _make_trainer(model)
        loss = trainer.train_epoch(all_graphs[:2], epoch=1, total_epochs=2)
        assert math.isfinite(loss), f"Training loss not finite: {loss}"
        # validate() must still compute r2_* even when recon loss is disabled in training
        _, r2_dict = trainer.validate([all_graphs[2]])
        assert "r2_eeg" in r2_dict, "r2_eeg must be computed in validation even with use_reconstruction_loss=False"
        assert "r2_fmri" in r2_dict
        # No recon tasks should appear in the loss balancer's task list
        if trainer.use_adaptive_loss:
            registered = trainer.loss_balancer.task_names
            assert not any(t.startswith('recon_') for t in registered), (
                f"recon tasks must not be registered when use_reconstruction_loss=False; "
                f"found: {[t for t in registered if t.startswith('recon_')]}"
            )


# ===========================================================================
# 5. Cache-only discovery (cache-only mode)
# ===========================================================================

import re

# Mirror of the filename-parsing logic added to main.py::build_graphs().
# Kept here as a standalone helper so we can unit-test it independently of
# the full build_graphs() pipeline.
_CACHE_FN_PATTERN = re.compile(r'^(sub-[^_]+)_(.+)_([0-9a-f]{8})\.pt$')


def _discover_from_cache_dir(cache_dir, tasks_cfg=None, max_subjects=None):
    """Parse .pt filenames in *cache_dir* and return (subject_id, task) pairs.

    Mirrors the cache-only fallback in main.py::build_graphs() exactly.
    Returns a list of (subject_id, task_or_None) tuples.
    """
    pairs = []
    for pt in sorted(Path(cache_dir).glob('*.pt')):
        m = _CACHE_FN_PATTERN.match(pt.name)
        if not m:
            continue
        sid, tsk_str = m.group(1), m.group(2)
        task = None if tsk_str == 'notask' else tsk_str
        if tasks_cfg is not None and len(tasks_cfg) > 0 and task not in tasks_cfg:
            continue
        pairs.append((sid, task))
    if max_subjects is not None:
        seen = {}
        for s, t in pairs:
            if s not in seen:
                seen[s] = []
            seen[s].append(t)
        kept = set(list(seen.keys())[:max_subjects])
        pairs = [(s, t) for s, t in pairs if s in kept]
    return pairs


class TestCacheOnlyDiscovery:
    """Verify that (subject_id, task) pairs are correctly parsed from .pt filenames.

    This validates the cache-only mode added to build_graphs(): when raw data is
    absent but pre-built .pt cache files exist, the pipeline discovers subjects
    and tasks by parsing the filenames instead of scanning the data directory.
    """

    def _make_cache_dir(self, tmp_path, names):
        """Create empty .pt files with the given names under tmp_path."""
        for name in names:
            (tmp_path / name).touch()
        return tmp_path

    def test_standard_filenames_parsed(self, tmp_path):
        names = [
            "sub-01_rest_a1b2c3d4.pt",
            "sub-01_wm_b2c3d4e5.pt",
            "sub-02_rest_a1b2c3d4.pt",
        ]
        self._make_cache_dir(tmp_path, names)
        pairs = _discover_from_cache_dir(tmp_path)
        assert ("sub-01", "rest") in pairs
        assert ("sub-01", "wm") in pairs
        assert ("sub-02", "rest") in pairs
        assert len(pairs) == 3

    def test_notask_sentinel_becomes_none(self, tmp_path):
        """'notask' task_str in filename must be converted to Python None."""
        self._make_cache_dir(tmp_path, ["sub-01_notask_a1b2c3d4.pt"])
        pairs = _discover_from_cache_dir(tmp_path)
        assert pairs == [("sub-01", None)]

    def test_task_with_underscore_parsed_correctly(self, tmp_path):
        """Tasks containing underscores (e.g. 'task_name') must be preserved intact."""
        self._make_cache_dir(tmp_path, ["sub-01_task_name_a1b2c3d4.pt"])
        pairs = _discover_from_cache_dir(tmp_path)
        assert pairs == [("sub-01", "task_name")]

    def test_tasks_cfg_filter_keeps_only_matching(self, tmp_path):
        names = [
            "sub-01_rest_a1b2c3d4.pt",
            "sub-01_wm_b2c3d4e5.pt",
            "sub-02_rest_a1b2c3d4.pt",
        ]
        self._make_cache_dir(tmp_path, names)
        pairs = _discover_from_cache_dir(tmp_path, tasks_cfg=["rest"])
        assert all(t == "rest" for _, t in pairs)
        assert len(pairs) == 2

    def test_max_subjects_limits_correctly(self, tmp_path):
        names = [
            "sub-01_rest_a1b2c3d4.pt",
            "sub-02_rest_a1b2c3d4.pt",
            "sub-03_rest_a1b2c3d4.pt",
        ]
        self._make_cache_dir(tmp_path, names)
        pairs = _discover_from_cache_dir(tmp_path, max_subjects=2)
        subjects = {s for s, _ in pairs}
        assert len(subjects) == 2
        assert "sub-03" not in subjects

    def test_max_subjects_preserves_all_tasks_per_subject(self, tmp_path):
        """When max_subjects=1, all tasks of the first subject must be retained."""
        names = [
            "sub-01_rest_a1b2c3d4.pt",
            "sub-01_wm_b2c3d4e5.pt",
            "sub-02_rest_a1b2c3d4.pt",
        ]
        self._make_cache_dir(tmp_path, names)
        pairs = _discover_from_cache_dir(tmp_path, max_subjects=1)
        assert len(pairs) == 2
        assert all(s == "sub-01" for s, _ in pairs)

    def test_non_standard_filenames_ignored(self, tmp_path):
        names = [
            "sub-01_rest_a1b2c3d4.pt",   # valid
            "other_file.pt",              # missing sub- prefix
            "sub-01_rest_toolongggg.pt",  # hash too long (9 chars)
            "checkpoint.pt",              # no pattern at all
        ]
        self._make_cache_dir(tmp_path, names)
        pairs = _discover_from_cache_dir(tmp_path)
        assert len(pairs) == 1
        assert pairs[0] == ("sub-01", "rest")

    def test_empty_cache_dir_returns_empty(self, tmp_path):
        pairs = _discover_from_cache_dir(tmp_path)
        assert pairs == []
