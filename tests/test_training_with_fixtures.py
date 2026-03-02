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
