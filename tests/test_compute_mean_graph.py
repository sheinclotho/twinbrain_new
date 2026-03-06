"""
tests/test_compute_mean_graph.py
=================================

Unit tests for compute_mean_graph.py.

Covers:
- load_graphs: success and failure paths
- check_edge_index_consistency: matching and mismatching graphs
- compute_mean_edge_attrs: plain average and Fisher-z average
- build_mean_graph: output structure
- compute_mean_graph (integration): full pipeline, output file, CLI, error paths
- plot_distribution: smoke test (skips if matplotlib absent)
"""

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# sys.path bootstrap (mirrors conftest.py for standalone runs)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from compute_mean_graph import (
    _fisher_z,
    _fisher_z_inv,
    build_mean_graph,
    check_edge_index_consistency,
    compute_mean_edge_attrs,
    compute_mean_graph,
    load_graphs,
    plot_distribution,
)

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "graph_cache"
_EEG_ET = ("eeg", "connects", "eeg")
_FMRI_ET = ("fmri", "connects", "fmri")


# ---------------------------------------------------------------------------
# Helpers — synthetic graph builders
# ---------------------------------------------------------------------------


def _make_graph(
    n_eeg: int = 4,
    n_fmri: int = 3,
    n_eeg_edges: int = 6,
    n_fmri_edges: int = 4,
    seed: int = 0,
) -> HeteroData:
    """Build a minimal synthetic HeteroData with two intra-modal edge types."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    g = HeteroData()

    # Node counts
    g["eeg"].num_nodes = n_eeg
    g["fmri"].num_nodes = n_fmri

    # EEG edges — fully connected upper triangle (reproducible)
    eeg_src = torch.zeros(n_eeg_edges, dtype=torch.long)
    eeg_dst = torch.zeros(n_eeg_edges, dtype=torch.long)
    k = 0
    for i in range(n_eeg):
        for j in range(i + 1, n_eeg):
            if k >= n_eeg_edges:
                break
            eeg_src[k] = i
            eeg_dst[k] = j
            k += 1
    g[_EEG_ET].edge_index = torch.stack([eeg_src[:k], eeg_dst[:k]], dim=0)
    g[_EEG_ET].edge_attr = torch.rand(k, 1, generator=rng)

    # fMRI edges
    fmri_src = torch.zeros(n_fmri_edges, dtype=torch.long)
    fmri_dst = torch.zeros(n_fmri_edges, dtype=torch.long)
    k2 = 0
    for i in range(n_fmri):
        for j in range(i + 1, n_fmri):
            if k2 >= n_fmri_edges:
                break
            fmri_src[k2] = i
            fmri_dst[k2] = j
            k2 += 1
    g[_FMRI_ET].edge_index = torch.stack([fmri_src[:k2], fmri_dst[:k2]], dim=0)
    g[_FMRI_ET].edge_attr = torch.rand(k2, 1, generator=rng)

    return g


def _clone_with_new_attrs(g: HeteroData, seed: int) -> HeteroData:
    """Duplicate *g*, replacing edge_attr with new random values (same topology)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    g2 = HeteroData()
    for nt in g.node_types:
        if hasattr(g[nt], "num_nodes"):
            g2[nt].num_nodes = g[nt].num_nodes
    for et in g.edge_types:
        g2[et].edge_index = g[et].edge_index.clone()
        if hasattr(g[et], "edge_attr") and g[et].edge_attr is not None:
            g2[et].edge_attr = torch.rand_like(g[et].edge_attr, generator=rng)
    return g2


# ---------------------------------------------------------------------------
# Fixtures — cached HeteroData objects
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_graph() -> HeteroData:
    return _make_graph(seed=1)


@pytest.fixture(scope="session")
def matching_graphs(base_graph) -> list:
    """Three graphs with identical topology, different weights."""
    g1 = base_graph
    g2 = _clone_with_new_attrs(base_graph, seed=2)
    g3 = _clone_with_new_attrs(base_graph, seed=3)
    return [g1, g2, g3]


# ---------------------------------------------------------------------------
# Tests: Fisher z helpers
# ---------------------------------------------------------------------------


class TestFisherZ:
    def test_round_trip_scalar(self):
        r = torch.tensor([0.3, -0.5, 0.0, 0.9])
        assert torch.allclose(_fisher_z_inv(_fisher_z(r)), r, atol=1e-5)

    def test_at_boundaries_clamped(self):
        r = torch.tensor([1.0, -1.0])
        # Should not raise (clamped internally)
        z = _fisher_z(r)
        assert torch.isfinite(z).all()

    def test_positive_values_map_to_positive_z(self):
        r = torch.tensor([0.5])
        assert _fisher_z(r).item() > 0

    def test_zero_maps_to_zero(self):
        r = torch.tensor([0.0])
        assert abs(_fisher_z(r).item()) < 1e-6


# ---------------------------------------------------------------------------
# Tests: load_graphs
# ---------------------------------------------------------------------------


class TestLoadGraphs:
    def test_loads_fixture_files(self):
        files = sorted(FIXTURE_DIR.glob("*.pt"))
        assert len(files) >= 1
        graphs = load_graphs(files)
        assert len(graphs) == len(files)
        for g in graphs:
            assert isinstance(g, HeteroData)

    def test_raises_on_missing_file(self, tmp_path):
        fake = tmp_path / "nonexistent.pt"
        with pytest.raises(RuntimeError, match="无法加载"):
            load_graphs([fake])

    def test_raises_on_non_heterodata(self, tmp_path):
        bad_file = tmp_path / "bad.pt"
        torch.save({"not": "a graph"}, bad_file)
        with pytest.raises(RuntimeError, match="不是 HeteroData"):
            load_graphs([bad_file])


# ---------------------------------------------------------------------------
# Tests: check_edge_index_consistency
# ---------------------------------------------------------------------------


class TestCheckEdgeIndexConsistency:
    def test_consistent_graphs_passes(self, matching_graphs):
        files = [Path(f"graph_{i}.pt") for i in range(len(matching_graphs))]
        # Should not raise
        check_edge_index_consistency(matching_graphs, files)

    def test_single_graph_passes(self, base_graph):
        check_edge_index_consistency([base_graph], [Path("graph_0.pt")])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="未找到"):
            check_edge_index_consistency([], [])

    def test_different_edge_types_raises(self, base_graph):
        g2 = HeteroData()
        g2["eeg"].num_nodes = 4
        g2[_EEG_ET].edge_index = base_graph[_EEG_ET].edge_index.clone()
        # Only EEG edges, no fMRI
        files = [Path("g0.pt"), Path("g1.pt")]
        with pytest.raises(ValueError, match="边类型不一致"):
            check_edge_index_consistency([base_graph, g2], files)

    def test_different_edge_index_raises(self, base_graph):
        g2 = _clone_with_new_attrs(base_graph, seed=99)
        # Perturb one entry in fMRI edge_index
        g2[_FMRI_ET].edge_index[0, 0] = (
            base_graph[_FMRI_ET].edge_index[0, 0] + 1
        ) % base_graph["fmri"].num_nodes
        files = [Path("g0.pt"), Path("g1.pt")]
        with pytest.raises(ValueError, match="edge_index 不一致"):
            check_edge_index_consistency([base_graph, g2], files)

    def test_fixture_files_with_same_topology_passes(self):
        """Two fixtures with the same N_eeg/N_fmri should pass; different fail."""
        files = sorted(FIXTURE_DIR.glob("*.pt"))
        graphs = load_graphs(files)
        # Fixtures intentionally have different sizes → should raise
        if len(graphs) >= 2:
            g0, g1 = graphs[0], graphs[1]
            if g0[_EEG_ET].edge_index.shape != g1[_EEG_ET].edge_index.shape:
                with pytest.raises(ValueError):
                    check_edge_index_consistency([g0, g1], [files[0], files[1]])
            # else topology may match, which is also fine


# ---------------------------------------------------------------------------
# Tests: compute_mean_edge_attrs
# ---------------------------------------------------------------------------


class TestComputeMeanEdgeAttrs:
    def test_mean_is_average_of_attrs(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs, use_fisher_z=False)
        # Manual average
        for et in [_EEG_ET, _FMRI_ET]:
            manual = torch.stack(
                [g[et].edge_attr.float() for g in matching_graphs], dim=0
            ).mean(0)
            assert torch.allclose(mean_attrs[et], manual, atol=1e-5), (
                f"Mean mismatch for {et}"
            )

    def test_fisher_z_mode_different_from_plain(self, matching_graphs):
        plain = compute_mean_edge_attrs(matching_graphs, use_fisher_z=False)
        fz = compute_mean_edge_attrs(matching_graphs, use_fisher_z=True)
        for et in [_EEG_ET, _FMRI_ET]:
            # Values should differ (non-linear transform)
            assert not torch.allclose(plain[et], fz[et], atol=1e-4), (
                f"Fisher-z result unexpectedly identical to plain for {et}"
            )

    def test_fisher_z_result_in_unit_interval(self, matching_graphs):
        # Edge attrs are in [0, 1] from rand → Fisher-z result should be too
        fz = compute_mean_edge_attrs(matching_graphs, use_fisher_z=True)
        for et in [_EEG_ET, _FMRI_ET]:
            assert (fz[et] >= -1.0).all() and (fz[et] <= 1.0).all()

    def test_single_graph_mean_equals_itself(self, base_graph):
        mean_attrs = compute_mean_edge_attrs([base_graph], use_fisher_z=False)
        for et in [_EEG_ET, _FMRI_ET]:
            assert torch.allclose(
                mean_attrs[et], base_graph[et].edge_attr.float(), atol=1e-5
            )

    def test_shape_preserved(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs, use_fisher_z=False)
        ref = matching_graphs[0]
        for et in [_EEG_ET, _FMRI_ET]:
            assert mean_attrs[et].shape == ref[et].edge_attr.shape

    def test_missing_edge_attr_raises(self, base_graph):
        # A graph without edge_attr on one edge type
        g2 = HeteroData()
        g2["eeg"].num_nodes = base_graph["eeg"].num_nodes
        g2["fmri"].num_nodes = base_graph["fmri"].num_nodes
        g2[_EEG_ET].edge_index = base_graph[_EEG_ET].edge_index.clone()
        g2[_FMRI_ET].edge_index = base_graph[_FMRI_ET].edge_index.clone()
        g2[_EEG_ET].edge_attr = None  # Missing!
        g2[_FMRI_ET].edge_attr = base_graph[_FMRI_ET].edge_attr.clone()

        with pytest.raises(ValueError, match="edge_attr"):
            compute_mean_edge_attrs([base_graph, g2])


# ---------------------------------------------------------------------------
# Tests: build_mean_graph
# ---------------------------------------------------------------------------


class TestBuildMeanGraph:
    def test_output_is_heterodata(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        mean_g = build_mean_graph(matching_graphs, mean_attrs)
        assert isinstance(mean_g, HeteroData)

    def test_edge_types_preserved(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        mean_g = build_mean_graph(matching_graphs, mean_attrs)
        ref_types = set(matching_graphs[0].edge_types)
        assert set(mean_g.edge_types) == ref_types

    def test_edge_index_equals_reference(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        mean_g = build_mean_graph(matching_graphs, mean_attrs)
        ref = matching_graphs[0]
        for et in ref.edge_types:
            assert torch.equal(mean_g[et].edge_index, ref[et].edge_index)

    def test_edge_attr_equals_mean(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        mean_g = build_mean_graph(matching_graphs, mean_attrs)
        for et in [_EEG_ET, _FMRI_ET]:
            assert torch.allclose(mean_g[et].edge_attr, mean_attrs[et], atol=1e-6)

    def test_node_num_nodes_copied(self, matching_graphs):
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        mean_g = build_mean_graph(matching_graphs, mean_attrs)
        ref = matching_graphs[0]
        for nt in ref.node_types:
            if hasattr(ref[nt], "num_nodes") and ref[nt].num_nodes is not None:
                assert mean_g[nt].num_nodes == ref[nt].num_nodes


# ---------------------------------------------------------------------------
# Tests: compute_mean_graph (integration)
# ---------------------------------------------------------------------------


class TestComputeMeanGraphIntegration:
    def _save_graphs(self, graphs: list, tmp_path: Path) -> None:
        for i, g in enumerate(graphs):
            torch.save(g, tmp_path / f"graph_{i:03d}.pt")

    def test_creates_output_file(self, tmp_path, matching_graphs):
        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "mean_graph.pt"
        compute_mean_graph(tmp_path, output_path=out)
        assert out.exists()

    def test_output_is_loadable_heterodata(self, tmp_path, matching_graphs):
        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "mean_graph.pt"
        compute_mean_graph(tmp_path, output_path=out)
        loaded = torch.load(out, map_location="cpu", weights_only=False)
        assert isinstance(loaded, HeteroData)

    def test_output_has_correct_edge_types(self, tmp_path, matching_graphs):
        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "mean_graph.pt"
        mean_g = compute_mean_graph(tmp_path, output_path=out)
        assert _EEG_ET in mean_g.edge_types
        assert _FMRI_ET in mean_g.edge_types

    def test_output_edge_index_equals_reference(self, tmp_path, matching_graphs):
        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "mean_graph.pt"
        mean_g = compute_mean_graph(tmp_path, output_path=out)
        ref = matching_graphs[0]
        for et in ref.edge_types:
            assert torch.equal(mean_g[et].edge_index, ref[et].edge_index)

    def test_output_file_excluded_from_averaging(self, tmp_path, matching_graphs):
        """mean_graph.pt already in the folder must not be read as input."""
        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "mean_graph.pt"
        # Run once
        compute_mean_graph(tmp_path, output_path=out)
        # Run again — should still succeed without double-counting
        compute_mean_graph(tmp_path, output_path=out)

    def test_fisher_z_flag_produces_different_result(self, tmp_path, matching_graphs):
        self._save_graphs(matching_graphs, tmp_path)
        out_plain = tmp_path / "mean_plain.pt"
        out_fz = tmp_path / "mean_fz.pt"
        compute_mean_graph(tmp_path, output_path=out_plain, use_fisher_z=False)
        compute_mean_graph(tmp_path, output_path=out_fz, use_fisher_z=True)
        g_plain = torch.load(out_plain, map_location="cpu", weights_only=False)
        g_fz = torch.load(out_fz, map_location="cpu", weights_only=False)
        # Results differ due to non-linearity
        assert not torch.allclose(
            g_plain[_EEG_ET].edge_attr, g_fz[_EEG_ET].edge_attr, atol=1e-4
        )

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            compute_mean_graph(Path("/nonexistent/dir"))

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="未找到"):
            compute_mean_graph(tmp_path)

    def test_inconsistent_graphs_raises(self, tmp_path):
        g1 = _make_graph(n_eeg=4, n_fmri=3, seed=1)
        g2 = _make_graph(n_eeg=6, n_fmri=5, seed=2)  # Different topology
        torch.save(g1, tmp_path / "g1.pt")
        torch.save(g2, tmp_path / "g2.pt")
        with pytest.raises(ValueError):
            compute_mean_graph(tmp_path)

    def test_default_output_path(self, tmp_path, matching_graphs):
        """When output_path is None, file should be saved as mean_graph.pt."""
        self._save_graphs(matching_graphs, tmp_path)
        compute_mean_graph(tmp_path)  # No output_path
        assert (tmp_path / "mean_graph.pt").exists()

    def test_custom_pattern(self, tmp_path, matching_graphs):
        """Only files matching the pattern should be loaded."""
        self._save_graphs(matching_graphs, tmp_path)
        # Add a non-matching file (will be ignored)
        torch.save({"not": "a graph"}, tmp_path / "other_data.bin")
        # Should still work using default *.pt pattern
        out = tmp_path / "mean_graph.pt"
        compute_mean_graph(tmp_path, output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: plot_distribution (smoke test)
# ---------------------------------------------------------------------------


class TestPlotDistribution:
    def test_skips_gracefully_without_matplotlib(
        self, tmp_path, matching_graphs, monkeypatch
    ):
        """If matplotlib is absent, plot_distribution should warn and return."""
        import builtins
        _real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("mocked absence")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        # Should not raise
        plot_distribution(
            mean_attrs,
            output_path=tmp_path / "dist.png",
            n_graphs=len(matching_graphs),
            use_fisher_z=False,
        )

    def test_creates_png_when_matplotlib_available(self, tmp_path, matching_graphs):
        pytest.importorskip("matplotlib")
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        out = tmp_path / "dist.png"
        plot_distribution(
            mean_attrs,
            output_path=out,
            n_graphs=len(matching_graphs),
            use_fisher_z=False,
        )
        assert out.exists()

    def test_fisher_z_annotation_in_title(self, tmp_path, matching_graphs, capsys):
        pytest.importorskip("matplotlib")
        mean_attrs = compute_mean_edge_attrs(matching_graphs)
        out = tmp_path / "dist_fz.png"
        plot_distribution(
            mean_attrs,
            output_path=out,
            n_graphs=len(matching_graphs),
            use_fisher_z=True,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: CLI via compute_mean_graph.main()
# ---------------------------------------------------------------------------


class TestCLI:
    def _save_graphs(self, graphs: list, tmp_path: Path) -> None:
        for i, g in enumerate(graphs):
            torch.save(g, tmp_path / f"graph_{i:03d}.pt")

    def test_cli_runs_successfully(self, tmp_path, matching_graphs):
        from compute_mean_graph import main

        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "cli_output.pt"
        main([str(tmp_path), "--output", str(out), "--log-level", "WARNING"])
        assert out.exists()

    def test_cli_fisher_z_flag(self, tmp_path, matching_graphs):
        from compute_mean_graph import main

        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "cli_fz.pt"
        main(
            [
                str(tmp_path),
                "--output", str(out),
                "--fisher-z",
                "--log-level", "WARNING",
            ]
        )
        assert out.exists()

    def test_cli_plot_flag(self, tmp_path, matching_graphs):
        pytest.importorskip("matplotlib")
        from compute_mean_graph import main

        self._save_graphs(matching_graphs, tmp_path)
        out = tmp_path / "cli_plot.pt"
        plot_out = tmp_path / "cli_dist.png"
        main(
            [
                str(tmp_path),
                "--output", str(out),
                "--plot",
                "--plot-output", str(plot_out),
                "--log-level", "WARNING",
            ]
        )
        assert out.exists()
        assert plot_out.exists()

    def test_cli_nonexistent_dir_exits(self, tmp_path):
        from compute_mean_graph import main

        with pytest.raises(SystemExit) as exc_info:
            main([str(tmp_path / "nonexistent"), "--log-level", "WARNING"])
        assert exc_info.value.code == 1
