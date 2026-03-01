# Test Graph Cache Fixtures

This directory contains small synthetic `.pt` files for use in automated tests.
They match the exact format produced by the TwinBrain training pipeline's graph
caching subsystem (`main.py → build_graphs → torch.save`).

## File naming convention

```
{subject_id}_{task}_{hash}.pt
```

In the training pipeline the 8-character hex hash is derived from the MD5 of
relevant config parameters (atlas, graph topology, etc.).  Test fixtures use the
suffix `testfixture` instead of a real hash so they are easy to identify.

## Cache file schema

Each `.pt` file is a **PyG `HeteroData`** object with:

| Attribute | Shape | dtype | Description |
|-----------|-------|-------|-------------|
| `graph['eeg'].x` | `[N_eeg, T_eeg, 1]` | float32 | z-scored EEG time series |
| `graph['eeg'].num_nodes` | scalar | int | Number of EEG electrodes |
| `graph['eeg'].sampling_rate` | scalar | float | Sampling rate in Hz (e.g. 250.0) |
| `graph['eeg'].pos` | `[N_eeg, 3]` | float32 | Electrode 3-D positions (mm) |
| `graph['fmri'].x` | `[N_fmri, T_fmri, 1]` | float32 | z-scored BOLD time series |
| `graph['fmri'].num_nodes` | scalar | int | Number of fMRI ROIs |
| `graph['fmri'].sampling_rate` | scalar | float | 1/TR in Hz (e.g. 0.5 for TR=2s) |
| `graph['fmri'].pos` | `[N_fmri, 3]` | float32 | ROI centroid MNI coords (mm) |
| `graph[('eeg','connects','eeg')].edge_index` | `[2, E_eeg]` | int64 | EEG connectivity graph |
| `graph[('eeg','connects','eeg')].edge_attr` | `[E_eeg, 1]` | float32 | \|Pearson r\| weights |
| `graph[('fmri','connects','fmri')].edge_index` | `[2, E_fmri]` | int64 | fMRI connectivity graph |
| `graph[('fmri','connects','fmri')].edge_attr` | `[E_fmri, 1]` | float32 | \|Pearson r\| weights |

> **Note:** The cross-modal edge `('eeg', 'projects_to', 'fmri')` is **not**
> stored in cache files; it is rebuilt dynamically at load time.

## Fixture inventory

| File | N_eeg | T_eeg | N_fmri | T_fmri | Notes |
|------|-------|-------|--------|--------|-------|
| `sub-test01_EOEC_testfixture.pt` | 16 | 50 | 10 | 25 | smallest fixture |
| `sub-test01_GRADON_testfixture.pt` | 19 | 60 | 12 | 30 | medium fixture |
| `sub-test02_EOEC_testfixture.pt` | 32 | 100 | 20 | 50 | larger fixture |

## Loading in tests

```python
from utils.helpers import load_subject_graph_from_cache

data = load_subject_graph_from_cache(
    "tests/fixtures/graph_cache/sub-test01_EOEC_testfixture.pt"
)
eeg  = data['eeg_timeseries']   # np.ndarray [N_eeg, T_eeg]
fmri = data['fmri_timeseries']  # np.ndarray [N_fmri, T_fmri]
```
