# FRANK: Reviewer Package

This repository contains the manuscripts and reproducibility code for two companion papers:

- **Paper 1**: *FRANK: A Brain-Inspired Modular Architecture for Extreme Length Generalization* ([`papers/Paper1_FRANK_Architecture.pdf`](papers/Paper1_FRANK_Architecture.pdf))
- **Paper 2**: *Emergent Specialization in Modular Recurrent Networks: Lesion Analysis of the FRANK Architecture* ([`papers/Paper2_Emergent_Specialization.pdf`](papers/Paper2_Emergent_Specialization.pdf))

Author: Izen Thornton (Independent Researcher, izen@cixate.com).

---

## What's in this package

```
frank-papers/
  README.md                  (this file)
  LICENSE                    (Research Use License; see below)
  requirements.txt           (just torch + numpy)
  paper_experiments.py       main experiment runner
  papers/
    Paper1_FRANK_Architecture.pdf
    Paper2_Emergent_Specialization.pdf
  frank/                     source package
    models/                  FRANK + 4 baseline classes (~500K params each)
    tasks/                   chain, copy, recall, sum
  scripts/
    aggregate_results.py     prints all paper tables from JSON output
  tests/
    test_integration.py      dataset, forward-pass, gradient sanity tests
```

`paper_experiments.py` produces the raw data for every table in both papers.

The four core baseline classes (`TransformerModel`, `GRUModel`, `ModularRecurrentModel`, `FrankModel`) live in `frank/models/`. The three derived variants used in the papers (Modular+Memory, RIMs, FRANK-NoLat) are defined inline in `paper_experiments.py`, alongside the training and eval loops, the chunked extreme-length evaluator, and the lesion routines.

---

## Quick start (for reviewers)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Smoke test (~5 minutes on a laptop CPU; verifies the pipeline runs)
python paper_experiments.py --phase 1 --quick --models gru --seeds 42

# 3. (Optional) confirm the package imports cleanly
python -m unittest tests.test_integration

# 4. Print paper tables from any results JSON you've produced
python scripts/aggregate_results.py
```

The smoke test trains a small GRU on all four tasks for 10 epochs each. The full 10-seed reproduction is much heavier; see *Compute requirements* below.

---

## Architecture summary

FRANK has five components, all active every timestep, with no routing:

| Component | Implementation | Brain analogue |
|---|---|---|
| **Brain** | 4 tau-gated recurrent modules (tau init = {5, 15, 30, 50}, learnable, clamped to [1, 100]) with lateral connections | cortex |
| **Active Memory** | 64 learned key-value slots, queried by previous brain state, injected into module 0 | hippocampus |
| **Reflex** | 3-layer feed-forward MLP, stateless | brainstem |
| **Anomaly Detector** | Small MLP outputting a scalar score; trains via gradient flow only (no direct output connection) | amygdala (dormant on these tasks) |
| **Veto** | `y = sigma(inh) * y_brain + (1 - sigma(inh)) * y_reflex`; bias initialized at -1.0 (FRANK starts ~73% reflex-dominated) | basal ganglia |

The recurrence is a tau-damped update without GRU/LSTM gates:

```
h_t = (1 - 1/tau) * h_{t-1} + (1/tau) * tanh(W_x x_t + W_h h_{t-1})
```

FRANK is the third stage of a development progression: Modular, then Modular+Memory, then FRANK. All seven models are matched at ~500K parameters.

| Model | Params | Notes |
|---|---|---|
| Transformer | 481K | 4 layers, 4 heads, d_model=128 |
| GRU | 500K | 2-layer, hidden_dim=214 |
| Modular | 502K | 4 tau-gated modules + laterals, MLP head |
| Modular+Memory | 512K | + active memory injection |
| RIMs | 500K | 4 GRU modules, soft top-2 input attention, multi-head comm |
| **FRANK** | 507K | + reflex + anomaly detector + veto |
| FRANK-NoLat | 507K | FRANK with laterals replaced by identity (ablation) |

---

## Reproducing the paper results

The runner has four phases. Re-running the same command resumes from the last completed unit (atomic JSON writes plus `progress.json`).

### Phase 1: train every (model, task, seed)
```bash
python paper_experiments.py --phase 1
```
Produces 7 models x 4 tasks x 10 seeds = 280 checkpoints in `paper1_checkpoints/`. All later phases load from these.

### Phase 2: extreme chain generalization (Paper 1, Tables 2 and 3)
```bash
python paper_experiments.py --phase 2
```
Streams chain sequences up to 100,000x training length (2,000,000 tokens) through FRANK, GRU, and FRANK-NoLat, chunked at 1,000 tokens. Per-sequence results are written incrementally so partial runs are not wasted.

### Phase 3: lesion studies (Paper 2, Tables 1, 2, 3, 4)
```bash
python paper_experiments.py --phase 3
```
Global lesion: random weight zeroing across the full network at {0, 10, 20, 30, 50}% damage, 10 patterns per level, every model, every task.

Targeted lesion: damage applied to a single FRANK component at a time at {0, 10, 20, 30, 50, 70}%, every task.

### Phase 4: standard generalization (Paper 1, Table 4)
```bash
python paper_experiments.py --phase 4
```
2x, 3x, 5x, 10x training-length evaluation across all models and all tasks.

### Run everything
```bash
python paper_experiments.py --phase all
```

### Splitting work across GPUs or machines
```bash
# GPU 0: half the seeds
python paper_experiments.py --seeds 42,123,456,789,1337 --gpu 0 \
    --results-dir paper1_results/gpu0

# GPU 1: the other half
python paper_experiments.py --seeds 2024,3141,4242,5555,6789 --gpu 1 \
    --results-dir paper1_results/gpu1
```

---

## Aggregating results into the paper tables

After phases finish, run:

```bash
python scripts/aggregate_results.py --results-dir paper1_results
```

This reads the JSON files Phases 1 through 4 produce and prints each table from both papers: extreme generalization per-seed, standard generalization mean across seeds, targeted lesion at 10%, global lesion vs. damage, and so on.

---

## Scope and a known omission

Everything in this folder maps directly to results reported in the two PDFs in `papers/`. The four phases of `paper_experiments.py` cover:

- Paper 1: Tables 2, 3, 4 (extreme generalization, scale-wise means, standard 2x to 10x).
- Paper 2: Tables 1, 2, 3, 4 (targeted-component lesion, degradation curves, specialization map, global lesion).

**Known omission: RAM-FRANK (Paper 1, Section 6).** The RAM-FRANK negative result (a FRANK variant with 16 memory slots and per-slot learnable write-gate biases initialized 0.0 to 2.0) is not bundled here. It lives on a separate ablation branch and is available on request; please email izen@cixate.com. RAM-FRANK is a negative-result ablation that supports the *minimalism-as-regularization* discussion. None of the headline numbers in either paper depend on it.

If you want to re-implement RAM-FRANK from the paper description, the only delta from FRANK is in the Active Memory: replace the single 64-slot memory in [`frank/models/frank.py`](frank/models/frank.py) with 16 slots, each carrying a learnable scalar write-gate bias initialized linearly across `[0.0, 2.0]`, and feed the slot's softmax-attended value through a sigmoid of that gate before injection into module 0.

---

## Compute requirements

- Phase 1: ~2 to 4 GPU-hours per (model, task, seed) on an RTX 3080 (~24 hours total per seed).
- Phase 2: dominated by the 100,000x sequences (~2M tokens each). ~30 to 60 min per FRANK seed at the largest scale.
- Phases 3 and 4: minutes per (model, seed) given existing checkpoints.

Published results: 10 seeds x ~24 hours = ~10 GPU-days, run in parallel on 10x RTX 3080s.

If you only want to verify the architecture and one or two seeds run end-to-end, the `--quick` flag with `--seeds 42` will produce the full pipeline output in well under an hour.

---

## Determinism and seeds

The 10 seeds used in the papers are fixed: `[42, 123, 456, 789, 1337, 2024, 3141, 4242, 5555, 6789]`.

`torch.manual_seed(seed)` and `np.random.seed(seed)` are set per training run. Per-sequence eval seeds in Phase 2 derive from `seed * 10000 + sequence_index` for full reproducibility of the extreme-length evaluation. No CUDA-deterministic algorithm flag is set, so single-token differences across hardware are possible. The trimodal distribution reported in Paper 1 (perfect, partial, chance) is robust across all hardware tested.

---

## Key headline results (for fast cross-checking)

### Paper 1: chain task at 100,000x training length (per-seed token accuracy)

| Seed | FRANK | GRU | FRANK-NoLat |
|---|---|---|---|
| 1337 | **100.0%** | 14.5% | 10.1% |
| 2024 | **100.0%** | 15.8% | **100.0%** |
| 3141 | **100.0%** | 27.3% | 9.9% |
| 42 | 91.8% | 16.4% | **100.0%** |
| 789 | 90.1% | 36.1% | 89.9% |
| 123 | 55.3% | 28.9% | 90.4% |
| 456 | 62.4% | **85.5%** | 10.2% |
| 4242 | 10.1% | 20.0% | 9.9% |
| 5555 | 9.9% | 16.1% | 10.1% |
| 6789 | 10.0% | 67.9% | 18.2% |
| **Mean** | **63.0%** | 32.8% | 44.9% |

### Paper 2: FRANK accuracy at 10% targeted component damage (seed 42)

| Component | Chain | Copy | Recall | Sum |
|---|---|---|---|---|
| Base (0%) | 100.0% | 95.5% | 100.0% | 96.7% |
| Brain 10% | 58.1% | 34.0% | 83.0% | 19.0% |
| Memory 10% | 79.2% | 77.8% | 99.6% | 13.0% |
| Reflex 10% | 96.9% | 95.5% | 100.0% | 20.5% |
| Anomaly 10% | 100.0% | 95.5% | 100.0% | 96.7% |

Each task has a different sensitivity profile; no two columns match. Specialization is emergent (no routing, no task labels, no component-specific loss).

---

## License and patent

Released under the **Research Use License** (see [LICENSE](LICENSE)): free for academic research, education, and reproducing the published results. Commercial use requires a separate license. The FRANK architecture is patent pending.

For commercial licensing inquiries: izen@cixate.com

---

## Citations

```bibtex
@article{thornton2026frank,
  title  = {FRANK: A Brain-Inspired Modular Architecture for Extreme Length Generalization},
  author = {Thornton, Izen},
  year   = {2026},
  note   = {Working Draft}
}

@article{thornton2026emergent,
  title  = {Emergent Specialization in Modular Recurrent Networks: Lesion Analysis of the FRANK Architecture},
  author = {Thornton, Izen},
  year   = {2026},
  note   = {Working Draft}
}
```
