#!/usr/bin/env python3
"""
Aggregate the JSON output of paper_experiments.py into the tables that
appear in Paper 1 and Paper 2.

Usage:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --results-dir paper1_results
  python scripts/aggregate_results.py --results-dir my_run --output tables.txt

The script prints whatever it finds; missing data is shown as "--".
Nothing here trains or evaluates anything; it only reads JSON.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 4242, 5555, 6789]
EXTREME_SCALES = [10, 100, 1000, 10000, 30000, 100000]
STANDARD_SCALES = [2, 3, 5, 10]
GLOBAL_DAMAGE = [0, 10, 20, 30, 50]
TARGETED_DAMAGE = [0, 10, 20, 30, 50, 70]
TASKS = ["chain", "copy", "recall", "sum"]
COMPONENTS = ["brain", "memory", "reflex", "anomaly_detector"]
COMPONENT_DISPLAY = {
    "brain": "Brain",
    "memory": "Memory",
    "reflex": "Reflex",
    "anomaly_detector": "Anomaly",
}
ALL_MODELS = [
    "transformer",
    "gru",
    "modular",
    "modular_memory",
    "rims",
    "frank",
    "frank_no_laterals",
]
MODEL_DISPLAY = {
    "transformer": "Transformer",
    "gru": "GRU",
    "modular": "Modular",
    "modular_memory": "Mod+Mem",
    "rims": "RIMs",
    "frank": "FRANK",
    "frank_no_laterals": "FRANK-NL",
}


def load(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fmt_pct(x):
    if x is None:
        return "  --  "
    return f"{x * 100:5.1f}%"


def header(title):
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}"


def subheader(title):
    return f"\n--- {title} ---"


# ---------------------------------------------------------------------------
# Paper 1, Table 2: Extreme chain (100,000x), per seed
# ---------------------------------------------------------------------------
def table_paper1_table2(data, out):
    if not data:
        return
    out.append(header("Paper 1, Table 2: Chain task accuracy at 100,000x (per seed)"))
    cols = ["frank", "gru", "frank_no_laterals"]
    out.append(f"{'Seed':>6}  " + "  ".join(f"{MODEL_DISPLAY[c]:>10}" for c in cols))
    out.append("-" * 50)
    for seed in SEEDS:
        row = [f"{seed:>6}"]
        for c in cols:
            val = data.get(c, {}).get(str(seed), {}).get("100000x")
            row.append(f"{fmt_pct(val):>10}")
        out.append("  ".join(row))
    out.append("-" * 50)
    means = []
    for c in cols:
        vals = [
            data.get(c, {}).get(str(s), {}).get("100000x")
            for s in SEEDS
            if data.get(c, {}).get(str(s), {}).get("100000x") is not None
        ]
        means.append(mean(vals) if vals else None)
    out.append(f"{'Mean':>6}  " + "  ".join(f"{fmt_pct(m):>10}" for m in means))


# ---------------------------------------------------------------------------
# Paper 1, Table 3: Mean chain accuracy across scales (10 seeds)
# ---------------------------------------------------------------------------
def table_paper1_table3(data, out):
    if not data:
        return
    out.append(header("Paper 1, Table 3: Mean chain accuracy across scales (10 seeds)"))
    cols = ["frank", "gru", "frank_no_laterals"]
    out.append(
        f"{'Scale':>10}  " + "  ".join(f"{MODEL_DISPLAY[c]:>10}" for c in cols)
        + f"  {'>=90% (FRANK)':>14}"
    )
    out.append("-" * 70)
    for scale in EXTREME_SCALES:
        scale_key = f"{scale}x"
        row = [f"{scale:>10,}x"]
        for c in cols:
            vals = [
                data.get(c, {}).get(str(s), {}).get(scale_key)
                for s in SEEDS
                if data.get(c, {}).get(str(s), {}).get(scale_key) is not None
            ]
            row.append(f"{fmt_pct(mean(vals) if vals else None):>10}")
        # >=90% count for FRANK
        frank_seed_vals = [
            data.get("frank", {}).get(str(s), {}).get(scale_key) for s in SEEDS
        ]
        n_seeds_with = sum(v is not None for v in frank_seed_vals)
        n_ge90 = sum(1 for v in frank_seed_vals if v is not None and v >= 0.90)
        row.append(f"{n_ge90}/{n_seeds_with:>2}".rjust(14))
        out.append("  ".join(row))


# ---------------------------------------------------------------------------
# Paper 1, Table 4: Standard generalization (2x to 10x), all models, all tasks
# Source: phase4_standard_gen_{task}.json AND phase1_training.json (for 1x baseline)
# ---------------------------------------------------------------------------
def table_paper1_table4(phase4_by_task, out):
    out.append(
        header("Paper 1, Table 4: Standard generalization, mean token accuracy (10 seeds)")
    )
    for task in TASKS:
        data = phase4_by_task.get(task)
        if not data:
            continue
        out.append(subheader(f"Task: {task.upper()}"))
        out.append(
            f"{'Model':<14}" + "".join(f"  {s}x".rjust(8) for s in STANDARD_SCALES)
        )
        out.append("-" * (14 + 8 * len(STANDARD_SCALES)))
        for model in ALL_MODELS:
            row = f"{MODEL_DISPLAY[model]:<14}"
            for scale in STANDARD_SCALES:
                key = f"{scale}x"
                vals = [
                    data.get(model, {}).get(str(s), {}).get(key)
                    for s in SEEDS
                    if data.get(model, {}).get(str(s), {}).get(key) is not None
                ]
                row += f"  {fmt_pct(mean(vals) if vals else None):>6}"
            out.append(row)


# ---------------------------------------------------------------------------
# Paper 2, Table 1: FRANK accuracy at 10% targeted damage, all tasks (seed 42)
# ---------------------------------------------------------------------------
def table_paper2_table1(targeted_by_task, out):
    if not any(targeted_by_task.values()):
        return
    out.append(header("Paper 2, Table 1: FRANK accuracy at 10% targeted damage (seed 42)"))
    out.append(f"{'Component':<14}" + "".join(f"{t.capitalize():>9}" for t in TASKS))
    out.append("-" * (14 + 9 * len(TASKS)))

    # Base row (0% damage on any component is identical -> use brain 0%)
    base_row = f"{'Base (0%)':<14}"
    for task in TASKS:
        data = targeted_by_task.get(task)
        val = (
            data.get("42", {}).get("brain", {}).get("0", {}).get("mean")
            if data
            else None
        )
        base_row += f"{fmt_pct(val):>9}"
    out.append(base_row)

    # Per-component 10% damage rows
    for component in COMPONENTS:
        row = f"{COMPONENT_DISPLAY[component] + ' 10%':<14}"
        for task in TASKS:
            data = targeted_by_task.get(task)
            val = (
                data.get("42", {}).get(component, {}).get("10", {}).get("mean")
                if data
                else None
            )
            row += f"{fmt_pct(val):>9}"
        out.append(row)


# ---------------------------------------------------------------------------
# Paper 2, Table 2: Chain task targeted lesion degradation (seed 42)
# ---------------------------------------------------------------------------
def table_paper2_table2(targeted_by_task, out):
    data = targeted_by_task.get("chain")
    if not data:
        return
    out.append(header("Paper 2, Table 2: Chain task targeted lesion degradation (seed 42)"))
    out.append(
        f"{'Component':<14}"
        + "".join(f"{str(d) + '%':>8}" for d in TARGETED_DAMAGE)
    )
    out.append("-" * (14 + 8 * len(TARGETED_DAMAGE)))
    for component in COMPONENTS:
        row = f"{COMPONENT_DISPLAY[component]:<14}"
        for damage in TARGETED_DAMAGE:
            val = (
                data.get("42", {}).get(component, {}).get(str(damage), {}).get("mean")
            )
            row += f"{fmt_pct(val):>8}"
        out.append(row)


# ---------------------------------------------------------------------------
# Paper 2, Table 3: Four-task specialization map (qualitative, derived from
# Table 1 magnitudes)
# ---------------------------------------------------------------------------
def table_paper2_table3(targeted_by_task, out):
    if not any(targeted_by_task.values()):
        return
    out.append(header("Paper 2, Table 3: Emergent specialization map"))
    out.append(
        "(category derived from base->10% drop; CRITICAL <50%, Essential 50-85%, "
        "Supporting 85-95%, Graceful >95% with non-zero drop, Dormant = no change)"
    )
    out.append(f"{'Component':<14}" + "".join(f"{t.capitalize():>14}" for t in TASKS))
    out.append("-" * (14 + 14 * len(TASKS)))

    def categorize(base, dmg):
        if base is None or dmg is None:
            return "  --  "
        # near-equal -> dormant
        if abs(base - dmg) < 0.005:
            return "Dormant"
        ratio = dmg / max(base, 1e-9)
        if ratio < 0.50:
            return "CRITICAL"
        if ratio < 0.85:
            return "Essential"
        if ratio < 0.95:
            return "Supporting"
        return "Graceful"

    for component in COMPONENTS:
        row = f"{COMPONENT_DISPLAY[component]:<14}"
        for task in TASKS:
            data = targeted_by_task.get(task)
            if data:
                base = (
                    data.get("42", {})
                    .get(component, {})
                    .get("0", {})
                    .get("mean")
                )
                dmg = (
                    data.get("42", {})
                    .get(component, {})
                    .get("10", {})
                    .get("mean")
                )
                cat = categorize(base, dmg)
            else:
                cat = "  --  "
            row += f"{cat:>14}"
        out.append(row)


# ---------------------------------------------------------------------------
# Paper 2, Table 4: Global lesion (chain task), seed 42
# ---------------------------------------------------------------------------
def table_paper2_table4(global_by_task, out):
    data = global_by_task.get("chain")
    if not data:
        return
    out.append(header("Paper 2, Table 4: Chain task global lesion (seed 42)"))
    out.append(
        f"{'Model':<14}" + "".join(f"{str(d) + '%':>8}" for d in GLOBAL_DAMAGE)
    )
    out.append("-" * (14 + 8 * len(GLOBAL_DAMAGE)))
    # Sort by 10% damage retention, descending
    rows = []
    for model in ALL_MODELS:
        seed_data = data.get(model, {}).get("42", {})
        if not seed_data:
            continue
        vals = {
            d: seed_data.get(str(d), {}).get("mean") for d in GLOBAL_DAMAGE
        }
        rows.append((model, vals))
    rows.sort(key=lambda r: -(r[1].get(10) or 0))
    for model, vals in rows:
        row = f"{MODEL_DISPLAY[model]:<14}"
        for d in GLOBAL_DAMAGE:
            row += f"{fmt_pct(vals.get(d)):>8}"
        out.append(row)


# ---------------------------------------------------------------------------
# Phase 1 sanity table: in-distribution test accuracy by (model, task)
# ---------------------------------------------------------------------------
def table_phase1_sanity(phase1, out):
    if not phase1:
        return
    out.append(
        header("Phase 1: in-distribution test token accuracy, mean across seeds")
    )
    out.append(f"{'Model':<14}" + "".join(f"  {t.capitalize():>9}" for t in TASKS))
    out.append("-" * (14 + 11 * len(TASKS)))
    for model in ALL_MODELS:
        row = f"{MODEL_DISPLAY[model]:<14}"
        for task in TASKS:
            vals = []
            for seed in SEEDS:
                key = f"{task}_{model}_seed{seed}"
                rec = phase1.get(key)
                if rec and "test" in rec:
                    v = rec["test"].get("token_accuracy")
                    if v is not None:
                        vals.append(v)
            row += f"  {fmt_pct(mean(vals) if vals else None):>9}"
        out.append(row)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate paper_experiments.py output into Paper 1/2 tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("paper1_results"),
        help="Directory containing the JSON output of paper_experiments.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write tables to (in addition to stdout).",
    )
    args = parser.parse_args()

    rd = args.results_dir
    if not rd.exists():
        print(f"Results directory not found: {rd}", file=sys.stderr)
        print(
            "Run paper_experiments.py first, or pass --results-dir to point at "
            "your output.",
            file=sys.stderr,
        )
        sys.exit(1)

    phase1 = load(rd / "phase1_training.json")
    phase2 = load(rd / "phase2_extreme_gen.json")
    phase4_by_task = {t: load(rd / f"phase4_standard_gen_{t}.json") for t in TASKS}
    targeted_by_task = {t: load(rd / f"phase3_targeted_lesion_{t}.json") for t in TASKS}
    global_by_task = {t: load(rd / f"phase3_global_lesion_{t}.json") for t in TASKS}

    out = []
    out.append(f"Aggregating results from: {rd.resolve()}")

    table_phase1_sanity(phase1, out)
    table_paper1_table2(phase2, out)
    table_paper1_table3(phase2, out)
    table_paper1_table4(phase4_by_task, out)
    table_paper2_table1(targeted_by_task, out)
    table_paper2_table2(targeted_by_task, out)
    table_paper2_table3(targeted_by_task, out)
    table_paper2_table4(global_by_task, out)

    text = "\n".join(out) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)
        print(f"\nWrote tables to {args.output}")


if __name__ == "__main__":
    main()
