"""
Analysis for the quantifier-reasoning experiment.

15-table framework (RQ1 - RQ6):
    1. task2_overall_accuracy        (model * prompt)
    2. task2_by_category             (simple vs multi)
    3. task2_by_type                 (negation vs non_negation)
    4. task2_by_label                (E/C/N recall — Neutral bias)
    5. task2_by_commonsense          (aligned vs conflict + diff)
    6. task1_accuracy                (aligned / conflict / overall)
    7. joint_analysis                (linked T1 <=> T2 pairs)
    8a. task2_cot_effect             (direct vs cot, overall diff)
    8b. task2_confusion_matrix       (per model * prompt)
    9. task2_cot_by_category
    10. task2_cot_by_type
    11. task2_cot_by_commonsense
    12. task2_by_complexity          (multi_quantifier Lv1/2/3)
    13. task2_by_category_type       (category * negation type)
    14. ambiguity_preference         (from ambiguity_*.csv: wide/narrow/both/neither/invalid)
    15. ambiguity_distribution       (from ambiguity_*.csv: raw E/C/N/Other per reading)
    
Usage:
    python analyze_results.py                     # latest results_*.csv (+ latest ambiguity_*.csv if present)
    python analyze_results.py path/to/file.csv    # specific results CSV
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd

RESULTS_DIR  = "results"
ANALYSIS_DIR = "analysis"

MODEL_ORDER  = ["bert_nli", "llama3_8b", "llama3_70b", "gpt4"]
PROMPT_ORDER = ["direct", "cot"]
NLI_LABELS   = ["Entailment", "Contradiction", "Neutral"]
PREFERENCE_ORDER = ["prefer_wide", "prefer_narrow", "both", "neither", "invalid"]


############
# helpers
############
def latest(pattern):
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def load_results(path):
    df = pd.read_csv(path)
    if df["correct"].dtype == object:
        df["correct"] = df["correct"].astype(str).str.lower() == "true"
    df["complexity"] = (
        df["complexity"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    )
    if "type" not in df.columns:
        df["type"] = ""
    df["type"] = df["type"].fillna("").astype(str)
    for col in ("linked_t1_id", "linked_t2_id"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df


def load_ambiguity(path):
    if not path:
        return None
    df = pd.read_csv(path)
    return df


# ##################################################
# Table 1 — Task 2 overall accuracy (model * prompt)
# ##################################################

def task2_overall_accuracy(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return None
    tbl = t2.groupby(["model", "prompt"])["correct"].mean().unstack("prompt")
    return _reorder(tbl, MODEL_ORDER, PROMPT_ORDER)


# #####################################
# Table 2 — Task 2 accuracy by category
# #####################################

def task2_by_category(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return None
    tbl = t2.groupby(["category", "model", "prompt"])["correct"].mean().unstack("prompt")
    return _reorder(tbl, None, PROMPT_ORDER)

def task2_by_type(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return None
    tbl = t2.groupby(["type", "model", "prompt"])["correct"].mean().unstack("prompt")
    return _reorder(tbl, None, PROMPT_ORDER)

# #######################################################
# Table 3 — Task 2 multi_quantifier by complexity
# #######################################################

def task2_by_complexity(df):
    t2 = df[(df["task"] == "task2") & (df["category"] == "multi_quantifier") & (df["complexity"] != "")]
    if t2.empty:
        return None
    tbl = t2.groupby(["complexity", "model", "prompt"])["correct"].mean().unstack("prompt")
    return _reorder(tbl, None, PROMPT_ORDER)


# ##################################################
# Table 4 — Task 2 recall per gold label (E / C / N)
# ##################################################

def task2_by_label(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return None
    tbl = t2.groupby(["gold", "model", "prompt"])["correct"].mean().unstack("prompt")
    return _reorder(tbl, None, PROMPT_ORDER)


# #########################################################
# Table 5 — Task 2 by commonsense (aligned vs conflict)
# #########################################################

def task2_by_commonsense(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return None
    tbl = (
        t2.groupby(["model", "prompt", "commonsense"])["correct"]
        .mean()
        .unstack("commonsense")
    )
    if "aligned" in tbl.columns and "conflict" in tbl.columns:
        tbl["diff (aligned-conflict)"] = tbl["aligned"] - tbl["conflict"]
    return tbl


# ############################################
# Table 6a — CoT effect: direct vs cot overall
# ############################################

def task2_cot_effect(df):
    t2 = df[(df["task"] == "task2") & (df["model"] != "bert_nli")]
    if t2.empty:
        return None
    tbl = t2.groupby(["model", "prompt"])["correct"].mean().unstack("prompt")
    if "direct" in tbl.columns and "cot" in tbl.columns:
        tbl["diff (cot-direct)"] = tbl["cot"] - tbl["direct"]
    return _reorder(tbl, MODEL_ORDER, None)


# ##########################
# Table 6b — CoT * category
# ##########################

def task2_cot_by_category(df):
    t2 = df[(df["task"] == "task2") & (df["model"] != "bert_nli")]
    if t2.empty:
        return None
    tbl = (
        t2.groupby(["model", "category", "prompt"])["correct"]
        .mean()
        .unstack("prompt")
    )
    if "direct" in tbl.columns and "cot" in tbl.columns:
        tbl["diff (cot-direct)"] = tbl["cot"] - tbl["direct"]
    return tbl


# ############################
# Table 6c — CoT * commonsense
# ############################

def task2_cot_by_commonsense(df):
    t2 = df[(df["task"] == "task2") & (df["model"] != "bert_nli")]
    if t2.empty:
        return None
    tbl = (
        t2.groupby(["model", "commonsense", "prompt"])["correct"]
        .mean()
        .unstack("prompt")
    )
    if "direct" in tbl.columns and "cot" in tbl.columns:
        tbl["diff (cot-direct)"] = tbl["cot"] - tbl["direct"]
    return tbl


# ########################################
# Table — CoT * type (negation vs non_negation)
# ########################################

def task2_cot_by_type(df):
    t2 = df[(df["task"] == "task2") & (df["model"] != "bert_nli")]
    if t2.empty:
        return None
    tbl = (
        t2.groupby(["model", "type", "prompt"])["correct"]
        .mean()
        .unstack("prompt")
    )
    if "direct" in tbl.columns and "cot" in tbl.columns:
        tbl["diff (cot-direct)"] = tbl["cot"] - tbl["direct"]
    return tbl


def task2_by_category_type(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return None
    tbl = (
        t2.groupby(["category", "type", "model", "prompt"])["correct"]
        .mean()
        .unstack("prompt")
    )
    return _reorder(tbl, None, PROMPT_ORDER)

# ######################################################
# Table 7 — Task 2 confusion matrix (per model * prompt)
# ######################################################

def task2_confusion_matrices(df):
    t2 = df[df["task"] == "task2"]
    if t2.empty:
        return []
    out = []
    for model in MODEL_ORDER:
        for prompt in PROMPT_ORDER:
            sub = t2[(t2["model"] == model) & (t2["prompt"] == prompt)]
            if sub.empty:
                continue
            mat = pd.crosstab(
                sub["gold"], sub["prediction"], dropna=False
            ).reindex(index=NLI_LABELS, columns=NLI_LABELS, fill_value=0)
            out.append((f"{model} / {prompt}", mat))
    return out


# ########################################################
# Table 8 — Task 1 accuracy (aligned / conflict / overall)
# ########################################################

def task1_accuracy(df):
    t1 = df[df["task"] == "task1"]
    if t1.empty:
        return None
    by_src = (
        t1.groupby(["model", "prompt", "source_type"])["correct"]
        .mean()
        .unstack("source_type")
    )
    overall = t1.groupby(["model", "prompt"])["correct"].mean().rename("overall")
    tbl = by_src.join(overall)
    return tbl


# ##################################################
# Table 9 — Joint analysis on linked T1 <=> T2 pairs
# ##################################################

def joint_analysis(df):
    t1 = df[(df["task"] == "task1") & (df["linked_t2_id"] != "")][
        ["id", "linked_t2_id", "model", "prompt", "correct"]
    ].rename(columns={"id": "t1_id", "correct": "t1_correct"})
    t2 = df[(df["task"] == "task2") & (df["linked_t1_id"] != "")][
        ["id", "linked_t1_id", "model", "prompt", "correct"]
    ].rename(columns={"id": "t2_id", "correct": "t2_correct"})

    if t1.empty or t2.empty:
        return None

    # Inner join drops BERT-on-task2 rows automatically (task1 has no BERT rows).
    merged = t1.merge(
        t2,
        left_on=["linked_t2_id", "model", "prompt"],
        right_on=["t2_id", "model", "prompt"],
        how="inner",
    )
    if merged.empty:
        return None

    merged["both_correct"] = merged["t1_correct"] & merged["t2_correct"]
    merged["t1_only"]      = merged["t1_correct"] & ~merged["t2_correct"]

    tbl = merged.groupby(["model", "prompt"]).agg(
        n_pairs=("t1_id", "count"),
        t1_acc=("t1_correct", "mean"),
        t2_acc=("t2_correct", "mean"),
        both_correct=("both_correct", "mean"),
        t1_only=("t1_only", "mean"),
    )
    return tbl


# ################################################################
# Table 11 — Ambiguity preference (pairing wide + narrow readings)
# ################################################################

def ambiguity_preference(amb_df):
    if amb_df is None or amb_df.empty or "reading" not in amb_df.columns:
        return None
    wide = amb_df[amb_df["reading"] == "wide"][
        ["id", "model", "prompt", "prediction"]
    ].rename(columns={"prediction": "pred_wide"})
    narrow = amb_df[amb_df["reading"] == "narrow"][
        ["id", "model", "prompt", "prediction"]
    ].rename(columns={"prediction": "pred_narrow"})
    pairs = wide.merge(narrow, on=["id", "model", "prompt"], how="inner")
    if pairs.empty:
        return None

    def classify(w, n):
        if w == "Entailment" and n == "Neutral":    return "prefer_wide"
        if w == "Neutral"    and n == "Entailment": return "prefer_narrow"
        if w == "Entailment" and n == "Entailment": return "both"
        if w == "Neutral"    and n == "Neutral":    return "neither"
        return "invalid"

    pairs["preference"] = [
        classify(w, n) for w, n in zip(pairs["pred_wide"], pairs["pred_narrow"])
    ]

    rows = []
    for model in MODEL_ORDER:
        for prompt in PROMPT_ORDER:
            sub = pairs[(pairs["model"] == model) & (pairs["prompt"] == prompt)]
            if sub.empty:
                continue
            counts = sub["preference"].value_counts()
            row = {"model": model, "prompt": prompt, "n": int(len(sub))}
            for pref in PREFERENCE_ORDER:
                row[pref] = int(counts.get(pref, 0))
            rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index(["model", "prompt"])


# #####################################################
# Table 12 — Ambiguity raw distribution (per reading)
# #####################################################

def ambiguity_distribution(amb_df):
    if amb_df is None or amb_df.empty or "reading" not in amb_df.columns:
        return None
    rows = []
    for reading in ("wide", "narrow"):
        for model in MODEL_ORDER:
            for prompt in PROMPT_ORDER:
                sub = amb_df[
                    (amb_df["reading"] == reading)
                    & (amb_df["model"] == model)
                    & (amb_df["prompt"] == prompt)
                ]
                if sub.empty:
                    continue
                counts = sub["prediction"].value_counts()
                row = {
                    "reading": reading,
                    "model": model,
                    "prompt": prompt,
                    "n": int(len(sub)),
                }
                for lbl in NLI_LABELS:
                    row[lbl] = int(counts.get(lbl, 0))
                other = int(counts.drop(labels=NLI_LABELS, errors="ignore").sum())
                row["Other"] = other
                rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index(["reading", "model", "prompt"])


# #########
# Rendering
# #########

def _reorder(tbl, index_order, column_order):
    if index_order:
        actual = tbl.index.get_level_values(0).unique().tolist()
        keep = [m for m in index_order if m in actual] + [m for m in actual if m not in index_order]
        if keep:
            tbl = tbl.reindex(keep, level=0)
    if column_order:
        keep = [c for c in column_order if c in tbl.columns]
        if keep:
            tbl = tbl[keep + [c for c in tbl.columns if c not in keep]]
    return tbl


def render(table, float_pct=True):
    if table is None:
        return "_(no data)_"
    fmt = "{:.1%}".format if float_pct else None
    return table.to_string(float_format=fmt)


def section(title, table, float_pct=True):
    return f"## {title}\n\n```\n{render(table, float_pct=float_pct)}\n```\n"


def section_matrices(title, matrices):
    if not matrices:
        return f"## {title}\n\n_(no data)_\n"
    parts = [f"## {title}\n"]
    for name, mat in matrices:
        parts.append(f"### {name}\n\n```\n{mat.to_string()}\n```\n")
    return "\n".join(parts)


# #####
# main
# #####

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="path to results CSV (default: latest in results/)")
    args = ap.parse_args()

    results_path = args.path or latest(f"{RESULTS_DIR}/results_*.csv")
    if not results_path:
        sys.exit(f"No results CSV found in {RESULTS_DIR}/")
    df = load_results(results_path)

    amb_path = latest(f"{RESULTS_DIR}/ambiguity_*.csv")
    amb_df = load_ambiguity(amb_path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    out_path = f"{ANALYSIS_DIR}/summary_{stamp}.md"

    header = [
        "# Results analysis",
        "",
        f"- Results CSV: `{results_path}`",
        f"- Ambiguity CSV: `{amb_path}`" if amb_path else "- Ambiguity CSV: _(none)_",
        f"- Rows: {len(df)}  (task1: {(df['task'] == 'task1').sum()}, task2: {(df['task'] == 'task2').sum()})",
        f"- Models: {sorted(df['model'].unique().tolist())}",
        "",
    ]

    blocks = [
        section("1. Task 2 — overall accuracy (model * prompt)", task2_overall_accuracy(df)),
        section("2. Task 2 — accuracy by category", task2_by_category(df)),
        section("3. Task 2 — accuracy by type (negation vs non_negation)", task2_by_type(df)),
        section("4. Task 2 — recall per gold label (E/C/N, Neutral bias)", task2_by_label(df)),
        section("5. Task 2 — aligned vs conflict", task2_by_commonsense(df)),
        section("6. Task 1 — accuracy (aligned / conflict / overall)", task1_accuracy(df)),
        section("7. Joint analysis — linked T1 <=> T2 pairs", joint_analysis(df)),
        section("8a. CoT effect — overall (direct vs cot)", task2_cot_effect(df)),
        section_matrices("8b. Task 2 — confusion matrices (gold * prediction)", task2_confusion_matrices(df)),
        section("9. CoT * category", task2_cot_by_category(df)),
        section("10. CoT * type", task2_cot_by_type(df)),
        section("11. CoT * commonsense", task2_cot_by_commonsense(df)),
        section("12. Task 2 multi_quantifier — accuracy by complexity", task2_by_complexity(df)),
        section("13. Task 2 — category * type", task2_by_category_type(df)),
        section("14. Ambiguity set — preference distribution (wide vs narrow scope)",
                ambiguity_preference(amb_df), float_pct=False),
        section("15. Ambiguity set — raw prediction distribution per reading",
                ambiguity_distribution(amb_df), float_pct=False),
    ]

    body = "\n".join(header) + "\n" + "\n".join(blocks)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)

    print(body)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
