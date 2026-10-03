"""
Main experiment: Task 1 (Commonsense Truth) + Task 2 (Quantifier NLI) with accuracy.

Models:
  - BERT NLI   (local, Task 2 only — no prompt interface)
  - Llama-3 8B / 70B (OpenRouter)
  - GPT-4.1    (OpenAI)

Prompts: Zero-shot Direct + Zero-shot CoT (BERT: direct only).
"""

import csv
import glob
import json
import os
from datetime import datetime

from utils.models import (
    CLOUD_MODELS,
    call_bert_nli,
    load_bert_nli,
    run_cloud_jobs_parallel,
)
from utils.prompts import (
    parse_nli_label,
    parse_true_false,
    t1_prompt_cot,
    t1_prompt_direct,
    t2_prompt_cot,
    t2_prompt_direct,
)
from utils.display import (
    print_accuracy_summary,
    print_prediction,
    print_sample_header,
)

DATA_DIR    = "data"
TASK1_DIR = f"{DATA_DIR}/task1_updated"
TASK2_DIR = f"{DATA_DIR}/task2_updated"
RESULTS_DIR = "results"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_items(dir_path):
    """Load and concatenate every JSON array in dir_path/*.json (sorted by filename)."""
    items = []
    for path in sorted(glob.glob(f"{dir_path}/*.json")):
        with open(path) as f:
            items.extend(json.load(f))
    return items

FIELDNAMES = [
    "task", "id", "source_type", "category", "type", "commonsense",
    "complexity", "gold", "model", "prompt",
    "raw_output", "prediction", "correct",
    "linked_t1_id", "linked_t2_id",
]


def run_task1():
    print("\n" + "═" * 56)
    print("TASK 1: Commonsense Truth Judgment")
    print("═" * 56 + "\n")

    data = load_items(TASK1_DIR)

    results = []

    for ex in data:
        pid    = ex["id"]
        stmt   = ex["statement"]
        gold   = ex["gold_truth_label"]
        stype  = ex["source_type"]
        link_t2 = ex.get("linked_t2_id") or ""

        print_sample_header(pid, stype)
        print(f"  Statement: {stmt}")
        print(f"  Gold:      {gold}")

        prompts = {
            "direct": t1_prompt_direct(stmt),
            "cot":    t1_prompt_cot(stmt),
        }
        raw_by_key = run_cloud_jobs_parallel(prompts)

        for ptype in ["direct", "cot"]:
            for model_name in CLOUD_MODELS:
                raw     = raw_by_key[(ptype, model_name)]
                pred    = parse_true_false(raw)
                correct = (pred == gold)
                print_prediction(model_name, ptype, pred, correct)
                results.append({
                    "task":          "task1",
                    "id":            pid,
                    "source_type":   stype,
                    "gold":          gold,
                    "model":         model_name,
                    "prompt":        ptype,
                    "raw_output":    raw,
                    "prediction":    pred,
                    "correct":       correct,
                    # task2-only fields (blank for task1)
                    "category":      "",
                    "type": 		 "",
                    "commonsense":   "",
                    "complexity":    "",
                    "linked_t1_id":  "",
                    "linked_t2_id":  link_t2,
                })
        print()

    return results


def run_task2(bert_pipe):
    print("\n" + "═" * 56)
    print("TASK 2: Quantifier Reasoning (NLI)")
    print("═" * 56 + "\n")

    data = load_items(TASK2_DIR)

    results = []

    for ex in data:
        pid     = ex["id"]
        prem    = ex["premise"]
        hyp     = ex["hypothesis"]
        gold    = ex["gold_label"]
        cat     = ex["category"]
        etype = ex.get("negation_attribute") or ""
        cs      = ex["commonsense_condition"]
        comp    = ex.get("complexity_level") or ""
        link_t1 = ex.get("linked_t1_id") or ""
        
        print_sample_header(pid, cat, f"cs={cs}")
        print(f"  Premise:    {prem}")
        print(f"  Hypothesis: {hyp}")
        print(f"  Gold:       {gold}")

        # BERT: direct only, local inference
        bert_pred    = call_bert_nli(bert_pipe, prem, hyp)
        bert_correct = (bert_pred == gold)
        print_prediction("bert_nli", "direct", bert_pred, bert_correct)
        results.append({
            "task":          "task2",
            "id":            pid,
            "source_type":   "",
            "gold":          gold,
            "model":         "bert_nli",
            "prompt":        "direct",
            "raw_output":    bert_pred,
            "prediction":    bert_pred,
            "correct":       bert_correct,
            "category":      cat,
            "type":			 etype,
            "commonsense":   cs,
            "complexity":    comp,
            "linked_t1_id":  link_t1,
            "linked_t2_id":  "",
        })

        # Cloud models in parallel: 3 models * 2 prompt types
        prompts = {
            "direct": t2_prompt_direct(prem, hyp),
            "cot":    t2_prompt_cot(prem, hyp),
        }
        raw_by_key = run_cloud_jobs_parallel(prompts)

        for ptype in ["direct", "cot"]:
            for model_name in CLOUD_MODELS:
                raw     = raw_by_key[(ptype, model_name)]
                pred    = parse_nli_label(raw)
                correct = (pred == gold)
                print_prediction(model_name, ptype, pred, correct)
                results.append({
                    "task":          "task2",
                    "id":            pid,
                    "source_type":   "",
                    "gold":          gold,
                    "model":         model_name,
                    "prompt":        ptype,
                    "raw_output":    raw,
                    "prediction":    pred,
                    "correct":       correct,
                    "category":      cat,
                    "type":          etype,
                    "commonsense":   cs,
                    "complexity":    comp,
                    "linked_t1_id":  link_t1,
                    "linked_t2_id":  "",
                })
        print()

    return results


def main():
    bert_pipe = load_bert_nli()

    t1_results = run_task1()
    t2_results = run_task2(bert_pipe)
    all_results = t1_results + t2_results

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = f"{RESULTS_DIR}/results_{TIMESTAMP}.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nAll results saved => {output_file}")
    print_accuracy_summary(all_results)


if __name__ == "__main__":
    main()
