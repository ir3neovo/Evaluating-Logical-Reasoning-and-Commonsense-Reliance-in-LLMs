"""
Ambiguity experiment: run models on ambiguous samples. Outputs prediction distribution, not accuracy.

Data: data/data_ambiguity.json.
"""

import csv
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
    t2_prompt_cot,
    t2_prompt_direct,
)
from utils.display import (
    print_distribution_summary,
    print_prediction,
    print_sample_header,
)

DATA_PATH   = "data/ambiguity/data_ambiguity.json"
RESULTS_DIR = "results"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")

FIELDNAMES = [
    "id", "ambiguity_type", "reading",
    "model", "prompt", "raw_output", "prediction",
]


def run_ambiguity(bert_pipe):
    print("\n" + "=" * 56)
    print("AMBIGUITY: Scope (dual-hypothesis, distribution only)")
    print("=" * 56 + "\n")

    with open(DATA_PATH) as f:
        data = json.load(f)

    results = []

    for ex in data:
        pid   = ex["id"]
        prem  = ex["premise"]
        atype = ex["ambiguity_type"]
        hyps  = {"wide": ex["hypothesis_wide"], "narrow": ex["hypothesis_narrow"]}

        print_sample_header(pid, atype, "")
        print(f"  Premise:         {prem}")
        print(f"  Hyp (wide):      {hyps['wide']}")
        print(f"  Hyp (narrow):    {hyps['narrow']}")

        # BERT: direct only, one call per reading
        for reading, hyp in hyps.items():
            bert_pred = call_bert_nli(bert_pipe, prem, hyp)
            print_prediction(f"bert_nli/{reading}", "direct", bert_pred)
            results.append({
                "id":             pid,
                "ambiguity_type": atype,
                "reading":        reading,
                "model":          "bert_nli",
                "prompt":         "direct",
                "raw_output":     bert_pred,
                "prediction":     bert_pred,
            })

        # Cloud: batch 2 readings * 2 prompt types = 4 prompts in one parallel call
        prompts = {}
        for reading, hyp in hyps.items():
            prompts[f"{reading}|direct"] = t2_prompt_direct(prem, hyp)
            prompts[f"{reading}|cot"]    = t2_prompt_cot(prem, hyp)
        raw_by_key = run_cloud_jobs_parallel(prompts)

        for reading in ("wide", "narrow"):
            for ptype in ("direct", "cot"):
                for model_name in CLOUD_MODELS:
                    raw  = raw_by_key[(f"{reading}|{ptype}", model_name)]
                    pred = parse_nli_label(raw)
                    print_prediction(f"{model_name}/{reading}", ptype, pred)
                    results.append({
                        "id":             pid,
                        "ambiguity_type": atype,
                        "reading":        reading,
                        "model":          model_name,
                        "prompt":         ptype,
                        "raw_output":     raw,
                        "prediction":     pred,
                    })
        print()

    return results


def main():
    bert_pipe = load_bert_nli()
    results = run_ambiguity(bert_pipe)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = f"{RESULTS_DIR}/ambiguity_{TIMESTAMP}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved => {output_file}")
    print_distribution_summary(results)


if __name__ == "__main__":
    main()
