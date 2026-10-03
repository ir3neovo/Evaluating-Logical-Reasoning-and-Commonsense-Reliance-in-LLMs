
from collections import Counter

NLI_LABELS = ["Entailment", "Contradiction", "Neutral"]


def print_sample_header(pid, *meta_fields):
    parts = [pid] + [str(m) for m in meta_fields if m != ""]
    print(f"── {' | '.join(parts)} ──")


def print_prediction(model, prompt, pred, correct=None):
    """correct=None => ambiguity mode"""
    tag = "" if correct is None else f"  {'✅' if correct else '❌'}"
    print(f"  [{model:<12} / {prompt:<6}] => {pred}{tag}")


def print_accuracy_summary(all_results):
    print("=" * 58)
    print("ACCURACY SUMMARY")
    print("=" * 58)

    for task_name in ["task1", "task2"]:
        subset = [r for r in all_results if r["task"] == task_name]
        if not subset:
            continue
        print(f"\n  {'─'*20} {task_name.upper()} {'─'*20}")
        models = ["bert_nli", "llama3_8b", "llama3_70b", "gpt4"] if task_name == "task2" \
                 else ["llama3_8b", "llama3_70b", "gpt4"]
        for model in models:
            prompts = ["direct"] if model == "bert_nli" else ["direct", "cot"]
            for ptype in prompts:
                rows = [r for r in subset if r["model"] == model and r["prompt"] == ptype]
                if rows:
                    n_correct = sum(r["correct"] for r in rows)
                    acc = n_correct / len(rows)
                    print(f"  {model:<14} / {ptype:<8} => {acc:.0%}  ({n_correct}/{len(rows)})")


def print_distribution_summary(all_results):
    print("=" * 58)
    print("PREDICTION DISTRIBUTION (ambiguity set)")
    print("=" * 58)
    n_samples = len({r["id"] for r in all_results})
    print(f"  {n_samples} sample(s) per (model * prompt)\n")

    models = ["bert_nli", "llama3_8b", "llama3_70b", "gpt4"]
    for model in models:
        prompts = ["direct"] if model == "bert_nli" else ["direct", "cot"]
        for ptype in prompts:
            rows = [r for r in all_results if r["model"] == model and r["prompt"] == ptype]
            if not rows:
                continue
            counts = Counter(r["prediction"] for r in rows)
            parts = [f"{lbl}={counts.get(lbl, 0)}" for lbl in NLI_LABELS]
            other = sum(v for k, v in counts.items() if k not in NLI_LABELS)
            if other:
                parts.append(f"Other={other}")
            print(f"  {model:<14} / {ptype:<8} => {'  '.join(parts)}")
