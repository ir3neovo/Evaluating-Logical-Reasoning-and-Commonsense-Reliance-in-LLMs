"""
Extract Task 1 items from Task 2 conflict items and back-write linked_t1_id
into every Task 2 item (conflict -> T1_<CAT>_<NNN>, aligned -> null).

Task 1 statement = first sentence of the Task 2 premise, verbatim.
"""

import json
import os

TASK2_DIR = "data/task2"
TASK1_DIR = "data/task1"

FILES = [
    "task_simple.json",
    "task_mq_level1.json",
    "task_mq_level2.json",
    "task_mq_level3.json",
    "task_negation.json",
]


def first_sentence(premise):
    idx = premise.find(". ")
    return premise if idx == -1 else premise[: idx + 1]


def t2_to_t1_id(t2_id):
    assert t2_id.startswith("T2_"), t2_id
    return "T1_" + t2_id[3:]


def write_json(path, items):
    with open(path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
        f.write("\n")


def process():
    os.makedirs(TASK1_DIR, exist_ok=True)
    t1_summary, t2_summary = [], []
    all_t1, all_t2 = [], []

    for name in FILES:
        with open(f"{TASK2_DIR}/{name}") as f:
            t2_items = json.load(f)

        t1_items = []
        n_conflict = n_aligned = 0
        for item in t2_items:
            if item["commonsense_condition"] == "conflict":
                t1_id = t2_to_t1_id(item["id"])
                t1_items.append({
                    "id": t1_id,
                    "statement": first_sentence(item["premise"]),
                    "gold_truth_label": "False",
                    "source_type": "conflict",
                    "linked_t2_id": item["id"],
                })
                item["linked_t1_id"] = t1_id
                n_conflict += 1
            else:
                item["linked_t1_id"] = None
                n_aligned += 1

        write_json(f"{TASK1_DIR}/{name}", t1_items)
        write_json(f"{TASK2_DIR}/{name}", t2_items)

        t1_summary.append((name, n_conflict))
        t2_summary.append((name, n_conflict, n_aligned))
        all_t1.extend(t1_items)
        all_t2.extend(t2_items)

    return t1_summary, t2_summary, all_t1, all_t2


def validate(all_t1, all_t2):
    errors = []
    t1_by_id = {x["id"]: x for x in all_t1}
    t2_by_id = {x["id"]: x for x in all_t2}

    if len(t1_by_id) != len(all_t1):
        errors.append("Duplicate T1 ids")
    conflict_cnt = sum(1 for x in all_t2 if x["commonsense_condition"] == "conflict")
    if len(all_t1) != conflict_cnt:
        errors.append(f"T1 count {len(all_t1)} != T2 conflict count {conflict_cnt}")

    for t1 in all_t1:
        t2 = t2_by_id.get(t1["linked_t2_id"])
        if t2 is None:
            errors.append(f"T1 {t1['id']} -> missing T2 {t1['linked_t2_id']}")
            continue
        if first_sentence(t2["premise"]) != t1["statement"]:
            errors.append(f"T1 {t1['id']} statement != first sentence of T2 premise")
        if t1["gold_truth_label"] != "False":
            errors.append(f"T1 {t1['id']} gold_truth_label not 'False'")

    for t2 in all_t2:
        if "linked_t1_id" not in t2:
            errors.append(f"T2 {t2['id']} missing linked_t1_id field")
            continue
        lt1 = t2["linked_t1_id"]
        if t2["commonsense_condition"] == "conflict":
            if lt1 not in t1_by_id:
                errors.append(f"T2 {t2['id']} -> missing T1 {lt1}")
            elif t1_by_id[lt1]["linked_t2_id"] != t2["id"]:
                errors.append(f"Asymmetric link: T2 {t2['id']} <-> T1 {lt1}")
        elif lt1 is not None:
            errors.append(f"T2 {t2['id']} aligned but linked_t1_id = {lt1!r}")

    return errors


def main():
    t1_sum, t2_sum, all_t1, all_t2 = process()
    errors = validate(all_t1, all_t2)

    print("=== Task 1 Extraction ===")
    total = 0
    for name, n in t1_sum:
        print(f"  {name:22s} {n} conflict items extracted")
        total += n
    print(f"  {'Total Task 1:':22s} {total} items")

    print("\n=== Task 2 Back-write ===")
    for name, c, a in t2_sum:
        print(f"  {name:22s} {c} conflict linked, {a} aligned set to null")

    print("\n=== Bidirectional Check ===")
    if not errors:
        print("  All links verified: OK")
    else:
        print("  All links verified: FAIL")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
