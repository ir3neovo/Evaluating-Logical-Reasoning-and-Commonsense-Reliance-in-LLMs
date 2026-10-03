# Quantifier Reasoning Experiment

Evaluates BERT NLI, Llama-3.1 (8B/70B) and GPT-4.1 on commonsense truth judgment (Task 1) and quantifier NLI (Task 2), plus a separate distribution study on scope-ambiguity samples (wide- vs narrow-scope reading preference).

## Structure

```
.
|-- data/
|   |-- task1/                # Commonsense truth statements (True/False), one file per task2 category
|   |   `-- task_{simple,negation,mq_level1,mq_level2,mq_level3}.json
|   |-- task2/                # Quantifier NLI (E/C/N), split by category / complexity
|   |   `-- task_{simple,negation,mq_level1,mq_level2,mq_level3}.json
|   `-- ambiguity/
|       `-- data_ambiguity.json   # Scope-ambiguity samples (20 items; each has hypothesis_wide + hypothesis_narrow; no gold label)
|-- data_example/             # Snapshot of the earlier pilot data (pre-expansion)
|-- utils/
|   |-- models.py             # BERT + cloud callers + parallel helper
|   |-- prompts.py            # Prompt templates + output parsers
|   `-- display.py            # Console printers + summary tables
|-- run_main.py               # Task 1 + Task 2 (accuracy)
|-- run_ambiguity.py          # Ambiguity set (label distribution, no accuracy)
|-- analyze_results.py        # Analysis -> analysis/summary_*.md
|-- results/                  # Raw CSVs (results_*.csv, ambiguity_*.csv)
`-- analysis/                 # Markdown summaries
```

### Cross-task linking

Conflict items in Task 1 and Task 2 are paired 1:1 via `linked_t2_id` / `linked_t1_id`. Each task 1 item is a statement extracted verbatim from the first sentence of a task 2 conflict premise (e.g. premise "Several toddlers operate every piece of heavy machinery. All toddlers who..." -> task 1 statement "Several toddlers operate every piece of heavy machinery.", gold `False`). This lets `analyze_results.py` compute a joint table (e.g. "model got the commonsense truth right but failed the quantifier inference built on top of it"). Aligned task-2 items carry `linked_t1_id: null`; ambiguity items have no linking fields at all.

## How to run

**1. Setup**

```bash
conda create -n 550_project python=3.10 -y
conda activate 550_project
pip install -r requirements.txt
cp .env_example .env    # fill in OPENROUTER_API_KEY, OPENAI_API_KEY, HF_TOKEN
```

**2. Main experiment** (Task 1 + Task 2, with accuracy)

```bash
python run_main.py
```

Writes `results/results_<timestamp>.csv` and prints an accuracy summary.

**3. Ambiguity experiment** (distribution only, no accuracy)

```bash
python run_ambiguity.py
```

Writes `results/ambiguity_<timestamp>.csv` and prints a prediction-distribution summary (E / C / N counts per model * prompt, aggregated across both reading hypotheses).

**4. Analyze results**

```bash
python analyze_results.py                       # latest results_*.csv (+ latest ambiguity_*.csv if present)
python analyze_results.py results/<file>.csv    # specific file
```

Writes `analysis/summary_<timestamp>.md` with the table framework:

| #   | Table                          | Scope                              |
| --- | ------------------------------ | ---------------------------------- |
| 1   | Overall accuracy               | Task 2, model * prompt             |
| 2   | By category                    | simple / multi / negation          |
| 3   | Recall per gold label          | E / C / N (Neutral bias)           |
| 4   | aligned vs conflict (+ diff)   | commonsense effect                 |
| 5   | Task 1 accuracy                | aligned / conflict / overall       |
| 6   | Joint analysis                 | linked T1 <-> T2 pairs             |
| 7a  | CoT effect (overall)           | direct vs cot diff                 |
| 7b  | Confusion matrices             | per model * prompt                 |
| 8   | CoT * category                 | interaction                        |
| 9   | CoT * commonsense              | interaction                        |
| 10  | Multi-quantifier by complexity | Lv 1 / 2 / 3                       |
| 11  | Ambiguity preference           | wide / narrow / both / neither / invalid (pairing wide+narrow hypotheses) |
| 12  | Ambiguity raw distribution     | E / C / N / Other per reading      |
