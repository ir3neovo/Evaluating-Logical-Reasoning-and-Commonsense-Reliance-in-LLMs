"""Prompt templates and output parsers for Task 1 (truth) and Task 2 (NLI)."""

# ##################################
# TASK 1: Commonsense truth judgment
# ##################################

def t1_prompt_direct(statement):
    return f"""Statement: {statement}

Question: Is this statement true in the real world?

Respond with only one of the following words: True, False"""


def t1_prompt_cot(statement):
    return f"""Statement: {statement}

Question: Is this statement true in the real world?
Think briefly step by step, then give your final answer in exactly this format:
Final Answer: True
or
Final Answer: False"""


# #######################
# TASK 2: Quantifier NLI
# #######################

def t2_prompt_direct(premise, hypothesis):
    return f"""Premise: {premise}
Hypothesis: {hypothesis}

Assume the premise is true. Based only on the premise, is the hypothesis:
Entailment, Contradiction, or Neutral?

Respond with only one of the following words: Entailment, Contradiction, Neutral"""


def t2_prompt_cot(premise, hypothesis):
    return f"""Premise: {premise}
Hypothesis: {hypothesis}

Assume the premise is true. Based only on the premise, reason briefly step by step.

Then give your final answer in exactly this format:
Final Answer: Entailment
or
Final Answer: Contradiction
or
Final Answer: Neutral"""


# ########
# PARSERS
# ########

def _extract_answer_region(text):
    """
    Try to isolate the 'answer' portion of the output.
    Priority:
      1. Everything after the last "Final Answer"
      2. Everything after the last "Answer:"
      3. The last line of the text
      4. Full text as fallback
    """
    low = text.lower()

    # Try "final answer"
    idx = low.rfind("final answer")
    if idx != -1:
        return text[idx:]

    # Try "answer:"
    idx = low.rfind("answer:")
    if idx != -1:
        return text[idx:]

    # Try "the answer is"
    idx = low.rfind("the answer is")
    if idx != -1:
        return text[idx:]

    # Last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        return lines[-1]

    return text


def _find_label(text, labels):
    """
    Find which label appears in text. If multiple labels appear,
    return the one that appears LAST (closest to the end = most likely the final answer).
    labels: list of (display_name, keyword) tuples
    """
    low = text.lower()
    best_label = None
    best_pos = -1

    for display, kw in labels:
        pos = low.rfind(kw)
        if pos != -1 and pos > best_pos:
            best_label = display
            best_pos = pos

    return best_label


def parse_true_false(text):
    """Parse True/False from model output. Returns 'True', 'False', or 'UNPARSED: ...'."""
    if not text or not text.strip():
        return "UNPARSED: [empty output]"

    region = _extract_answer_region(text)
    label = _find_label(region, [("True", "true"), ("False", "false")])

    if label:
        return label

    return f"UNPARSED: {text[:200]}"


def parse_nli_label(text):
    """Parse Entailment/Contradiction/Neutral from model output."""
    if not text or not text.strip():
        return "UNPARSED: [empty output]"

    region = _extract_answer_region(text)
    label = _find_label(region, [
        ("Entailment",    "entailment"),
        ("Contradiction", "contradiction"),
        ("Neutral",       "neutral"),
    ])

    if label:
        return label

    return f"UNPARSED: {text[:200]}"
