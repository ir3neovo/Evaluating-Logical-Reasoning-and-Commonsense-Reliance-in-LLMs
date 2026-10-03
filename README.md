# Quantifier Reasoning Experiment

This project evaluates how language models distinguish between **commonsense truth** and **premise-based logical reasoning**, with a focus on quantifier reasoning and scope ambiguity.

The experiments compare **BERT NLI, Llama-3.1 8B, Llama-3.1 70B, and GPT-4.1** across:

- **Task 1: Commonsense Truth Judgment**  
  Evaluates whether models can correctly identify whether standalone statements are true or false in the real world.

- **Task 2: Quantifier Reasoning (NLI)**  
  Evaluates whether models can classify premise-hypothesis pairs as Entailment, Contradiction, or Neutral while following the premise even when it conflicts with commonsense knowledge.

- **Scope Ambiguity Analysis**  
  Examines model preferences between wide-scope and narrow-scope interpretations for ambiguous sentences.

The benchmark includes varying levels of logical complexity, aligned and commonsense-conflict conditions, and both direct and chain-of-thought prompting. Cross-task linking between Task 1 and Task 2 also allows comparison between factual knowledge and logical inference on related examples.

## Contributors

- Winnie Xiong
- Xiyuan Fan
- Irene Wang
