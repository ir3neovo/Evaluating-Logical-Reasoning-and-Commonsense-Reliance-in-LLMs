"""Model loading and callers: BERT NLI, OpenRouter (Llama), OpenAI (GPT-4)."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
HF_TOKEN           = os.getenv("HF_TOKEN")

CLOUD_MODELS = ["llama3_8b", "llama3_70b", "gpt4"]


def load_bert_nli():
    print("Loading BERT NLI model (first run downloads ~500MB)...")
    pipe = pipeline(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=-1,
    )
    print("BERT NLI ready.\n")
    return pipe


def call_bert_nli(pipe, premise, hypothesis):
    """BERT: NLI direct only. Returns Entailment/Contradiction/Neutral."""
    result = pipe({"text": premise, "text_pair": hypothesis})
    if isinstance(result, list):
        top = max(result, key=lambda x: x["score"])
    else:
        top = result
    label = top["label"].upper()
    if "ENTAIL" in label:
        return "Entailment"
    if "CONTRADICT" in label:
        return "Contradiction"
    return "Neutral"


def call_openrouter(prompt_text, model):
    """Llama-3.1 8B/70B via OpenRouter."""
    try:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1", timeout=60)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=300,
        )
        if not resp.choices:
            return f"ERROR: empty response from {model}"
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def call_gpt4(prompt_text):
    """GPT-4.1 via OpenAI."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=60)
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=300,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


def call_model(model_name, prompt_text):
    """Dispatch to the right cloud backend."""
    if model_name == "llama3_8b":
        return call_openrouter(prompt_text, "meta-llama/llama-3.1-8b-instruct")
    if model_name == "llama3_70b":
        return call_openrouter(prompt_text, "meta-llama/llama-3.1-70b-instruct")
    if model_name == "gpt4":
        return call_gpt4(prompt_text)
    return f"ERROR: unknown model {model_name}"


def run_cloud_jobs_parallel(prompts, max_workers=6):
    """Run {ptype: prompt_text} * CLOUD_MODELS concurrently.
    Returns {(ptype, model_name): raw_output}.
    """
    jobs = [(ptype, model) for ptype in prompts for model in CLOUD_MODELS]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(call_model, model, prompts[ptype]): (ptype, model)
            for ptype, model in jobs
        }
        return {future_map[f]: f.result() for f in as_completed(future_map)}
