import os, json
from openai import OpenAI
from tqdm import tqdm
from typing import Tuple, List, Dict, Union
import numpy as np
import re

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def save_json(save_path: str, data: Union[dict, list]) -> None:
    """Save data to a JSON file, creating directories if needed."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def judge(instruction: str, prompt: str, model_response: str) -> Tuple[str, int]:
    """
    Judge a model response and return both the full judgment text and extracted score.
    Returns: (judgment_text, score)
    """
    RUBRIC = """You are a strict grader. Score 1–10 how well MODEL_RESPONSE follows INSTRUCTION for the PROMPT. Focus especially on INSTRUCTION and less on PROMPT. Reason explicitly about the response, then output a single integer from 1 to 10 that is boxed in []."""

    case = (
        f"INSTRUCTION:\n{instruction}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL_RESPONSE:\n{model_response}\n"
    )
    judgment_text = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": case}
        ],
        temperature=0,
    ).choices[0].message.content

    # Extract score from judgment text (look for [number] or just a number)
    score_match = re.search(r'\[(\d+)\]', judgment_text)
    if not score_match:
        score_match = re.search(r'\b(\d+)\b', judgment_text)
    score = int(score_match.group(1)) if score_match else 0

    return judgment_text, score

def evaluate(instruction: str, conversations: List[Dict[str, str]], save_path: str) -> dict:
    """
    Evaluate conversations against an instruction.

    Args:
        instruction: The instruction to evaluate against
        conversations: List of dicts with 'prompt' and 'response' keys
        save_path: Path to save the results JSON file

    Returns:
        Dictionary with scores, statistics, and full results
    """
    results = []
    scores = []

    for item in tqdm(conversations):
        prompt = item["prompt"]
        response = item["response"]
        judgment_text, score = judge(instruction, prompt, response)

        results.append({
            "prompt": prompt,
            "response": response,
            "judgment": judgment_text,
            "score": score
        })
        scores.append(score)

    # Save full results
    save_data = {
        "instruction": instruction,
        "statistics": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": int(np.min(scores)),
            "max": int(np.max(scores)),
        },
        "results": results
    }
    save_json(save_path, save_data)

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

# --- minimal example ---
if __name__ == "__main__":
    instruction = "Keep your answer short and concise."
    items = [
        {"prompt": "What is overfitting?", "response": """Overfitting happens when a model learns not just the underlying pattern in the training data but also the random noise. It’s like a student who memorizes all the answers to past exam questions instead of understanding the concepts—great performance on the practice tests, terrible performance on new questions.

        In machine learning terms:
            •	During training, the model’s error keeps dropping.
            •	During validation or testing, the error starts increasing.

        That’s the telltale sign: the model fits the training data too well and fails to generalize.

        A few key causes and signs:
            •	Too complex model: Too many parameters relative to data (like a high-degree polynomial fitting 5 points perfectly).
            •	Too little data: The model has no choice but to “memorize” small quirks.
            •	Too few constraints: Lack of regularization (e.g., no dropout, no L2 penalty).

        Typical ways to reduce overfitting:
            •	Add more training data.
            •	Simplify the model.
            •	Use regularization (L1/L2, dropout, early stopping).
            •	Use cross-validation to detect overfitting early.

        If you want, I can walk you through a visual or mathematical example (say, fitting polynomials of different degrees to the same dataset). It makes the intuition click sharply."""},
        {"prompt": "What is overfitting?", "response": "Overfitting is a phenomenon where a model performs well on the training data but poorly on the test data."},
    ]
    save_path = "results/overfitting.json"
    evaluate(instruction, items, save_path)
