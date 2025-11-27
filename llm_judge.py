import os
import json
import re
import numpy as np
import argparse
from typing import Tuple, List, Dict, Union
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

def save_json(save_path: str, data: Union[dict, list]) -> None:
    """Save data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(load_path: str) -> Union[dict, list]:
    with open(load_path, "r", encoding="utf-8") as f:
        return json.load(f)

def judge(client, instruction: str, prompt: str, model_response: str) -> Tuple[str, int, int]:
    """
    Judge a model response and return judgment text and two scores: correctness and instruction_following.
    Returns: (judgment_text, correctness_score, instruction_following_score)
    """

    RUBRIC = """You are a strict grader. You will evaluate the MODEL_RESPONSE based on two criteria:
1. Correctness: Does the response accurately answer the PROMPT? (Score 1-10)
2. Instruction Following: Does the response follow the specific INSTRUCTION provided? (Score 1-10)

IMPORTANT: The Instruction Following score must be independent of Correctness. If the response follows the formatting or constraints of the INSTRUCTION but the answer is factually incorrect, it should still receive a high Instruction Following score.

Output your evaluation in the following format:
Correctness Reasoning: [Your reasoning for the correctness score]
Correctness Score: [Score]
Instruction Reasoning: [Your reasoning for the instruction following score]
Instruction Following Score: [Score]"""

    case = (
        f"INSTRUCTION:\n{instruction}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL_RESPONSE:\n{model_response}\n"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RUBRIC},
                {"role": "user", "content": case}
            ],
            temperature=0,
        )
        judgment_text = completion.choices[0].message.content
    except Exception as e:
        print(f"Judge error: {e}")
        return str(e), 0, 0

    # Extract scores
    correctness_match = re.search(r'Correctness Score:\s*\[?(\d+)\]?', judgment_text)
    instruction_match = re.search(r'Instruction Following Score:\s*\[?(\d+)\]?', judgment_text)
    c_score = int(correctness_match.group(1)) if correctness_match else 0
    i_score = int(instruction_match.group(1)) if instruction_match else 0

    return judgment_text, c_score, i_score

def evaluate_conversations(client, instruction: str, conversations: List[Dict[str, str]]) -> dict:
    """Evaluate conversations against an instruction using LLM judge."""
    results = []
    c_scores = []
    i_scores = []

    def run_one(item: Dict[str, str]):
        prompt = item["prompt"]
        response = item["response"]
        judgment_text, c_score, i_score = judge(client, instruction, prompt, response)
        return prompt, response, judgment_text, c_score, i_score

    # Parallelize with 10 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        for prompt, response, judgment_text, c_score, i_score in tqdm(
            executor.map(run_one, conversations),
            total=len(conversations),
            desc="Evaluating with LLM judge",
        ):
            results.append({
                "prompt": prompt,
                "response": response,
                "judgment": judgment_text,
                "correctness_score": c_score,
                "instruction_following_score": i_score
            })
            c_scores.append(c_score)
            i_scores.append(i_score)

    return {
        "instruction": instruction,
        "statistics": {
            "correctness": {
                "mean": float(np.mean(c_scores)) if c_scores else 0.0,
                "std": float(np.std(c_scores)) if c_scores else 0.0
            },
            "instruction_following": {
                "mean": float(np.mean(i_scores)) if i_scores else 0.0,
                "std": float(np.std(i_scores)) if i_scores else 0.0
            }
        },
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Run LLM Judge on generated responses.")
    parser.add_argument("--experiments_dir", type=str, default="experiments", help="Base experiments directory.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY or OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key)

    # Walk through experiments directory to find model_responses.json
    for root, dirs, files in os.walk(args.experiments_dir):
        if "model_responses.json" in files:
            response_path = os.path.join(root, "model_responses.json")
            eval_path = os.path.join(root, "llm_judge_evaluation.json")

            # Check if already evaluated
            if os.path.exists(eval_path):
                print(f"Skipping {eval_path}, already exists.")
                continue

            print(f"Processing {response_path}...")

            try:
                data = load_json(response_path)
                instruction = data.get("instruction", "")
                conversations = data.get("conversations", [])

                if not conversations:
                    print(f"No conversations found in {response_path}")
                    continue

                eval_results = evaluate_conversations(client, instruction, conversations)
                save_json(eval_path, eval_results)

                stats = eval_results["statistics"]
                print(f"Saved evaluation to {eval_path}")
                print(f"Correctness: {stats['correctness']['mean']:.2f}")
                print(f"Instruction Following: {stats['instruction_following']['mean']:.2f}")

            except Exception as e:
                print(f"Error evaluating {response_path}: {e}")

if __name__ == "__main__":
    main()
