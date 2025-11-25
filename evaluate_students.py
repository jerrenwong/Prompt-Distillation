
import os
import json
import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# --- Configuration ---
DOMAINS = ["trivia", "math", "general"]
STUDENT_MODELS = [
    "models/Qwen_Qwen2.5-3B-Instruct",
    "models/meta-llama_Llama-3.2-3B-Instruct",
]
# --- Helper Functions ---
def load_jsonl(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_json(save_path: str, data: Union[dict, list]) -> None:
    """Save data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)

client = get_openai_client()

def judge(instruction: str, prompt: str, model_response: str) -> Tuple[str, int, int]:
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

def generate_responses(model, tokenizer, prompts: List[str], device, max_new_tokens: int = 512) -> List[str]:
    """Generate responses from the model for given prompts."""
    model.eval()
    responses = []

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating responses"):

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if torch.cuda.is_available():
                # Use new autocast syntax if possible, or just run
                with torch.amp.autocast('cuda', enabled=True):
                     outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        top_k=20,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

    return responses

def evaluate_domain(domain: str, student_model_path: str, device):
    print(f"Processing Domain: {domain}, Model: {student_model_path}")

    # Paths
    questions_path = os.path.join("experiments", domain, "questions", "test_selected.jsonl")
    template_path = os.path.join("experiments", domain, "template.txt")
    criteria_path = os.path.join("experiments", domain, "judge_criteria.txt")

    if not os.path.exists(questions_path):
        print(f"File not found: {questions_path}")
        return

    # Load Data
    questions_data = load_jsonl(questions_path)

    # Load Template
    template_content = ""
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            template_content = f.read().strip()
    else:
        print(f"Template not found for {domain}")
        return

    # Load Judge Criteria
    judge_instruction = ""
    if os.path.exists(criteria_path):
        with open(criteria_path, "r") as f:
            judge_instruction = f.read().strip()
    else:
        print(f"Judge criteria not found for {domain}")
        return

    # Format Prompts
    # Template typically looks like: "[QUESTION] ... instructions ..."
    # We replace [QUESTION] with the actual question.
    formatted_prompts = []
    original_questions = []

    for item in questions_data:
        q = item.get("question", "")
        if not q: continue

        if "[QUESTION]" in template_content:
            prompt = template_content.replace("[QUESTION]", q)
        else:
            # If template doesn't have placeholder, maybe append?
            # Assuming it has placeholder as per standard practice in this repo
            prompt = f"{q}\n{template_content}"

        formatted_prompts.append(prompt)
        original_questions.append(q)

    # Load Model
    print(f"Loading model from {student_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(student_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load model {student_model_path}: {e}")
        return

    # Generate
    responses = generate_responses(model, tokenizer, formatted_prompts, device)

    # Evaluate
    results = []
    c_scores = []
    i_scores = []

    print("Evaluating with Judge...")
    for q, prompt, resp in tqdm(zip(original_questions, formatted_prompts, responses), total=len(responses)):
        judgment, c, i = judge(judge_instruction, prompt, resp)
        results.append({
            "question": q,
            "prompt_with_template": prompt,
            "response": resp,
            "judgment": judgment,
            "correctness_score": c,
            "instruction_following_score": i
        })
        c_scores.append(c)
        i_scores.append(i)

    # Statistics
    stats = {
        "correctness": {
            "mean": float(np.mean(c_scores)),
            "std": float(np.std(c_scores))
        },
        "instruction_following": {
            "mean": float(np.mean(i_scores)),
            "std": float(np.std(i_scores))
        }
    }

    # Save
    model_name = os.path.basename(student_model_path)
    output_file = os.path.join("experiments", domain, "evaluation", f"eval_{model_name}_test_selected.json")
    save_data = {
        "domain": domain,
        "model": model_name,
        "instruction": judge_instruction,
        "template": template_content,
        "statistics": stats,
        "results": results
    }
    save_json(output_file, save_data)

    print(f"Results saved to {output_file}")
    print(f"Correctness: {stats['correctness']['mean']:.2f}")
    print(f"Instruction Following: {stats['instruction_following']['mean']:.2f}")

    del model
    del tokenizer
    torch.cuda.empty_cache()

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for domain in DOMAINS:
        for student_path in STUDENT_MODELS:
            if not os.path.exists(student_path):
                print(f"Model path not found: {student_path}")
                continue
            evaluate_domain(domain, student_path, device)

if __name__ == "__main__":
    main()
