import os
import json
import torch
import argparse
from typing import List, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    dir_name = os.path.dirname(save_path)
    os.makedirs(dir_name if dir_name else ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_responses(model, tokenizer, prompts: List[str], device, max_new_tokens: int = 512) -> List[str]:
    """Generate responses using tokenizer's chat template (always applied)."""
    model.eval()
    responses = []

    use_cuda = torch.cuda.is_available() and device.type == "cuda"

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating responses"):
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(device)

            attention_mask = torch.ones_like(inputs)
            model_inputs = {"input_ids": inputs, "attention_mask": attention_mask}

            if use_cuda:
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    outputs = model.generate(
                        **model_inputs,
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
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_length = model_inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

    return responses


def run_domain(domain: str, student_model_path: str, device):
    print(f"Processing Domain: {domain}, Model: {student_model_path}")

    # Paths
    questions_path = os.path.join("experiments", domain, "questions", "test_selected.jsonl")
    template_path = os.path.join("experiments", domain, "template.txt")

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

    # Format Prompts
    formatted_prompts = []
    original_questions = []

    for item in questions_data:
        q = item.get("question", "")
        if not q:
            continue

        if "[QUESTION]" in template_content:
            prompt = template_content.replace("[QUESTION]", q)
        else:
            prompt = f"{q}\n{template_content}"

        formatted_prompts.append(prompt)
        original_questions.append(q)

    # Load Model
    print(f"Loading model from {student_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(student_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                student_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Model loaded on CUDA with device_map='auto'")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                student_model_path,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            print("Model loaded on CPU/MPS")
    except Exception as e:
        print(f"Failed to load model {student_model_path}: {e}")
        return

    # Generate with template
    print("Generating responses (With Template)...")
    responses_with_template = generate_responses(model, tokenizer, formatted_prompts, device)

    # Generate without template
    print("Generating responses (Without Template)...")
    responses_no_template = generate_responses(model, tokenizer, original_questions, device)

    # Build conversations for each setting
    conversations_with = [
        {
            "prompt": prompt_t,        # what the model actually saw
            "response": resp_t
        }
        for prompt_t, resp_t in zip(formatted_prompts, responses_with_template)
    ]

    conversations_no = [
        {
            "prompt": q,              # raw question
            "response": resp_q
        }
        for q, resp_q in zip(original_questions, responses_no_template)
    ]

    # Determine model name (directory-friendly)
    model_name = os.path.basename(student_model_path.rstrip("/"))

    # Save with-template runs
    out_dir_with = os.path.join("experiments", domain, "runs", f"{model_name}_with_template")
    out_path_with = os.path.join(out_dir_with, "model_responses.json")
    save_json(out_path_with, {"conversations": conversations_with})
    print(f"With-template responses saved to {out_path_with}")

    # Save no-template runs
    out_dir_no = os.path.join("experiments", domain, "runs", f"{model_name}_no_template")
    out_path_no = os.path.join(out_dir_no, "model_responses.json")
    save_json(out_path_no, {"conversations": conversations_no})
    print(f"No-template responses saved to {out_path_no}")

    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")


def main():
    parser = argparse.ArgumentParser(description="Run a student model on a specific domain and save responses.")
    parser.add_argument("--domain", type=str, required=True, help="Domain to run (e.g., trivia, math, general)")
    parser.add_argument("--model", type=str, required=True, help="Path to the student model")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")

    if not os.path.exists(args.model):
        print(f"Model path not found: {args.model}")
        return

    run_domain(args.domain, args.model, device)


if __name__ == "__main__":
    main()
