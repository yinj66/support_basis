import sys
sys.path.append('/data/hchen/pretrained_models/LLaDA-8B-Instruct')
sys.path.append('..')

from datasets import load_dataset
from pathlib import Path
import torch
import tqdm
import json
import re
from transformers import AutoTokenizer, AutoModel
import argparse
from generate import generate


def hellaswag_prompt(context, endings):
    prompt = f"""
You're given a context followed by four possible endings. Choose the best option to complete the sentence.

Context: {context}

Options:
A. {endings[0]}
B. {endings[1]}
C. {endings[2]}
D. {endings[3]}

Answer with the letter (A, B, C, or D) that best completes the context.
Answer:"""
    return prompt


def extract_hellaswag_answer(text):
    match = re.search(r'\b([A-D])\b', text.strip())
    if match:
        return match.group(1)
    return None


def chat_hellaswag(args, steps, block_length):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.config.hybrid_exact_ratio = args.exact_ratio
    model.config.hybrid_chebyshev_degree = args.chebyshev_degree

    # Load dataset from HuggingFace
    dataset = load_dataset("hellaswag", split="validation")
    gen_length = 32

    results = []
    correct = 0
    total = 0

    for idx, datum in enumerate(tqdm.tqdm(dataset)):
        context = datum["ctx"]
        endings = datum["endings"]
        correct_label = int(datum["label"])
        correct_letter = ["A", "B", "C", "D"][correct_label]

        user_input = hellaswag_prompt(context, endings)
        chat_input = [{"role": "user", "content": user_input}]
        formatted_prompt = tokenizer.apply_chat_template(chat_input, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(formatted_prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        prompt = input_ids

        out, _ = generate(
            model, prompt, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=0., cfg_scale=0.,
            remasking='low_confidence'
        )

        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        predicted_letter = extract_hellaswag_answer(answer)

        is_correct = (predicted_letter == correct_letter)

        results.append({
            "context": context,
            "options": endings,
            "predicted": predicted_letter,
            "gold": correct_letter,
            "full_answer": answer,
            "correct": is_correct
        })

        # print(f"Context: {context}")
        # print(f"Predicted: {predicted_letter} | Correct: {correct_letter}")
        # print(f"Raw Answer: {answer}")
        # print('-' * 60)

        total += 1
        correct += is_correct

    out_dir = Path(f"hellaswag_results/steps{steps}_block{block_length}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"HellaSwag Accuracy: {100 * correct / total:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Softmax / Chebyshev Attention args")

    parser.add_argument(
        "--model_name",
        type=str
    )

    parser.add_argument(
        "--exact_ratio",
        type=float,
        default=0.2,
        help="Proportion of entries computed exactly (top+bottom). Example: 0.2 means 20% total."
    )

    parser.add_argument(
        "--chebyshev_degree",
        type=int,
        default=6,
        help="Chebyshev polynomial degree for exp approximation."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chat_hellaswag(args, steps=8, block_length=32)
