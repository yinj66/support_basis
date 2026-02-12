import sys
import argparse
from datasets import load_dataset
import torch
import tqdm
import json
from transformers import AutoTokenizer, AutoModel
# from configuration_llada import LLaDAConfig
from generate import generate
import re
from pathlib import Path


def arc_prompt(question, choices_text, choices_label):
    choice_lines = "\n".join(
        [f"{label}. {text}" for label, text in zip(choices_label, choices_text)]
    )
    prompt = f"""
You're given a science question with multiple choices. Read the question carefully and choose the best answer. Respond with only the letter of the correct choice.

Question: {question}

Choices:
{choice_lines}

Answer:"""
    return prompt


def extract_arc_answer(text):
    match = re.search(r'\b([A-E])\b', text.strip())
    if match:
        return match.group(1)
    return None


def chat_arc(args, steps, block_length):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config = LLaDAConfig.from_pretrained('/data/hchen/pretrained_models/LLaDA-8B-Instruct')
    # model = AutoModel.from_pretrained(
    #     '/data/hchen/pretrained_models/LLaDA-8B-Instruct',
    #     config=config,
    #     trust_remote_code=True,
    #     torch_dtype=torch.float32
    # ).to(device).eval()

    # model = AutoModel.from_pretrained(
    #     'GSAI-ML/LLaDA-8B-Instruct',
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16
    # ).to(device).eval()


    # model = AutoModel.from_pretrained('/Users/zhc/Downloads/LLaDA-8B-Instruct', torch_dtype=torch.bfloat16)

    # tokenizer = AutoTokenizer.from_pretrained('/Users/zhc/Downloads/LLaDA-8B-Instruct-AS23')
    # model = AutoModel.from_pretrained('/Users/zhc/Downloads/LLaDA-8B-Instruct-AS23', torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code = True
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    # 修改你新加的 config 字段
    model.config.hybrid_exact_ratio = args.exact_ratio
    model.config.hybrid_chebyshev_degree = args.chebyshev_degree


    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", download_mode="force_redownload")  # You can also try "ARC-Easy"
    # dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", download_mode="force_redownload")  # You can also try "ARC-Easy"
    gen_length = 32

    results = []
    correct = 0
    total = 0

    for idx, example in enumerate(tqdm.tqdm(dataset)):
        question = example["question"]
        choices_text = example["choices"]["text"]
        choices_label = example["choices"]["label"]
        gold = example["answerKey"]

        user_input = arc_prompt(question, choices_text, choices_label)
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
        predicted = extract_arc_answer(answer)

        is_correct = (predicted == gold)

        results.append({
            "question": question,
            "choices": list(zip(choices_label, choices_text)),
            "gold": gold,
            "predicted": predicted,
            "full_answer": answer,
            "correct": is_correct
        })

        # print(f"Q: {question}")
        # print(f"Predicted: {predicted} | Correct: {gold}")
        # print(f"Full answer: {answer}")
        # print("-" * 60)

        total += 1
        correct += is_correct

    out_dir = Path(f"arc_results/steps{steps}_block{block_length}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"ARC Easy Accuracy: {100 * correct / total:.2f}%")



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
    chat_arc(args, steps=8, block_length=32)
