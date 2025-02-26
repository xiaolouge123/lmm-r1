import json
import argparse
from transformers import AutoTokenizer, AutoProcessor

processor = AutoProcessor.from_pretrained("/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2.5-VL-3B-Instruct")


system = "You are a helpful assistant good at solving math problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>."

def make_conv(question: str):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def main(args):
    with open(args.input_path, "r") as f:
        data = json.load(f)
    formatted_data = []
    for d in data:
        assert d[0]["from"].lower() == "human", d
        question = d[0]["value"]
        conv = make_conv(question)
        text = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True
        )
        assert d[1]["from"].lower() == "assistant", d
        answer = d[1]["ground_truth"]['value']
        formatted_data.append({
            "prompt": text,
            "answer": answer
        })
    with open(args.output_path, "w") as f:
        json.dump(formatted_data, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

# python make_math_rl_chatml_data.py --input_path /data/true_nas/zfs_share1/zyc/workspace/lmm-r1/examples/data/orz_math_57k_collected.json --output_path /data/true_nas/zfs_share1/zyc/workspace/lmm-r1/examples/data/orz_math_57k_collected_chatml.json