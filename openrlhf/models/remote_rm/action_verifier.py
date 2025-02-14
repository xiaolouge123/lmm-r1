import json
import os
import random
import re
from argparse import ArgumentParser
from multiprocessing import Process, Queue


from flask import Flask, jsonify, request
from utils import action_accuracy_reward as accuracy_reward_func, format_reward as format_reward_func

app = Flask(__name__)

problem_to_answer = {}


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    rewards = []
    for q, problem in zip(data["query"], data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            rewards.append(0.0)
            continue
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400

        format_reward = float(format_reward_func(response))
        acc_reward = float(accuracy_reward_func(response, answer))
        do_print = random.randint(1, 20) == 1
        if do_print:
            info = f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\n"
            info = re.sub(r"<\|.*?\|>", "", info)
            print(info)

        rewards.append(0.5 * format_reward + 0.5 * acc_reward)
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math_dataset", help="Dataset to use")
    parser.add_argument("--prompt-template", type=str, default="chatml", help="Prompt template")
    parser.add_argument("--input_key", type=str, default="prompt", help="The key name of prompt.")
    args = parser.parse_args()

    if args.dataset.endswith("json"):
        with open(args.dataset, "r") as f:
            dataset = json.load(f)
    elif args.dataset.endswith("jsonl"):
        with open(args.dataset, "r") as f:
            dataset = [json.loads(l) for l in f.readlines()]
    else:
        raise ValueError(f"Unknown dataset format: {args.dataset}")

    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        problem_to_answer[problem] = answer

    if args.prompt_template == "chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template == "qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template == "base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.dataset}")

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
