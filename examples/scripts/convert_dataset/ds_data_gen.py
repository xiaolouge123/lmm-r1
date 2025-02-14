import argparse
import json
import os
import re
from .utils import ACTION_SPACE, ACTION_OUTPUT_FORMAT, QUERY_TEMPLATE


def convert_to_distill_data(basic_data: list, task="long_cot") -> list:
    def make_conv(system_prompt: str = "", user_prompt: str = "", image_path: str = "", assistant_answer: str = ""):
        conv = []
        if system_prompt:
            conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        if user_prompt and image_path:
            conv.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}, {"type": "image", "image": image_path}],
                }
            )
        elif user_prompt:
            conv.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
        if assistant_answer:
            conv.append({"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]})
        return json.dumps(conv, ensure_ascii=False)

    data = []
    for item in basic_data:
        user_prompt = QUERY_TEMPLATE.substitute(
            instruction=item["instruction"], action_spaces=ACTION_SPACE, output_action_format=ACTION_OUTPUT_FORMAT
        )
        answer = ""
        if task == "long_cot":
            answer = f"<think>{item['long_cot']}</think><answer>{item['answer'] if not item['answer'].startswith('<answer>') else item['answer'].lstrip('<answer>').rstrip('</answer>')}</answer>"
        elif task == "short_cot":
            answer = f"<think>{item['short_cot']}</think><answer>{item['answer'] if not item['answer'].startswith('<answer>') else item['answer'].lstrip('<answer>').rstrip('</answer>')}</answer>"
        elif task == "action":
            answer = f"<answer>{item['answer'] if not item['answer'].startswith('<answer>') else item['answer'].lstrip('<answer>').rstrip('</answer>')}</answer>"
        else:
            raise ValueError(f"Unsupported task: {task}")
        message_str = make_conv(
            system_prompt="", user_prompt=user_prompt, image_path=item["image_path"], assistant_answer=answer
        )
        data.append(
            {
                "message": message_str,
                "question": item["instruction"],
            }
        )
    return data


def convert_to_rl_data(basic_data: list) -> list:
    def make_conv(system_prompt: str = "", user_prompt: str = "", image_path: str = ""):
        conv = []
        if system_prompt:
            conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        if user_prompt and image_path:
            conv.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}, {"type": "image", "image": image_path}],
                }
            )
        elif user_prompt:
            conv.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})
        return json.dumps(conv, ensure_ascii=False)

    data = []
    for item in basic_data:
        user_prompt = QUERY_TEMPLATE.substitute(
            instruction=item["instruction"], action_spaces=ACTION_SPACE, output_action_format=ACTION_OUTPUT_FORMAT
        )
        message_str = make_conv(system_prompt="", user_prompt=user_prompt, image_path=item["image_path"])
        data.append(
            {
                "message": message_str,
                "answer": item["answer"],
                "question": item["instruction"],
            }
        )
    return data


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def main(args):
    if args.input_path.endswith(".jsonl"):
        with open(args.input_path, "r") as f:
            data = [json.loads(line) for line in f]
    elif args.input_path.endswith(".json"):
        with open(args.input_path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {args.input_path}")

    # get basic data fields
    basic_data = []
    # TODO 上游的数据要标准化一下，把相关重要字段都准备好
    # 比如 image_path，instruction，answer，long_cot, short_cot,
    for item in data:
        basic_data.append(
            {
                "instruction": item["instruction"],
                "image_path": item["image_path"],
                "answer": item["answer"],
                "long_cot": item.get("long_cot", ""),
                "short_cot": item.get("short_cot", ""),
                # 后续需要继续补充
            }
        )

    if args.data_type == "distill":
        # 生成distill数据
        data = convert_to_distill_data(basic_data)
    elif args.data_type == "rl":
        # 生成rl数据
        data = convert_to_rl_data(basic_data)
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}")

    # 保存数据
    with open(args.output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True, default="distill", choices=["distill", "rl"])
    args = parser.parse_args()
    main(args)
