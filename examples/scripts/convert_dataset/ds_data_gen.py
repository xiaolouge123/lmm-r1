import argparse
import json
import os
import re
from .utils import ACTION_SPACE, ACTION_OUTPUT_FORMAT, QUERY_TEMPLATE


def convert_to_distill_data(basic_data: list) -> list:
    pass


def convert_to_rl_data(basic_data: list) -> list:
    pass


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
    # hack
    image_dir = "/data/true_nas/zfs_share1/zyc/data/data/Yuxiang007/AMEX/AMEX/screenshot"

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
