from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json

processor = AutoProcessor.from_pretrained("/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

if __name__ == "__main__":
    file_path = '/data/true_nas/zfs_share1/zyc/workspace/lmm-r1/examples/data/AMEX_acton_rl_conv.json'
    output_path = '/data/true_nas/zfs_share1/zyc/workspace/lmm-r1/examples/data/AMEX_acton_rl_chatml.json'
    data = json.loads(open(file_path).read())
    for item in data:
        msg = item['message']
        text = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        item['prompt'] = text
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
