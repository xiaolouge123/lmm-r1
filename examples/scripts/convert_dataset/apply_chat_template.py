from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm


processor = AutoProcessor.from_pretrained("/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "http://path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Base64 encoded image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "data:image;base64,/9j/..."},
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
    for item in tqdm(data):
        msg = json.loads(item['message'])
        for i, m in enumerate(msg):
            if m['role'] == 'user':
                for c in m['content']:
                    if c['type'] == 'image':
                        c['image'] = c['image'] if c['image'].startswith('file://') else f'file://{c["image"]}'
                # put the image to the first position
                m['content'] = [c for c in m['content'] if c['type'] == 'image'] + [c for c in m['content'] if c['type'] != 'image']

        text = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        item['prompt'] = text
        item['message'] = json.dumps(msg, ensure_ascii=False)
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
