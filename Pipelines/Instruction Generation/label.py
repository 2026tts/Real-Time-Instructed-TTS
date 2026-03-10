
import os
import torch
import csv
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import ceil

import gc
# ========== 配置 ==========
model_name = "Qwen/Qwen2.5-7B-Instruct"
model_dir = "../caption/qwen2.5"  # 模型缓存路径（可与之前一样）
input_folder = "../caption" # 存放8个txt的文件夹
output_folder ="../wenetspeech4tts/labels"       # 输出目录
os.makedirs(output_folder, exist_ok=True)

# ========== GPU编号 & 文件匹配 ==========
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
all_txts = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
assert len(all_txts) >= gpu_id + 1, f"GPU {gpu_id} 没有对应txt文件"
caption_list_path = os.path.join(input_folder, all_txts[gpu_id])
base_name = os.path.splitext(os.path.basename(caption_list_path))[0]
output_tsv = os.path.join(output_folder, f"{base_name}_labels.tsv")
output_tsv = os.path.abspath(output_tsv)
print(output_tsv)

# ========== 加载模型 ==========
print(f"[GPU {gpu_id}] Loading model {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=model_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 关键设置：decoder-only 用左补齐 + 左截断
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# ✅ 确保有 pad_token（很多 decoder-only 默认没有）
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id



# ========== Prompt 模板 ==========
PROMPT_TEMPLATE = """
You are an annotation assistant. Your task is to read the given speech description and assign four categorical labels: tempo, emotion, energy, and pitch.

You must follow these strict rules:
1. Each label must be chosen ONLY from the predefined sets below.
2. You must use EXACTLY one of the given words for each label.
3. You MUST NOT invent, modify, or combine any new words.
4. If the description does not provide enough information, use "unknown" for that category.

Predefined sets:
- tempo: [high, normal, low]
- energy: [high, normal, low]
- pitch: [high, normal, low]
- emotion: [happy, sad, angry, surprise, neutral, fear, disgusted, contempt, unknown]

Output Format Rules:
- Output ONLY four plain lines, with no explanations, no extra text, and no punctuation other than the colon.
- Each line must strictly follow this format:
  tempo: <label>
  energy: <label>
  pitch: <label>
  emotion: <label>

Focus Criteria:
- Base your judgment strictly on the **speaker’s delivery style**, such as tone, pace, intensity, and emotional expression.
- ⚠️ DO NOT consider the meaning or content of the words.
- ⚠️ DO NOT explain your choice or add any text other than the required four lines.

Speech description:
{caption_text}
"""


# ========== 解析函数 ==========
def parse_labels_from_output(output_text: str):
    result = {"tempo": "unknown", "energy": "unknown", "pitch": "unknown", "emotion": "unknown"}
    for key in result.keys():
        match = re.search(rf"{key}\s*:\s*([a-zA-Z]+)", output_text, re.IGNORECASE)
        if match:
            result[key] = match.group(1).lower()
    return result

# ========== 读取所有 caption ==========
with open(caption_list_path, "r", encoding="utf-8", errors="ignore") as f:
    caption_paths = [line.strip() for line in f if line.strip()]

captions = []
for path in caption_paths:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fin:
            text = fin.read().strip()
        captions.append((path, text))
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")

# ========== 批处理参数 ==========
batch_size = 5
num_batches = ceil(len(captions) / batch_size)
rows = []

# ========== 主循环 ==========
for b in tqdm(range(num_batches), desc=f"[GPU {gpu_id}] Batching"):
    batch = captions[b * batch_size : (b + 1) * batch_size]
    if not batch:
        continue

    prompts = [PROMPT_TEMPLATE.format(caption_text=text) for _, text in batch]
    messages_batch = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]

    # === 批量tokenize ===
    inputs = tokenizer(messages_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # === 批量生成 ===
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
        )

    # === 官方推荐写法裁剪输入部分 ===
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


    # === 解析输出 ===
    for (path, _), output_text in zip(batch, responses):
        parsed = parse_labels_from_output(output_text)
        id_name = os.path.splitext(os.path.basename(path))[0]
        rows.append({
            "id": id_name,
            "caption": path,
            "tempo": parsed["tempo"],
            "energy": parsed["energy"],
            "pitch": parsed["pitch"],
            "emotion": parsed["emotion"]
        })
    del inputs, generated_ids, responses
    del prompts, messages_batch, batch
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


# ========== 保存TSV ==========
with open(output_tsv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "caption", "tempo", "energy", "pitch", "emotion"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)



print(f"[GPU {gpu_id}] Done! Saved to {output_tsv}")

