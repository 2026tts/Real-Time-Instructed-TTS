
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

list_path = sys.argv[1]  # 比如 filelist_0.txt
gpu_id = sys.argv[2]     # 比如 0

# ----------- 配置路径 -----------
input_dir = "../wenetspeech4tts"      # 原始 txt 文件夹
output_dir = "../wenetspeech4tts/short_caption_diverse"              # 输出结果文件夹
os.makedirs(output_dir, exist_ok=True)

# ----------- 加载模型 -----------
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# ----------- Prompt 模板 -----------
base_prompt = """
You will be given a long description of a speech recording. Your task is to
rewrite it into SIX different one-sentence speaking-style descriptions, each
corresponding to a distinct format and language. All outputs must describe ONLY:

- tempo (slow / medium / fast)
- emotion (happy / sad / angry / surprised / fear / disgusted / contempt)
- pitch (low / medium / high / trembling / stable)
- energy (low / medium / high)

Do NOT mention:
- content of the speech
- environment, background noise, microphone quality
- speaker identity, age, gender, region, appearance
- body language or psychology
- quoted text or semantic content

============================================================
REQUIRED OUTPUT SET (all six must be produced):
============================================================

(1) Natural Descriptive Sentence — English  
(2) Natural Descriptive Sentence — Chinese  
(3) Freer Expressive Phrasing — English  
(4) Freer Expressive Phrasing — Chinese  
(5) Label-like Keyword Combination — English  
(6) Label-like Keyword Combination — Chinese

Rules:
- Each sentence must be ONE single sentence.
- Each style must be noticeably different.
- Avoid template reuse.
- Output only the six sentences.
- Use the exact format:

[1] ...
[2] ...
[3] ...
[4] ...
[5] ...
[6] ...

============================================================
FEW-SHOT EXAMPLE
============================================================

Example input description:
The audio clip begins with a brief, high-pitched, non-verbal vocalization—
a soft exhalation or sigh—indicative of the speaker preparing to speak.
Immediately following this, an older male voice, characterized by a gentle,
weary, and slightly raspy timbre, enters and speaks in clear Mandarin.
His speech is slow, deliberate, and emotionally heavy, with vocal tremor
and a sense of resignation. He recounts a personal conflict, speaking with
melancholy and vulnerability, ending with a low, fatigued sigh. The room is
quiet, private, and close-miked with no ambient noise.

Example Output:

[1] Natural-English:
His voice moves slowly with a weary, trembling tone and very low energy, carrying a sense of restrained sadness.

[2] Natural-Chinese:
他的语速缓慢、情绪低沉，声线微微颤抖，整体能量弱而带着无奈。

[3] Expressive-English:
A weary, trembling voice drifts forward in a slow, heavy rhythm, as if each word sinks under quiet sorrow.

[4] Expressive-Chinese:
那声音缓慢而沉重，带着微颤的无力感，像是被压在深深的悲伤里。

[5] Label-English:
slow, sad, trembling pitch, low-energy

[6] Label-Chinese:
慢、悲伤、颤音、低能量

============================================================
NOW PROCESS THE FOLLOWING DESCRIPTION:
============================================================
"""

# ----------- 读取并处理文件 -----------
with open(list_path, "r", encoding="utf-8") as f:
    files = [line.strip() for line in f if line.strip()]

for filename in tqdm(files, desc=f"GPU {gpu_id}"):
    name = filename.split("/")[-1]
    input_path = filename
    output_path = os.path.join(output_dir, name)



    if os.path.exists(output_path):
        print(filename)
        print(output_path)
        continue

    # 读取 description
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            description = f.read().strip()
    except UnicodeDecodeError:
        print(f"[SKIP] Cannot decode file (not UTF-8): {input_path}")
        continue
    except Exception as e:
        print(f"[SKIP] Error reading {input_path}: {e}")
        continue

    # 构造 prompt
    full_prompt = f"{base_prompt}\n\nDescription:\n{description}"

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": full_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 推理
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 去掉输入部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    #output_path_new = os.path.join(output_dir, name)

    # 保存输出
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response)

    #print(f"✅ Processed: {filename}")

print("🎉 All files processed successfully!")


