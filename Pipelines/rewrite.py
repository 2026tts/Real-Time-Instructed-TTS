import os
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ======================
# argument parser (for split list)
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--list", type=str, required=True, help="Path to split txt list")
parser.add_argument("--output", type=str, required=False, default="rewritten_output", help="Output folder")
args = parser.parse_args()

list_path = args.list
output_folder = args.output
os.makedirs(output_folder, exist_ok=True)

# ======================
# Model Load (per GPU process)
# ======================
model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading model on GPU: {torch.cuda.current_device()} ...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)
model.eval()
torch.set_grad_enabled(False)

base_prompt = """
You will be given a single-sentence speaking-style caption. It already describes
ONLY the speaking delivery characteristics (tempo, emotion, pitch, and energy).
Rewrite it into SIX new one-sentence captions with clearly distinct styles.
Do NOT change the speaking-style attributes (tempo, emotion, pitch, energy).
Only rephrase.

============================================================
REQUIRED SIX OUTPUTS (STRICT)
============================================================

(1) Natural-English
    - Objective, neutral, factual.
    - No metaphors, no imagery.

(2) Natural-Chinese
    - Same style as (1), but in Chinese.

(3) Expressive-English
    - Uses imagery / metaphors / emotional shading.
    - Must NOT resemble (1).

(4) Expressive-Chinese
    - Expressive, metaphorical Chinese.
    - Must NOT resemble (2).

(5) Label-English
    - MUST be 4 comma-separated labels:
        tempo (slow/medium/fast)
        emotion (happy/sad/angry/fearful/surprised/disgusted/contempt)
        pitch (low/medium/high/trembling/stable)
        energy (low/medium/high)
    - No full sentences.

(6) Label-Chinese
    - MUST be 4 comma-separated Chinese labels:
        速度: 慢/中等/快
        情绪: 开心/悲伤/愤怒/害怕/惊讶/厌恶/轻蔑
        音调: 低/中/高/颤音/稳定
        能量: 低/中/高
    - No full sentences.

============================================================
EXAMPLE
============================================================

Example Input Caption:
His slow, trembling delivery reveals low energy and restrained sadness.

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
NOW PROCESS THIS INPUT CAPTION:
"""


# ======================
# Regex
# ======================
pattern_caption = re.compile(r"caption_4:\s*(.*)")
pattern_wav = re.compile(r"wav:\s*(.*)")
pattern_header = re.compile(r"^\[(\d)\]\s*([A-Za-z\-]+):\s*$")

# ======================
# Read split list
# ======================
with open(list_path, "r", encoding="utf-8") as f:
    files_to_process = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(files_to_process)} files from {list_path}")

# ======================
# Process each txt file
# ======================
for input_path in tqdm(files_to_process, desc="Processing"):

    fname = os.path.basename(input_path)
    output_path = os.path.join(output_folder, fname)

    # read txt
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # parse wav
    wav_match = pattern_wav.search(text)
    wav_path = wav_match.group(1).strip() if wav_match else ""

    # parse caption_4
    cap_match = pattern_caption.search(text)
    caption = cap_match.group(1).strip() if cap_match else ""

   

    full_prompt = base_prompt + caption + "\n\nOutput ONLY the six sentences."

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": full_prompt}
    ]

    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    output_text = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()
    #print(output_text)
    lines = output_text.split("\n")
    # parse final 6 outputs
    extracted = [""] * 6
    current_idx = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Step 1: detect header "[1] Natural-English:"
        m = pattern_header.match(line)
        if m:
            current_idx = int(m.group(1)) - 1
            continue

        # Step 2: the next non-empty line after header is the actual content
        if current_idx is not None:
            extracted[current_idx] = line
            current_idx = None
    #print(extracted)

    # ======================
    # write output txt
    # ======================
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"wav: {wav_path}\n\n")
        f.write(f"[1] Natural-English:\n{extracted[0]}\n\n")
        f.write(f"[2] Natural-Chinese:\n{extracted[1]}\n\n")
        f.write(f"[3] Expressive-English:\n{extracted[2]}\n\n")
        f.write(f"[4] Expressive-Chinese:\n{extracted[3]}\n\n")
        f.write(f"[5] Label-English:\n{extracted[4]}\n\n")
        f.write(f"[6] Label-Chinese:\n{extracted[5]}\n\n")


        # generated 6 captions
        
print("Done! Output saved to:", output_folder)
