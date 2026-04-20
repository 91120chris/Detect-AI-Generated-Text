"""
HW2 - Detect AI Generated Text
Part 3: Adversarial Attack with Local LLM (Mistral-7B)
Hardware: NVIDIA RTX 4090 (24GB VRAM)

前置條件：
  - Part 2 已完成，saved_model_bert-large-cased（或 bert-base-cased）目錄存在
"""

import os
import gc
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import warnings
warnings.filterwarnings('ignore')

# ===================== 設定 =====================
DETECTOR_PATH = (
    "./saved_model_bert-large-cased"
    if os.path.exists("./saved_model_bert-large-cased")
    else "./saved_model_bert-base-cased"
)

# 生成模型選擇：
GEN_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

NUM_SAMPLES = 10   # 選取 5~10 筆 human 樣本
MAX_LEN     = 512
SEED        = 42

# ===================== 1. 清理 VRAM =====================
gc.collect()
torch.cuda.empty_cache()
print(f"使用偵測器模型: {DETECTOR_PATH}")
print(f"使用生成模型:   {GEN_MODEL_NAME}")

# ===================== 2. 載入並選取 Human 樣本 =====================
print("\n載入資料集，挑選 Human 驗證樣本...")
df = pd.read_csv("train_v2_drcat_02.csv")
df = df[['text', 'label']].dropna()

_, X_val, _, y_val = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=SEED, stratify=df['label']
)

val_df  = pd.DataFrame({'text': X_val.values, 'label': y_val.values})
human_val = val_df[val_df['label'] == 0].reset_index(drop=True)

# 挑選長度適中的樣本（避免過短無意義或過長超出限制）
human_val['word_count'] = human_val['text'].apply(lambda x: len(str(x).split()))
human_samples = (
    human_val[(human_val['word_count'] >= 100) & (human_val['word_count'] <= 600)]
    .sample(n=NUM_SAMPLES, random_state=SEED)
    .reset_index(drop=True)
)
print(f"已選取 {len(human_samples)} 筆 Human 樣本（100~600 詞）")

# ===================== 3. 載入生成 LLM =====================
print(f"\n載入生成模型 {GEN_MODEL_NAME}（fp16，device_map=auto）...")
generator = pipeline(
    "text-generation",
    model=GEN_MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)
print("生成模型載入完成！")

# ===================== 4. 攻擊提示詞設計 =====================
# 使用不同風格的提示詞（增加多樣性）
PROMPTS = [
    "Rewrite the following essay to make it sound like it was written by a high school student. Keep the same ideas but use simpler vocabulary, shorter sentences, and a more casual tone:\n\n{text}",
    "Revise the essay below to sound more personal and emotional, as if a teenager wrote it for a class assignment. Use first-person perspective naturally:\n\n{text}",
    "Rewrite the following essay with more colloquial expressions, some minor grammatical imperfections, and a less structured format—like a real student wrote it:\n\n{text}",
]

def build_prompt(text: str, style_idx: int) -> list:
    """建構 chat-style prompt"""
    user_content = PROMPTS[style_idx % len(PROMPTS)].format(text=text[:1500])  # 截斷避免超長
    return [
        {"role": "system", "content": "You are a helpful writing assistant."},
        {"role": "user",   "content": user_content},
    ]

# ===================== 5. 生成攻擊樣本 =====================
print(f"\n開始生成 {NUM_SAMPLES} 筆攻擊樣本...")
attack_texts = []

for i, row in human_samples.iterrows():
    print(f"  生成第 {i+1}/{NUM_SAMPLES} 筆...", end=" ", flush=True)
    prompt_msgs = build_prompt(row['text'], style_idx=i)

    try:
        output = generator(
            prompt_msgs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        generated = output[0]['generated_text'][-1]['content'].strip()
        print(f"完成（{len(generated.split())} 詞）")
    except Exception as e:
        print(f"失敗: {e}")
        generated = row['text']  # fallback 使用原文

    attack_texts.append(generated)

# 清理生成模型的 VRAM
del generator
gc.collect()
torch.cuda.empty_cache()
print("生成模型已釋放 VRAM")

# ===================== 6. 載入 BERT 偵測器 =====================
print(f"\n載入 BERT 偵測器: {DETECTOR_PATH}")
detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to("cuda")
detector.eval()

def predict_ai_prob(text: str) -> float:
    """回傳文本被判定為 AI 的機率"""
    inputs = detector_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to("cuda")
    with torch.no_grad():
        logits = detector(**inputs).logits
        probs  = torch.softmax(logits, dim=1)
    return probs[0][1].item()  # label=1 (AI) 的機率

# ===================== 7. 偵測原文 vs 攻擊樣本 =====================
print("\n開始偵測原文與攻擊樣本...")
print("=" * 70)

records = []
for i in range(NUM_SAMPLES):
    orig_text   = human_samples.loc[i, 'text']
    attack_text = attack_texts[i]

    prob_orig   = predict_ai_prob(orig_text)
    prob_attack = predict_ai_prob(attack_text)

    label_orig   = "AI-Generated" if prob_orig   > 0.5 else "Human-Written"
    label_attack = "AI-Generated" if prob_attack > 0.5 else "Human-Written"

    fooled = (prob_attack <= 0.5)  # 被騙判定為 Human

    records.append({
        "sample_id":      i + 1,
        "orig_ai_prob":   prob_orig,
        "attack_ai_prob": prob_attack,
        "orig_label":     label_orig,
        "attack_label":   label_attack,
        "fooled":         fooled,
        "orig_snippet":   orig_text[:200],
        "attack_snippet": attack_text[:200],
    })

    print(f"樣本 {i+1:02d}:")
    print(f"  原文    → {label_orig:<15} (AI prob: {prob_orig:.4f})")
    print(f"  攻擊後  → {label_attack:<15} (AI prob: {prob_attack:.4f}) {'★ FOOLED!' if fooled else ''}")
    print()

# ===================== 8. 結果統計 =====================
results_df = pd.DataFrame(records)
fooled_count = results_df['fooled'].sum()

print("=" * 70)
print("Part 3 結果統計")
print("=" * 70)
print(f"攻擊成功（判為 Human）: {fooled_count} / {NUM_SAMPLES}  ({fooled_count/NUM_SAMPLES:.0%})")
print(f"攻擊失敗（仍判為 AI）: {NUM_SAMPLES - fooled_count} / {NUM_SAMPLES}")
print(f"\n原文 AI 機率    平均: {results_df['orig_ai_prob'].mean():.4f}")
print(f"攻擊後 AI 機率  平均: {results_df['attack_ai_prob'].mean():.4f}")
print(f"機率變化        平均: {(results_df['attack_ai_prob'] - results_df['orig_ai_prob']).mean():+.4f}")

# ===================== 9. 儲存結果與圖表 =====================
# 儲存詳細結果到 JSON（報告用）
output_records = []
for i, row in results_df.iterrows():
    output_records.append({
        "sample_id":      int(row['sample_id']),
        "original_text":  human_samples.loc[i, 'text'],
        "attack_text":    attack_texts[i],
        "orig_ai_prob":   round(float(row['orig_ai_prob']), 4),
        "attack_ai_prob": round(float(row['attack_ai_prob']), 4),
        "fooled":         bool(row['fooled']),
    })

with open("part3_results.json", "w", encoding="utf-8") as f:
    json.dump(output_records, f, ensure_ascii=False, indent=2)
print("\n詳細結果已儲存為 part3_results.json")

# 繪製 AI 機率對比圖
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(NUM_SAMPLES)
width = 0.35

bars1 = ax.bar(x - width/2, results_df['orig_ai_prob'],   width, label='Original (Human)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, results_df['attack_ai_prob'], width, label='After LLM Attack',  color='tomato',   alpha=0.8)

ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.2, label='Decision Threshold (0.5)')
ax.set_xlabel("Sample ID")
ax.set_ylabel("AI Probability (BERT Detector)")
ax.set_title("Part 3: BERT Detector Score — Original vs Adversarial Samples")
ax.set_xticks(x)
ax.set_xticklabels([f"S{i+1}" for i in range(NUM_SAMPLES)])
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)

# 在 bar 上標示數值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("part3_adversarial_results.png", dpi=150)
plt.close()
print("對比圖已儲存為 part3_adversarial_results.png")

print("\nPart 3 完成！")
