"""
HW2 - Detect AI Generated Text
Part 2: BERT Fine-Tuning & Scaling (bert-base-cased vs bert-large-cased)
Hardware: NVIDIA RTX 4090 (24GB VRAM)
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# ===================== 設定 =====================
MAX_LEN    = 512
EPOCHS     = 3
SEED       = 42

# 兩個模型的批次大小：4090 的 24GB VRAM
#   bert-base-cased  (110M)  → batch 32
#   bert-large-cased (340M)  → batch 16 + grad_accum 2 (模擬等效 batch 32)
MODEL_CONFIGS = [
    {"name": "bert-base-cased",  "batch_size": 32, "grad_accum": 1},   # effective batch = 32
    {"name": "bert-large-cased", "batch_size": 16, "grad_accum": 2},   # effective batch = 32
]

# ===================== 1. 載入並切分資料 =====================
print("=" * 60)
print("載入資料集...")
df = pd.read_csv("train_v2_drcat_02.csv")
df = df[['text', 'label']].dropna()

X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=SEED, stratify=df['label']
)

train_df = pd.DataFrame({'text': X_train.values, 'label': y_train.values})
val_df   = pd.DataFrame({'text': X_val.values,   'label': y_val.values})

print(f"訓練集: {len(train_df)} 筆，驗證集: {len(val_df)} 筆")

# ===================== 2. 自訂評估函數 =====================
def compute_metrics(eval_pred):
    """計算 ROC-AUC（Trainer 呼叫）"""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    auc = roc_auc_score(labels, probs)
    return {"roc_auc": auc}

# ===================== 3. 訓練函數 =====================
def train_bert(model_name: str, batch_size: int, grad_accum: int = 1):
    print(f"\n{'=' * 60}")
    print(f"開始訓練: {model_name}  (batch={batch_size}, grad_accum={grad_accum}, effective_batch={batch_size*grad_accum}, epochs={EPOCHS})")
    print(f"{'=' * 60}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    hf_train = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    hf_val   = Dataset.from_pandas(val_df).map(tokenize_fn,   batched=True)

    # 移除不需要的欄位
    hf_train = hf_train.remove_columns(["text"])
    hf_val   = hf_val.remove_columns(["text"])
    hf_train.set_format("torch")
    hf_val.set_format("torch")

    # 模型
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    output_dir = f"./results_{model_name.replace('/', '_')}"
    save_dir   = f"./saved_model_{model_name.replace('/', '_')}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",        # 每 epoch 評估
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        gradient_accumulation_steps=grad_accum,  # 模擬等效 batch size
        fp16=True,                    # RTX 4090 混合精度（加速 + 省 VRAM）
        logging_steps=100,
        report_to="none",
        seed=SEED,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # 取最佳模型結果
    eval_results = trainer.evaluate()
    best_auc = eval_results.get("eval_roc_auc", 0.0)
    print(f"\n{model_name} 最終 ROC-AUC: {best_auc:.4f}")

    # 儲存模型
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"模型已儲存至 {save_dir}")

    # 提取訓練 log（Loss 曲線）
    logs = trainer.state.log_history
    train_loss = [(e['epoch'], e['loss'])        for e in logs if 'loss'      in e and 'eval_loss' not in e]
    eval_loss  = [(e['epoch'], e['eval_loss'])   for e in logs if 'eval_loss' in e]
    eval_auc   = [(e['epoch'], e['eval_roc_auc'])for e in logs if 'eval_roc_auc' in e]

    return best_auc, train_loss, eval_loss, eval_auc, model_name

# ===================== 4. 依序訓練兩個模型 =====================
results = []
all_logs = {}

for cfg in MODEL_CONFIGS:
    best_auc, train_loss, eval_loss, eval_auc, name = train_bert(cfg["name"], cfg["batch_size"], cfg["grad_accum"])
    results.append({"model": name, "roc_auc": best_auc})
    all_logs[name] = {
        "train_loss": train_loss,
        "eval_loss":  eval_loss,
        "eval_auc":   eval_auc,
    }

# ===================== 5. 繪製 Loss & AUC 曲線 =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['steelblue', 'tomato']

for idx, cfg in enumerate(MODEL_CONFIGS):
    name = cfg["name"]
    log  = all_logs[name]
    color = colors[idx]
    label = "Base" if "base" in name else "Large"

    if log["train_loss"]:
        epochs_tr, losses_tr = zip(*log["train_loss"])
        axes[0].plot(epochs_tr, losses_tr, label=f"{label} Train", color=color, linestyle='--', marker='o', markersize=3)
    if log["eval_loss"]:
        epochs_ev, losses_ev = zip(*log["eval_loss"])
        axes[0].plot(epochs_ev, losses_ev, label=f"{label} Val",   color=color, linestyle='-',  marker='s')
    if log["eval_auc"]:
        epochs_auc, aucs = zip(*log["eval_auc"])
        axes[1].plot(epochs_auc, aucs, label=f"{label}", color=color, marker='s')

axes[0].set_title("Training & Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation ROC-AUC per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("ROC-AUC")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Part 2: BERT Base vs Large - Training Curves", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("part2_training_curves.png", dpi=150)
plt.close()
print("\nLoss & AUC 曲線已儲存為 part2_training_curves.png")

# ===================== 6. 結果摘要 =====================
print("\n" + "=" * 60)
print("Part 2 結果摘要")
print("=" * 60)
print(f"{'Model':<30} {'ROC-AUC':>10}")
print("-" * 42)
for r in results:
    print(f"{r['model']:<30} {r['roc_auc']:>10.4f}")

if len(results) == 2:
    diff = results[1]['roc_auc'] - results[0]['roc_auc']
    print(f"\nbert-large vs bert-base 差異: {diff:+.4f}")
    if diff > 0.01:
        print("結論：Large 模型明顯優於 Base 模型（差距 > 0.01）")
    else:
        print("結論：兩模型表現相近，Scale-up 效益有限（此任務特性使然）")
