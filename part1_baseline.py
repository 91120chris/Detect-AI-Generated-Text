"""
HW2 - Detect AI Generated Text
Part 1: Data Analysis (EDA) & TF-IDF Baseline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 無顯示器環境下使用
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 載入資料 =====================
print("=" * 60)
print("載入資料集...")
df = pd.read_csv("train_v2_drcat_02.csv")
df = df[['text', 'label']].dropna()

print(f"資料總筆數: {len(df)}")
print(f"Human (label=0): {(df['label'] == 0).sum()} 筆")
print(f"AI    (label=1): {(df['label'] == 1).sum()} 筆")
print(f"類別比例 (AI%)：{df['label'].mean():.2%}")

# ===================== 2. EDA =====================
print("\n" + "=" * 60)
print("執行探索性資料分析 (EDA)...")

# 計算每筆文本的詞數
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# 計算詞彙豐富度（Type-Token Ratio）：不重複詞數 / 總詞數
def vocab_richness(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

df['vocab_richness'] = df['text'].apply(vocab_richness)

# 分組統計
human = df[df['label'] == 0]
ai    = df[df['label'] == 1]

print("\n--- 詞數統計 ---")
print(f"Human 文本 - 平均詞數: {human['word_count'].mean():.1f}, 中位數: {human['word_count'].median():.1f}, 標準差: {human['word_count'].std():.1f}")
print(f"AI    文本 - 平均詞數: {ai['word_count'].mean():.1f}, 中位數: {ai['word_count'].median():.1f}, 標準差: {ai['word_count'].std():.1f}")

print("\n--- 詞彙豐富度 (Type-Token Ratio) ---")
print(f"Human 文本 - 平均: {human['vocab_richness'].mean():.4f}, 標準差: {human['vocab_richness'].std():.4f}")
print(f"AI    文本 - 平均: {ai['vocab_richness'].mean():.4f}, 標準差: {ai['vocab_richness'].std():.4f}")

# ===================== 3. 繪製 EDA 圖表 =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Part 1 EDA: Human vs AI Text Analysis", fontsize=14, fontweight='bold')

# 圖 1：詞數分布（直方圖）
ax1 = axes[0, 0]
ax1.hist(human['word_count'], bins=50, alpha=0.6, color='steelblue', label='Human', density=True)
ax1.hist(ai['word_count'],    bins=50, alpha=0.6, color='tomato',    label='AI',    density=True)
ax1.set_title("Word Count Distribution")
ax1.set_xlabel("Word Count")
ax1.set_ylabel("Density")
ax1.legend()

# 圖 2：詞數箱型圖
ax2 = axes[0, 1]
ax2.boxplot([human['word_count'], ai['word_count']],
            labels=['Human', 'AI'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
ax2.set_title("Word Count Box Plot")
ax2.set_ylabel("Word Count")

# 圖 3：詞彙豐富度分布
ax3 = axes[1, 0]
ax3.hist(human['vocab_richness'], bins=50, alpha=0.6, color='steelblue', label='Human', density=True)
ax3.hist(ai['vocab_richness'],    bins=50, alpha=0.6, color='tomato',    label='AI',    density=True)
ax3.set_title("Vocabulary Richness (Type-Token Ratio) Distribution")
ax3.set_xlabel("TTR")
ax3.set_ylabel("Density")
ax3.legend()

# 圖 4：類別分布 Bar Chart
ax4 = axes[1, 1]
counts = df['label'].value_counts().sort_index()
ax4.bar(['Human (0)', 'AI (1)'], counts.values, color=['steelblue', 'tomato'], alpha=0.8)
for i, v in enumerate(counts.values):
    ax4.text(i, v + 100, str(v), ha='center', fontweight='bold')
ax4.set_title("Class Distribution")
ax4.set_ylabel("Sample Count")

plt.tight_layout()
plt.savefig("part1_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA 圖表已儲存為 part1_eda.png")

# ===================== 4. 切分資料 =====================
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"\n訓練集: {len(X_train)} 筆，驗證集: {len(X_val)} 筆")

# ===================== 5. TF-IDF + Logistic Regression =====================
print("\n" + "=" * 60)
print("訓練 TF-IDF + Logistic Regression Baseline...")

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

clf = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# ===================== 6. 評估 =====================
val_probs = clf.predict_proba(X_val_tfidf)[:, 1]
val_preds = clf.predict(X_val_tfidf)

auc = roc_auc_score(y_val, val_probs)
acc = accuracy_score(y_val, val_preds)

print(f"\n Baseline TF-IDF + LR 結果:")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"  Accuracy : {acc:.4f}")

# 繪製 ROC Curve
fpr, tpr, _ = roc_curve(y_val, val_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'TF-IDF LR (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - TF-IDF Baseline")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("part1_roc_curve.png", dpi=150)
plt.close()
print("ROC Curve 已儲存為 part1_roc_curve.png")

print("\n" + "=" * 60)
print(f"Part 1 完成！Baseline ROC-AUC = {auc:.4f}（BERT 必須超越此分數）")
