#!/usr/bin/env python3
"""
MobileNetV3-Large (minimalistic) – 테스트 & 메트릭 계산
1. 학습에 사용한 config.yaml을 그대로 불러와 동일한 전처리·분할 재현
2. best_model_saved/ 또는 best.keras를 읽어 모델 복원
3. 테스트셋 전체에 대해
   ▪︎ 정확도(Accuracy)
   ▪︎ 클래스별 Precision / Recall / F1
   ▪︎ 클래스별 정확도
   ▪︎ Confusion Matrix
4. 결과를 콘솔 출력 (추후 CSV·그래프 저장도 쉽게 확장 가능)
"""

import os, yaml, collections
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from tqdm import tqdm

# ── 0. 설정 로드 ────────────────────────────────────────────────
# ── 0. 설정 ─────────────────────────────────────────────
CFG = yaml.safe_load(open("config.yaml"))
IMG = (CFG["img_height"], CFG["img_width"])
IMG_SIZE   = (CFG["img_height"], CFG["img_width"])
BATCH = CFG["batch_size"]; SEED = CFG["seed"]
VAL_RATE = CFG["valtest_split"]; DATA = Path(CFG["data_dir"])
E_HEAD, E_FINE = CFG["epochs_head"], CFG["epochs_fine"]
LR_HEAD, LR_FINE = CFG["initial_lr"], CFG["fine_tune_lr"]
UNFREEZE_AT = CFG["unfreeze_from"]; ALPHA = CFG["depth_multiplier"]

tf.keras.utils.set_random_seed(SEED)

tf.keras.utils.set_random_seed(SEED)

# ── 1. 테스트셋 재구성 (학습 스크립트와 동일한 분할 방식) ───────
train_ds = keras.utils.image_dataset_from_directory(
    DATA, label_mode="int", image_size=IMG, batch_size=BATCH,
    validation_split=VAL_RATE, subset="training", seed=SEED)
valtest_ds = keras.utils.image_dataset_from_directory(
    DATA, label_mode="int", image_size=IMG, batch_size=BATCH,
    validation_split=VAL_RATE, subset="validation", seed=SEED)
val_cnt = tf.data.experimental.cardinality(valtest_ds).numpy() // 2
val_ds  = valtest_ds.take(val_cnt); test_ds = valtest_ds.skip(val_cnt)

CLASSES = train_ds.class_names; N = len(CLASSES)
N       = len(CLASSES)
print("클래스:", CLASSES)

prep = keras.applications.mobilenet_v3.preprocess_input
test_ds = test_ds.map(lambda x,y: (prep(x), y)).prefetch(tf.data.AUTOTUNE)

# ── 2. 모델 복원 ────────────────────────────────────────────────
MODEL_DIR = Path("best_model_saved")     # SavedModel 디렉터리
WEIGHT_KR = Path("best.keras")           # 가중치 파일

if MODEL_DIR.exists():
    model = keras.models.load_model(MODEL_DIR)
elif WEIGHT_KR.exists():
    # 학습 스크립트와 동일 구조로 모델 재정의 후 가중치 로드
    base = keras.applications.MobileNetV3Large(
        input_shape=IMG_SIZE+(3,), include_top=False, minimalistic=True,
        alpha=CFG["depth_multiplier"], weights=None
    )
    inputs = keras.Input(shape=IMG_SIZE+(3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(N, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.load_weights(WEIGHT_KR)
else:
    raise FileNotFoundError("복원할 모델이 없습니다.")

print("✅ 모델 로드 완료")

# ── 3. 예측 수행 ────────────────────────────────────────────────
y_true, y_pred = [], []
for batch_x, batch_y in tqdm(test_ds):
    preds = model.predict(batch_x, verbose=0)
    y_true.extend(batch_y.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ── 4-A. 전체 정확도 ───────────────────────────────────────────
acc = (y_true == y_pred).mean()
print(f"\n📊  Test Accuracy: {acc*100:5.2f}%")

# ── 4-B. 클래스별 Precision / Recall / F1 ─────────────────────
report = classification_report(
    y_true, y_pred, target_names=CLASSES, digits=4
)
print("\n=== Classification Report ===")
print(report)

# ── 4-C. 클래스별 정확도만 별도 계산 ───────────────────────────
cls_correct = collections.Counter()
cls_total   = collections.Counter()
for t, p in zip(y_true, y_pred):
    cls_total[int(t)]   += 1
    if t == p:
        cls_correct[int(t)] += 1
print("=== Per-class Accuracy ===")
for i, name in enumerate(CLASSES):
    acc_i = cls_correct[i] / cls_total[i]
    print(f"{name:<25}: {acc_i*100:5.2f}%  ({cls_correct[i]}/{cls_total[i]})")

# ── 4-D. Confusion Matrix ─────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=range(N))
print("\n=== Confusion Matrix (raw counts) ===")
print(cm)

# 필요 시: np.savetxt("confusion_matrix.csv", cm, fmt="%d", delimiter=",")
