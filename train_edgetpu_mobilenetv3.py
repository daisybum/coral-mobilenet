#!/usr/bin/env python3
"""
MobileNetV3-Large (minimalistic) EdgeTPU 호환 학습 스크립트
1) feature extractor(헤드) 학습 → 2) fine-tune
최종 SavedModel 및 H5 가중치 저장
"""
import os, yaml, time, math, random
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ───────────────── 설정 로드 ───────────────────────────
CFG = yaml.safe_load(open("config.yaml", "r"))

IMG_SIZE   = (CFG["img_height"], CFG["img_width"])
BATCH      = CFG["batch_size"]
SEED       = CFG["seed"]
VAL_RATE   = CFG["valtest_split"]
DATA_DIR   = Path(CFG["data_dir"])

EPOCHS_HEAD = CFG["epochs_head"]
EPOCHS_FINE = CFG["epochs_fine"]
LR_HEAD     = CFG["initial_lr"]
LR_FINE     = CFG["fine_tune_lr"]
UNFREEZE_AT = CFG["unfreeze_from"]
ALPHA       = CFG["depth_multiplier"]

# reproducibility
tf.keras.utils.set_random_seed(SEED)

# ───────────────── 데이터셋 로드 ───────────────────────
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    validation_split=VAL_RATE,
    subset="training",
    seed=SEED,
)

valtest_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    validation_split=VAL_RATE,
    subset="validation",
    seed=SEED,
)

val_cnt = tf.data.experimental.cardinality(valtest_ds).numpy() // 2
val_ds  = valtest_ds.take(val_cnt)
test_ds = valtest_ds.skip(val_cnt)

CLASS_NAMES = train_ds.class_names
N_CLASSES   = len(CLASS_NAMES)
print("클래스:", CLASS_NAMES)

# ───────────────── 전처리 & 증강 ───────────────────────
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),        # [0,1]
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# 성능 최적화
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ───────────────── 모델 정의 ───────────────────────────
base_model = keras.applications.MobileNetV3Large(
    input_shape=IMG_SIZE + (3,),
    alpha=ALPHA,
    include_top=False,
    minimalistic=True,      # EdgeTPU 호환 핵심 옵션
    weights="imagenet",
)
base_model.trainable = False           # 1단계: feature extractor

# 분류 헤드
inputs  = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)          # 추론 시에도 정규화 일관성 유지
x = keras.applications.mobilenet_v3.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(N_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(LR_HEAD),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# ───────────────── 콜백 ────────────────────────────────
ckpt_cb = callbacks.ModelCheckpoint(
    "ckpt_head.keras", monitor="val_accuracy",
    save_best_only=True, save_weights_only=True
)
es_cb   = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# ───────────────── 1단계 학습 ──────────────────────────
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[ckpt_cb, es_cb],
)

# ───────────────── 2단계 fine-tune ─────────────────────
base_model.trainable = True
for layer in base_model.layers[:UNFREEZE_AT]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_FINE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
ckpt_cb_fine = callbacks.ModelCheckpoint(
    "ckpt_finetune.keras", monitor="val_accuracy",
    save_best_only=True, save_weights_only=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[ckpt_cb_fine, es_cb],
)

# ───────────────── 평가 & 저장 ─────────────────────────
print("테스트 정확도:")
model.evaluate(test_ds)

model.save("saved_model_edgetpu")      # SavedModel
model.save("edgetpu_final.h5")         # 가중치별도

print("SavedModel 경로: saved_model_edgetpu")
