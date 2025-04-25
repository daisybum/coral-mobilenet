#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QAT-MobileNetV3-Large 7-클래스 학습 스크립트
(1) 헤드 학습 → (2) 미세조정(QAT 포함)
"""

import yaml, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
import tensorflow_model_optimization as tfmot

# ─── 1. 설정 로드 ─────────────────────────────────────────
cfg = yaml.safe_load(open("config.yaml"))
IMG_SIZE      = (cfg["img_height"], cfg["img_width"])
BATCH_SIZE    = cfg["batch_size"]
EPOCHS_HEAD   = cfg["epochs_head"]
EPOCHS_FINE   = cfg["epochs_fine"]
INIT_LR       = cfg["initial_lr"]
FINE_TUNE_LR  = cfg["fine_tune_lr"]
DATA_DIR      = Path(cfg["data_dir"])
UNFREEZE_FROM = cfg["unfreeze_from"]
SEED          = 1337
VALTEST_RATE  = 0.30

# ─── 2. 데이터셋 분할 ─────────────────────────────────────
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, label_mode="categorical", image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, validation_split=VALTEST_RATE,
    subset="training", seed=SEED, shuffle=True)
valtest_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, label_mode="categorical", image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, validation_split=VALTEST_RATE,
    subset="validation", seed=SEED, shuffle=True)

val_cnt = tf.data.experimental.cardinality(valtest_ds).numpy()
val_ds  = valtest_ds.take(val_cnt // 2)
test_ds = valtest_ds.skip(val_cnt // 2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds, val_ds = train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)

NUM_CLASSES = len(train_ds.class_names)

# ─── 3. 데이터 증강 + 정규화(-1~1) ─────────────────────────
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augment")

# ─── 4. 베이스 모델 ─ minimalistic MobileNetV3 ─────────────
base = keras.applications.MobileNetV3Large(
    input_shape=IMG_SIZE + (3,),
    minimalistic=True,          # Edge-TPU 친화
    include_top=False,
    weights="imagenet",
    include_preprocessing=False,
    dropout_rate=0.0)
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = layers.Rescaling(scale=1/127.5, offset=-1)(inputs)  # [-1,1]
x = augment(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_fp32 = keras.Model(inputs, outputs, name="mnv3_qat")

# ─── 5. 콜백 ───────────────────────────────────────────────
ckpt = callbacks.ModelCheckpoint("models/best_model.h5", save_best_only=True,
                                 monitor="val_accuracy", mode="max")
early = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                monitor="val_accuracy", mode="max")
reduce = callbacks.ReduceLROnPlateau(factor=0.2, patience=3,
                                     monitor="val_loss", mode="min")

# ─── 6. 1단계(헤드) 학습 ───────────────────────────────────
model_fp32.compile(optimizer=keras.optimizers.Adam(INIT_LR),
                   loss="categorical_crossentropy", metrics=["accuracy"])
model_fp32.fit(train_ds, epochs=EPOCHS_HEAD, validation_data=val_ds,
               callbacks=[ckpt, early, reduce])

# ─── 7. 2단계(QAT 미세조정) ───────────────────────────────
base.trainable = True
for l in base.layers[:UNFREEZE_FROM]:
    l.trainable = False

qat_model = tfmot.quantization.keras.quantize_model(model_fp32)

qat_model.compile(optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
qat_model.fit(train_ds, epochs=EPOCHS_HEAD + EPOCHS_FINE,
              initial_epoch=qat_model.history.epoch[-1] + 1
              if qat_model.history.epoch else EPOCHS_HEAD,
              validation_data=val_ds,
              callbacks=[ckpt, early, reduce])

# ─── 8. 선택적 테스트 ──────────────────────────────────────
test_dir = DATA_DIR / "test"
if test_dir.exists():
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, label_mode="categorical", image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=False).prefetch(AUTOTUNE)
    _, acc = qat_model.evaluate(test_ds)
    print(f"✅ 테스트 정확도: {acc:.4f}")
