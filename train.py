#!/usr/bin/env python3
"""
MobileNetV3-Large 7-클래스 분류 학습 스크립트
"""

import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path

# ──── 1. 설정 로드 ───────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH, "r"))

IMG_SIZE      = (cfg["img_height"], cfg["img_width"])
BATCH_SIZE    = cfg["batch_size"]
EPOCHS_HEAD   = cfg["epochs_head"]
EPOCHS_FINE   = cfg["epochs_fine"]
INITIAL_LR    = cfg["initial_lr"]
FINE_TUNE_LR  = cfg["fine_tune_lr"]
DATA_DIR      = Path(cfg["data_dir"])
UNFREEZE_FROM = cfg["unfreeze_from"]

# ──── 2. 데이터 로딩 & 내부 분할 ──────────────────────────────────
SEED         = 1337
VALTEST_RATE = 0.30          # train: 70%, 나머지 30%를 val+test로

# 1) train vs (val+test) 1차 분할
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,                     # dataset/hazy, dataset/normal, …
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALTEST_RATE,
    subset="training",
    seed=SEED,
    shuffle=True,
)
valtest_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALTEST_RATE,
    subset="validation",
    seed=SEED,
    shuffle=True,                 # 섞어서 가져온 뒤 아래에서 두 등분
)

# 2) (val+test) → val / test 2차 분할 (동일 비율: 15 %씩)
valtest_count = tf.data.experimental.cardinality(valtest_ds).numpy()
val_ds  = valtest_ds.take(valtest_count // 2)
test_ds = valtest_ds.skip(valtest_count // 2)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ──── 3. 전처리·증강 ─────────────────────────────────────────────
augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="augmentation",
)

# ──── 4. 모델 정의 ──────────────────────────────────────────────
base_model = keras.applications.MobileNetV3Large(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
    dropout_rate=0.2,
    include_preprocessing=True,
)
base_model.trainable = False  # 1단계: 헤드만 학습

inputs  = keras.Input(shape=IMG_SIZE + (3,))
x       = augment(inputs)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs, name="mnv3large_weather")

# ──── 5. 콜백 ───────────────────────────────────────────────────
ckpt_cb   = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True,
                                     monitor="val_accuracy", mode="max")
early_cb  = callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                    monitor="val_accuracy", mode="max")
reduce_cb = callbacks.ReduceLROnPlateau(factor=0.2, patience=3,
                                        monitor="val_loss", mode="min")

# ──── 6. 1단계 학습 ─────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(INITIAL_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=EPOCHS_HEAD,
    validation_data=val_ds,
    callbacks=[ckpt_cb, early_cb, reduce_cb],
)

# ──── 7. 2단계 미세조정 ────────────────────────────────────────
base_model.trainable = True
for layer in base_model.layers[:UNFREEZE_FROM]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,
    initial_epoch=model.history.epoch[-1] + 1,
    validation_data=val_ds,
    callbacks=[ckpt_cb, early_cb, reduce_cb],
)

# ──── 8. 테스트(optional) ──────────────────────────────────────
test_dir = DATA_DIR / "test"
if test_dir.exists():
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    ).prefetch(AUTOTUNE)
    loss, acc = model.evaluate(test_ds)
    print(f"\n✅ 테스트 정확도: {acc:.4f}")
