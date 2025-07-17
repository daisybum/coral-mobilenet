#!/usr/bin/env python3
"""
MobileNetV3-Large (minimalistic) – EdgeTPU 학습 스크립트 v2
* 이중 스케일 문제 제거
* 클래스 불균형 대응 (class_weight)
* Dropout + CosineDecayLR 적용
"""
import os, yaml, collections
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ── 0. 설정 ─────────────────────────────────────────────
CFG = yaml.safe_load(open("config.yaml"))
IMG = (CFG["img_height"], CFG["img_width"])
BATCH = CFG["batch_size"]; SEED = CFG["seed"]
VAL_RATE = CFG["valtest_split"]; DATA = Path(CFG["data_dir"])
E_HEAD, E_FINE = CFG["epochs_head"], CFG["epochs_fine"]
LR_HEAD, LR_FINE = CFG["initial_lr"], CFG["fine_tune_lr"]
UNFREEZE_AT = CFG["unfreeze_from"]; ALPHA = CFG["depth_multiplier"]

tf.keras.utils.set_random_seed(SEED)

# ── 1. 데이터셋 ────────────────────────────────────────
train_ds = keras.utils.image_dataset_from_directory(
    DATA, label_mode="int", image_size=IMG, batch_size=BATCH,
    validation_split=VAL_RATE, subset="training", seed=SEED)
valtest_ds = keras.utils.image_dataset_from_directory(
    DATA, label_mode="int", image_size=IMG, batch_size=BATCH,
    validation_split=VAL_RATE, subset="validation", seed=SEED)
val_cnt = tf.data.experimental.cardinality(valtest_ds).numpy() // 2
val_ds  = valtest_ds.take(val_cnt); test_ds = valtest_ds.skip(val_cnt)

CLASSES = train_ds.class_names; N = len(CLASSES)
print("클래스:", CLASSES)

# ── 1-b. 클래스 가중치 계산 ────────────────────────────
label_counts = collections.Counter()
for _, y in train_ds.unbatch(): label_counts[int(y)] += 1
total = sum(label_counts.values())
class_weight = {i: total/(N*cnt) for i, cnt in label_counts.items()}
print("클래스 가중치:", class_weight)

# ── 2. 증강 (Rescaling 제거!) ──────────────────────────
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

prep = keras.applications.mobilenet_v3.preprocess_input

def preprocess(x, y):
    x = augment(x, training=True)
    x = prep(x)          # 입력을 [-1,1] 로 스케일 (단일 단계)
    return x, y

train_ds = train_ds.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(lambda x,y:(prep(x),y)).cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(lambda x,y:(prep(x),y)).cache().prefetch(tf.data.AUTOTUNE)

# ── 3. 모델 ────────────────────────────────────────────
base = keras.applications.MobileNetV3Large(
    input_shape=IMG+(3,), include_top=False, minimalistic=True,
    alpha=ALPHA, weights="imagenet")
base.trainable = False

inputs = keras.Input(shape=IMG+(3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)              # 규제 추가
outputs = layers.Dense(N, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# ── 4. 콜백 & 학습 스케줄 ──────────────────────────────
schedule_h = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_HEAD, decay_steps=len(train_ds)*E_HEAD)
schedule_f = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_FINE, decay_steps=len(train_ds)*E_FINE)

ckpt = callbacks.ModelCheckpoint(
    "best.keras",             # ← 최고 가중치 (weights-only)
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)
es   = callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)

# ── 5-1. 헤드 학습 ────────────────────────────────────
model.compile(optimizer=keras.optimizers.Adam(schedule_h),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=E_HEAD,
          class_weight=class_weight, callbacks=[ckpt, es])

# ── 5-2. 미세조정 ─────────────────────────────────────
base.trainable = True
for l in base.layers[:UNFREEZE_AT]:
    l.trainable = False

model.compile(optimizer=keras.optimizers.Adam(schedule_f),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=E_FINE,
          class_weight=class_weight, callbacks=[ckpt, es])

print("테스트 성능:")
model.evaluate(test_ds)

model.save("saved_model_edgetpu")

# ── 6. 최고 모델 저장 ─────────────────────────────────
# best.keras → 모델 구조에 주입 후 SavedModel 로 보관
model.load_weights("best.keras")          # 최고 성능 가중치 불러오기
model.save("best_model_saved")            # 완전한 SavedModel 디렉터리
print("✔️  최고 성능 모델 SavedModel 로 저장: best_model_saved/")
