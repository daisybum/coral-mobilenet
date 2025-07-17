#!/usr/bin/env python3
"""
SavedModel → INT8 TFLite 변환 스크립트 (Raspberry Pi 5 LiteRT용)
출력: cls_model_int8.tflite
"""
import numpy as np
import tensorflow as tf

SAVED_DIR          = "best_model_saved"
TFLITE_OUT         = "cls_model_int8.tflite"
REP_DATASET_SIZE   = 300          # 대표 샘플 수
IMG_SIZE           = (224, 224)   # 모델 입력 크기

# ───────────────── representative 데이터셋 ─────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    label_mode=None,
    image_size=IMG_SIZE,
    batch_size=1,
    shuffle=True,
)

rep_ds = train_ds.take(REP_DATASET_SIZE)   # 300 개 배치(=샘플) 사용

def representative_gen():
    """
    LiteRT 양자화 스케일 산출용 제너레이터
    · float32, 0~1 스케일 권장
    """
    for img in rep_ds:
        yield [img.numpy().astype(np.float32) / 255.0]

# ───────────────── TFLite 변환기 설정 ────────────────────
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR)
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_gen

# 라즈베리 파이용 XNNPACK delegate가 지원하는 INT8 연산만 사용
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# ★ 입력·출력 dtype을 모두 int8로 지정 (XNNPACK 권장)
converter.inference_input_type   = tf.int8
converter.inference_output_type  = tf.int8

# ───────────────── 변환 & 저장 ───────────────────────────
tflite_model = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

print("INT8 TFLite 모델 저장 완료 →", TFLITE_OUT)
