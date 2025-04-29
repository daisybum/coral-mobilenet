#!/usr/bin/env python3
"""
SavedModel → 풀 INT8 양자화 TFLite 변환 스크립트
출력: model_quant.tflite
"""
import numpy as np
import tensorflow as tf

SAVED_DIR = "saved_model_edgetpu"
TFLITE_OUT = "model_quant.tflite"
REP_DATASET_SIZE = 300    # representative 샘플 수

# ───────────────── 데이터셋 준비 ───────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    label_mode=None,
    image_size=(224, 224),
    batch_size=1,
    shuffle=True,
)
rep_ds = train_ds.take(REP_DATASET_SIZE)

def representative_gen():
    for img in rep_ds:
        yield [img.numpy().astype(np.float32)]

# ───────────────── 변환기 설정 ─────────────────────────
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR)
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type   = tf.uint8
converter.inference_output_type  = tf.uint8

tflite_model = converter.convert()
open(TFLITE_OUT, "wb").write(tflite_model)
print("INT8 TFLite 모델 저장:", TFLITE_OUT)
