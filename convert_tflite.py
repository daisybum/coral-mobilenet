#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
best_model.h5 → FP32 / Dynamic / INT8-EdgeTPU TFLite 변환
"""

import tensorflow as tf, numpy as np
from tensorflow import keras
from pathlib import Path

MODEL_PATH   = "models/best_model.h5"
FP32_PATH    = "models/mobilenet_fp32.tflite"
DYN_PATH     = "models/mobilenet_dynamic.tflite"
INT8_PATH    = "models/mobilenet_int8.tflite"
CALIB_DIR    = "calib_samples"      # 대표 이미지 폴더
IMG_SIZE     = (224, 224)

model = keras.models.load_model(MODEL_PATH, compile=False)

# FP32
conv = tf.lite.TFLiteConverter.from_keras_model(model)
Path(FP32_PATH).write_bytes(conv.convert())

# Dynamic-range
conv.optimizations = [tf.lite.Optimize.DEFAULT]
Path(DYN_PATH).write_bytes(conv.convert())

# Edge-TPU INT8
def rep():
    imgs = sorted(Path(CALIB_DIR).rglob("*.[jp][pn]g"))[:100]
    for p in imgs:
        arr = keras.preprocessing.image.img_to_array(
              keras.preprocessing.image.load_img(p, target_size=IMG_SIZE))
        arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
        yield [arr.astype(np.float32)]

conv.representative_dataset      = rep
conv.optimizations               = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type        = tf.uint8
conv.inference_output_type       = tf.uint8
Path(INT8_PATH).write_bytes(conv.convert())

print("✅ 변환 완료")
