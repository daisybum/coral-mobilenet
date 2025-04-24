"""
best_model.h5 → MobileNetV3-Large TFLite 변환 스크립트
(1) FP32   (2) Dynamic-range   (3) Edge-TPU full-INT8
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras

# ──── 0. 경로 및 파라미터 ──────────────────────────────────────────
MODEL_PATH          = "models/best_model.h5"          # 학습 완료 모델
FP32_PATH           = "models/mobilenet_fp32.tflite"  # (선택) 순수 float 모델
DYNAMIC_PATH        = "models/mobilenet_dynamic.tflite"
INT8_PATH           = "models/mobilenet_int8.tflite"  # Edge TPU 컴파일용
CALIB_DIR           = "calib_samples"          # 대표 샘플 50~100장 보관 폴더
IMG_SIZE            = (224, 224)               # train.py의 img_height, img_width
BATCH_FOR_REP       = 1                        # Edge TPU는 1 배치 권장

# ──── 1. Keras 모델 로드 ──────────────────────────────────────────
model = keras.models.load_model(MODEL_PATH, compile=False)

# ──── 2-A. FP32(TFLite) ──────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
fp32_tflite = converter.convert()
Path(FP32_PATH).write_bytes(fp32_tflite)

# ──── 2-B. Dynamic-range 양자화 ───────────────────────────────────
converter.optimizations = [tf.lite.Optimize.DEFAULT]
dynamic_tflite = converter.convert()
Path(DYNAMIC_PATH).write_bytes(dynamic_tflite)

# ──── 2-C. Edge-TPU 풀-INT8 양자화 ────────────────────────────────
def representative_data_gen():
    """
    Edge TPU 칼리브레이션용 대표 데이터 50~100장.
    `calib_samples/**/*.jpg` 구조를 가정.
    """
    imgs = sorted(Path(CALIB_DIR).rglob("*.[jp][pn]g"))[:100]
    for img_path in imgs:
        img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        arr = keras.preprocessing.image.img_to_array(img)
        # MobileNetV3 preprocess: [-1, 1] 구간 정규화
        arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
        yield [arr.astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type   = tf.uint8   # Edge TPU 요건
converter.inference_output_type  = tf.uint8

int8_tflite = converter.convert()
Path(INT8_PATH).write_bytes(int8_tflite)

print("✅ 변환 완료:")
print(f"  • FP32   : {FP32_PATH}")
print(f"  • Dynamic: {DYNAMIC_PATH}")
print(f"  • INT8   : {INT8_PATH} (Edge TPU 컴파일 대상)")
