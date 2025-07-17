# 라즈베리파이4 실시간 날씨 분류 모델 학습·추론 파이프라인

## 📌 프로젝트 개요
각 사이트(거치 위치)별로 촬영된 하늘 이미지를 분석하여 **맑음·흐림·안개** 등 날씨 상태를 자동 분류하고, 결과를 서버로 전송하는 Edge-AI 시스템입니다. 학습은 GPU 서버에서 수행하고, **Raspberry Pi 4**(CPU 전용)에서 경량화된 TFLite 모델을 실시간 추론에 사용합니다.

> 초기에는 Google EdgeTPU 가속(USB Coral) 사용을 고려했으나, Pi 4의 CPU 성능이 충분하다고 판단되어 **현재 버전은 EdgeTPU를 사용하지 않습니다**. 관련 스크립트·옵션은 참고용으로 유지되어 있습니다.

---

## 🗂️ 디렉터리 구조
```
coral-mobilenet/
│  README.md            # ← (현재 파일)
│  config.yaml          # 하이퍼파라미터 & 경로 설정
│  requirements.txt     # Python 의존성 목록
│
├─ train.py             # MobileNetV3 학습 스크립트
├─ evaluate.py          # 테스트·메트릭 계산
├─ convert_to_tflite.py # SavedModel ➜ INT8 TFLite 변환
│
├─ dataset/             # 이미지 데이터셋 (클래스별 폴더)
│
└─ docker/
   ├─ Dockerfile        # GPU 학습용 컨테이너 정의
   └─ docker-compose.yml
```

---

## ⚙️ 사전 준비
1. **데이터셋 구성**  
   `dataset/클래스명/이미지파일.jpg` 형태로 이미지를 배치합니다. 예:
   ```
   dataset/
     clear/
       img_0001.jpg
       img_0002.jpg
     cloudy/
       img_1001.jpg
     foggy/
       ...
   ```
   💾 **NAS 원본 데이터 경로**  
   `Z:\\DEV_AI\\AI Learning Data\\2025 기상분류AI\\weather_detection_dataset`
2. **Python 3.10+ 환경** (로컬 실행 시) 또는 **Docker** (권장) 사용
3. NVIDIA GPU 학습 시 드라이버 & CUDA 12.4 호환성 확인

---

## 🚀 빠른 시작 (Docker 권장)
```bash
# 1) 이미지를 빌드하고 컨테이너 진입
$ cd docker
$ docker compose up --build -d   # 백그라운드 실행
$ docker compose exec mnv3_train bash  # 컨테이너 쉘 진입

# 2) 학습 실행
root@container:/workspace$ python train.py

# 3) (선택) 양자화 모델 생성
root@container:/workspace$ python convert_to_tflite.py

# 4) 테스트/평가
root@container:/workspace$ python evaluate.py
```

컨테이너는 `/workspace`에 호스트 프로젝트를 마운트하므로 결과(`best_model_saved/`, `cls_model_int8.tflite` 등)가 로컬에도 바로 나타납니다.

### 로컬 실행
```bash
# 의존성 설치 (Python 3.10 이상)
$ python -m venv .venv && source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt

# 학습
$ python train.py
```

---

## 🏋️‍♀️ 학습 파이프라인
1. **데이터 분리**  
   `config.yaml`의 `valtest_split`(예: 0.30)을 기준으로 **훈련 70 % / 검증 15 % / 테스트 15 %** 분할
2. **증강 & 전처리**  
   - Random Flip / Rotation / Zoom  
   - `mobilenet_v3.preprocess_input` 으로  [1m[-1,1] [0m 스케일 적용 (이중 스케일 문제 제거)
3. **모델**  
   - `MobileNetV3Large(minimalistic, α=depth_multiplier)` + GAP + Dropout + Dense N
   - 2-stage 학습
     1. **Head 학습** : 베이스 Freeze, LR = `initial_lr`, Epochs = `epochs_head`
     2. **Fine-tune** : 상위 `unfreeze_from` 이전 레이어 Freeze 해제, LR = `fine_tune_lr`, Epochs = `epochs_fine`
4. **클래스 불균형 보정** : `class_weight` 자동 계산 후 `model.fit(..., class_weight=...)`
5. **콜백** : ModelCheckpoint(`best.keras`), EarlyStopping(patience = 6)
6. **출력**
   - `saved_model_edgetpu/` : 전체 모델 (추론·재학습용)
   - `best_model_saved/`     : 검증 최고 성능 SavedModel

---

## 📊 평가
`evaluate.py`는 학습과 동일한 분할·전처리 방식을 재현하여 다음을 출력합니다.
* 전체 Accuracy
* 클래스별 Precision / Recall / F1
* 클래스별 Accuracy
* Confusion Matrix

필요 시 결과를 CSV/그래프로 저장하도록 쉽게 확장할 수 있습니다.

---

## 📦 TFLite 양자화 (선택)
라즈베리파이 4는 FP32 모델도 실시간 추론이 가능하지만, 파일 크기 감소·추론 지연 최소화를 위해 INT8 양자화를 지원합니다.

```bash
$ python convert_to_tflite.py   # → cls_model_int8.tflite 생성
```

* 대표 300개 이미지를 사용해 Post-Training Quantization 수행
* 입력·출력 타입 `int8` 지정 → XNNPACK delegate 최적화

> EdgeTPU 컴파일(.tflite ➜ .tpu)은 현재 불필요하므로 제외했습니다.

---

## 🍓 Raspberry Pi 4 배포
1. **필수 패키지 설치** (Raspberry Pi OS 64-bit 권장)
   ```bash
   $ sudo apt update && sudo apt install -y python3-pip
   $ pip3 install tensorflow==2.15.0
   ```
2. **모델 복사** : `cls_model_int8.tflite`(또는 SavedModel) + 라벨 목록 파일
3. **추론 스크립트 예시**
   ```python
   import tensorflow as tf, cv2, numpy as np

   interpreter = tf.lite.Interpreter(model_path="cls_model_int8.tflite")
   interpreter.allocate_tensors()
   inp_idx = interpreter.get_input_details()[0]['index']
   out_idx = interpreter.get_output_details()[0]['index']

   img = cv2.imread("test.jpg")
   img = cv2.resize(img, (224,224))
   img = img.astype(np.int8) - 128  # [-128,127] 스케일 예시

   interpreter.set_tensor(inp_idx, img[None])
   interpreter.invoke()
   probs = interpreter.get_tensor(out_idx)[0]
   pred  = np.argmax(probs)
   print("예측 결과:", pred)
   ```
4. **서버 전송** : HTTP/ MQTT 등 원하는 방식으로 추론 결과 업로드

---

## 🛠️ 설정 파일(`config.yaml`) 설명
| 키 | 설명 | 기본값 |
|---|---|---|
| `data_dir` | 이미지 데이터셋 폴더 | `dataset` |
| `valtest_split` | 검증+테스트 비율(0~1) | `0.30` |
| `img_height`, `img_width` | 입력 해상도 | `224` |
| `batch_size` | 배치 크기 | `32` |
| `epochs_head` | Head 학습 Epochs | `20` |
| `epochs_fine` | Fine-tune Epochs | `20` |
| `initial_lr` | Head 학습 Learning Rate | `1.0e-3` |
| `fine_tune_lr` | Fine-tune Learning Rate | `1.0e-4` |
| `depth_multiplier` | MobileNetV3 α 값(0.35-1.0) | `1.0` |
| `unfreeze_from` | Fine-tune 시 Freezing 해제 기준 레이어 인덱스 | `100` |

---

## 🖥️ 의존성
`requirements.txt` (최소)
```
tensorflow==2.15.0
PyYAML>=6.0
h5py>=3.1.0
```

추가 라이브러리:
* `opencv-python`, `tqdm`, `scikit-learn` 등은 필요 시 `pip install` 하세요.

---

## 🤝 Contributing
이슈·PR 환영합니다! 특히
* 다양한 날씨·조도 환경 데이터셋 제보
* Raspberry Pi 실측 벤치마크
* 코드 리팩터링 / 문서 개선

---

## 📄 라이선스
MIT License © 2024