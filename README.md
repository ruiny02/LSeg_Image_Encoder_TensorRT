### 0. NGC([nvcr.io](http://nvcr.io/)) 계정/API Key 준비 (필수)

이 레포의 Dockerfile은 **NGC 베이스 이미지**를 사용합니다.

- x86: `nvcr.io/nvidia/pytorch:24.08-py3`
- Jetson: `nvcr.io/nvidia/l4t-jetpack:r36.4.0`

따라서 [**nvcr.io](http://nvcr.io/) 로그인**이 필요합니다.

### (1) NGC API Key 만들기

1. NVIDIA NGC 사이트 로그인
2. 계정 설정(Setup)에서 **API Key 생성**
3. 발급된 키를 복사해 둡니다.

### (2) [nvcr.io](http://nvcr.io/) 로그인

아래 2가지 중 편한 방법 사용:

**방법 A: 인터랙티브 로그인**

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key 붙여넣기>

```

## 1. 코드 받기 (git clone --recurse-submodules)

> CPP_Project/third_party/cnpy 등이 submodule일 수 있어 --recurse-submodules 권장
> 

```bash
git clone --recurse-submodules https://github.com/ruiny02/LSeg_Image_Encoder_TensorRT/tree/last_fix
cd LSeg_Image_Encoder_TensorRT

```

이미 clone을 해버렸다면:

```bash
git submodule update --init --recursive

```

---

## 2. Docker 이미지 빌드
 

### 2.1 x86 (Ubuntu + NVIDIA GPU) 이미지 빌드

```bash
cd LSeg_Image_Encoder_TensorRT
git switch real-time

docker build \
  -f Dockerfile.x86 \
  -t ruiny022/lseg-trt:x86-u22.04 \
  .

```

### 2.2 Jetson (JetPack 6.x) 이미지 빌드 (Jetson에서 실행)

```bash
cd LSeg_Image_Encoder_TensorRT

docker build \
  -f Dockerfile.jetson \
  -t ruiny022/lseg-trt:jetson-jp62 \
  .

```

> Jetson Dockerfile은 기본적으로 C++ 바이너리 빌드를 build 단계에서 스킵합니다.
> 
> 
> (Jetson은 docker build 단계에서 DLA/cuDLA 라이브러리 마운트가 없어 링크가 깨질 수 있음)
> 
> 필요하면 컨테이너 실행 후 runtime에서 빌드하는 방식을 권장합니다. (아래 8절 참고)
> 

---

## 3. 컨테이너 실행 (X11 + 카메라)

### 3.1 카메라 디바이스 확인

**호스트**에서 어떤 `/dev/videoX`인지 먼저 확인하세요.

```bash
ls -l /dev/video*
v4l2-ctl --list-devices || true
```

- 보통 x86: `/dev/video0` 또는 `/dev/video2` … (PC마다 다름)
- 보통 Jetson USB: `/dev/video0`

이 문서에서는 예시로:

- x86: `/dev/video4`
- Jetson: `/dev/video0`
을 사용합니다. **본인 환경에 맞게 바꿔주세요.**

### 3.2 (공통) X11 권한 열기

호스트에서:

```bash
xhost +local:root

```

### 3.3 x86에서 컨테이너 실행 (docker run)

```bash
cd LSeg_Image_Encoder_TensorRT

docker run --rm -it \
  --gpus all \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $PWD:/workspace \
  -v $HOME/.cache:/root/.cache \
  --device /dev/video4:/dev/video4 \
  ruiny022/lseg-trt:x86-u22.04

```

### 3.4 Jetson에서 컨테이너 실행 (docker run)

Jetson 호스트에서:

```bash
cd LSeg_Image_Encoder_TensorRT

docker run -it \
  --runtime nvidia \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $PWD:/workspace \
  -v $HOME/.cache:/root/.cache \
  --device /dev/video0:/dev/video0 \
  ruiny022/lseg-trt:jetson-jp62

```


---

## 4. LSeg 체크포인트(.ckpt) 받기

컨테이너 안에서 아래를 실행하면 **호스트의 repo 폴더(models/weights)**에 저장됩니다.

```bash
cd /workspace
mkdir -p models/weights/ViT models/weights/Resnet

pip3 install -U gdown

# ViT-L/16 (demo_e200.ckpt)
gdown 'https://drive.google.com/uc?id=1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7' -O models/weights/ViT/demo_e200.ckpt

# ViT-L/16 (FSS, fss_l16.ckpt)
gdown 'https://drive.google.com/uc?id=1Nplkc_JsHIS55d--K2vonOOC3HrppzYy' -O models/weights/ViT/fss_l16.ckpt

# **Nano는 이것만 설치하시면 됩니다.**
# ResNet101 (ZS, fss_rn101.ckpt)
gdown 'https://drive.google.com/uc?id=1UIj49Wp1mAopPub5M6O4WW-Z79VB1bhw' -O models/weights/Resnet/fss_rn101.ckpt

```

> 체크포인트 출처: LSeg 공식 레포(lang-seg). 라이선스/배포 정책에 따라 본 레포에는 포함되어 있지 않습니다.
> 

---

## 5. 실행 (따라치기)

> 아래 명령들은 **컨테이너 내부(/workspace)**에서 실행합니다.
> 

### 5.1 실시간 USB 카메라 데모 (TensorRT, 기본)

첫 실행 시 자동으로:

- ONNX 생성: `models/onnx_engines/*.onnx`
- TensorRT 엔진 생성: `models/trt_engines/*.trt`

을 수행합니다. (처음만 시간이 걸립니다)

### (1) ViT 데모 (Nano환경에서 구현 불가능)

```bash
python3 realtime/lseg_realtime.py \
  --device /dev/video4 \
  --backend trt \
  --weights models/weights/ViT/demo_e200.ckpt \
  --labels "person, chair, desk, monitor, keyboard, background"

```

### (2) ResNet(ZS) 데모

```bash
python3 realtime/lseg_realtime.py \
  --device /dev/video4 \
  --backend trt \
  --weights models/weights/Resnet/fss_rn101.ckpt \
  --labels "person, chair, desk, monitor, keyboard, background"

```

**(옵션) Nano 환경에서 onnx로 인해 자동 엔진 빌드 실패 시 FP32 ONNX를 그대로 사용하면서 아래 명령으로 FP16 TensorRT 엔진을 수동으로 생성**

```jsx
python3 conversion/onnx_to_trt.py \
--onnx models/onnx_engines/lseg_img_enc_rn101_fss_rn101.onnx \
--engine models/trt_engines/lseg_img_enc_rn101_fss_rn101_288x512_fp16.trt \
--fp16 \
--min_hw 288 512 --opt_hw 288 512 --max_hw 288 512 \
--workspace 1073741824
```

**(옵션) 그 후 다시 데모 수행**

```jsx
python3 realtime/lseg_realtime.py \
  --device /dev/video0 \
  --backend trt \
  --weights models/weights/Resnet/fss_rn101.ckpt \
  --labels "person, chair, desk, monitor, keyboard, background"
```

- 종료: `q` 또는 `ESC`
- 종료 시 콘솔에 Avg(ms), FPS 요약이 출력됩니다.

> **Jetson에서는 --device /dev/video0로 바꿔서 실행하세요.**
> 

### 5.2 PyTorch(기준)로 실행 (Nano환경에서 구현 불가능)

```bash
python3 realtime/lseg_realtime.py \
  --device /dev/video4 \
  --backend torch \
  --weights models/weights/ViT/demo_e200.ckpt \
  --labels "person, chair, desk, monitor, keyboard, background"

```


---

## 6. ONNX / TensorRT를 수동으로 만들기 (옵션)

실시간 스크립트가 자동으로 만들어주지만, 수동으로도 가능합니다.

### 6.1 PyTorch(.ckpt) → ONNX

**ViT** (Nano환경에서 구현 불가능)

```bash
python3 conversion/model_to_onnx.py \
  --weights models/weights/ViT/demo_e200.ckpt
# 출력: models/onnx_engines/lseg_img_enc_vit_demo_e200.onnx

```

**ResNet101(ZS)**

```bash
python3 conversion/model_to_onnx_zs.py \
  --weights models/weights/Resnet/fss_rn101.ckpt
# 출력: models/onnx_engines/lseg_img_enc_rn101_fss_rn101.onnx

```

### 6.2 ONNX → TensorRT (FP16 예시)

입력 해상도는 실시간 데모 기본값(288x512)에 맞춘 예시입니다.

```bash
python3 conversion/onnx_to_trt.py \
  --onnx models/onnx_engines/lseg_img_enc_vit_demo_e200.onnx \
  --fp16 \
  --min_hw 288 512 --opt_hw 288 512 --max_hw 288 512 \
  --workspace $((1<<30))

```

엔진은 기본적으로 `models/trt_engines/` 아래에 자동 이름으로 저장됩니다.

(직접 파일명을 지정하려면 `--engine <PATH>` 옵션 사용)

### 6.3 입력 해상도 다양하게 엔진 수동 빌드

1) TensorRT 엔진 생성 **(Input: 160×256)**

```jsx
python3 conversion/onnx_to_trt.py \
--onnx models/onnx_engines/lseg_img_enc_rn101_fss_rn101.onnx \
--engine models/trt_engines/lseg_img_enc_rn101_fss_rn101_160x256_fp16.trt \
--fp16 \
--min_hw 160 256 --opt_hw 160 256 --max_hw 160 256 \
--workspace 1073741824
```

2) 실시간 데모 실행 **(Engine: 160×256)**

```jsx
python3 realtime/lseg_realtime.py \
  --device /dev/video0 \
  --backend trt \
  --weights models/weights/Resnet/fss_rn101.ckpt \
  --labels "person, chair, desk, monitor, keyboard, background"
  --engine models/trt_engines/lseg_img_enc_rn101_fss_rn101_160x256_fp16.trt
```

1) TensorRT 엔진 생성 **(Input: 224×352)**

```jsx
python3 conversion/onnx_to_trt.py \
--onnx models/onnx_engines/lseg_img_enc_rn101_fss_rn101.onnx \
--engine models/trt_engines/lseg_img_enc_rn101_fss_rn101_224x352_fp16.trt \
--fp16 \
--min_hw 224 352 --opt_hw 224 352 --max_hw 224 352 \
--workspace 1073741824
```

2) 실시간 데모 실행 **(Engine: 224×352)**

```jsx
python3 realtime/lseg_realtime.py \
  --device /dev/video0 \
  --backend trt \
  --weights models/weights/Resnet/fss_rn101.ckpt \
  --labels "person, chair, desk, monitor, keyboard, background"
  --engine models/trt_engines/lseg_img_enc_rn101_fss_rn101_224x352_fp16.trt
```

---

## 8. (옵션) C++ 바이너리 빌드/사용

- x86 Dockerfile은 build 단계에서 C++ 바이너리를 미리 빌드해 `/opt/prebuilt`에 넣고,
컨테이너 시작 시 `/workspace/CPP_Project/**/build/`로 복사합니다.
- Jetson은 기본적으로 build 단계 C++ 빌드를 스킵하므로, **컨테이너 내부에서 직접 빌드**를 권장합니다.

Jetson 컨테이너 내부에서:

```bash
cd /workspace/CPP_Project/Inference_Time_Tester
cmake -B build -S .
cmake --build build -j"$(nproc)"

cd /workspace/CPP_Project/Feature_Extractor
cmake -B build -S .
cmake --build build -j"$(nproc)"

```

---

## 9. 자주 나는 문제(트러블슈팅)

### 9.1 `docker build`에서 [nvcr.io](http://nvcr.io/) pull이 실패

- `docker login nvcr.io`가 되어 있는지 확인
- Username이 **$oauthtoken** 인지 확인
- API Key가 만료/삭제되지 않았는지 확인

### 9.2 컨테이너에서 GUI 창이 안 뜸 (cv2.imshow)

- 호스트에서 `xhost +local:root` 실행했는지 확인
- `e DISPLAY=$DISPLAY` 와 `/tmp/.X11-unix` 마운트 확인
- Wayland 환경이면 X11 설정이 추가로 필요할 수 있습니다.

### 9.3 카메라 오픈 실패

- `/dev/videoX`가 컨테이너에 전달되었는지 확인 (`-device ...`)
- 호스트에서 다른 프로그램이 카메라를 잡고 있지 않은지 확인
- 카메라 해상도/프레임이 지원되는지 확인 (`-cam_w/--cam_h/--cam_fps` 변경)

### 9.4 Jetson에서 TensorRT 엔진이 안 맞거나 로드 실패

- 엔진은 Jetson에서 직접 빌드해야 합니다.
- 기존에 x86에서 만든 `models/trt_engines/*.trt`가 남아있으면 삭제 후 재빌드:
    
    ```bash
    rm -f models/trt_engines/*.trt
    
    ```
