<div align="center">

# yt-fakenews-classifier

**YouTube 자막을 추출해 가짜뉴스 여부를 판별하는 NLP 파이프라인**
**NLP pipeline that pulls YouTube captions and labels them REAL / FAKE**

![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
[![Verify](https://github.com/jumincho/yt-fakenews-classifier/actions/workflows/verify.yml/badge.svg)](https://github.com/jumincho/yt-fakenews-classifier/actions/workflows/verify.yml)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-XLM--RoBERTa-FFD43B?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green)
![Year](https://img.shields.io/badge/year-2023-blue)

**한국어** · [English](#english) · [中文](./README.zh-CN.md)

</div>

---

## 개요

> YouTube 영상 URL → 자막 텍스트 → REAL/FAKE 라벨

YouTube URL 하나만 있으면 오디오를 받아 자막을 뽑고, 그 자막을 다국어
XLM-RoBERTa 분류기에 통과시켜 **REAL / FAKE** 라벨을 출력합니다. 영상 콘텐츠의
발화 내용을 텍스트로 환원한 뒤 가짜뉴스 분류 모델을 적용해 본 학생 프로젝트입니다.

## 파이프라인

```
YouTube URL
    │
    ▼  src/extract_subtitles.py · notebooks/01_subtitle_extraction.ipynb
오디오(.mp3) → WhisperX → 자막(.srt) → 정제(.txt)
    │
    ▼  src/train.py · src/predict.py · notebooks/02_fake_news_classification.ipynb
XLM-RoBERTa (binary head) → REAL / FAKE
```

## 두 단계로 나뉘어진 이유

WhisperX는 **torch 1.13.1 (cu117)** 을, 분류기는 **PyTorch ≥ 2.0** 을 요구해서
같은 환경에서 공존하기 어렵습니다. 그래서 자막 추출 단계는 별도 가상환경
(`whisper-env`)을 두고 `whisperx` 바이너리를 서브프로세스로 호출하는 형태로
분리했습니다. 가운데 산출물(`.txt`)이 두 단계를 잇는 인터페이스입니다.

## 주요 기능

- **자막 추출 파이프라인** — `yt-dlp`로 오디오(.mp3) → WhisperX(large-v2 + wav2vec2 alignment)로 `.srt` → 정제된 `.txt`
- **이진 분류기** — `symanto/xlm-roberta-base-snli-mnli-anli-xnli`의 백본을 그대로 두고 분류 헤드만 2-way로 새로 학습 (`num_labels=2` + `ignore_mismatched_sizes=True`)
- **노트북 · CLI 동시 지원** — `notebooks/`에서 클릭으로 돌릴 수도 있고, `python -m src.train` / `python -m src.predict` 로 스크립트화도 가능

## 기술 스택

- **언어**: Python 3.10+
- **음성 인식**: [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 정렬)
- **오디오 다운로드**: `yt-dlp`, `ffmpeg`
- **분류**: Hugging Face `transformers` (≥ 4.41, < 4.46), `datasets`, `evaluate`, `accelerate`
- **학습 프레임워크**: PyTorch ≥ 2.0 + `Trainer`
- **데이터**: `pandas`, `scikit-learn`
- **하드웨어**: GPU (Colab T4 등) 권장

## 프로젝트 구조

```
yt-fakenews-classifier/
├── src/
│   ├── __init__.py
│   ├── extract_subtitles.py   # YouTube → mp3 → srt → txt (WhisperX)
│   ├── data.py                # CSV 로드, NewsDataset, train/val/test 분할
│   ├── model.py               # 토크나이저 · 분류기 팩토리 (binary head)
│   ├── train.py               # 학습 파이프라인 (노트북·CLI 공용)
│   └── predict.py             # 텍스트 파일 추론 (노트북·CLI 공용)
├── notebooks/
│   ├── 01_subtitle_extraction.ipynb    # src.extract_subtitles 를 사용하는 진입 노트북
│   └── 02_fake_news_classification.ipynb # src.train + src.predict 진입 노트북
├── data/
│   ├── dataset.zip            # fake_or_real_news.csv (~6,300건, 학습용)
│   └── sample_input.txt       # 추론 테스트용 자막 샘플
├── requirements.txt
├── .gitignore
└── README.md
```

## 사용 방법

### Google Colab

각 노트북을 GPU 런타임으로 바로 열 수 있습니다.

- 자막 추출:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/01_subtitle_extraction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
- 가짜뉴스 분류:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/02_fake_news_classification.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

Colab에서는 저장소를 `git clone`한 뒤 `notebooks/` 안의 파일을 열어 주세요
(노트북이 `src/` 를 임포트합니다).

### 로컬

```bash
git clone https://github.com/jumincho/yt-fakenews-classifier.git
cd yt-fakenews-classifier
pip install -r requirements.txt

# 학습 데이터는 첫 실행 시 dataset.zip 에서 자동 압축 해제됩니다.

# (1) 자막 추출 — WhisperX 가상환경을 한 번 만들고 시작
virtualenv whisper-env
whisper-env/bin/pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchaudio==0.13.1
whisper-env/bin/pip install git+https://github.com/m-bain/whisperx.git@v2.0.1
export WHISPERX_BIN=$(pwd)/whisper-env/bin/whisperx

python -m src.extract_subtitles "https://www.youtube.com/watch?v=4sC-k-92JBE" --out-dir output

# (2) 분류기 학습
python -m src.train --epochs 3 --batch-size 8 --seed 42

# (3) 영상 자막에 대해 REAL/FAKE 예측
python -m src.predict output/audio.txt --model-dir output/final
```

GPU (CUDA) 환경 권장. 학습은 T4 기준 약 30분.

## 데이터셋

- **학습**: `data/dataset.zip` 안의 `fake_or_real_news.csv` (~6,300건의 영문 뉴스, `title` · `text` · `label(REAL/FAKE)`)
- **추론 예시**: `data/sample_input.txt` — notebook 01 에서 만들 수 있는 자막 텍스트의 예시

## 스크린샷

![008](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/615e65f5-edec-464c-bcd2-a72d8efc989b)
![009](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/477220aa-c59e-4c3e-8929-7b6923b35394)
![010](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/57fb6b07-950d-472b-a9ce-ab30414bd363)
![011](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/40f9dab0-8884-45b9-aa6b-147c5125b51f)
![013](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/5fd12381-ff93-4d20-9f95-5568f83714e3)

## 라이선스

[MIT License](./LICENSE)

---

<a name="english"></a>

## English

> NLP pipeline: YouTube URL → caption text → REAL/FAKE label.

Give the pipeline a YouTube URL: it pulls audio, generates captions with
WhisperX, then passes the text through a multilingual XLM-RoBERTa classifier
with a binary REAL/FAKE head. Built as a student project on whether spoken
video content can be evaluated for veracity by reducing it to text and applying
a fake-news classifier.

### Pipeline

```
YouTube URL
    │
    ▼  src/extract_subtitles.py · notebooks/01_subtitle_extraction.ipynb
audio (.mp3) → WhisperX → captions (.srt) → cleaned (.txt)
    │
    ▼  src/train.py · src/predict.py · notebooks/02_fake_news_classification.ipynb
XLM-RoBERTa (binary head) → REAL / FAKE
```

### Why split into two stages

WhisperX needs **torch 1.13.1 (cu117)**; the classifier needs **PyTorch ≥ 2.0**.
They cannot coexist in one environment, so subtitle extraction lives in a
separate virtualenv (`whisper-env`) and `whisperx` is invoked as a subprocess.
The intermediate artifact (the `.txt`) is the interface between the two halves.

### Features

- **Caption-extraction pipeline** — `yt-dlp` for audio → [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 alignment) → cleaned `.txt`.
- **Binary classifier** — fine-tunes `symanto/xlm-roberta-base-snli-mnli-anli-xnli` with the 3-way NLI head swapped for a freshly initialized 2-way head (`num_labels=2` + `ignore_mismatched_sizes=True`).
- **Notebook & CLI parity** — every step runs from `notebooks/` or from `python -m src.<step>`.

### Tech stack

- **Language**: Python 3.10+
- **Speech recognition**: [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 alignment)
- **Audio**: `yt-dlp`, `ffmpeg`
- **Classifier**: Hugging Face `transformers` (≥ 4.41, < 4.46), `datasets`, `evaluate`, `accelerate`
- **Training**: PyTorch ≥ 2.0 + `Trainer`
- **Data**: `pandas`, `scikit-learn`
- **Hardware**: GPU recommended (Colab T4 or similar)

### Layout

```
yt-fakenews-classifier/
├── src/
│   ├── __init__.py
│   ├── extract_subtitles.py   # YouTube → mp3 → srt → txt (WhisperX)
│   ├── data.py                # CSV load, NewsDataset, train/val/test split
│   ├── model.py               # tokenizer / classifier factory (binary head)
│   ├── train.py               # training pipeline (notebook & CLI)
│   └── predict.py             # text-file inference (notebook & CLI)
├── notebooks/
│   ├── 01_subtitle_extraction.ipynb    # src.extract_subtitles entry notebook
│   └── 02_fake_news_classification.ipynb # src.train + src.predict entry notebook
├── data/
│   ├── dataset.zip            # fake_or_real_news.csv (~6,300 rows, training)
│   └── sample_input.txt       # sample caption text for inference
├── requirements.txt
├── .gitignore
└── README.md
```

### Run

#### Google Colab

- Caption extraction:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/01_subtitle_extraction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
- Classification:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/02_fake_news_classification.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

In Colab, `git clone` the repo and open the notebooks from inside `notebooks/`
(they import from `src/`).

#### Local

```bash
git clone https://github.com/jumincho/yt-fakenews-classifier.git
cd yt-fakenews-classifier
pip install -r requirements.txt

# Training data auto-extracts from dataset.zip on first run.

# (1) Subtitle extraction — bootstrap the WhisperX virtualenv once
virtualenv whisper-env
whisper-env/bin/pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchaudio==0.13.1
whisper-env/bin/pip install git+https://github.com/m-bain/whisperx.git@v2.0.1
export WHISPERX_BIN=$(pwd)/whisper-env/bin/whisperx

python -m src.extract_subtitles "https://www.youtube.com/watch?v=4sC-k-92JBE" --out-dir output

# (2) Train the classifier
python -m src.train --epochs 3 --batch-size 8 --seed 42

# (3) Predict REAL/FAKE for a transcript
python -m src.predict output/audio.txt --model-dir output/final
```

GPU (CUDA) recommended. ~30 minutes on a T4.

### Dataset

- **Training**: `fake_or_real_news.csv` inside `data/dataset.zip` (~6,300 English news articles with `title`, `text`, `label(REAL/FAKE)` columns).
- **Inference example**: `data/sample_input.txt` — a sample caption text produced by notebook 01.

### Screenshots

![008](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/615e65f5-edec-464c-bcd2-a72d8efc989b)
![009](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/477220aa-c59e-4c3e-8929-7b6923b35394)
![010](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/57fb6b07-950d-472b-a9ce-ab30414bd363)
![011](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/40f9dab0-8884-45b9-aa6b-147c5125b51f)
![013](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/5fd12381-ff93-4d20-9f95-5568f83714e3)

### License

[MIT License](./LICENSE)
