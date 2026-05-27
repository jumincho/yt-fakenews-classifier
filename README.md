<div align="center">

# yt-fakenews-classifier

**YouTube 자막을 추출해 가짜뉴스 여부를 판별하는 NLP 파이프라인**
**NLP pipeline that pulls YouTube captions and labels them REAL / FAKE**

![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-XLM--RoBERTa-FFD43B?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green)
![Year](https://img.shields.io/badge/year-2023-blue)

**한국어** · [English](#english)

</div>

---

## 개요

> YouTube 동영상의 자막을 추출하고, 그 내용을 가짜뉴스인지 자동으로 판별하는 NLP 파이프라인

YouTube 영상 URL만 있으면 오디오를 받아 자막을 뽑고, 그 자막 텍스트를 다국어
XLM-RoBERTa 분류기에 통과시켜 **REAL / FAKE** 라벨을 출력합니다.
정보의 진위 판단이 점점 어려워지는 영상 콘텐츠 환경에서, 발화 내용을 텍스트로
바꾸어 가짜뉴스 분류 모델을 적용해본 프로젝트입니다.

## 주요 기능

- **자막 추출 파이프라인** — YouTube URL 입력 → `yt-dlp`로 오디오(.mp3) 다운로드
  → [WhisperX](https://github.com/m-bain/whisperx)로 음성 인식 및 정렬 → `.srt`
  → 정제된 `.txt`
- **가짜뉴스 분류 모델** — 영문 [Fake or Real News 데이터셋](https://www.kaggle.com/datasets/rajatkumar30/fake-news)
  으로 `symanto/xlm-roberta-base-snli-mnli-anli-xnli`를 파인튜닝, REAL/FAKE 이진 분류
- **추론** — 추출된 자막 텍스트 파일을 입력으로 받아 학습된 모델이 진위를 예측

## 파이프라인

```
YouTube URL
    │
    ▼  (notebooks/01_subtitle_extraction.ipynb)
오디오(.mp3) → WhisperX → 자막(.srt) → 정제(.txt)
    │
    ▼  (notebooks/02_fake_news_classification.ipynb)
XLM-RoBERTa 분류기 → REAL / FAKE
```

## 기술 스택

- **언어**: Python 3.10 (Google Colab 환경 기준)
- **음성 인식**: [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 alignment)
- **오디오 처리**: `yt-dlp`, `ffmpeg`
- **분류 모델**: Hugging Face `transformers` — `symanto/xlm-roberta-base-snli-mnli-anli-xnli`
- **학습 프레임워크**: PyTorch + `Trainer` API, `accelerate`, `evaluate`
- **데이터 처리**: `pandas`, `scikit-learn`, `datasets`
- **하드웨어**: GPU (Colab T4 등) 권장

## 프로젝트 구조

```
.
├── notebooks/
│   ├── 01_subtitle_extraction.ipynb       # YouTube URL → .txt 자막
│   └── 02_fake_news_classification.ipynb  # 자막 .txt → REAL/FAKE 예측
├── data/
│   ├── dataset.zip                         # fake_or_real_news.csv (학습용, 12MB 바이너리)
│   └── sample_input.txt                    # 추론 테스트용 예시 자막
├── requirements.txt
├── .gitignore
└── README.md
```

## 사용 방법

### 1. Google Colab에서 실행 (권장)

각 노트북을 Colab에서 바로 열 수 있습니다.

- 자막 추출:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/01_subtitle_extraction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
- 가짜뉴스 분류:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/02_fake_news_classification.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

런타임 유형은 **GPU (T4)** 가 권장됩니다. 학습 데이터(`fake_or_real_news.csv`)는
`data/dataset.zip` 만 두면 노트북이 첫 셀에서 자동으로 풀어 사용합니다. Colab 환경에서는 `dataset.zip` 또는 `fake_or_real_news.csv` 를 세션에 업로드해도 됩니다.

### 2. 로컬에서 실행

```bash
git clone https://github.com/jumincho/yt-fakenews-classifier.git
cd yt-fakenews-classifier

# 학습 데이터 압축 해제
unzip data/dataset.zip -d data/

# 의존성 설치 (CUDA 가능 환경 권장)
pip install -r requirements.txt
pip install git+https://github.com/m-bain/whisperx.git@v2.0.1

# Jupyter에서 노트북 열기
jupyter lab notebooks/
```

## 데이터셋

- **학습 데이터**: `data/dataset.zip` 안의 `fake_or_real_news.csv`
  (약 6,300건의 영문 뉴스 기사, `title`, `text`, `label(REAL/FAKE)` 컬럼)
- **추론 예시**: `data/sample_input.txt` — 노트북 1에서 만들어 노트북 2에 입력으로 넣을 수 있는 자막 텍스트 샘플

## 스크린샷

![008](https://github.com/jumincho/YouTube-fake-news-detector/assets/77545063/615e65f5-edec-464c-bcd2-a72d8efc989b)
![009](https://github.com/jumincho/YouTube-fake-news-detector/assets/77545063/477220aa-c59e-4c3e-8929-7b6923b35394)
![010](https://github.com/jumincho/YouTube-fake-news-detector/assets/77545063/57fb6b07-950d-472b-a9ce-ab30414bd363)
![011](https://github.com/jumincho/YouTube-fake-news-detector/assets/77545063/40f9dab0-8884-45b9-aa6b-147c5125b51f)
![013](https://github.com/jumincho/YouTube-fake-news-detector/assets/77545063/5fd12381-ff93-4d20-9f95-5568f83714e3)

## 라이선스

[MIT License](./LICENSE)

---

<a name="english"></a>

## English

> NLP pipeline that extracts YouTube captions and classifies them as REAL or FAKE news.

Give the pipeline a YouTube URL: it pulls audio, generates captions, then passes the
text through a multilingual XLM-RoBERTa classifier to get a **REAL / FAKE** label.
Built as an experiment on whether spoken video content can be evaluated for veracity
by reducing it to text and applying a fake-news classifier.

### Features

- **Caption-extraction pipeline** — YouTube URL → `yt-dlp` for audio → [WhisperX](https://github.com/m-bain/whisperx) for speech recognition and alignment → `.srt` → cleaned `.txt`.
- **Fake-news classifier** — fine-tunes `symanto/xlm-roberta-base-snli-mnli-anli-xnli` on the English [Fake or Real News dataset](https://www.kaggle.com/datasets/rajatkumar30/fake-news) for REAL/FAKE binary classification.
- **Inference** — feeds the extracted caption text into the trained model and outputs a label.

### Pipeline

```
YouTube URL
    │
    ▼  (notebooks/01_subtitle_extraction.ipynb)
audio (.mp3) → WhisperX → captions (.srt) → cleaned (.txt)
    │
    ▼  (notebooks/02_fake_news_classification.ipynb)
XLM-RoBERTa classifier → REAL / FAKE
```

### Tech stack

- **Language**: Python 3.10 (Google Colab baseline)
- **Speech recognition**: [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 alignment)
- **Audio**: `yt-dlp`, `ffmpeg`
- **Classifier**: Hugging Face `transformers` — `symanto/xlm-roberta-base-snli-mnli-anli-xnli`
- **Training**: PyTorch + `Trainer` API, `accelerate`, `evaluate`
- **Data**: `pandas`, `scikit-learn`, `datasets`
- **Hardware**: GPU recommended (Colab T4 or similar)

### Layout

```
.
├── notebooks/
│   ├── 01_subtitle_extraction.ipynb       # YouTube URL → .txt captions
│   └── 02_fake_news_classification.ipynb  # captions .txt → REAL/FAKE prediction
├── data/
│   ├── dataset.zip                         # fake_or_real_news.csv (training data, 12MB binary)
│   └── sample_input.txt                    # sample caption text for inference
├── requirements.txt
├── .gitignore
└── README.md
```

### Run

#### 1. Google Colab (recommended)

- Caption extraction:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/01_subtitle_extraction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
- Classification:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/02_fake_news_classification.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

A GPU runtime (T4) is recommended. Training data (`fake_or_real_news.csv`) ships as
`data/dataset.zip`; the notebook auto-unzips it on first run. In Colab, uploading either `dataset.zip` or the raw `fake_or_real_news.csv` to the session works.

#### 2. Local

```bash
git clone https://github.com/jumincho/yt-fakenews-classifier.git
cd yt-fakenews-classifier

# unzip training data
unzip data/dataset.zip -d data/

# install deps (CUDA recommended)
pip install -r requirements.txt
pip install git+https://github.com/m-bain/whisperx.git@v2.0.1

# open notebooks in Jupyter
jupyter lab notebooks/
```

### Dataset

- **Training**: `fake_or_real_news.csv` inside `data/dataset.zip`
  (~6,300 English news articles with `title`, `text`, `label(REAL/FAKE)` columns).
- **Inference example**: `data/sample_input.txt` — caption text from notebook 1, usable as input for notebook 2.

### License

[MIT License](./LICENSE)
