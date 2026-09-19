<div align="center">

# yt-fakenews-classifier

**NLP pipeline that pulls YouTube captions and labels them REAL / FAKE**

![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
[![Verify](https://github.com/jumincho/yt-fakenews-classifier/actions/workflows/verify.yml/badge.svg)](https://github.com/jumincho/yt-fakenews-classifier/actions/workflows/verify.yml)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-XLM--RoBERTa-FFD43B?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green)
![Year](https://img.shields.io/badge/year-2023-blue)

</div>

---

## Overview

Give the pipeline a YouTube URL: it pulls the audio, generates captions with
WhisperX, then passes the text through a multilingual XLM-RoBERTa classifier
with a binary REAL/FAKE head. Built as a student project on whether spoken
video content can be evaluated for veracity by reducing it to text and applying
a fake-news classifier.

## Pipeline

```
YouTube URL
    │
    ▼  src/extract_subtitles.py · notebooks/01_subtitle_extraction.ipynb
audio (.mp3) → WhisperX → captions (.srt) → cleaned (.txt)
    │
    ▼  src/train.py · src/predict.py · notebooks/02_fake_news_classification.ipynb
XLM-RoBERTa (binary head) → REAL / FAKE
```

### Why two stages

WhisperX needs **torch 1.13.1 (cu117)**; the classifier needs **PyTorch ≥ 2.0**.
They cannot coexist in one environment, so subtitle extraction lives in a
separate virtualenv (`whisper-env`) and `whisperx` is invoked as a subprocess.
The intermediate artifact (the `.txt`) is the interface between the two halves.

## Features

- **Caption-extraction pipeline** — `yt-dlp` for audio → [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 alignment) → cleaned `.txt`.
- **Binary classifier** — fine-tunes `symanto/xlm-roberta-base-snli-mnli-anli-xnli` with the 3-way NLI head swapped for a freshly initialized 2-way head (`num_labels=2` + `ignore_mismatched_sizes=True`).
- **Notebook & CLI parity** — every step runs from `notebooks/` or from `python -m src.<step>`.

## Tech stack

- **Language**: Python 3.10+
- **Speech recognition**: [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 alignment)
- **Audio**: `yt-dlp`, `ffmpeg`
- **Classifier**: Hugging Face `transformers` (≥ 4.41, < 4.46), `datasets`, `evaluate`, `accelerate`
- **Training**: PyTorch ≥ 2.0 + `Trainer`
- **Data**: `pandas`, `scikit-learn`
- **Hardware**: GPU recommended (Colab T4 or similar)

## Project layout

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
│   ├── 01_subtitle_extraction.ipynb      # src.extract_subtitles entry notebook
│   └── 02_fake_news_classification.ipynb # src.train + src.predict entry notebook
├── data/
│   ├── dataset.zip            # fake_or_real_news.csv (~6,300 rows, training)
│   └── sample_input.txt       # sample caption text for inference
├── requirements.txt
├── .gitignore
└── README.md
```

## Run

### Google Colab

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

### Local

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

GPU (CUDA) recommended. About 30 minutes on a T4.

## Dataset

- **Training**: `fake_or_real_news.csv` inside `data/dataset.zip` (~6,300 English news articles with `title`, `text`, `label(REAL/FAKE)` columns).
- **Inference example**: `data/sample_input.txt` — a sample caption text produced by notebook 01.

## Screenshots

![Subtitle extraction](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/615e65f5-edec-464c-bcd2-a72d8efc989b)
![Training](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/477220aa-c59e-4c3e-8929-7b6923b35394)
![Evaluation](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/57fb6b07-950d-472b-a9ce-ab30414bd363)
![Prediction](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/40f9dab0-8884-45b9-aa6b-147c5125b51f)
![Result](https://github.com/jumincho/yt-fakenews-classifier/assets/77545063/5fd12381-ff93-4d20-9f95-5568f83714e3)

## License

[MIT License](./LICENSE)
