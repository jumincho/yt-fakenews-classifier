<div align="center">

# yt-fakenews-classifier

**提取 YouTube 字幕并判别是否为虚假新闻的 NLP 流水线**

![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-XLM--RoBERTa-FFD43B?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green)
![Year](https://img.shields.io/badge/year-2023-blue)

[한국어](./README.md) · [English](./README.md#english) · **中文**

</div>

---

## 概览

> YouTube 视频 URL → 字幕文本 → REAL / FAKE 标签

只需提供一个 YouTube URL,流水线会下载音频、提取字幕,然后将文本输入多语言
XLM-RoBERTa 分类器,输出 **REAL / FAKE** 标签。这是一个学生项目,目的是探索
能否将视频中的口头内容还原为文本,再套用虚假新闻分类模型来判断真伪。

## 流水线

```
YouTube URL
    │
    ▼  src/extract_subtitles.py · notebooks/01_subtitle_extraction.ipynb
音频 (.mp3) → WhisperX → 字幕 (.srt) → 清洗后的 (.txt)
    │
    ▼  src/train.py · src/predict.py · notebooks/02_fake_news_classification.ipynb
XLM-RoBERTa (二分类头) → REAL / FAKE
```

## 为什么拆成两个阶段

WhisperX 需要 **torch 1.13.1 (cu117)**,而分类器需要 **PyTorch ≥ 2.0**,二者
无法共存。因此字幕提取阶段被放在独立的虚拟环境 (`whisper-env`) 中,通过
子进程调用 `whisperx` 可执行文件。两个阶段之间通过中间产物 (`.txt`) 衔接。

## 主要功能

- **字幕提取流水线** — `yt-dlp` 下载音频 → WhisperX (large-v2 + wav2vec2 对齐) → 清洗后的 `.txt`
- **二分类器** — 基于 `symanto/xlm-roberta-base-snli-mnli-anli-xnli`,将原本 3 类的 NLI 头替换为重新初始化的 2 类头 (`num_labels=2` + `ignore_mismatched_sizes=True`)
- **Notebook 与 CLI 双入口** — 每个步骤都可以通过 `notebooks/` 点击运行,也可以通过 `python -m src.<step>` 在脚本中调用

## 技术栈

- **语言**: Python 3.10+
- **语音识别**: [WhisperX](https://github.com/m-bain/whisperx) v2.0.1 (Whisper large-v2 + wav2vec2 对齐)
- **音频下载**: `yt-dlp`、`ffmpeg`
- **分类**: Hugging Face `transformers` (≥ 4.36, < 4.46)、`datasets`、`evaluate`、`accelerate`
- **训练框架**: PyTorch ≥ 2.0 + `Trainer`
- **数据**: `pandas`、`scikit-learn`
- **硬件**: 推荐 GPU (Colab T4 等)

## 项目结构

```
yt-fakenews-classifier/
├── src/
│   ├── __init__.py
│   ├── extract_subtitles.py   # YouTube → mp3 → srt → txt (WhisperX)
│   ├── data.py                # CSV 加载、NewsDataset、train/val/test 划分
│   ├── model.py               # 分词器 / 分类器工厂 (二分类头)
│   ├── train.py               # 训练流水线 (notebook 与 CLI 共用)
│   └── predict.py             # 文本文件推理 (notebook 与 CLI 共用)
├── notebooks/
│   ├── 01_subtitle_extraction.ipynb    # 调用 src.extract_subtitles 的入口 notebook
│   └── 02_fake_news_classification.ipynb # 调用 src.train + src.predict 的入口 notebook
├── data/
│   ├── dataset.zip            # fake_or_real_news.csv (~6,300 条,训练用)
│   └── sample_input.txt       # 推理测试用字幕样本
├── requirements.txt
├── .gitignore
└── README.md
```

## 使用方法

### Google Colab

每个 notebook 都可以直接在 GPU 运行时中打开。

- 字幕提取:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/01_subtitle_extraction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
- 虚假新闻分类:
  <a target="_blank" href="https://colab.research.google.com/github/jumincho/yt-fakenews-classifier/blob/main/notebooks/02_fake_news_classification.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

在 Colab 中请先 `git clone` 仓库,然后从 `notebooks/` 目录打开
(notebook 会从 `src/` 导入代码)。

### 本地

```bash
git clone https://github.com/jumincho/yt-fakenews-classifier.git
cd yt-fakenews-classifier
pip install -r requirements.txt

# 训练数据首次运行时会从 dataset.zip 自动解压。

# (1) 字幕提取 —— 首次需要先创建 WhisperX 虚拟环境
virtualenv whisper-env
whisper-env/bin/pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.1 torchaudio==0.13.1
whisper-env/bin/pip install git+https://github.com/m-bain/whisperx.git@v2.0.1
export WHISPERX_BIN=$(pwd)/whisper-env/bin/whisperx

python -m src.extract_subtitles "https://www.youtube.com/watch?v=4sC-k-92JBE" --out-dir output

# (2) 训练分类器
python -m src.train --epochs 3 --batch-size 8 --seed 42

# (3) 对字幕文件预测 REAL/FAKE
python -m src.predict output/audio.txt --model-dir output/final
```

推荐使用 GPU (CUDA),在 T4 上约 30 分钟。

## 数据集

- **训练**: `data/dataset.zip` 内的 `fake_or_real_news.csv` (~6,300 条英文新闻,字段为 `title` · `text` · `label(REAL/FAKE)`)
- **推理示例**: `data/sample_input.txt` —— notebook 01 产生的字幕文本示例

## 许可证

[MIT License](./LICENSE)
