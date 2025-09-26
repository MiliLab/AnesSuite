
<p align="center">

  <h2 align="center"><strong>AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs</strong></h2>

<div align="center">
<h5>
<em>Xiang Feng<sup>1 *</sup>, Wentao Jiang<sup>1 *</sup>, Zengmao Wang<sup>1</sup>, Yong Luo<sup>1 †</sup>, Pingbo Xu<sup>2,3</sup>, Baosheng Yu<sup>4</sup>,<br/> Hua Jin<sup>5,6</sup>, Bo Du<sup>1 †</sup>, Jing Zhang<sup>1 †</sup> </em>
    <br><br>
       	<sup>1</sup> School of Computer Science, Wuhan University, China,<br/>
        <sup>2</sup> Department of Anesthesiology, Zhejiang Cancer Hospital, China,<br/> 
        <sup>3</sup> Institute of Medicine, Chinese Academy of Sciences, Hangzhou, Zhejiang, China<br/> 
        <sup>4</sup> Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore<br/> 
        <sup>5</sup> Department of Anesthesiology, First People’s Hospital of Yunnan Province, China<br/> 
        <sup>6</sup> Kunming University of Science and Technology, China<br/> 
</h5>
<h5>
<sup>∗</sup> Equal contribution, <sup>†</sup> Corresponding author
</h5>
</div>



<h5 align="center">
<a href="https://mililab.github.io/anesbench.ai/"> <img src="https://img.shields.io/badge/Project_Page-AnesSuite-test?logo=Github&color=green"></a> <a href="https://arxiv.org/abs/2504.02404"> <img src="https://img.shields.io/badge/Arxiv-2504.02404-b31b1b.svg?logo=arXiv"></a> <a href="https://huggingface.co/datasets/MiliLab/AnesBench"></a>
</h5>

<figure>
<div align="center">
<img src=figs/logo.png width="20%">
</div>
</figure>

# 🐨  Contents

- [🔥 Update](#-update)
- [🌞 Intro](#-intro)
- [🔍 Overview](#-overview)
- [📖 Datasets](#-datasets)
  - [AnesBench](#anesbench)
  - [AnesCorpus](#anescorpus)
  - [AnesQA](#anesqa)
  - [AnesR1](#anesr1)
- [🐎 Leaderboard](#-leaderboard)
- [🔨 Evaluation](#-evaluation)
- [🛠️ Training with LLaMA-Factory](#️-training-with-llama-factory)
- [⭐ Citation](#-citation)



# 🔥 Update
**2025.09.26**
- We updated the latest progress.

**2025.05.14**
- We released the evaluation code along with usage instructions.

**2025.04.04**
- We uploaded our work on [arXiv](https://arxiv.org/abs/2504.02404).

**2025.03.31**
- We released the [AnesSuite project page](https://mililab.github.io/anesbench.ai/).


# 🌞 Intro
AnesSuite is a benchmark and dataset suite for advancing LLM reasoning in anesthesiology. It provides bilingual benchmark and curated training resources (AnesCorpus, AnesQA, AnesR1) to support CPT, SFT, and RLVR. 

Built on this foundation, Morpheus is first baseline model collection (7B & 14B) for anesthesiology reasnoning. Together, AnesSuite and Morpheus offer a practical infrastructure for research and development of advanced anesthesiology LLMs.


# 🔍 Overview
<figure>
<div align="center">
<img src="figs/overview.png">
</div>
<div align="center">
<figcaption align = "center"><b>Figure 1: Overview of the AnesSuite. 
 </b></figcaption>
</div>
</figure>

# 📖 Datasets

## AnesBench

**AnesBench** is designed to assess anesthesiology-related reasoning capabilities of Large Language Models (LLMs). It contains 7,972 anesthesiology MCQs (≈4.4k English / 3.5k Chinese). Each question is labeled with a three-level categorization of cognitive demands, enabling evaluation of LLMs’ knowledge, application, and clinical reasoning abilities across diverse linguistic contexts.

## AnesCorpus

**AnesCorpus** is a large-scale, domain-specific corpus constructed for CPT in the field of anesthesiology.

| Language | Rows    |
|----------|---------|
| English  | ~1.8M   |
| Chinese  | ~0.6M   |

This curated dataset provides a rich foundation for pretraining language models to understand anesthesiology-related concepts, terminology, and clinical context.

## AnesQA

**AnesQA** is a QA dataset designed for SFT. The QA pairs are generated and filtered using advanced large language models.

| Language | QA Pairs |
|----------|----------|
| English  | ~20K   |

AnesQA enables the development of instruction-tuned models with robust reasoning and answering capabilities in the anesthesiology domain.

## AnesR1
AnesR1 contains over 10k instances, each featuring a verifiable MCQ and a detailed reasoning chain, making it well-suited for both SFT and RLVR.

| Language | QA Pairs |
|----------|----------|
| English  | ~3.2K   |
| Chinese  | ~7K   |


### Recommended Usage

- AnesBench: Use as the primary evaluation benchmark to measure LLM performance across factual recall, hybrid reasoning, and complex decision-making in anaesthesiology.

- AnesCorpus: Apply for CPT to enhance domain knowledge before fine-tuning.

- AnesQA: Use for SFT.

- AnesR1: Use for SFT or RLVR to strengthen reasoning capability.

# 🔨 Evaluation

---

## 📁 0. Clone the Repository & Download Benchmark

Clone Repository:

```bash
git clone https://github.com/MiliLab/AnesBench
cd AnesBench
```

Download Benchmark:
```bash
cd benchmark
huggingface-cli download --repo-type dataset  MiliLab/AnesBench --local-dir ./
```
---

## 🧱 1. Prepare the Runtime Environment

Before starting, ensure that `CUDA` and its compiler `nvcc` are properly installed and accessible.

### Check:
```bash
nvcc --version
```

We recommend separating the SGLang service environment from the inference environment.

### SGLang service environment

```bash
conda create -n sglang_server python==3.10
conda activate sglang_server
```

Then, install the required `sglang` and `flashinfer` packages.

```bash
pip install "sglang[all]"
pip install sglang-router 
```
Download the wheel file for your environment from [https://github.com/flashinfer-ai/flashinfer/releases](https://github.com/flashinfer-ai/flashinfer/releases).

```bash
pip install /path/to/flashinfer-wheel
```

### Inference environment

Create a new environment and install the packages based on the requirements file.

```bash
conda create -n inference python==3.10
conda activate inference
cd eval
pip install -r requirements.txt
```
---

### Environment Variables

Prepare environment variables in the `.env` file.

```bash
export RESULT_SAVE_PATH=/path/to/result_save_dir
export MODEL_PATH=/path/to/model
export BENCHMARK_PATH=/path/to/benchmark
```

and run:

```bash
source .env
```

## ▶️ 2. Run Evaluation

### For SGLang service
```bash
bash sglang_server.sh 
```

### For Inference
```bash
python ./evaluate.py --config ./config.yaml 
```

---

# 🛠️ Training with LLaMA-Factory

To train with **AnesCorpus** (for CPT) and **AnesQA** (for SFT) using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), follow the steps below:


## 1️. Install LLaMA-Factory

Follow the [LLaMA-Factory official installation guide](https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html), or use the following scripts:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## 2. Convert Data to LLaMA-Factory Format

We provide scripts to convert the raw Parquet files into the required JSON format.

> 📌 The `--split` argument can be set to:
> - `en`: English data only  
> - `cn`: Chinese data only  
> - `all`: merge both English and Chinese

#### For AnesCorpus (CPT):
```bash
python tools/anescorpus2json.py \
    --local-dir /path/to/anescorpus/parquet_files \
    --save-dir ./data \
    --split en
```
This will generate:  
`
./data/AnesCorpus_en.json
`


#### For AnesQA (SFT):
```bash
python tools/anescorpus2json.py \
    --local-dir /path/to/anesqa/parquet_files \
    --save-dir ./data \
    --split en \
    --instruction "Please answer the following question based on the anesthesiology context."
```

This will generate:  
`
./data/AnesCorpus_en.json
`

## 3. Register the Dataset
Move your dataset in `LLaMA-Factory/data`, and register your dataset entries in `LLaMA-Factory/data/dataset_info.json/`. 


```json
{
  "anescorpus_en": {
    "file_name": "AnesCorpus_en.json",
    "columns": {
      "prompt": "text"
    }
  },
  "anesqa_en": {
    "file_name": "AnesQA_en.json",
  }
}
```

For more details on dataset registration and formatting, refer to the official data preparation guide in [manual](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html) and [github](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).

## 4. Set Config File
You can use or modify the example config files we provide in `configs/`.

Edit them to set paths like:

```yaml
// Example snippet
dataset_dir: LLaMA-Factory/data    // Directory contains "dataset_info.json"
dataset: anesqa_en
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
output_dir: ./output/llama3.1-anesqa-sft
...
```
More details can be found in [official guide](https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html).

## 5. Launch Training from CLI
### Continuous Pre-training (CPT)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train configs/qwen2.5-7b-pt-anesthesia.yaml
```

### Supervised Fine-Tuning (SFT)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train configs/qwen2.5-7b-sft-anesthesia.yaml
```


# ⭐ Citation

If you find AnesBench helpful, please consider giving this repo a ⭐ and citing:

```latex
@article{AnesBench,
  title={AnesBench: Multi-Dimensional Evaluation of LLM Reasoning in Anesthesiology},
  author={Xiang Feng and Wentao Jiang and Zengmao Wang and Yong Luo and Pingbo Xu and Baosheng Yu and Hua Jin and Bo Du and Jing Zhang},
  journal={arXiv preprint arXiv:2504.02404},
  year={2025}
}
```
