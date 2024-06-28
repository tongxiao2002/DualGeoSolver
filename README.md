# DualGeoSolver

Official implementation of the IJCAI-24 paper [Learning to Solve Geometry Problems via Simulating Human Dual-Reasoning Process](https://arxiv.org/abs/2405.06232).

## Datasets & Checkpoints

All the datasets (GeoQA and GeoQA+) and checkpoints can be found in [here](https://rec.ustc.edu.cn/share/fd0812c0-1dc5-11ef-9485-a15417b8b58d).

Unzip the tarball by running:

```bash
tar -zxvf GeoQA-Data.tar.gz
```

The GeoQA dataset is in `GeoQA-Pro` folder and GeoQA+ dataset is in `GeoQA+` folder. The GeoQA+ dataset is slightly different from the [original one](https://github.com/SCNU203/GeoQA-Plus), because we have fix some annotation errors in original GeoQA+. For example, there are some program sequences which have only one operand following a binary operator in the original GeoQA+ dataset.

Our constructed knowledge base is presented in `geometry-knowledges-chinese.json` and `geometry-knowledges-english.json` for Chinese version and English version respectively (in directory of each dataset). While training, we only utilize the Chinese version. The `GPT-3.5-Turbo` annotated geometry knowledge for each reasoning step is presented in `geometry-knowledge-data-map.json`.

## Run

### Preparation

Firstly, download ViTMAE checkpoint from [here](https://rec.ustc.edu.cn/share/fd0812c0-1dc5-11ef-9485-a15417b8b58d) and put it in your working directory.

Secondly, download Chinese-RoBERTa checkpoint from [huggingface](https://huggingface.co/hfl/chinese-roberta-wwm-ext) and put it in `chinese-roberta-wwm-ext` directory.

Now, your working directory should looks like this:

```
|--config
|
|--chinese-roberta-wwm-ext
|
|--vit-64-base-160epoch
|
|--GeoQA-Data
|  |--GeoQA-Pro
|  |--GeoQA+
|
|--ManualProgram
|
|--requirements.txt
|
|--train.py
...
```

### Training

You can adjust hyper parameters in `config/solver.json`. After you finish this, run:

```bash
allennlp train config/solver.json --include-package train -s /path/to/save
```

### Evaluating

Run:

```
allennlp evaluate /path/to/checkpoint_dir GeoQA-Data/GeoQA-Pro/pro_test.pk --include-package evaluate --cuda-device 0
```

## Refactor codes are coming soon ...
