# Inspiration-to-Structure

Official reproduction code for AAAI2026: **Is Symbolic Music a Specific Language? Exploring Inspiration-to-Structure Machine Composition via LLMs**.

## Data

Training dataset on HuggingFace:

```text
cszhu09876/ios-disco-abc
```

Original section-level POP909 annotation package:

```text
data/pop909_section.zip
```

The zip file contains the original section-level MIDI and ABC data. The HuggingFace dataset contains the chat-format data used by the training script.

## Installation

```bash
conda create -n ios python=3.10 -y
conda activate ios
pip install -r requirements.txt
```

If needed, install the Unsloth build matching your CUDA/PyTorch version before running training.

## Check Dataset

Check the HuggingFace dataset:

```bash
python scripts/inspect_data.py \
  --data_path cszhu09876/ios-disco-abc \
  --split train \
  --max_scan 1000
```

Download the dataset locally:

```bash
python scripts/download_dataset.py
```

This creates:

```text
data/train_dataset_abc/
data/eval_dataset_abc/
```

Check the local dataset:

```bash
python scripts/inspect_data.py \
  --data_path ./data/train_dataset_abc \
  --max_scan 1000
```

## Train

Train directly from HuggingFace:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train.py \
  --config configs/llama32_1b.yaml \
  --logging_steps 100 \
  --log_contrastive_every 100 \
  --save_steps 1000 \
  --optim adamw_8bit
```

Train from local downloaded data:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/train.py \
  --train_dataset ./data/train_dataset_abc \
  --eval_dataset ./data/eval_dataset_abc \
  --model_name unsloth/Llama-3.2-1B-Instruct \
  --output_dir ./outputs/llama1b_ios_disco \
  --max_seq_length 4096 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --lambda_contrastive 0.5 \
  --gamma 0.1 \
  --logging_steps 100 \
  --log_contrastive_every 100 \
  --save_steps 1000 \
  --optim adamw_8bit
```

The trained LoRA adapter and tokenizer are saved to:

```text
outputs/llama1b_ios_disco/
```

## Inference

Free-form prompt:

```bash
python scripts/infer.py \
  --model_path ./outputs/llama1b_ios_disco \
  --prompt "Generate a chorus following this ABC melody: X: 1\nM: 4/4\nL: 1/8\nK:C\nCDEF GABc|" \
  --max_new_tokens 1024
```

ABC section prompt:

```bash
python scripts/infer.py \
  --model_path ./outputs/llama1b_ios_disco \
  --abc "X: 1\nM: 4/4\nL: 1/8\nK:C\nCDEF GABc|" \
  --section chorus \
  --max_new_tokens 1024
```

## Files

```text
configs/llama32_1b.yaml      Training config for Llama-3.2-1B-Instruct.

ios_disco/data.py            Dataset loading, split selection, and dataset inspection.
ios_disco/modeling.py        Unsloth Llama model/tokenizer loading.
ios_disco/span_utils.py      Token span detection for ABC blocks wrapped by {{...}}.
ios_disco/trainer.py         SFT trainer with the DiSCO contrastive objective.

scripts/download_dataset.py  Downloads cszhu09876/ios-disco-abc into ./data.
scripts/inspect_data.py      Prints dataset columns, section labels, and ABC block counts.
scripts/train.py             Fine-tunes Llama with SFT and DiSCO.
scripts/infer.py             Runs inference with a trained LoRA adapter.

data/pop909_section.zip      Original section-level POP909 MIDI/ABC annotation package.
data/README.md               Data source and download notes.

examples/free_prompt.txt     Example free-form prompt for inference.
```

## Citation

```bibtex
@inproceedings{hu2026symbolic,
  title={Is Symbolic Music a Specific Language? Exploring Inspiration-to-Structure Machine Composition via LLMs},
  author={Hu, Zhejing and Liu, Yan and Zhang, Zhi and Zhang, Aiwei and Zhong, Sheng-hua and Yu, Bruce XB and Chen, Gong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={3},
  pages={1837--1845},
  year={2026}
}
```
