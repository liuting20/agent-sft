# Who_and_When SFT for Qwen

这个项目用于把 `Kevin355/Who_and_When` 数据集整理成监督微调格式，并使用 `TRL + PEFT + QLoRA` 微调 Qwen 模型。

默认配置面向：

- 基座模型：`Qwen/Qwen3-8B`
- 训练方法：`SFT + LoRA`
- 显存优化：`4-bit QLoRA`

`Qwen3-8B-Thinking` 这个名字在实际使用时通常对应“Qwen3 8B 基座 + Thinking 模式/推理模板”。为了兼容 Hugging Face 上更常见的模型命名，这个项目默认用 `Qwen/Qwen3-8B` 作为 `--model_name_or_path`，你也可以替换成你自己实际要训练的模型仓库名。

## 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

如果你使用 CUDA，请确保本机 `PyTorch`、`bitsandbytes` 和显卡驱动匹配。

## 2. 准备数据

```bash
bash run_prepare.sh
```

它会：

- 下载 `Kevin355/Who_and_When`
- 默认读取配置 `Algorithm-Generated`
- 自动识别问题列和答案列
- 生成 `train/validation` 划分
- 保存到 `data/processed`

如果自动识别失败，可以手动指定字段：

```bash
python3 scripts/prepare_dataset.py \
  --dataset_name Kevin355/Who_and_When \
  --dataset_config Algorithm-Generated \
  --question_field question \
  --answer_field answer \
  --output_dir data/processed
```

## 3. 开始训练

如果你是在 A100 服务器上训练，建议直接用下面的脚本。

```bash
bash run_train.sh
```

如果你的卡型明确，也可以直接选：

```bash
bash run_train_a100_40g.sh
```

```bash
bash run_train_a100_80g.sh
```

或者手动执行：

```bash
python3 train_sft.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_path data/processed \
  --output_dir outputs/qwen3-8b-thinking-whoandwhen \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --max_seq_length 2048 \
  --use_bf16 \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2
```

## 4. 推理测试

```bash
python3 inference.py \
  --adapter_path outputs/qwen3-8b-thinking-whoandwhen \
  --question "Who was the president of the United States in 1995?"
```

## 5. 常用参数

- `--disable_4bit`：关闭 QLoRA，改成普通精度加载
- `--use_bf16`：使用 `bfloat16`
- `--gradient_checkpointing`：降低显存占用
- `--question_field` / `--answer_field`：手动指定数据集列名
- `--val_size`：验证集比例

## 6. 显存建议

- `Qwen 8B + QLoRA`：通常建议至少 `24GB` 显存，实际取决于序列长度和梯度累计
- 如果显存不足，可以把 `--max_seq_length` 改成 `1024`
- 也可以进一步降低 `LoRA r` 或减少 batch size

如果你用的是 A100：

- `A100 40GB`：建议从 `batch_size=4, grad_acc=8, seq_len=2048` 起步
- `A100 80GB`：可以尝试 `batch_size=8, grad_acc=4, seq_len=4096`
- 默认优先使用 `bf16 + flash_attention_2 + QLoRA`
- 如果服务器环境里还没装 `flash-attn`，把 `--attn_implementation flash_attention_2` 去掉也能跑

## 7. 输出结果

训练完成后，LoRA adapter 会保存在：

```text
outputs/qwen3-8b-thinking-whoandwhen
```

如果你想把 LoRA adapter 合并成完整模型，可以执行：

```bash
python3 merge_lora.py \
  --adapter_path outputs/qwen3-8b-thinking-whoandwhen \
  --output_dir outputs/qwen3-8b-thinking-whoandwhen-merged
```
