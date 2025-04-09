# Brahma LLM Training Guide

This comprehensive guide explains how to train and fine-tune your own language models using the Brahma LLM platform.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Preparing Training Data](#preparing-training-data)
4. [Model Configuration](#model-configuration)
5. [Training Process](#training-process)
6. [Fine-tuning Existing Models](#fine-tuning-existing-models)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Training Performance Tips](#training-performance-tips)
9. [Monitoring Training Progress](#monitoring-training-progress)
10. [Evaluating Models](#evaluating-models)
11. [Troubleshooting](#troubleshooting)

## Overview

The Brahma LLM platform makes it easy to train your own language models from scratch or fine-tune existing models on custom datasets. The platform supports:

- **Pre-training**: Training a new model from scratch on large text corpora
- **Fine-tuning**: Adapting pre-trained models to specific tasks or domains
- **Instruction tuning**: Teaching models to follow instructions in a conversational setting
- **LoRA/QLoRA**: Efficient parameter-efficient fine-tuning methods

## Prerequisites

Before starting training, make sure you have:

- A machine with at least one CUDA-compatible GPU (recommended: 24GB+ VRAM)
- For larger models: Multiple GPUs or access to a cluster
- Sufficient disk space for datasets and checkpoints (500GB+ recommended)
- Python 3.8+ and all requirements installed

## Preparing Training Data

### Data Formats

Brahma supports the following data formats:

1. **Plain Text** (.txt): Used for pre-training
   ```
   This is an example of plain text data.
   Each line can be a sentence or paragraph.
   ```

2. **JSON Lines** (.jsonl): Used for fine-tuning
   ```json
   {"text": "This is a training example"}
   {"text": "Each line is a separate JSON object"}
   ```

3. **Chat/Instruction Format** (.jsonl): For conversational tuning
   ```json
   {"messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "What is Brahma LLM?"}, {"role": "assistant", "content": "Brahma LLM is a framework for training and using language models."}]}
   ```

4. **Completion Format** (.jsonl): For instruction following
   ```json
   {"prompt": "Write a poem about AI:", "completion": "In circuits deep, where data flows..."}
   ```

### Data Preparation Tools

The Brahma platform includes tools to help prepare your training data:

- **Data Cleaning**: Filter and clean your text corpora
  ```bash
  python -m brahma.tools.data_cleaner --input my_data.txt --output clean_data.txt
  ```

- **Data Conversion**: Convert between different formats
  ```bash
  python -m brahma.tools.data_converter --input my_data.txt --output my_data.jsonl --format chat
  ```

- **Data Augmentation**: Expand your dataset
  ```bash
  python -m brahma.tools.data_augment --input my_data.jsonl --output augmented_data.jsonl --techniques paraphrase,backtranslation
  ```

### Uploading Data

You can upload your data through:

1. **Web Interface**: Navigate to the "Datasets" tab and use the upload form
2. **API Endpoint**: Use the `/upload-dataset` endpoint
3. **Direct File System**: Place files in the `data/` directory

## Model Configuration

### Model Size Options

Brahma supports various model sizes:

| Size | Parameters | Layers | Hidden Size | Attention Heads | Min GPU VRAM |
|------|------------|--------|------------|-----------------|--------------|
| Tiny | 1B         | 16     | 1024       | 16              | 8GB          |
| Small| 3B         | 24     | 2048       | 24              | 16GB         |
| Medium| 7B        | 32     | 4096       | 32              | 24GB         |
| Large| 13B        | 40     | 5120       | 40              | 40GB         |
| XL   | 30B        | 60     | 6144       | 48              | 80GB         |
| XXL  | 70B        | 80     | 8192       | 64              | 160GB        |

### Architecture Configuration

The core architecture can be configured through the web UI or a configuration file:

```json
{
  "model_type": "brahma",
  "vocab_size": 32000,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-6,
  "use_cache": true,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "rope_theta": 10000.0
}
```

## Training Process

### Starting Training

There are multiple ways to start training:

1. **Web Interface**: 
   - Navigate to the "Training" tab
   - Select your model configuration and dataset
   - Set your hyperparameters
   - Click "Start Training"

2. **API Endpoint**:
   ```http
   POST /start-training
   Authorization: Bearer your_token
   Content-Type: application/json

   {
     "model_config": {...},
     "training_config": {...},
     "data_config": {...}
   }
   ```

3. **Command Line**:
   ```bash
   python -m brahma.train --config my_config.json
   ```

### Training Configuration

The training process itself is configured with these parameters:

```json
{
  "training_config": {
    "lr": 1e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 1000,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "fp16": true,
    "bf16": false,
    "seed": 42
  }
}
```

## Fine-tuning Existing Models

### Preparing for Fine-tuning

To fine-tune an existing model:

1. Upload or select the base model
2. Prepare your fine-tuning dataset
3. Configure fine-tuning parameters:
   - Typically use a lower learning rate (1e-5 to 5e-6)
   - Fewer epochs (1-5)
   - Consider parameter-efficient methods like LoRA

### LoRA / QLoRA Fine-tuning

For efficient fine-tuning with limited resources:

```json
{
  "lora_config": {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
  },
  "quantization": {
    "bits": 4,
    "group_size": 128
  }
}
```

## Hyperparameter Optimization

Brahma includes tools for automatically optimizing hyperparameters:

```bash
python -m brahma.tools.hyperopt \
  --config base_config.json \
  --trials 20 \
  --optimize lr,batch_size,warmup_steps \
  --metric perplexity
```

## Training Performance Tips

### Memory Optimization

- **Gradient Checkpointing**: Trades computation for memory
  ```json
  {"gradient_checkpointing": true}
  ```

- **Mixed Precision**: Use FP16 or BF16 for faster training
  ```json
  {"fp16": true, "bf16": false}
  ```

- **DeepSpeed**: For distributed training
  ```json
  {"deepspeed_config": "ds_config.json"}
  ```

### Multi-GPU Training

For training across multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=4 -m brahma.train --config my_config.json
```

## Monitoring Training Progress

### Web Interface

The training progress can be monitored through:

- Real-time loss graphs
- Learning rate schedules
- GPU utilization charts
- Checkpoint management

### Logs and TensorBoard

For detailed monitoring:

```bash
# View logs
tail -f logs/training_run_name.log

# Start TensorBoard
tensorboard --logdir runs/
```

## Evaluating Models

### Built-in Evaluation

The Brahma platform includes several evaluation benchmarks:

```bash
python -m brahma.evaluate --model path/to/model --tasks hellaswag,mmlu,truthfulqa
```

### Custom Evaluation

You can create custom evaluations using:

```python
from brahma.evaluation import Evaluator

evaluator = Evaluator(model_path="path/to/model")
results = evaluator.evaluate_custom(your_test_data)
print(results)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training
   - Try LoRA/QLoRA instead of full fine-tuning

2. **Slow Training**
   - Increase batch size (if memory allows)
   - Use mixed precision
   - Check data loading bottlenecks
   - Optimize model size for your hardware

3. **Poor Convergence**
   - Adjust learning rate
   - Check data quality
   - Increase warmup steps
   - Try different optimizers

### Getting Help

If you encounter issues:

1. Check the logs for detailed error messages
2. Review the [Github issues](https://github.com/yourusername/brahma-llm/issues)
3. Join our community Discord for support
