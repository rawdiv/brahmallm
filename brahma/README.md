# Brahma: Your Own LLM Project

Brahma is a comprehensive framework for training and using your own large language model (LLM) similar to ChatGPT. This project includes everything you need to train, fine-tune, and deploy your custom LLM.

## Features

- **Complete Model Architecture**: Transformer-based decoder-only architecture with modern techniques like RoPE embeddings, grouped-query attention, and more
- **Training Pipeline**: Efficient training with mixed precision, gradient accumulation, and learning rate scheduling
- **Inference Tools**: Text generation and chat interfaces for easy model usage
- **Customizable**: Configure model size, training parameters, and more via config files
- **Web Platform**: Full-featured web interface for managing models, datasets, and user accounts
- **API Integration**: Comprehensive RESTful API for programmatic access
- **Database Backend**: SQLAlchemy-based database for user management and platform metrics
- **Monitoring System**: Detailed logging and metrics collection for production deployments

## Key Features

### Model Training and Inference
- Train models of various sizes (1B to 70B parameters)
- Fine-tune on custom datasets
- Efficient inference with optional quantization
- Batched generation for higher throughput

### Web Platform
- User authentication and account management
- Model selection and parameter configuration
- Training job submission and monitoring
- Usage statistics and API key management

### API Access
- Text generation endpoints
- Chat completion
- Model and dataset management
- User preference settings

### Deployment Options
- Local development setup
- Production deployment with ASGI servers
- Docker containerization
- Cloud deployment guides

## Documentation

Brahma comes with comprehensive documentation:

- [API Documentation](./API_DOCUMENTATION.md) - Detailed API endpoint reference
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment instructions
- [Training Guide](./TRAINING.md) - How to train and fine-tune models

## Project Structure

```
brahma/
├── api/              # API server and endpoints
│   ├── auth.py       # Authentication and user management
│   ├── database.py   # Database models and functions
│   ├── monitoring.py # Logging and metrics collection
│   ├── server.py     # FastAPI server implementation
│   ├── templates/    # HTML templates for web UI
│   └── webui.py      # Gradio web interface
├── config/           # Configuration files
├── data/             # Training data
├── models/           # Model architecture
├── training/         # Training pipeline
├── utils/            # Utility functions
├── inference/        # Inference code
├── train.py          # Training script
└── README.md         # This file
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data (Optional)

```bash
python train.py --create_sample_data --sample_data_size 1000
```

### 2. Train Your Model

```bash
python train.py --config config/default_config.json
```

For distributed training:

```bash
python -m torch.distributed.launch --nproc_per_node=N train.py --config config/default_config.json
```

### 3. Chat with Your Model

```bash
python -m inference.chat --model_path checkpoints/checkpoint-final.pt
```

### 4. Generate Text with Your Model

```bash
python -m inference.generate --model_path checkpoints/checkpoint-final.pt --prompt "Once upon a time"
```

## Training on Your Own Data

1. Prepare your data in jsonl format with a text field:
   
   ```json
   {"text": "Your training example here"}
   ```

2. Update the data configuration in `config/default_config.json`:
   
   ```json
   "data_config": {
     "dataset_path": "path/to/your/data",
     "train_split": "train",
     "validation_split": "validation",
     "text_column": "text"
   }
   ```

3. Start training as described above

## Model Configuration

You can customize the model architecture by editing `config/default_config.json`:

```json
"model_config": {
  "vocab_size": 32000,
  "hidden_size": 4096,  # Model dimension
  "intermediate_size": 11008,  # MLP hidden dimension
  "num_hidden_layers": 32,  # Number of layers
  "num_attention_heads": 32,  # Number of attention heads
  "num_key_value_heads": 8,  # For grouped-query attention
  "max_position_embeddings": 4096  # Maximum sequence length
}
```

Scale the model according to your needs:
- Small (~1B parameters): hidden_size=2048, num_hidden_layers=24
- Medium (~7B parameters): hidden_size=4096, num_hidden_layers=32
- Large (~13B parameters): hidden_size=5120, num_hidden_layers=40

## Training Configuration

Adjust training parameters in `config/default_config.json`:

```json
"training_config": {
  "learning_rate": 1e-4,
  "weight_decay": 0.01,
  "warmup_steps": 500,
  "max_steps": 100000,
  "gradient_accumulation_steps": 1,
  "mixed_precision": true,
  "batch_size": 8
}
```

## Using a Custom Tokenizer

For real-world applications, it's recommended to use a SentencePiece tokenizer:

1. Train a SentencePiece tokenizer on your corpus
2. Update the tokenizer configuration in `config/default_config.json`
3. Use the tokenizer during training and inference

## Advanced Usage

### Fine-tuning

To fine-tune an existing checkpoint:

```bash
python train.py --config config/finetune_config.json --resume_from_checkpoint path/to/checkpoint.pt
```

### Evaluating Your Model

```bash
python -m evaluation.evaluate --model_path checkpoints/checkpoint-final.pt --eval_dataset path/to/evaluation/data
```

### Exporting to Different Formats

To export your model to ONNX format:

```bash
python -m conversion.export_onnx --model_path checkpoints/checkpoint-final.pt --output_path model.onnx
```

## Acknowledgements

This project is inspired by modern LLM architectures like Llama, GPT, Mistral, and others.

## License

[MIT License](LICENSE)
