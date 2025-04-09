# Brahma LLM API Documentation

This document provides a comprehensive guide to the Brahma LLM Platform API endpoints, including authentication, request/response formats, and example usage.

## Table of Contents

1. [Authentication](#authentication)
2. [Text Generation API](#text-generation-api)
3. [Chat API](#chat-api)
4. [Model Management API](#model-management-api)
5. [Dataset Management API](#dataset-management-api)
6. [Training API](#training-api)
7. [User Management API](#user-management-api)
8. [API Key Management](#api-key-management)
9. [Utility Endpoints](#utility-endpoints)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [WebSocket API](#websocket-api)

## Authentication

The Brahma LLM API uses JWT (JSON Web Tokens) for authentication. To access protected endpoints, you need to:

1. Obtain an access token by logging in
2. Include the token in the `Authorization` header of subsequent requests

### Getting an Access Token

```http
POST /api/token
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

#### Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Using the Access Token

Include the token in the `Authorization` header:

```http
GET /protected-endpoint
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Using API Keys

For programmatic access, you can use API keys instead of JWT tokens:

```http
POST /api/generate
Authorization: Bearer your_api_key
Content-Type: application/json

{
  "prompt": "Hello, world!"
}
```

## Text Generation API

Generate text from a prompt using a language model.

### Generate Text

```http
POST /api/generate
Authorization: Bearer your_token_or_api_key
Content-Type: application/json

{
  "prompt": "Once upon a time in a galaxy far, far away",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "seed": 42
}
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | string | The input text to generate from | *Required* |
| `max_new_tokens` | integer | Maximum number of new tokens to generate | 100 |
| `temperature` | float | Controls randomness: higher values increase diversity | 0.7 |
| `top_k` | integer | Only sample from the top k tokens | 50 |
| `top_p` | float | Only sample from tokens with cumulative probability < top_p | 0.9 |
| `repetition_penalty` | float | Penalty for repeating tokens | 1.1 |
| `do_sample` | boolean | Whether to use sampling (true) or greedy decoding (false) | true |
| `seed` | integer | Random seed for reproducibility | null |

#### Response

```json
{
  "generated_text": "Once upon a time in a galaxy far, far away, there existed a civilization of beings who had mastered interstellar travel...",
  "prompt": "Once upon a time in a galaxy far, far away",
  "config": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": true
  }
}
```

## Chat API

Interact with the language model in a chat format.

### Chat Completion

```http
POST /api/chat
Authorization: Bearer your_token_or_api_key
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "seed": null
}
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `messages` | array | Array of message objects with `role` and `content` | *Required* |
| `max_new_tokens` | integer | Maximum number of new tokens to generate | 100 |
| `temperature` | float | Controls randomness: higher values increase diversity | 0.7 |
| `top_k` | integer | Only sample from the top k tokens | 50 |
| `top_p` | float | Only sample from tokens with cumulative probability < top_p | 0.9 |
| `repetition_penalty` | float | Penalty for repeating tokens | 1.1 |
| `seed` | integer | Random seed for reproducibility | null |

#### Response

```json
{
  "response": "The capital of France is Paris.",
  "config": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }
}
```

## Model Management API

Manage language models.

### List Models

```http
GET /list-models
Authorization: Bearer your_token
```

#### Response

```json
{
  "models": [
    {
      "name": "checkpoint-final.pt",
      "path": "checkpoints/checkpoint-final.pt",
      "size": 5368709120,
      "modified": 1649278362.0
    },
    {
      "name": "brahma-7b.pt",
      "path": "checkpoints/brahma-7b.pt",
      "size": 14000000000,
      "modified": 1649279362.0
    }
  ]
}
```

### Reload Model

```http
POST /reload-model
Authorization: Bearer your_token
Content-Type: application/json

{
  "model_path": "checkpoints/brahma-7b.pt",
  "config_path": null,
  "tokenizer_path": null
}
```

#### Response

```json
{
  "status": "success",
  "message": "Model reloaded from checkpoints/brahma-7b.pt"
}
```

### Get Model Configuration

```http
GET /model-config
Authorization: Bearer your_token
```

#### Response

```json
{
  "model_config": {
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
}
```

## Dataset Management API

Manage training datasets.

### List Datasets

```http
GET /list-datasets
Authorization: Bearer your_token
```

#### Response

```json
{
  "datasets": [
    {
      "name": "instruction-tuning",
      "path": "data/instruction-tuning",
      "has_training": true,
      "has_validation": true,
      "train_size": 1245678
    },
    {
      "name": "custom-data",
      "path": "data/custom-data",
      "has_training": true,
      "has_validation": false,
      "train_size": 456789
    }
  ]
}
```

### Upload Dataset

```http
POST /upload-dataset
Authorization: Bearer your_token
Content-Type: multipart/form-data

name=my-dataset
description=Custom training data
file=@/path/to/file.jsonl
```

#### Response

```json
{
  "status": "success",
  "dataset": {
    "name": "my-dataset",
    "path": "data/my-dataset",
    "file_count": 1,
    "size": 1245678
  }
}
```

## Training API

Manage model training jobs.

### Start Training

```http
POST /start-training
Authorization: Bearer your_token
Content-Type: application/json

{
  "model_config": {
    "model_type": "brahma",
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
  },
  "training_config": {
    "lr": 1e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "epochs": 3,
    "warmup_steps": 100
  },
  "data_config": {
    "train_file": "data/custom-data/train.jsonl",
    "validation_file": "data/custom-data/validation.jsonl"
  },
  "tokenizer_config": {
    "tokenizer_path": "tokenizers/brahma-tokenizer"
  }
}
```

#### Response

```json
{
  "status": "started",
  "message": "Training started in background",
  "training_id": "training-2025-04-08-12345"
}
```

### Get Training Status

```http
GET /training-status/training-2025-04-08-12345
Authorization: Bearer your_token
```

#### Response

```json
{
  "status": "running",
  "message": "Training in progress",
  "progress": 0.45,
  "metrics": {
    "loss": 2.1,
    "perplexity": 8.2,
    "learning_rate": 9.5e-6,
    "step": 500,
    "epoch": 1.5
  }
}
```

## User Management API

Manage users and user data.

### Register User

```http
POST /register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "New User"
}
```

#### Response

```json
{
  "username": "newuser",
  "email": "user@example.com",
  "full_name": "New User",
  "is_admin": false
}
```

### Get Current User Info

```http
GET /users/me
Authorization: Bearer your_token
```

#### Response

```json
{
  "username": "newuser",
  "email": "user@example.com",
  "full_name": "New User",
  "is_admin": false,
  "disabled": false
}
```

### Update User Info

```http
PUT /users/me
Authorization: Bearer your_token
Content-Type: application/json

{
  "email": "newemail@example.com",
  "full_name": "Updated Name",
  "bio": "AI enthusiast and researcher"
}
```

#### Response

```json
{
  "username": "newuser",
  "email": "newemail@example.com",
  "full_name": "Updated Name",
  "bio": "AI enthusiast and researcher",
  "is_admin": false,
  "disabled": false
}
```

### Change Password

```http
PUT /users/me/password
Authorization: Bearer your_token
Content-Type: application/json

{
  "current_password": "secure_password",
  "new_password": "even_more_secure_password"
}
```

#### Response

```json
{
  "message": "Password updated successfully"
}
```

### Update User Preferences

```http
PUT /users/me/preferences
Authorization: Bearer your_token
Content-Type: application/json

{
  "default_model": "brahma-7b",
  "default_temperature": 0.8,
  "stream_responses": true,
  "theme": "dark"
}
```

#### Response

```json
{
  "message": "Preferences updated successfully",
  "preferences": {
    "default_model": "brahma-7b",
    "default_temperature": 0.8,
    "stream_responses": true,
    "theme": "dark"
  }
}
```

## API Key Management

Manage API keys for programmatic access.

### Create API Key

```http
POST /api-keys
Authorization: Bearer your_token
Content-Type: application/x-www-form-urlencoded

name=My API Key
```

#### Response

```json
{
  "id": 1,
  "name": "My API Key",
  "key": "brm_api_1234567890abcdef",
  "created_at": "2025-04-08T12:00:00Z"
}
```

### List API Keys

```http
GET /api-keys
Authorization: Bearer your_token
```

#### Response

```json
[
  {
    "id": 1,
    "name": "My API Key",
    "key": "brm_api_1234************",
    "created_at": "2025-04-08T12:00:00Z",
    "last_used": "2025-04-08T13:30:00Z"
  },
  {
    "id": 2,
    "name": "Production Key",
    "key": "brm_api_5678************",
    "created_at": "2025-04-07T10:00:00Z",
    "last_used": null
  }
]
```

### Delete API Key

```http
DELETE /api-keys/1
Authorization: Bearer your_token
```

#### Response

```json
{
  "message": "API key deleted successfully"
}
```

## Utility Endpoints

Miscellaneous utility endpoints.

### API Status

```http
GET /api/status
```

#### Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "uptime": 45678
}
```

### Default Configuration

```http
GET /default-config
Authorization: Bearer your_token
```

#### Response

```json
{
  "model_config": {
    "model_type": "brahma",
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32
  },
  "training_config": {
    "lr": 1e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "epochs": 3,
    "warmup_steps": 100
  },
  "data_config": {
    "train_file": "",
    "validation_file": ""
  },
  "tokenizer_config": {
    "tokenizer_path": "tokenizers/brahma-tokenizer"
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- `200 OK`: The request succeeded
- `201 Created`: A new resource was created
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failed
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error responses have a consistent format:

```json
{
  "detail": "Error message explaining what went wrong"
}
```

## Rate Limiting

API requests are subject to rate limiting to prevent abuse. Rate limits are applied:

- Per IP address
- Per authenticated user
- Per API key

Rate limits are specified in the response headers:

- `X-RateLimit-Limit`: Maximum requests per time window
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Time in seconds until the rate limit resets

When a rate limit is exceeded, the API returns a `429 Too Many Requests` response.

## WebSocket API

For streaming responses, use the WebSocket API.

### Streaming Text Generation

Connect to the WebSocket endpoint:

```
ws://your-domain.com/ws/generate
```

Send a JSON message:

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "stream": true,
  "api_key": "your_api_key"
}
```

The server will send multiple messages with generated tokens:

```json
{"token": "Once"}
{"token": " upon"}
{"token": " a"}
{"token": " time"}
{"token": ","}
{"token": " there"}
...
{"token": "[EOS]", "finished": true}
```
