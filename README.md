# mini-TGI

A lightweight inference server with OpenAI API compatibility, built on top of Hugging Face Transformers.

## Overview

mini-TGI aims to provide a simple, portable inference engine that supports the broadest range of models possible—targeting any model loadable via the Transformers library.

### Key Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI API endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- **Continuous Batching**: Efficient request handling with automatic batching
- **Streaming Support**: Real-time token streaming via Server-Sent Events (SSE)
- **Multi-GPU Support**: Distributed inference via Hugging Face Accelerate
- **Broad Model Support**: Works with any Transformers-compatible Language model

## Installation

### From source

```bash
git clone https://github.com/your-org/mini-TGI.git
cd mini-TGI
pip install -e .
```

## Quickstart

### Start the server

```bash
mini-tgi serve --model-id meta-llama/Llama-3.2-1B-Instruct
```

The server will start on `http://0.0.0.0:8000` by default.

### Make a request

Using curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## CLI Options

```
mini-tgi serve [OPTIONS]

Server options:
  --host TEXT              Host to bind the server to (default: 0.0.0.0)
  --port INTEGER           Port to bind the server to (default: 8000)

Generator options:
  --model-id TEXT          Model ID to load, supports revision syntax:
                           model_id@revision (default: meta-llama/Llama-3.2-1B-Instruct)
  --dtype TEXT             Data type for model weights: auto, float16, bfloat16, float32
  --trust-remote-code      Trust remote code from the model repository
  --attn-implementation    Attention implementation: eager, sdpa, flash_attention_2
  --seed INTEGER           Random seed for reproducibility
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI compatible) |
| `/v1/completions` | POST | Text completion (OpenAI compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## License

MIT License
