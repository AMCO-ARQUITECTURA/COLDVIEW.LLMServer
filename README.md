# COLDVIEW.LLMServer
A web service for deploying and managing GGUF LLM models (Llama 3.2 and compatible models) with configurable instances and load balancing.
## Features
- Deploy multiple LLM models simultaneously
- Configurable model instances with min/max scaling
- REST API for text generation
- Support for GGUF format models
- Custom system prompts per model
- Health monitoring and status endpoints
## Installation on MacOS Apple Silicon
### Generacion de ambiente virtual con conda
conda create -n venv_name python=3.11 -y
conda activate venv_name
### Instalacion de dependencias
conda install -c conda-forge cmake pkg-config numpy -y
CMAKE_ARGS="-DLLAMA_METAL=on" $(conda info --base)/envs/llm-server/bin/pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
$(conda info --base)/envs/llm-server/bin/pip install fastapi uvicorn[standard] pydantic
### Ejecucion con conda
$(conda info --base)/envs/llm-server/bin/python run_server.py 

## Instalation on other platforms
```bash
pip install -r requirements.txt
```
## Configuration
Create a `config.json` file with your model configurations:
```json
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0"
  },
  "models": [
    {
      "model_id": "llama-3.2-1b",
      "model_path": "./models/llama-3.2-1b.gguf",
      "system_prompt_path": "./prompts/default_system.txt",
      "context_size": 4096,
      "batch_size": 512,
      "min_instances": 1,
      "max_instances": 3
    }
  ]
}
```
## Usage
Start the server:
```bash
python run_server.py --config config.json
```
## API Endpoints
- `GET /health` - Health check
- `GET /status` - Server status and model info
- `GET /models` - List available models
- `POST /generate` - Generate text
### Example API Usage
```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llama-3.2-1b",
    "prompt": "What is artificial intelligence?",
    "max_tokens": 512
  }'
```
## Directory Structure
```
COLDVIEW.LLMServer/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model_manager.py
│   ├── server.py
│   └── main.py
├── prompts/
│   ├── default_system.txt
│   └── assistant_system.txt
├── models/          # Place your .gguf files here
├── config.json
├── requirements.txt
├── run_server.py
└── README.md
```
```
To get started:
```bash
pip install -r requirements.txt
```
```bash
mkdir models prompts
```
```bash
python run_server.py --config config.json

