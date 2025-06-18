from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
import logging
import uvicorn
import asyncio
import time
import uuid

from .config import Config
from .model_manager import ModelManager

# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"]

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# Legacy endpoint models (keeping for backward compatibility)
class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: Optional[int] = 512

class GenerateResponse(BaseModel):
    model_id: str
    response: str

# Server status models
class ModelInfo(BaseModel):
    model_id: str
    context_size: int
    batch_size: int
    min_instances: int
    max_instances: int

class ServerStatus(BaseModel):
    status: str
    available_models: List[str]
    models_info: List[ModelInfo]

# OpenAI Models list response
class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "coldview"

class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]

class LLMServer:
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(title="COLDVIEW LLM Server", version="1.0.0")
        self.model_manager = ModelManager()
        self._models_loaded = False
        
        self._setup_routes()
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (roughly 4 characters per token)"""
        return len(text) // 4
    
    def _setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            """Load models on FastAPI startup"""
            if not self._models_loaded:
                await self.load_models()
        
        # OpenAI-compatible endpoints
        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest):
            try:
                # Validate model exists
                if request.model not in self.model_manager.get_available_models():
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Model '{request.model}' not found"
                    )
                
                # Convert Pydantic messages to dict format
                messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                
                # Prepare stop sequences
                stop_sequences = []
                if request.stop:
                    if isinstance(request.stop, str):
                        stop_sequences.append(request.stop)
                    else:
                        stop_sequences.extend(request.stop)
                
                # Generate response using the new message-based approach
                response_text = await self.model_manager.generate_async(
                    model_id=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=stop_sequences
                )
                
                # Estimate token usage
                total_content = " ".join([msg["content"] for msg in messages])
                prompt_tokens = self._estimate_tokens(total_content)
                completion_tokens = self._estimate_tokens(response_text)
                
                # Create response
                completion_response = ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=response_text),
                            finish_reason="stop"
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                )
                
                return completion_response
                
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logging.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/v1/models", response_model=ModelsResponse)
        async def list_models():
            """OpenAI-compatible models endpoint"""
            available_models = self.model_manager.get_available_models()
            models_data = [
                ModelObject(
                    id=model_id,
                    created=int(time.time()),
                    owned_by="coldview"
                )
                for model_id in available_models
            ]
            
            return ModelsResponse(data=models_data)
        
        # Health and status endpoints
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
        
        @self.app.get("/status", response_model=ServerStatus)
        async def get_status():
            available_models = self.model_manager.get_available_models()
            models_info = [
                ModelInfo(
                    model_id=model.model_id,
                    context_size=model.context_size,
                    batch_size=model.batch_size,
                    min_instances=model.min_instances,
                    max_instances=model.max_instances
                )
                for model in self.config.models
            ]
            
            return ServerStatus(
                status="running",
                available_models=available_models,
                models_info=models_info
            )
        
        # Legacy endpoint (keeping for backward compatibility)
        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate_text(request: GenerateRequest):
            try:
                # Convert legacy format to messages format
                messages = [{"role": "user", "content": request.prompt}]
                
                response = await self.model_manager.generate_async(
                    model_id=request.model_id,
                    messages=messages,
                    max_tokens=request.max_tokens
                )
                
                return GenerateResponse(
                    model_id=request.model_id,
                    response=response
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logging.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def load_models(self):
        """Load models asynchronously"""
        logging.info("Loading models...")
        # Run model loading in thread pool since it's CPU intensive
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.model_manager.load_models, self.config.models)
        self._models_loaded = True
        logging.info("All models loaded successfully")
    
    def run(self):
        """Run the server"""
        uvicorn.run(
            self.app,
            host=self.config.server.host,
            port=self.config.server.port,
            log_level="info"
        )
