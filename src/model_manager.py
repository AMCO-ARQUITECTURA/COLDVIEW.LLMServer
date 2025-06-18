import asyncio
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from queue import Queue
import threading
import time
import platform

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not installed. Install with Metal support:")
    print("CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir")
    raise

from .config import ModelConfig

@dataclass
class ModelInstance:
    instance_id: str
    model: Llama
    system_prompt: str
    is_busy: bool = False
    last_used: float = 0

class ModelPool:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.instances: List[ModelInstance] = []
        self.available_queue = Queue()
        self.lock = threading.Lock()
        self.system_prompt = model_config.load_system_prompt()
        
        # Initialize minimum instances
        for i in range(model_config.min_instances):
            self._create_instance(f"{model_config.model_id}-{i}")
    
    def _create_instance(self, instance_id: str) -> ModelInstance:
        """Create a new model instance optimized for Apple Silicon"""
        logging.info(f"Creating model instance: {instance_id}")
        
        # Apple Silicon optimizations
        is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
        
        model_kwargs = {
            'model_path': self.config.model_path,
            'n_ctx': self.config.context_size,
            'n_batch': self.config.batch_size,
            'verbose': False,
            'chat_format': 'chatml',  # Use ChatML format for better system prompt handling
            # 'chat_format': 'alpaca',  # Use ChatML format for better system prompt handling
            # performance optimizations 
        }
        
        if is_apple_silicon:
            # Enable Metal acceleration and optimize for Apple Silicon
            model_kwargs.update({
                'n_gpu_layers': -1,  # Use all layers on GPU (Metal)
                'n_threads': 8,      # Optimal for M3 MAX
                'n_threads_batch': 8,
                'use_mlock': True,   # Lock memory for better performance
                'use_mmap': True,    # Memory mapping for efficiency,
                "low_vram": False,       # Disable for speed
                "f16_kv": True,          # Half precision KV cache
                "logits_all": False,     
                "vocab_only": False,
                "rope_scaling_type": 1,  # Enable RoPE scaling for efficiency
                "rope_freq_base": 10000.0
            })
            logging.info(f"Configuring {instance_id} with Metal acceleration for Apple Silicon")
        
        model = Llama(**model_kwargs)
        
        instance = ModelInstance(
            instance_id=instance_id,
            model=model,
            system_prompt=self.system_prompt
        )
        
        with self.lock:
            self.instances.append(instance)
            self.available_queue.put(instance)
        
        logging.info(f"Model instance {instance_id} created with system prompt configured")
        return instance
    
    def get_instance(self, timeout: float = 30.0) -> Optional[ModelInstance]:
        """Get an available model instance"""
        try:
            instance = self.available_queue.get(timeout=timeout)
            instance.is_busy = True
            instance.last_used = time.time()
            return instance
        except:
            # Try to create new instance if under max limit
            with self.lock:
                if len(self.instances) < self.config.max_instances:
                    new_id = f"{self.config.model_id}-{len(self.instances)}"
                    instance = self._create_instance(new_id)
                    instance.is_busy = True
                    instance.last_used = time.time()
                    return instance
            return None
    
    def return_instance(self, instance: ModelInstance):
        """Return an instance to the pool"""
        instance.is_busy = False
        self.available_queue.put(instance)
    
    def generate(self, instance: ModelInstance, messages: List[Dict[str, str]], 
                max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, 
                stop: Optional[List[str]] = None) -> str:
        """Generate text using the model instance with proper system prompt handling"""
        
        # Prepare messages for the model
        # The system prompt is already configured in the model, so we just need to handle the conversation
        chat_messages = []
        
        # Add the configured system prompt as the base
        chat_messages.append({
            "role": "system",
            "content": instance.system_prompt
        })
        
        # Add user messages (including any additional system context)
        for message in messages:
            if message["role"] == "system":
                # Append additional system context to the base system prompt
                chat_messages.append({
                    "role": "system", 
                    "content": message["content"]
                })
            else:
                chat_messages.append(message)
        
        # Default stop sequences
        default_stop = ["<|im_end|>", "</s>"]
        if stop:
            stop_sequences = stop + default_stop
        else:
            stop_sequences = default_stop
        
        try:
            response = instance.model.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences
            )
            
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logging.error(f"Error in model generation: {e}")
            # Fallback to simple completion if chat completion fails
            prompt = self._messages_to_simple_prompt(chat_messages)
            response = instance.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
                echo=False
            )
            return response['choices'][0]['text'].strip()
    
    def _messages_to_simple_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Fallback: Convert messages to simple prompt format"""
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

class ModelManager:
    def __init__(self):
        self.pools: Dict[str, ModelPool] = {}
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system information for debugging"""
        is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
        if is_apple_silicon:
            logging.info("Detected Apple Silicon - Metal acceleration will be used")
        else:
            logging.info(f"System: {platform.system()} {platform.machine()}")
    
    def load_models(self, model_configs: List[ModelConfig]):
        """Load all configured models"""
        for config in model_configs:
            logging.info(f"Loading model: {config.model_id}")
            self.pools[config.model_id] = ModelPool(config)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self.pools.keys())
    
    async def generate_async(self, model_id: str, messages: List[Dict[str, str]], 
                           max_tokens: int = 512, temperature: float = 0.7, 
                           top_p: float = 0.9, stop: Optional[List[str]] = None) -> str:
        """Generate text asynchronously"""
        if model_id not in self.pools:
            raise ValueError(f"Model {model_id} not found")
        
        pool = self.pools[model_id]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, pool, messages, max_tokens, temperature, top_p, stop
        )
    
    def _generate_sync(self, pool: ModelPool, messages: List[Dict[str, str]], 
                      max_tokens: int, temperature: float, top_p: float, 
                      stop: Optional[List[str]]) -> str:
        """Synchronous generation helper"""
        instance = pool.get_instance()
        if not instance:
            raise RuntimeError("No available model instances")
        
        try:
            return pool.generate(instance, messages, max_tokens, temperature, top_p, stop)
        finally:
            pool.return_instance(instance)
