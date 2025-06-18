from dataclasses import dataclass
from typing import List
import json
import os

@dataclass
class ServerConfig:
    port: int
    host: str

@dataclass
class ModelConfig:
    model_id: str
    model_path: str
    system_prompt_path: str
    context_size: int
    batch_size: int
    min_instances: int
    max_instances: int
    
    def load_system_prompt(self) -> str:
        """Load system prompt from file"""
        if os.path.exists(self.system_prompt_path):
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return ""

@dataclass
class Config:
    server: ServerConfig
    models: List[ModelConfig]
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        server_config = ServerConfig(**data['server'])
        model_configs = [ModelConfig(**model) for model in data['models']]
        
        return cls(server=server_config, models=model_configs)
    
    def validate(self) -> None:
        """Validate configuration"""
        for model in self.models:
            if not os.path.exists(model.model_path):
                raise FileNotFoundError(f"Model file not found: {model.model_path}")
            
            if model.min_instances < 1:
                raise ValueError(f"min_instances must be >= 1 for model {model.model_id}")
            
            if model.max_instances < model.min_instances:
                raise ValueError(f"max_instances must be >= min_instances for model {model.model_id}")