import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .server import LLMServer

def main():
    parser = argparse.ArgumentParser(description="COLDVIEW LLM Server")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config.from_json(args.config)
        config.validate()
        
        # Create and start server
        server = LLMServer(config)
        
        logging.info(f"Starting server on {config.server.host}:{config.server.port}")
        server.run()
        
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
