# src_rag/logger_manager.py
import logging
import sys

class LoggerManager:
    @staticmethod
    def setup_logging():
        """Sets up logging configuration."""
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
