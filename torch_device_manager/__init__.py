"""
Torch Device Manager - Automatic hardware detection and memory optimization for PyTorch
"""

import logging
from typing import Tuple
import torch

__version__ = "0.1.0"
__author__ = "Ali B.M."

# Set up logging
logger = logging.getLogger(__name__)

class DeviceManager:
    """Manage device selection and memory optimization for different hardware"""
    
    def __init__(self, device: str = "auto", mixed_precision: bool = True):
        self.device = self._detect_device(device)
        self.mixed_precision = mixed_precision
        self.scaler = None
        
        logger.info(f"Device Manager initialized:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Mixed Precision: {self.mixed_precision}")
        
        if self.mixed_precision and self.device != "cpu":
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info(f"  - Gradient Scaler: Enabled")
    
    def _detect_device(self, device: str) -> str:
        """Detect the best available device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA detected: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info(f"Apple Silicon MPS detected")
            else:
                device = "cpu"
                logger.info(f"Using CPU")
        else:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"
            elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                device = "cpu"
        
        return device
    
    def get_device(self):
        """Get the torch device object"""
        return torch.device(self.device)
    
    def get_memory_info(self):
        """Get memory information for the current device"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        elif self.device == "mps":
            # MPS doesn't provide detailed memory info like CUDA
            return {"device": "mps", "info": "Memory info not available for MPS"}
        else:
            return {"device": "cpu", "info": "Memory info not available for CPU"}
    
    def log_memory_usage(self):
        """Log current memory usage"""
        memory_info = self.get_memory_info()
        if "allocated_gb" in memory_info:
            logger.info(f"Memory Usage: {memory_info['allocated_gb']:.2f}GB allocated, "
                       f"{memory_info['free_gb']:.2f}GB free")
    
    def optimize_for_memory(self, model, batch_size: int) -> Tuple[int, int]:
        """Optimize batch size and gradient accumulation for available memory"""
        
        if self.device == "cpu":
            # CPU: Use smaller batches
            optimized_batch_size = min(batch_size, 8)
            gradient_steps = max(1, batch_size // optimized_batch_size)
            logger.info(f"CPU optimization: batch_size={optimized_batch_size}, gradient_steps={gradient_steps}")
            
        elif self.device == "mps":
            # Apple Silicon: Conservative settings
            optimized_batch_size = min(batch_size, 4)
            gradient_steps = max(1, batch_size // optimized_batch_size)
            logger.info(f"MPS optimization: batch_size={optimized_batch_size}, gradient_steps={gradient_steps}")
            
        elif self.device == "cuda":
            # CUDA: Check available memory
            memory_info = self.get_memory_info()
            total_memory = memory_info["total_gb"]
            
            if total_memory < 8:  # Less than 8GB
                optimized_batch_size = min(batch_size, 4)
                gradient_steps = max(1, batch_size // optimized_batch_size)
                logger.info(f"CUDA <8GB optimization: batch_size={optimized_batch_size}, gradient_steps={gradient_steps}")
            elif total_memory < 16:  # Less than 16GB
                optimized_batch_size = min(batch_size, 8)
                gradient_steps = max(1, batch_size // optimized_batch_size)
                logger.info(f"CUDA <16GB optimization: batch_size={optimized_batch_size}, gradient_steps={gradient_steps}")
            else:  # 16GB or more
                optimized_batch_size = batch_size
                gradient_steps = 1
                logger.info(f"CUDA >=16GB: using full batch_size={optimized_batch_size}")
        
        return optimized_batch_size, gradient_steps

# Make DeviceManager easily importable
__all__ = ["DeviceManager"]