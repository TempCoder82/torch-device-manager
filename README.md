# Torch Device Manager

A lightweight PyTorch utility for automatic hardware detection and memory optimization across different devices (CPU, CUDA, MPS).

## Features

- 🔍 **Automatic Device Detection**: Detects the best available hardware (CUDA, Apple Silicon MPS, or CPU)
- 🧠 **Memory Optimization**: Automatically adjusts batch sizes and gradient accumulation based on available memory
- ⚡ **Mixed Precision Support**: Optional automatic mixed precision with gradient scaling
- 📊 **Memory Monitoring**: Real-time memory usage tracking and logging
- 🛡️ **Fallback Protection**: Graceful fallback to CPU when requested devices aren't available

## Installation

```bash
pip install torch-device-manager
```

## Quick Start

```python
from torch_device_manager import DeviceManager
import torch

# Initialize device manager (auto-detects best device)
device_manager = DeviceManager(device="auto", mixed_precision=True)

# Get the torch device
device = device_manager.get_device()

# Move your model to the optimal device
model = YourModel().to(device)

# Optimize batch size based on available memory
optimal_batch_size, gradient_steps = device_manager.optimize_for_memory(
    model=model, 
    batch_size=32
)

print(f"Using device: {device}")
print(f"Optimized batch size: {optimal_batch_size}")
print(f"Gradient accumulation steps: {gradient_steps}")
```

## Usage in Training Scripts

### Basic Integration

```python
import torch
import torch.nn as nn
from torch_device_manager import DeviceManager

def train_model():
    # Initialize device manager
    device_manager = DeviceManager(device="auto", mixed_precision=True)
    device = device_manager.get_device()
    
    # Setup model
    model = YourModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Optimize memory usage
    batch_size, gradient_steps = device_manager.optimize_for_memory(model, 32)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Use mixed precision if available
            if device_manager.mixed_precision and device_manager.scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                device_manager.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_steps == 0:
                    device_manager.scaler.step(optimizer)
                    device_manager.scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if (batch_idx + 1) % gradient_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Log memory usage
        device_manager.log_memory_usage()
```

### Advanced Usage

```python
from torch_device_manager import DeviceManager

# Force specific device
device_manager = DeviceManager(device="cuda", mixed_precision=False)

# Check memory info
memory_info = device_manager.get_memory_info()
print(f"Available memory: {memory_info}")

# Manual memory optimization
if memory_info.get("free_gb", 0) < 2.0:
    print("Low memory detected, reducing batch size")
    batch_size = 4
```

## API Reference

### DeviceManager

#### Constructor
- `device` (str, default="auto"): Device to use ("auto", "cuda", "mps", "cpu")
- `mixed_precision` (bool, default=True): Enable mixed precision training

#### Methods
- `get_device()`: Returns torch.device object
- `get_memory_info()`: Returns memory information dict
- `log_memory_usage()`: Logs current memory usage
- `optimize_for_memory(model, batch_size)`: Returns optimized (batch_size, gradient_steps)

## Device Support

- **CUDA**: Full support with memory optimization and mixed precision
- **Apple Silicon (MPS)**: Basic support with conservative memory settings
- **CPU**: Fallback support with optimized batch sizes

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.1

## License

MIT License

## Contributing

Contributions are welcome! 