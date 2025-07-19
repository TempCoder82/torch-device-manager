"""
Example training script showing how to use torch-device-manager in a real training scenario
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from torch_device_manager import DeviceManager

# Setup logging
logging.basicConfig(level=logging.INFO)

class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy data for demonstration"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)

def train_with_device_manager():
    """Training function using DeviceManager"""
    
    # Initialize device manager
    print("Initializing Device Manager...")
    device_manager = DeviceManager(device="auto", mixed_precision=True)
    device = device_manager.get_device()
    
    # Create model and move to device
    print("Setting up model...")
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy dataset
    print("Creating dataset...")
    dataset = create_dummy_data()
    
    # Optimize batch size for available memory
    original_batch_size = 64
    optimized_batch_size, gradient_accumulation_steps = device_manager.optimize_for_memory(
        model, original_batch_size
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=optimized_batch_size, 
        shuffle=True
    )
    
    print(f"Training configuration:")
    print(f"  - Device: {device}")
    print(f"  - Original batch size: {original_batch_size}")
    print(f"  - Optimized batch size: {optimized_batch_size}")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  - Mixed precision: {device_manager.mixed_precision}")
    
    # Training loop
    num_epochs = 5
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        accumulation_count = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Use mixed precision if available
            if device_manager.mixed_precision and device_manager.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target) / gradient_accumulation_steps
                
                device_manager.scaler.scale(loss).backward()
                accumulation_count += 1
                
                # Update weights after accumulating gradients
                if accumulation_count >= gradient_accumulation_steps:
                    device_manager.scaler.step(optimizer)
                    device_manager.scaler.update()
                    optimizer.zero_grad()
                    accumulation_count = 0
                    
            else:
                output = model(data)
                loss = criterion(output, target) / gradient_accumulation_steps
                loss.backward()
                accumulation_count += 1
                
                # Update weights after accumulating gradients
                if accumulation_count >= gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulation_count = 0
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item() * gradient_accumulation_steps:.4f}')
        
        # Handle any remaining accumulated gradients
        if accumulation_count > 0:
            if device_manager.mixed_precision and device_manager.scaler is not None:
                device_manager.scaler.step(optimizer)
                device_manager.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')
        
        # Log memory usage after each epoch
        device_manager.log_memory_usage()
    
    print("Training completed!")
    
    # Final memory check
    print("\nFinal memory state:")
    memory_info = device_manager.get_memory_info()
    print(memory_info)

if __name__ == "__main__":
    train_with_device_manager()