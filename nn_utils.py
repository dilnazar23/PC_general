import torch
import csv
import os
from datetime import datetime
from collections import OrderedDict

class TrainingLogger:
    def __init__(self, model, optimizer, log_dir='logs'):
        self.model = model
        self.optimizer = optimizer
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%m%d_%H%M')
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.csv')
        
        # Initialize the CSV file with headers and hyperparameters
        self._init_log_file()
        
    def _get_model_structure(self):
        """Returns a string representation of the model structure"""
        return str(self.model)
    
    def _get_optimizer_params(self):
        """Returns a dictionary of optimizer parameters"""
        params = {}
        for param_group in self.optimizer.param_groups:
            for key, value in param_group.items():
                if key != 'params':  # Skip the actual parameters
                    params[key] = value
        return params
    
    def _init_log_file(self):
        """Initialize the log file with hyperparameters and headers"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write hyperparameters
            writer.writerow(['=== Hyperparameters ==='])
            writer.writerow(['Optimizer:', self.optimizer.__class__.__name__])
            
            # Write optimizer parameters
            opt_params = self._get_optimizer_params()
            for key, value in opt_params.items():
                writer.writerow([f'Optimizer {key}:', value])
            
            # Write model structure
            writer.writerow(['Model Structure:'])
            writer.writerow([self._get_model_structure()])
            
            # Add a blank line
            writer.writerow([])
            
            # Write headers for training progress
            writer.writerow([
                'Epoch',
                'Training Loss',
                'Validation Loss',
                *[f'{name} (grad)' for name, _ in self.model.named_parameters()],
                *[f'{name} (value)' for name, _ in self.model.named_parameters()]
            ])
    
    def log_progress(self, epoch, train_loss, val_loss):
        """Log the training progress including parameters and gradients"""
        if epoch == 0 or (epoch + 1) % 1000 == 0:  # Log at start and every 1000 epochs
            # Collect parameter values and gradients
            param_data = OrderedDict()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_data[f'{name} (grad)'] = param.grad.data.norm(2).item()
                else:
                    param_data[f'{name} (grad)'] = 0.0
                param_data[f'{name} (value)'] = param.data.norm(2).item()
            
            # Write to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_loss,
                    *[param_data[f'{name} (grad)'] for name, _ in self.model.named_parameters()],
                    *[param_data[f'{name} (value)'] for name, _ in self.model.named_parameters()]
                ])

# Example usage:
"""
# Initialize your model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create logger
logger = TrainingLogger(model, optimizer)

# In your training loop:
for epoch in range(num_epochs):
    # Training code here
    train_loss = ...
    val_loss = ...
    
    # Log progress
    logger.log_progress(epoch, train_loss, val_loss)
"""
def export_model(model, device='cuda', save_path="layout1_model", input_shape=(1,32)):
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, f"{save_path}.pth")
    
    # Prepare model for ONNX export
    model.eval()    
    # Create dummy input tensor
    dummy_input = torch.randn(input_shape, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f"{save_path}.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model saved as {save_path}.pth and {save_path}.onnx")