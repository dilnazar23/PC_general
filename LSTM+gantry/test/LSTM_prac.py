import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from datetime import datetime

class SequenceDataset(Dataset):
    def __init__(self, train_path, seq_length=40):
        self.inputs, self.targets = torch.load(train_path)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.inputs) - self.seq_length + 1
    
    def __getitem__(self, idx):
        return (self.inputs[idx:idx + self.seq_length], 
                self.targets[idx + self.seq_length - 1])

class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.norm = nn.LayerNorm(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def export_model(model, device='cuda', input_shape=(40,1,32)):
    
    curr_time = datetime.now().strftime("%m-%d_%H:%M")
    save_path = f"lstm_{curr_time}"
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

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(num_hid, optimizer_type, learning_rate, epochs, data_dir):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading 
    for file in Path('test').glob(f'{data_dir}*.pt'):
        print(file)
        ## add val_dataset here
        train_dataset = SequenceDataset(file)
        break
    
    train_loader = DataLoader(train_dataset, batch_size=320, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model initialization
    model = LSTMModel(num_features=32, hidden_size=num_hid).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # TensorBoard setup
    # writer = SummaryWriter('runs/lstm_training')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #print(f'debug print, bat_idx: {batch_idx}, {inputs.shape}, {targets.shape}')

            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation every 100 epochs
        if (epoch + 1) % 10 == 0:
        #     val_loss = validate_model(model, val_loader, criterion, device)
            print(f'Epoch [{epoch+1}/{epochs}]',
                  f'Training Loss: {avg_train_loss:.4f}')
        #           f'Validation Loss: {val_loss:.4f}')
            
        # Log to TensorBoard
        # writer.add_scalar('Training Loss', avg_train_loss, epoch)
            # writer.add_scalar('Validation Loss', val_loss, epoch)
    
    # Save the model
    # curr_time = datetime.now().strftime("%m-%d_%H:%M")
    # torch.save(model.state_dict(), f'lstm_{curr_time}.pth')
    # writer.close()
    export_model(model)

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--num_hid', type=int, default=64,
                      help='number of hidden units')
    parser.add_argument('--optimizer', type=str, default='adam',
                      help='optimizer type (adam or sgd)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='number of epochs')
    parser.add_argument('--data_dir', type=str, default='./data',
                      help='directory containing the data files')
    
    args = parser.parse_args()
    
    train_model(
        num_hid=args.num_hid,
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        epochs=args.epochs,
        data_dir=args.data_dir
    )

if __name__ == "__main__":
    main()