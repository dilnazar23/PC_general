import torch
import torch.nn as nn
import numpy as np
import math

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=10, sequence_length=10):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)  # Removed batch_size
        self.linear = nn.Linear(hidden_size, 1)  # Output a single value
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[-1]
        output = self.linear(last_output)
        return output

# Create model
model = SimpleLSTM()
model.eval()  # Set to evaluation mode

# Create dummy input
# Shape: (sequence_length, batch_size, features)
dummy_input = abs(torch.randn(10, 1, 32))
print("\nDummy input shape:", dummy_input.shape)
print("First sequence of dummy input:")
print(dummy_input.numpy())  # Print first 5 features of first sequence

# Get model output before export
with torch.no_grad():
    output = model(dummy_input)
print("\nModel output before ONNX export:")
print(output.numpy())

# Export the model
torch.onnx.export(model,                  # model being run
                 dummy_input,             # model input
                 "dummy_lstm.onnx",       # where to save the model
                 export_params=True,      # store the trained parameter weights inside the model file
                 opset_version=11,        # the ONNX version to export the model to
                 do_constant_folding=True,# whether to execute constant folding for optimization
                 input_names=['input'],   # the model's input names
                 output_names=['output'], # the model's output names
                 dynamic_axes={'input' : {0 : 'sequence_length'},    # variable length axes
                             'output' : {0 : 'sequence_length'}})

print("\nModel has been exported to 'dummy_lstm.onnx'")

# Verify the model using ONNX Runtime
import onnxruntime as ort

ort_session = ort.InferenceSession("dummy_lstm.onnx")

# Run the model in ONNX Runtime
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

print("\nModel output after ONNX export:")
print(ort_outs[0])