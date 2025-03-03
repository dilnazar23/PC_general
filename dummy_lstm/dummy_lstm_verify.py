import torch
import torch.nn as nn
import numpy as np
import math

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=10, sequence_length=10):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)  # Removed batch_size
        self.linear = nn.Linear(hidden_size, 2)  # Output 2 value
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[-1]
        output = self.linear(last_output)
        return output
# define vanilla recurrent net
class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Basic Recurrent Neural Network
        
        Args:
            input_size (int): Size of input features at each time step
            hidden_size (int): Size of hidden state
            output_size (int): Size of output
            num_layers (int): Number of recurrent layers
        """
        super(BasicRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            #batch_first=True  # (batch, seq, feature)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            output: Tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate through RNN
        # out shape: (batch_size, seq_length, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        out, hn = self.rnn(x, h0)
        
        # We take the output from the last time step
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        
        return out
# Create model
#model = SimpleLSTM()
model = BasicRNN(input_size=32, hidden_size=10, output_size=2, num_layers=1)
model.eval()  # Set to evaluation mode

# Create dummy input
# Shape: (sequence_length, batch_size, features)
dummy_input = np.array([
    [[1.2355309, 0.40005484, 1.4679403, 0.50303423, 1.7926798, 0.38952476,
      1.5690275, 0.08054668, 1.0924711, 1.0150241, 1.4997307, 1.7835981,
      0.39210245, 1.0739399, 0.36519253, 0.8144836, 1.3015168, 0.05195837,
      1.8977833, 0.02696723, 0.8608237, 1.1391039, 0.04551141, 0.2583735,
      2.0456958, 1.4061794, 2.3365052, 0.06573228, 0.289699, 0.65864664,
      0.60642934, 0.62207514]],
    
    [[0.82539016, 0.04569487, 1.6152793, 0.02593499, 0.3972043, 1.4612042,
      1.536507, 1.2764013, 0.39470562, 1.461827, 0.06037264, 0.3954381,
      0.20735003, 0.7100783, 1.3670148, 0.08119939, 0.763295, 0.06552369,
      0.24669558, 0.8201339, 0.43034, 1.2691871, 0.65646976, 1.7284923,
      0.6736582, 0.02556144, 0.38839966, 1.5417147, 0.85914373, 0.25074857,
      0.16787857, 1.3932378]],
    
    [[1.991467, 0.00610504, 0.9340074, 1.5871189, 0.17631707, 1.7929045,
      1.4380469, 0.5235318, 0.868078, 0.19921236, 1.0856292, 0.70544386,
      0.44438583, 0.30641788, 1.1706436, 1.2088658, 0.84161067, 0.00705302,
      1.2972226, 0.50975734, 1.3111117, 1.9307247, 1.5082372, 2.1114476,
      2.1411178, 0.72238374, 0.9818974, 0.54147345, 1.5542388, 0.12023494,
      0.18006125, 0.690067]],
    
    [[0.18348677, 0.5642189, 0.2928422, 0.28644076, 0.08568436, 0.8431264,
      1.1330025, 0.80602694, 0.16315182, 1.627708, 1.2000788, 0.04264363,
      0.36713418, 0.47860587, 0.8977224, 0.22594705, 0.30521932, 1.4282109,
      0.53864855, 1.1672262, 0.4766473, 0.10331412, 1.2448044, 0.3492704,
      0.48975003, 0.88562185, 0.97177243, 0.41028935, 1.6238736, 1.0387245,
      0.25963926, 0.05016821]],
    
    [[1.543461, 0.8237618, 0.0821252, 0.83990616, 1.1917579, 0.795873,
      1.0082617, 1.6203598, 2.1426618, 0.13467196, 0.46390575, 0.37554747,
      1.6568421, 0.03884806, 0.7521789, 0.05326528, 1.8195418, 1.2249823,
      0.11699186, 0.06229394, 0.41346708, 0.12000363, 0.7282623, 2.9900556,
      0.08488282, 0.48273703, 0.83242685, 1.3281412, 0.0815366, 0.09089628,
      0.73454654, 0.3231045]],
    
    [[2.048797, 0.7168671, 1.0575343, 0.1280611, 0.2878252, 0.17941381,
      0.2154229, 1.0834404, 1.7229892, 0.3044021, 0.5421018, 1.7866935,
      1.3242388, 0.34347022, 1.1750343, 0.8953654, 0.37773338, 0.75670344,
      0.9054142, 0.34942085, 0.31255826, 2.0284317, 0.40677208, 1.3159069,
      0.15128404, 1.577652, 0.7726027, 1.3139398, 0.11544372, 1.2304851,
      1.1666131, 0.05299658]],
    
    [[1.8633745, 1.8570817, 0.01418534, 0.3016143, 1.0577863, 0.882737,
      0.31097248, 0.09892187, 0.8078669, 0.6312144, 1.0872098, 1.3681774,
      1.6445256, 0.11338906, 1.2763649, 0.39142698, 0.20617922, 0.6128252,
      1.1535773, 1.3963906, 1.1761514, 1.2124695, 0.39823967, 0.7003009,
      0.7481429, 1.0090246, 1.1012594, 1.1888478, 1.2257744, 0.7761592,
      1.2276686, 0.4756252]],
    
    [[0.7700781, 0.5409796, 0.8695787, 0.91977954, 1.7561224, 0.9321872,
      1.0788547, 0.34315076, 0.01393411, 1.3468617, 1.3110188, 0.8888961,
      0.02955047, 0.58429116, 0.04661578, 0.7517879, 1.5857775, 0.28935382,
      0.61099404, 1.1742952, 0.19267298, 1.0482812, 0.9378664, 0.653717,
      1.2232997, 0.24444039, 0.5765979, 0.33776292, 0.35993683, 1.6637557,
      0.36189, 1.1094456]],
    
    [[0.16022974, 0.3479977, 0.5206106, 1.4345257, 0.6695661, 1.8405027,
      1.235711, 0.78278756, 0.2800403, 0.37458134, 1.7747028, 0.61112666,
      1.8331939, 0.3289957, 0.308823, 1.2831739, 0.21013534, 0.5600507,
      1.4778328, 1.2439957, 0.11488748, 2.972461, 0.52062476, 1.384482,
      1.1016655, 1.6480241, 2.5545063, 0.566783, 0.02890932, 0.6794103,
      0.12329698, 0.30623493]],
    
    [[1.189318, 0.30893934, 0.73358446, 0.776966, 0.7189575, 1.0406969,
      0.17916808, 0.15517002, 0.35176295, 1.118514, 0.53134406, 0.9063453,
      0.5660747, 0.2109012, 0.64602363, 1.0279154, 0.75325006, 0.35831767,
      1.1813455, 1.5584852, 0.7186114, 0.04611504, 0.42244217, 0.17144065,
      2.0324352, 0.6853077, 0.03641406, 0.04360988, 0.6290482, 1.4833324,
      0.74401015, 0.83549625]]], dtype=np.float32)

# # print("\nDummy input shape:", dummy_input.shape)
# print("First sequence of dummy input:")
# print(dummy_input.numpy())  # Print first 5 features of first sequence

# # Get model output before export
# with torch.no_grad():
#     output = model(dummy_input)
#     print("\nModel output before ONNX export:")
#     print(output.numpy())

# Export the model to onnx
# torch.onnx.export(model,                  # model being run
#                  dummy_input,             # model input
#                  "dummy_lstm.onnx",       # where to save the model
#                  export_params=True,      # store the trained parameter weights inside the model file
#                  opset_version=11,        # the ONNX version to export the model to
#                  do_constant_folding=True,# whether to execute constant folding for optimization
#                  input_names=['input'],   # the model's input names
#                  output_names=['output'], # the model's output names
#                  dynamic_axes={'input' : {0 : 'sequence_length'},    # variable length axes
#                              'output' : {0 : 'sequence_length'}})

# print("\nModel has been exported to 'dummy_lstm.onnx'")
# print("\nCheck model input size")
# print(dummy_input.shape)
# print([dummy_input[0,:,:]])

# Verify the model using ONNX Runtime
# import onnxruntime as ort

# ort_session = ort.InferenceSession("dummy_lstmfinn_verify.onnx")

# # Run the model in ONNX Runtime

# for i in range(10):
#     input_data = dummy_input[:i+1,:,:]
#     # print(f"sequence {i} data:{input_data}")
#     ort_inputs = onnx_inputs = {
#         "input": input_data
#     }
#     print(f"\nSequence {i} model output:")
#     ort_outs = ort_session.run(None, ort_inputs)

#     print(ort_outs)

# Convert the model to TFLite
import ai_edge_torch
# convert dummy input to tesor
dummy_input_tensor = torch.from_numpy(dummy_input)
dummy_input_tuple = (dummy_input_tensor,)  # Convert to tensor and wrap in tuple

edge_model = ai_edge_torch.convert(model, dummy_input_tuple)
edge_model.export('dummy_RNNfinn_verify.tflite')
import model_explorer
model_explorer.visualize('dummy_RNNfinn_verify.tflite')

# # Convert resnet18 model to tflite
# import torchvision

# resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
# sample_inputs = (torch.randn(1, 3, 224, 224),)
# torch_output = resnet18(*sample_inputs)
     
# edge_model = ai_edge_torch.convert(resnet18, sample_inputs)

# edge_output = edge_model(*sample_inputs)

# if np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5):
#     print("Inference result with Pytorch and TfLite was within tolerance")
# else:
#     print("Something wrong with Pytorch --> TfLite")
#     edge_model.export('resnet.tflite')

# Download the tflite flatbuffer which can be used with the existing TfLite APIs.
# from google.colab import files
# files.download('resnet.tflite')





