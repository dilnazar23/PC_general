# define model class
# import model from onnx file
# define model input
# run inference at each time step

import onnxruntime as ort
import numpy as np

# the following is the code generate by Claude
def load_input(input_set_path):
    data = np.load(input_set_path)    
    # print(hardware_val_data.files)
    # print(data['m_inputs_1'])
    return data

def load_and_run_lstm_model(model_path,data_path):
    # Load the input data
    hardware_val_data = load_input(data_path)
    # Load the ONNX model
    print(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Get model input name
    input_name = session.get_inputs()[0].name
    print(f"Model input name: {input_name}")
    
    # Get input shape details
    input_shape = session.get_inputs()[0].shape
    print(f"Model input shape: {input_shape}")
    
    # Generate dummy input data: 10 timestamps, 32 features each
    # For LSTM models, the input shape is typically [batch_size, sequence_length, input_features]
    # Here we use batch_size=1, sequence_length=10, input_features=32

    #dummy_input = np.random.rand(1, 10, 32).astype(np.float32)
    
    print(f"Using input with shape: {hardware_val_data.shape}")
    
    # Run inference
    outputs = session.run(None, {input_name: hardware_val_data})
    
    # Print model output
    print("\nModel output:")
    print(outputs)
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
        
        # Check if this output contains sequence data (has timestamp dimension)
        # LSTM outputs are typically shaped as [batch_size, sequence_length, hidden_size]
        if len(output.shape) >= 3:
            print("Sequence output detected, showing output at each timestamp:")
            sequence_length = output.shape[1]
            for t in range(sequence_length):
                print(f"  Timestamp {t}: {output[0, t]}")
        else:
            print(f"Output {i} values: {output}")
        
    return outputs


if __name__ == "__main__":
    # Replace with your actual ONNX model path
    model_path = "py model/lstm_03-10_16:14.onnx"
    val_data_path = 'data/val_1_input.npy'
    
    try:
        print("=== Running full sequence inference ===")
        outputs = load_and_run_lstm_model(model_path,val_data_path)
        
            
    except Exception as e:
        print(f"Error occurred: {e}")