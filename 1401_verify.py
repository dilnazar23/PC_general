import onnxruntime as ort
import numpy as np

def load_and_infer_onnx(model_path, input_data):
    # Load ONNX model
    ort_session = ort.InferenceSession(model_path)
    
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Prepare input data - reshape to match model's expected input shape
    # Assuming the model expects a 2D array with shape (1, 4)
    input_data = np.array([input_data], dtype=np.float32)
    
    # Run inference
    outputs = ort_session.run(None, {input_name: input_data})
    
    return outputs[0]

# Input data
input_values = [4685.0, 4367.0, 5095.0, 4632.0]

try:
    # Run inference
    result = load_and_infer_onnx("1401.onnx", input_values)
    
    print("Input:", input_values)
    print("Output:", result)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")