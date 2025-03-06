#!/usr/bin/env python3
import numpy as np
import os
import sys
import datetime

def npz_to_c_header(npz_file, output_header_file=None):
    """
    Convert NumPy NPZ file to C header file with array declarations.
    
    Args:
        npz_file (str): Path to the input .npz file
        output_header_file (str, optional): Path to output C header file.
            If None, will use the same name as the input file with .h extension.
    
    Returns:
        str: Path to the generated header file
    """
    # Load the NPZ file
    try:
        data = np.load(npz_file)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        sys.exit(1)
    
    # Create output file name if not provided
    if output_header_file is None:
        output_header_file = os.path.splitext(npz_file)[0] + ".h"
    
    # Extract base filename for symbol names
    base_name = os.path.basename(os.path.splitext(npz_file)[0])
    safe_name = base_name.replace('-', '_').replace(' ', '_')
    
    with open(output_header_file, 'w') as f:
        # Write header guards and comments
        f.write(f"/**\n")
        f.write(f" * Auto-generated C header from NumPy array file: {npz_file}\n")
        f.write(f" * Generated on: {datetime.datetime.now()}\n")
        f.write(f" */\n\n")
        
        header_guard = f"{safe_name.upper()}_H"
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")
        
        f.write("#include <stdint.h>\n\n")
        
        # Write array metadata structure
        f.write("// Array metadata structure\n")
        f.write("typedef struct {\n")
        f.write("    const char* name;\n")
        f.write("    int ndim;\n")
        f.write("    const int* shape;\n")
        f.write("    const char* dtype;\n")
        f.write("    const void* data;\n")
        f.write("    size_t data_size;\n")
        f.write("} numpy_array_t;\n\n")
        
        # Count number of arrays
        array_count = len(data.files)
        f.write(f"// Number of arrays in this file\n")
        f.write(f"#define {safe_name.upper()}_ARRAY_COUNT {array_count}\n\n")
        
        # Process each array in the NPZ file
        for i, name in enumerate(data.files):
            array = data[name]
            safe_array_name = name.replace('-', '_').replace(' ', '_').replace('.', '_')
            
            # Determine C type based on numpy dtype
            if array.dtype == np.float32:
                c_type = "float"
            elif array.dtype == np.float64:
                c_type = "double"
            elif array.dtype == np.int8:
                c_type = "int8_t"
            elif array.dtype == np.int16:
                c_type = "int16_t"
            elif array.dtype == np.int32:
                c_type = "int32_t"
            elif array.dtype == np.int64:
                c_type = "int64_t"
            elif array.dtype == np.uint8:
                c_type = "uint8_t"
            elif array.dtype == np.uint16:
                c_type = "uint16_t"
            elif array.dtype == np.uint32:
                c_type = "uint32_t"
            elif array.dtype == np.uint64:
                c_type = "uint64_t"
            elif array.dtype == np.bool_:
                c_type = "uint8_t"  # C doesn't have a bool type, use uint8
            else:
                print(f"Warning: Unsupported dtype {array.dtype} for array '{name}', using uint8_t")
                c_type = "uint8_t"
            
            # Write shape information
            f.write(f"// Shape for array '{name}'\n")
            shape_var = f"{safe_name}_{safe_array_name}_shape"
            f.write(f"static const int {shape_var}[{array.ndim}] = {{{', '.join(str(dim) for dim in array.shape)}}};\n\n")
            
            # Flatten array for easier C representation
            flat_data = array.flatten()
            
            # Write array data
            f.write(f"// Data for array '{name}' ({array.dtype})\n")
            data_var = f"{safe_name}_{safe_array_name}_data"
            f.write(f"static const {c_type} {data_var}[{len(flat_data)}] = {{\n    ")
            
            # Format data in rows for readability
            elements_per_row = 10
            for j in range(0, len(flat_data), elements_per_row):
                row_data = flat_data[j:j + elements_per_row]
                if array.dtype == np.bool_:
                    # Convert boolean to integer values
                    f.write(", ".join(str(int(val)) for val in row_data))
                else:
                    f.write(", ".join(str(val) for val in row_data))
                if j + elements_per_row < len(flat_data):
                    f.write(",\n    ")
            
            f.write("\n};\n\n")
        
        # Create metadata array
        f.write("// Array metadata collection\n")
        f.write(f"static const numpy_array_t {safe_name}_arrays[{array_count}] = {{\n")
        
        for i, name in enumerate(data.files):
            array = data[name]
            safe_array_name = name.replace('-', '_').replace(' ', '_').replace('.', '_')
            shape_var = f"{safe_name}_{safe_array_name}_shape"
            data_var = f"{safe_name}_{safe_array_name}_data"
            
            f.write(f"    {{\n")
            f.write(f"        .name = \"{name}\",\n")
            f.write(f"        .ndim = {array.ndim},\n")
            f.write(f"        .shape = {shape_var},\n")
            f.write(f"        .dtype = \"{array.dtype}\",\n")
            f.write(f"        .data = (const void*){data_var},\n")
            f.write(f"        .data_size = sizeof({data_var})\n")
            f.write(f"    }}")
            if i < array_count - 1:
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n")
        
        # Add helper function prototypes
        f.write("// Helper function prototypes\n")
        f.write(f"const numpy_array_t* {safe_name}_get_array_by_name(const char* name);\n")
        f.write(f"const numpy_array_t* {safe_name}_get_array_by_index(int index);\n\n")
        
        # Implement helper functions
        f.write("// Helper function implementations\n")
        
        # Function to get array by name
        f.write(f"const numpy_array_t* {safe_name}_get_array_by_name(const char* name) {{\n")
        f.write(f"    for (int i = 0; i < {safe_name.upper()}_ARRAY_COUNT; i++) {{\n")
        f.write(f"        if (strcmp({safe_name}_arrays[i].name, name) == 0) {{\n")
        f.write(f"            return &{safe_name}_arrays[i];\n")
        f.write(f"        }}\n")
        f.write(f"    }}\n")
        f.write(f"    return NULL; // Array not found\n")
        f.write(f"}}\n\n")
        
        # Function to get array by index
        f.write(f"const numpy_array_t* {safe_name}_get_array_by_index(int index) {{\n")
        f.write(f"    if (index >= 0 && index < {safe_name.upper()}_ARRAY_COUNT) {{\n")
        f.write(f"        return &{safe_name}_arrays[index];\n")
        f.write(f"    }}\n")
        f.write(f"    return NULL; // Index out of bounds\n")
        f.write(f"}}\n\n")
        
        # Close header guard
        f.write(f"#endif // {header_guard}\n")
    
    print(f"Successfully converted '{npz_file}' to C header file '{output_header_file}'")
    return output_header_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_to_c_header.py <input_npz_file> [output_header_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    npz_to_c_header(input_file, output_file)
