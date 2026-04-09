# ONNX Debugger

A command-line tool for debugging ONNX models by capturing and visualizing intermediate tensor values at every node during inference.

## Features

- **Full Tensor Capture**: Runs inference with all intermediate outputs exposed, capturing shapes, data types, and statistics for every tensor.
- **HTML Report Generation**: Produces an interactive HTML report showing the model graph with detailed tensor information.
- **Node Inspection**: Allows detailed inspection of individual nodes' input/output tensors.
- **JSON Export**: Optional export of raw debug data to JSON format.
- **ONNX Compatibility**: Works with standard ONNX models and supports various input formats.

## Installation

### Prerequisites

- Python 3.6+
- ONNX model file (.onnx)
- Input data in NumPy format (.npy)

### Dependencies

Install the required Python packages:

```bash
pip install onnx onnxruntime numpy
```

## Usage

### Basic Usage

```bash
python cli.py <model.onnx> <input.npy>
```

This will generate an HTML debug report with the default name `<model>_debug.html`.

### Advanced Options

- **Specify Output File**:
  ```bash
  python cli.py resnet18.onnx input.npy --output custom_report.html
  ```

- **Inspect a Specific Node**:
  ```bash
  python cli.py resnet18.onnx input.npy --inspect Conv_0
  ```

- **Export Raw Data to JSON**:
  ```bash
  python cli.py resnet18.onnx input.npy --json
  ```

### Command Line Arguments

- `model`: Path to the ONNX model file (.onnx)
- `input`: Path to the input data file (.npy)
- `--output`, `-o`: Output HTML file path (optional, defaults to `<model>_debug.html`)
- `--inspect`, `-i`: Node ID to inspect in detail (optional)
- `--json`, `-j`: Export raw debug results to JSON file (optional)

## Input Data Format

The input data should be a NumPy array saved in `.npy` format. For models with multiple inputs, save a dictionary of arrays:

```python
import numpy as np

# Single input
input_data = np.random.randn(1, 3, 224, 224)
np.save('input.npy', input_data)

# Multiple inputs
inputs = {
    'input1': np.random.randn(1, 3, 224, 224),
    'input2': np.random.randn(1, 10)
}
np.save('inputs.npy', inputs)
```

## Output

### HTML Report

The HTML report provides:
- Interactive model graph visualization
- Per-node tensor details (shapes, data types, statistics)
- Min/max values, means, and standard deviations for each tensor
- Node operation types and attributes

### JSON Export

When using `--json`, exports a structured dictionary containing:
- Node information (operation type, attributes)
- Input/output tensor metadata for each node
- Statistical summaries of tensor values

## Project Structure

```
onnx_debugger/
├── debugger.py          # Main debugger interface
├── core/
│   ├── runner.py        # ONNX inference execution
│   ├── model_loader.py  # Model loading utilities
│   └── graph_patcher.py # Graph modification for intermediate capture
├── inspector/
│   ├── node_info.py     # Node static information extraction
│   └── tensor_viewer.py # Tensor description and statistics
└── report/
    └── html_builder.py  # HTML report generation
```

## How It Works

1. **Model Patching**: Modifies the ONNX graph to expose all intermediate outputs
2. **Inference Execution**: Runs the model with ONNX Runtime, capturing all tensor values
3. **Data Analysis**: Computes statistics and metadata for each tensor
4. **Report Generation**: Builds an interactive HTML visualization

## Limitations

- Requires ONNX Runtime for inference execution
- Memory intensive for large models with many intermediate tensors
- Currently supports NumPy input formats only

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source. Please check the license file for details.