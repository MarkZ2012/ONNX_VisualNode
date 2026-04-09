# ONNX Debugger

A comprehensive command-line tool for debugging ONNX models by capturing and visualizing intermediate tensor values at every node during inference.

## Features

- **Full Tensor Capture**: Runs inference with all intermediate outputs exposed, capturing shapes, data types, and statistics for every tensor.
- **Interactive HTML Report**: Produces a feature-rich HTML report with three visualization modes:
  - **Debug View**: Detailed node-by-node tensor inspection with statistics
  - **Graph View**: Interactive computational graph visualization with SVG rendering
  - **Statistics**: Comprehensive operator statistics and model analysis
- **Graph Visualization**: Interactive computational graph with:
  - SVG-based rendering with pan and zoom support
  - Topological layout algorithm for clear visualization
  - Smart edge routing to avoid crossings
  - Node search and filtering
  - Click-to-inspect node details
  - Color-coded operator categories
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
python onnx_visualnode.py <model.onnx> <input.npy>
```

This will generate an HTML debug report with the default name `<model>_debug.html`.

**Note**: You can also use the modular version:
```bash
python cli.py <model.onnx> <input.npy>
```

### Advanced Options

- **Specify Output File**:
  ```bash
  python onnx_visualnode.py resnet18.onnx input.npy --output custom_report.html
  ```

- **Inspect a Specific Node**:
  ```bash
  python onnx_visualnode.py resnet18.onnx input.npy --inspect Conv_0
  ```

- **Export Raw Data to JSON**:
  ```bash
  python onnx_visualnode.py resnet18.onnx input.npy --json
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

The HTML report provides three interactive visualization modes:

#### 1. Debug View
- List of all nodes with operation types
- Per-node tensor details (shapes, data types, statistics)
- Min/max values, means, and standard deviations for each tensor
- Node operation types and attributes
- Search and filter functionality

#### 2. Graph View
- Interactive SVG-based computational graph visualization
- Topological layout with clear data flow
- Pan (drag) and zoom (scroll) controls
- Click nodes to view detailed information
- Smart edge routing to minimize crossings
- Color-coded operator categories
- Node search and filtering
- Legend showing operator types

#### 3. Statistics
- Total node count and unique operator types
- Operator breakdown with counts and percentages
- Visual bar charts for operator distribution
- Model metadata and analysis timestamp

### JSON Export

When using `--json`, exports a structured dictionary containing:
- Node information (operation type, attributes)
- Input/output tensor metadata for each node
- Statistical summaries of tensor values

## Project Structure

### Single-File Version (Recommended)
```
onnx_visualnode.py       # All-in-one single file version with all features
```

### Modular Version
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