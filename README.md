# Recursive Network Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains experimental code to test the hypothesis that recursive functions create temporal networks whose properties can predict performance with high correlation. 

Based on the paper: *"Recursion Requires Networks: Why Isolated Computation Cannot Exist"*

**Key Finding**: Network properties (nodes, edges, diameter) predict recursive performance with **0.98 average correlation** across tested algorithms.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Build-In-Public-University/recursion_networks.git
cd recursion_networks

# Install dependencies
pip install -r requirements.txt

# Run the main experiment
python recursive_network_test.py
```

## What This Does

The script:
1. **Traces recursive execution** to build temporal networks
2. **Auto-calibrates** hardware-specific performance constants
3. **Tests correlations** between network properties and execution time
4. **Validates network topologies** match theoretical predictions
5. **Generates visualizations** of the correlations

## Results Summary

| Algorithm | Correlation | Network Type | Status |
|-----------|------------|--------------|---------|
| Factorial | 0.995 | Linear Chain | ✓ Validated |
| Fibonacci | 1.000 | Binary Tree | ✓ Validated |
| Quicksort | 0.991 | Divide-Conquer | ✓ Validated |
| Mergesort | 0.998 | Balanced Tree | ✓ Validated |
| Binary Search | 0.922 | Logarithmic | ✓ Validated |

**Average Correlation: 0.981**

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install all dependencies:
```bash
pip install numpy scipy matplotlib
```

## Project Structure

```
recursion_networks/
│
├── recursive_network_test.py    # Main experiment script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── results/                     # Output directory (created on run)
│   ├── correlations.png        # Correlation plots
│   └── network_metrics.csv     # Raw network data
└── examples/                    # Example outputs
    └── sample_output.txt        # Sample terminal output
```

## How It Works

### 1. Network Construction

Each recursive call becomes a node, each call/return relationship becomes an edge:

```python
factorial(3)
├── factorial(2)      # Node 1 → Node 2 (call edge)
│   ├── factorial(1)  # Node 2 → Node 3 (call edge)
│   └── return        # Node 3 → Node 2 (return edge)
└── return            # Node 2 → Node 1 (return edge)
```

### 2. Performance Prediction

Performance is predicted using only three network properties:

```
Time = (nodes × op_cost) + (edges × call_cost) + (diameter × stack_cost)
```

### 3. Auto-Calibration

The script automatically calibrates hardware-specific constants using least-squares optimization on sample measurements.

## Running Custom Tests

### Test Your Own Recursive Function

```python
# Add your function to the test suite
def your_recursive_function(n):
    if n <= 0:  # Base case
        return 1
    return your_recursive_function(n-1) + n  # Recursive case

# Add to test_cases in recursive_network_test.py
test_cases = [
    ('your_function', your_recursive_function, [10, 20, 30, 40, 50]),
    # ... existing test cases
]
```

### Adjust Test Parameters

```python
# In recursive_network_test.py

# Change number of measurement runs (default: 100)
measure_execution_time(func, input, num_runs=200)

# Change calibration sample size
for test_input in test_inputs[:5]:  # Use first 5 instead of 3

# Change input sizes
test_cases = [
    ('factorial', factorial, range(10, 200, 10)),  # Test larger inputs
    # ...
]
```

## Output Interpretation

### Terminal Output
```
Testing factorial...
  Correlation: 0.9947 (p=3.54e-09)
  R-squared: 0.9893
  Mean Absolute Error: 61.9%
  ✓ PASSED: Strong correlation
```

- **Correlation**: Pearson correlation between predicted and actual times (target: >0.95)
- **R-squared**: Proportion of variance explained by the model
- **p-value**: Statistical significance (smaller = more significant)
- **Mean Absolute Error**: Average prediction error percentage

### Network Topology Validation
```
1. Factorial (n=20)
   Nodes: 20 (expected: 20)
   Diameter: 19 (expected: 19)
   Branching: 1.0 (expected: 1.0)
   ✓ Confirmed: Linear chain structure
```

- **Nodes**: Total function calls
- **Diameter**: Maximum recursion depth
- **Branching**: Average calls per non-leaf node

## Troubleshooting

### Low Correlations

If you see correlations below 0.9:
1. **Measurement noise**: Increase `num_runs` for more stable measurements
2. **Too small inputs**: Use larger input sizes for better resolution
3. **System interference**: Close other programs, disable CPU throttling

### Import Errors

```bash
# Ensure you have the right Python version
python --version  # Should be 3.8+

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Memory Issues

For algorithms with exponential growth (like naive Fibonacci), limit input sizes:
```python
# Fibonacci gets expensive quickly
('fibonacci', fibonacci, [5, 8, 10, 12, 15])  # Keep n < 25
```

## Contributing

Contributions are welcome! Areas of interest:

1. **More algorithms**: Add other recursive patterns
2. **Different languages**: Port to other programming languages
3. **Parallel recursion**: Test concurrent recursive algorithms
4. **Optimization effects**: Compare with memoized versions

## Citation

If you use this code in research, please cite:

```bibtex
@software{recursive_network_analysis,
  title = {Recursive Network Analysis},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/Build-In-Public-University/recursion_networks}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Paper authors for the theoretical framework
- Python scientific computing community for NumPy/SciPy/Matplotlib

## Contact

For questions or issues, please open a GitHub issue or contact leo@buildinpublicuniversity.com

---

**Remember**: The key insight is that recursion doesn't just *use* networks—it *creates* them through temporal execution, and these networks have measurable, predictable properties.
