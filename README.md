# Algorithms from Scratch

This project aims to implement various machine learning algorithms from scratch using Python. The goal is to provide a deeper understanding of how these algorithms work under the hood.

## Project Structure

- `supervised/`: Implementations of supervised learning algorithms
- `unsupervised/`: Implementations of unsupervised learning algorithms
- `reinforcement/`: Implementations of reinforcement learning algorithms
- `utils/`: Utility functions for data preprocessing, evaluation, and visualization
- `datasets/`: Sample datasets for testing and examples
- `examples/`: Jupyter notebooks with usage examples
- `tests/`: Unit tests for the implemented algorithms

## Getting Started

1. Clone this repository
2. Install the required dependencies:  
    ```bash
    uv pip install -r requirements.txt
    ```
3. Create a virtual environment using `uv`:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```
4. Explore the implementations in the respective directories
5. Run the example notebooks to see the algorithms in action

## Data Management

Please note that large dataset files such as `.csv`, `.pkl`, `.parquet`, etc., are excluded from the Git repository. You can manually add your own datasets to the `datasets/` folder for testing purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
