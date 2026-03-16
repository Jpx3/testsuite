# 🧪 MLIAP Testsuite

This repository contains a comprehensive testsuite designed to evaluate Machine Learning Interatomic Potentials (MLIAP) using Atomic Simulation Environment (ASE) calculators.
The findings of these tests find use in my bachelors thesis `Evaluating Machine Learned Interatomic Potentials for Practical Simulations`,
which you can find 

## 📁 Directory Structure

* 🧠 `/driver`: MLIAP model implementations of the ASE calculator.
* 📈 `/plotting`: Scripts to generate visualization images from test results.
* 💾 `/results`: Generated test results stored in `.pkl` format.
* ⚙️ `/tests`: The individual test definitions executed against the models.

## 🛠️ Installation

```bash
  uv venv
  uv pip install -e .
```

## ⚙️ Tests

Available tests include:
* **Diatomic PE Curves**: Evaluates PE for two atoms across various distances, comparing against reference DFT calculations.
* **Phonon Bands**: Assesses the ability of MLIAPs to reproduce phonon dispersion relations, comparing against DFT reference data.
* **Widom Insertion**: Predicts the chemical potential of a system by inserting a test particle and evaluating the energy change.
* **CO2 Stability**: Evaluates the stability of CO2 molecules under various conditions.
* **Inference Time**: Measures the time taken for MLIAPs to make predictions, providing insights into computational efficiency.

## 🚀 Usage

### 1. Running Tests 🏃‍♂️

Tests are executed sequentially through the primary runner script in the root directory. Individual test execution is not supported. The results are automatically saved as `.pkl` files in the `/results` directory.

```bash
  python run_all_tests.py
```

If you want the full project with our results,
download https://richy.de/testsuite.rar (11GB) for the full repository.

### 2. Generating Plots 🎨


After generating results, you can create specific figures by calling the individual visualization scripts. These scripts automatically read the corresponding `.pkl` files from the `/results` directory.

```bash
  python plotting/plot_inference.py
```

The corresponding `.png` files can be found in `/plotting/figures/`.


## Citing
If you found use for this project 
```bibtex
@inproceedings{
    strunk2026evaluating,
    title={Evaluating Machine Learned Inter-Atomic Potentials for a Practical Simulation Workflow},
    author={Richard Strunk and Karnik Ram and Daniel Cremers},
    booktitle={The Blogpost Track at ICLR 2026},
    year={2026},
    url={https://openreview.net/forum?id=qTOa90sISn}
}
