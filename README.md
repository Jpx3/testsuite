# 🧪 MLIAP Testsuite

This repository contains a comprehensive testsuite designed to evaluate Machine Learning Interatomic Potentials (MLIAP) using Atomic Simulation Environment (ASE) calculators.

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

Please note that this step is optional—results are already provided.
To override provided results for verification, the corresponding files in `/results` need to be erased or renamed.

### 2. Generating Plots 🎨

After generating results, you can create specific figures by calling the individual visualization scripts. These scripts automatically read the corresponding `.pkl` files from the `/results` directory.

```bash
  python plotting/plot_inference.py
```

The corresponding `.png` files can be found in `/plotting/figures/`.