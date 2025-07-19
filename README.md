This repository explores various machine learning and deep learning techniques for analyzing the structural components of documents. It includes implementations using state-of-the-art deep learning frameworks like Detectron2, as well as classical machine learning algorithms such as Random Forest and Support Vector Machines (SVMs), for tasks like layout analysis, element classification, and information extraction from documents.

## Table of Contents

-   [Document Structure Analysis](#document-structure-analysis)
    -   [Table of Contents](#table-of-contents)
    -   [Features](#features)
    -   [Project Structure](#project-structure)
    -   [Installation](#installation)
        -   [Prerequisites](#prerequisites)
        -   [Cloning the Repository](#cloning-the-repository)
        -   [Setting up the Environment](#setting-up-the-environment)
        -   [Installing Dependencies](#installing-dependencies)
            -   [Common Dependencies](#common-dependencies)
            -   [Detectron2 Specific Dependencies](#detectron2-specific-dependencies)
    -   [Usage](#usage)
        -   [General Workflow](#general-workflow)
        -   [Detectron2](#detectron2)
        -   [Random Forest](#random-forest)
        -   [SVM](#svm)
    -   [API Documentation / Code Structure](#api-documentation--code-structure)
    -   [Contributing](#contributing)
    -   [License](#license)
    -   [Contact](#contact)

## Features

*   **Deep Learning with Detectron2:** Leverage a powerful deep learning framework for advanced document layout analysis, object detection (e.g., tables, figures, text blocks), and segmentation.
*   **Classical Machine Learning Approaches:** Implementations of Random Forest and Support Vector Machines for tasks such as document element classification or feature-based structural analysis.
*   **Modular Design:** Each technique is encapsulated in its own directory, allowing for easy comparison, experimentation, and extension.
*   **Scalability:** Designed to be adaptable for various document types and datasets.
*   **(Assumed) Pre-trained Models:** Support for using pre-trained models where applicable (e.g., Detectron2's model zoo).

## Project Structure

```
Document-Structure-Analysis/
├── Detectron2/
│   ├── configs/                 # Configuration files for Detectron2 models
│   ├── scripts/                 # Scripts for training, inference, data preparation
│   ├── models/                  # (Optional) Directory for pre-trained models or checkpoints
│   └── README.md                # Specific README for Detectron2 implementation
├── Random Forest/
│   ├── data/                    # Sample data or feature sets
│   ├── scripts/                 # Scripts for training, evaluation, prediction
│   └── README.md                # Specific README for Random Forest implementation
├── SVM/
│   ├── data/                    # Sample data or feature sets
│   ├── scripts/                 # Scripts for training, evaluation, prediction
│   └── README.md                # Specific README for SVM implementation
├── .gitignore                   # Git ignore file
├── requirements.txt             # Main Python dependencies
└── README.md                    # This comprehensive README file
```

## Installation

### Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package installer)
*   `git` (version control system)

### Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/DhruvBhalodia/Document-Structure-Analysis.git
cd Document-Structure-Analysis
```

### Setting up the Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Installing Dependencies

Dependencies are split into common ones and Detectron2-specific ones due to Detectron2's complex installation.

#### Common Dependencies

Install the core Python packages required for the Random Forest and SVM modules, and general utilities:

```bash
pip install -r requirements.txt
```

**`requirements.txt` (Example Content):**
```
numpy>=1.20
scikit-learn>=1.0
pandas>=1.3
Pillow>=9.0
opencv-python>=4.5
matplotlib>=3.5
tqdm>=4.60
```

#### Detectron2 Specific Dependencies

Detectron2 has specific installation requirements, especially concerning PyTorch and CUDA. Please refer to the official Detectron2 installation guide for the most up-to-date instructions: [Detectron2 Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Here's a common installation path assuming you have CUDA 11.3 and PyTorch 1.10:

```bash
# Install PyTorch (ensure matching CUDA version if you have a GPU)
# Example for PyTorch 1.10.0 and CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Or, for a specific version:
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/<YOUR_CUDA_VERSION>/torch<YOUR_PYTORCH_VERSION>/index.html
```

Make sure to be inside your activated virtual environment before installing these.

## Usage

Each subdirectory contains its own set of scripts and potentially its own `README.md` with more specific usage instructions.

### General Workflow

1.  **Prepare Data:** Ensure your document images and corresponding annotations (if required for training) are in the expected format for each module.
2.  **Train Model:** Run the training scripts within the respective module (`Detectron2`, `Random Forest`, `SVM`).
3.  **Evaluate Model:** Use evaluation scripts to assess model performance.
4.  **Run Inference:** Apply the trained models to new, unseen documents for structure analysis.

### Detectron2

The `Detectron2/` directory is designed for deep learning-based document analysis.

**Example: Training a Detectron2 Model**

```bash
# Navigate to the Detectron2 directory
cd Detectron2/scripts

# Example: Train a model (replace with your actual script name and config)
# You'll need to set up your dataset configuration in Detectron2/configs/
python train_model.py --config-file ../configs/my_doc_layout_config.yaml --num-gpus 1
```

**Example: Running Inference with Detectron2**

```bash
# Navigate to the Detectron2 directory
cd Detectron2/scripts

# Example: Run inference on a document image
python inference_on_document.py --input-image ../data/sample_doc.png --output-dir ../output/ --model-path ../models/trained_doc_model.pth
```

Refer to `Detectron2/README.md` for detailed instructions and configuration specifics.

### Random Forest

The `Random Forest/` directory contains implementations using Random Forest for document analysis tasks, often based on extracted features.

**Example: Training a Random Forest Model**

```bash
# Navigate to the Random Forest directory
cd Random Forest/scripts

# Example: Train a Random Forest classifier
# Assumes you have your features and labels ready (e.g., in CSV or pickle files)
python train_random_forest.py --features ../data/doc_features.csv --labels ../data/doc_labels.csv --output-model ../models/rf_model.pkl
```

**Example: Making Predictions with Random Forest**

```bash
# Navigate to the Random Forest directory
cd Random Forest/scripts

# Example: Predict document structure elements on new features
python predict_random_forest.py --input-features ../data/new_doc_features.csv --model ../models/rf_model.pkl --output-predictions ../output/rf_predictions.csv
```

Refer to `Random Forest/README.md` for more details.

### SVM

The `SVM/` directory holds implementations utilizing Support Vector Machines for document analysis.

**Example: Training an SVM Model**

```bash
# Navigate to the SVM directory
cd SVM/scripts

# Example: Train an SVM classifier
# Similar to Random Forest, requires prepared features and labels
python train_svm.py --features ../data/doc_features.csv --labels ../data/doc_labels.csv --output-model ../models/svm_model.pkl
```

**Example: Making Predictions with SVM**

```bash
# Navigate to the SVM directory
cd SVM/scripts

# Example: Predict document structure elements on new features
python predict_svm.py --input-features ../data/new_doc_features.csv --model ../models/svm_model.pkl --output-predictions ../output/svm_predictions.csv
```

Refer to `SVM/README.md` for more details.

## API Documentation / Code Structure

Each module (`Detectron2`, `Random Forest`, `SVM`) is self-contained. The `scripts/` subdirectory within each module typically contains the main executable Python files.

*   **Input Data:** Scripts generally expect input data (images, feature vectors, annotations) to be provided via command-line arguments or configuration files.
*   **Output:** Outputs usually include trained models, prediction files (e.g., CSV, JSON), or visualized results.
*   **Internal Modules:** Within each `scripts/` folder, you might find helper functions or classes (e.g., `data_loader.py`, `model_utils.py`) that encapsulate specific functionalities.
*   **Configuration:** For Detectron2, configuration is primarily handled via YAML files. For classical ML, parameters are often passed as command-line arguments.

For specific function signatures and class definitions, please inspect the source code files within each module.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/issue-description`.
3.  **Make your changes** and ensure they adhere to the existing code style.
4.  **Write clear, concise commit messages.**
5.  **Push your branch** to your forked repository.
6.  **Open a Pull Request** to the `main` branch of this repository, describing your changes in detail.

## License

This project is currently **Unlicensed**. This means that by default, all rights are reserved by the copyright holder (DhruvBhalodia).

**Recommendation:** For an open-source project, it is highly recommended to add a license. Common choices include:

*   **MIT License:** A permissive license that allows reuse with minimal restrictions.
*   **Apache 2.0 License:** Similar to MIT but includes a patent grant.

Please consider adding a LICENSE file to the root of the repository.
