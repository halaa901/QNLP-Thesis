# Multimodal Quantum Natural Language Processing: A Novel Framework for Using Quantum Methods to Analyse Real Data

This project is part of a thesis submitted in partial fulfilment of the requirements for the degree of Master of Science in Emerging Digital Technologies, University College London.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Libraries Used](#libraries-used)
- [Usage](#usage)
  - [Dataset 1 (Unstructured)](#dataset-1-unstructured)
  - [Dataset 2 (Structured)](#dataset-2-structured)


## Overview
The project focuses on developing a framework for multimodal quantum natural language processing (QNLP) using quantum methods to analyse real data, particularly integrating image and text classification tasks.

## Requirements

To run this project, you will need to install the following libraries. The installation can be done by running the `requirements.txt` file.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/halaa901/QNLP-Thesis.git
    ```

2. Navigate to the project directory:
    ```bash
    cd QNLP-Thesis
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Libraries Used

- **PyTorch**: For building and training neural networks.
- **Torchvision**: For working with image data.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Pillow (PIL)**: For handling image files.
- **Requests**: For making HTTP requests.
- **Matplotlib**: For creating visualizations.
- **Scikit-learn**: For model evaluation and data splitting.
- **Lambeq**: For quantum natural language processing.

## Usage

The project consists of two datasets:

- **Dataset 1**: Unstructured dataset.
- **Dataset 2**: Structured dataset.

### Dataset 1 (Unstructured)

1. **Preprocessing**:
    - The main dataset is stored in `svo_probes.csv`.
    - The script `sub-dataset.py` creates a smaller dataset with a fair distribution.
    - The script `clean_dataset.py` checks for incorrect URLs and removes or corrects them.
    - The script `feature_extraction.py` is used to extract image feature vectors from the dataset.

2. **Data Splitting**:
    - The `circuit.ipynb` script splits the dataset into training, validation, and test sets.
    - The split data is saved for reuse during the training of other models.

3. **Model Training and Testing**:
    - The `circuit.ipynb` script also handles the execution of string diagrams, quantum circuit representations, and runs the models for training and testing.
    - The results are saved in the `Outputs` folder.
   
### Dataset 2 (Structured)

1. **Preprocessing**:
    - A custom dataset `custom_dataset_v2.csv` was created with images stored in the `images` folder.
    - The script `feature_extraction_2.py` extracts image features from this dataset.

2. **Data Splitting**:
    - The `circuit2.ipynb` script splits the dataset into training, validation, and test sets.
    - The split data is saved for reuse during the training of other models.

3. **Model Training and Testing**:
    - The `circuit2.ipynb` script also handles the execution of string diagrams, quantum circuit representations, and runs the models for training and testing.
    - The results are saved in the `Output_2` folder.
  
## Results

The results of the experiments are stored in the `Dataset_1/Outputs` and `Dataset_2/Output_2` folders. For each model, accuracy and loss metrics were recorded to evaluate performance. The detailed results and performance metrics can be found in these respective folders.

