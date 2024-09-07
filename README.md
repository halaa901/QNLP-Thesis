# Project Title

A brief description of what this project does and who it's for.

## Table of Contents

..
## Overview
..
Provide a brief introduction to your project here. Explain the goals, the scope, and what this project achieves.

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
    - The results are saved in the `output_2` folder.

## File Structure
...
Outline the structure of your project folder to help users understand the organization.

