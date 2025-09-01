
# E-Style

E-Style is a Python project for classifying clothing product descriptions as either "tops" or "bottoms" using a BERT-based transformer model. The project is modular, supports reproducible research, and includes both scripts and Jupyter Notebooks for data analysis and experimentation.


## Features

- **ML Pipeline:** Classifies product descriptions as "top" or "bottom" using a fine-tuned BERT model.
- **Comprehensive Data Pipeline:** Organized directories for data storage (`data/`), pipeline components (`pipeline/`), and utility functions (`utils/`).
- **Interactive Analysis:** Jupyter Notebooks for EDA and experiments.
- **Modular Codebase:** Separated logic for scalability and maintenance.
- **Reproducible Environments:** All dependencies managed via `requirements.txt`.


## Directory Structure

```
E-Style/
│
├── data/                   # Data files: bottoms.csv, tops.csv (required)
├── notebooks/              # Jupyter notebooks for EDA and experimentation
│   ├── EDA.ipynb
│   └── train_experiments.ipynb
├── pipeline/               # Data pipeline and modeling scripts
├── utils/                  # Utility modules and helper functions
├── main.py                 # Main execution script (model training & evaluation)
└── requirements.txt        # Project dependencies
```


## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ajsike2310/E-Style.git
    cd E-Style
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare data:**
    - Place `bottoms.csv` and `tops.csv` in the `data/` directory. Each file should have at least `product_name` and `details` columns.



## Usage

- **Train and evaluate the top/bottom classifier:**
    ```bash
    python main.py
    ```
    This will train a BERT-based classifier to distinguish between tops and bottoms, print evaluation metrics, and show sample predictions.

- **Train and evaluate the top-bottom pair compatibility model:**
    ```bash
    python main_pair.py
    ```
    This will train a BERT-based model to predict whether a given top and bottom pair is a good match (1) or not (0), using the data in `data/top_bottom_pairs.csv`. The script prints per-pair predictions and evaluation metrics for the validation set.

- **Exploratory Data Analysis & Experimentation:**
    Open the notebooks in the `notebooks/` directory using JupyterLab or Jupyter Notebook.
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```


## How it works

- **main.py:** Loads and balances the data, combines product name and details, and splits into train/test sets. A BERT model is fine-tuned to classify each product as a top (1) or bottom (0). Prints accuracy, confusion matrix, and classification report. Example predictions for jeans and blazer descriptions are shown.

- **main_pair.py:** Loads labeled top-bottom pairs from `data/top_bottom_pairs.csv` (CSV with columns: top_product_name, top_details, bottom_product_name, bottom_details, label). Trains a BERT-based model to predict compatibility. Splits data into train/validation sets, prints per-pair predictions, accuracy, confusion matrix, and classification report.


## Sample predictions

After running `python main.py`, you will see output like:

```
Prediction for jeans: 0
Prediction for blazer: 1
```
Where 0 = bottom, 1 = top.

After running `python main_pair.py`, you will see output like:

```
Predictions for each pair in validation set:
Pair 1: Top='CHECKED SUIT BLAZER', Bottom='METALLIC-EFFECT JEANS' | True: 0 | Pred: 0
Pair 2: Top='100% WOOL SUIT BLAZER', Bottom='METALLIC-EFFECT JEANS' | True: 0 | Pred: 0
Pair 3: Top='CHECKED SUIT BLAZER', Bottom='WIDE FIT JEANS' | True: 1 | Pred: 1
...
Validation Evaluation:
Accuracy: 1.0
Confusion matrix:
 [[2 0]
    [0 5]]
Classification report:
 ...
```



## Notebooks

- [`EDA.ipynb`](notebooks/EDA.ipynb): Exploratory Data Analysis
- [`train_experiments.ipynb`](notebooks/train_experiments.ipynb): Model Training Experiments

## Contributing

Contributions are welcome and encouraged. Please open an issue or submit a pull request with suggestions, improvements, or bug fixes.

## License

This project is licensed under the terms of the MIT License.

---

For any inquiries or professional collaboration opportunities, please contact [ajsike2310](https://github.com/ajsike2310).
