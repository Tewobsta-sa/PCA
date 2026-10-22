# PCA Study: Does Dimensionality Reduction Help Classification?

An experiment with **scikit-learn** that compares machine-learning models **with and without Principal Component Analysis (PCA)**.

## What's inside
- **`main.py`:** Wine dataset. Logistic Regression with and without PCA (keeping 95% of the variance). It prints the eigenvalues, the explained and cumulative variance, a 5-fold cross-validation comparison and how many features PCA removed.
- **`newMain.py`:** a larger, more realistic comparison on several datasets. It tests **Logistic Regression, Random Forest and a Neural Network (MLP)** with and without PCA, using a stratified split, measuring:
  - test accuracy
  - training time
  - stratified 5-fold cross-validation score
- **`pca_analysis.png`, `realistic_pca_analysis.png`:** the result charts.

## Key idea
PCA reduces the number of features while keeping most of the information (variance). This can speed up training. These scripts measure when the speed gain is worth any change in accuracy.

## Run it
```bash
pip install numpy pandas matplotlib scikit-learn
python main.py
python newMain.py
```

## Tech
Python · scikit-learn · NumPy · Pandas · Matplotlib
