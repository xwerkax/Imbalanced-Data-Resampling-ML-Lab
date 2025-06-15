# Laboratory 6 - Imbalanced Data & Resampling
This laboratory focuses on generating and analyzing imbalanced classification problems with varying class distributions. Students will learn to create synthetic datasets with different imbalance ratios and visualize their characteristics using dimensionality reduction techniques.

## Task 1: Generate Imbalanced Datasets
Create three binary classification problems using make_classification:

### Dataset Parameters (all datasets):

- Samples: 5000
- Features: 8 (2 informative only)
- Classes: 2 (binary classification)

### Three Scenarios:

- Moderate imbalance: Class ratio 1:5
- Extreme imbalance: Class ratio 1:99
- Noisy imbalance: Class ratio 1:9 + 5% label noise

### Visualization:

- Extract 2 principal components using PCA
- Create plots showing class distribution for each dataset
- Display class imbalance ratios

### Expected Output
Three scatter plots showing:

- PCA-reduced feature space (2D)
- Different colors for each class
- Clear visualization of class imbalance
- Titles indicating class ratios
