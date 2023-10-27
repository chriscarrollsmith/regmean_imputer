# README

## Regularized Mean Imputation

### Introduction

Handling missing data is a common challenge in data analysis and machine learning. Regularized mean imputation offers a technique to fill missing values using a regularized mean based on specific grouping columns.

This package provides two main utilities for this purpose:
1. A standalone function `impute_column` which can be used for one-time imputation on a dataset.
2. A custom imputer class `RegularizedMeanImputer` which extends the Scikit-learn `_BaseImputer` class, allowing it to be integrated into Scikit-learn's pipeline and model-building process.

### How it Works

The imputation process works by grouping the data based on the specified columns and computing a regularized mean for each group. This regularized mean is a weighted average of the group mean and the global mean, adjusted by a regularization parameter. The regularization parameter is determined using cross-validation.

### Installation

```bash
pip install custom-imputers
```

### Usage

#### 1. Standalone Imputation using `impute_column`:

```python
from custom_imputers import impute_column

# Sample data
data = {
    'Age': [25, None, 30, 35, 40, None, 45],
    'Title': ['Mr', 'Mrs', 'Mr', 'Miss', 'Miss', 'Mrs', 'Mr'],
    'Pclass': [1, 2, 1, 3, 3, 2, 1]
}
df = pd.DataFrame(data=data)

# Impute the 'Age' column using 'Title' and 'Pclass' as group by columns
imputed_data = impute_column(data=df, impute_col='Age', group_by_cols=['Title', 'Pclass'])
print(imputed_data)
```

#### 2. Using `RegularizedMeanImputer` within Scikit-learn's pipeline:

```python
from custom_imputers import RegularizedMeanImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data
data = {
    'Age': [25, None, 30, 35, 40, None, 45],
    'Title': ['Mr', 'Mrs', 'Mr', 'Miss', 'Miss', 'Mrs', 'Mr'],
    'Pclass': [1, 2, 1, 3, 3, 2, 1]
}
df = pd.DataFrame(data=data)

# Define the imputer
imputer = RegularizedMeanImputer(impute_col='Age', group_by_cols=['Title', 'Pclass'])

# Use the imputer in a ColumnTransformer
transformers = [("impute_age", imputer, ['Age', 'Title', 'Pclass'])]
column_transform = ColumnTransformer(transformers, remainder='passthrough')

# Create a pipeline with the transformer
pipeline = Pipeline(steps=[('preprocess', column_transform)])

# Fit and transform the data using the pipeline
transformed_data = pipeline.fit_transform(df)
print(transformed_data)
```

### Parameters

- `impute_col`: Column to be imputed.
- `group_by_cols`: Columns used for grouping to compute the regularized mean.
- `m_values`: List of regularization parameters to be tested for optimal performance.
- `n_splits`: Number of splits for cross-validation during regularization evaluation.
- `missing_values`: The placeholder for the missing values. Default is `np.nan`.
- `add_indicator`: Whether to add an indicator column (or columns) that mark the missing values.

### Conclusion

Regularized mean imputation provides an efficient way to handle missing data, especially when certain columns can provide context on how the imputation should be done. The provided utilities in this package make it easier to apply this method both independently and within a Scikit-learn pipeline.