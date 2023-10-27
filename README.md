# README

## Regularized Mean Imputation

### Introduction

Handling missing data is a common challenge in data analysis and machine learning. Regularized mean imputation offers a technique to fill missing values using a regularized mean based on specific grouping columns.

This package provides one main utility for this purpose: a standalone function `impute_column` which can be used for imputation on a pandas DataFrame.

### How it Works

The imputation process works by grouping the data based on the specified columns and computing a regularized mean for each group. This regularized mean is a weighted average of the group mean and the global mean, adjusted by a regularization parameter. The regularization parameter is tuned using cross-validation.

### Installation

```bash
pip install regmean-imputer
```

### Usage

#### 1. Standalone Imputation using `impute_column`:

```python
from regmean_imputer import impute_column

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

### Parameters

- `impute_col`: Column to be imputed.
- `group_by_cols`: Columns used for grouping to compute the regularized mean.
- `m_values`: List of regularization parameters to be tested for optimal performance.
- `n_splits`: Number of splits for cross-validation during regularization evaluation.
- `missing_values`: The placeholder for the missing values. Default is `np.nan`.
- `add_indicator`: Whether to add an indicator column (or columns) that mark the missing values.

### Conclusion

Regularized mean imputation provides an efficient way to handle missing data, especially when certain columns can provide context on how the imputation should be done. The provided utilities in this package make it easier to apply this method.