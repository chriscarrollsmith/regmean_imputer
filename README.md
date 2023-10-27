# Regularized Mean Imputer

## Introduction

Handling missing data is a common challenge in data analysis and machine learning. Regularized mean imputation offers a technique to fill missing values using a regularized mean based on specific grouping columns.

This package provides one main utility for this purpose: a standalone function `impute_column` which can be used for imputation on a pandas DataFrame. This function is meant to be used in a machine learning preprocessing pipeline to impute missing values in both the training and testing datasets with averages only from the train set, preventing information leakage.

## How it Works

The imputation process works by grouping the data based on the specified columns and computing a regularized mean for each group. This regularized mean is a weighted average of the group mean and the global mean, adjusted by a regularization parameter. The regularization parameter is tuned using cross-validation.

## Installation

```bash
pip install regmean-imputer
```

## Usage

### Imputation using `impute_column` with separate train and test data:

```python
from regmean_imputer import impute_column

# Sample train data
train_data = {
    'Age': [25, None, 30, 35, 40],
    'Title': ['Mr', 'Mrs', 'Mr', 'Miss', 'Miss'],
    'Pclass': [1, 2, 1, 3, 3]
}

# Sample test data
test_data = {
    'Age': [None, 45],
    'Title': ['Mrs', 'Mr'],
    'Pclass': [2, 1]
}

train_df = pd.DataFrame(data=train_data)
test_df = pd.DataFrame(data=test_data)

# Impute the 'Age' column using 'Title' and 'Pclass' as group by columns
imputed_train_data, imputed_test_data = impute_column(train_data=train_df, test_data=test_df, impute_col='Age', group_by_cols=['Title', 'Pclass'])
print(imputed_train_data)
print(imputed_test_data)
```

This approach of separating the train and test data before imputation is crucial in a machine learning preprocessing pipeline to prevent information leakage. Information leakage happens when information that would not be available at prediction time is used when building the model. This can lead to overly optimistic performance estimates. For example, if we impute missing values in the entire dataset using the mean of a column, the mean is influenced by the test set values, which wouldn't be available at prediction time in a real-world scenario.

## Parameters

- `train_data` (pd.DataFrame): The training dataset.
- `test_data` (pd.DataFrame): The testing dataset.
- `impute_col` (str): Column to be imputed.
- `group_by_cols` (list): Columns used for grouping to compute the regularized mean.
- `m_values` (list, optional): List of regularization parameters to be tested for optimal performance. Default is [1,2,3,4,5,6,7,8,9,10].
- `n_splits` (int, optional): Number of splits for cross-validation during regularization evaluation. Default is 5.
- `verbose` (bool, optional): Whether to print the best regularization parameter. Default is False.
- `random_state` (int, optional): Random state for cross-validation. Default is None.

## Conclusion

Regularized mean imputation provides an efficient way to handle missing data, especially when certain columns can provide context on how the imputation should be done. The provided utilities in this package make it easier to apply this method.
