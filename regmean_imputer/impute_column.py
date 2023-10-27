from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def impute_with_regularization(m, train_data, test_data, impute_col, group_by_cols, global_mean) -> pd.DataFrame:
    """Compute regularized mean for imputation."""
    # Combine the train and test datasets
    data = pd.concat([train_data, test_data])
    
    def regularized_mean(group) -> float:
        n = len(group)
        sample_mean = group.mean()
        return (n * sample_mean + m * global_mean) / (n + m)
    
    regularized_means = data.groupby(group_by_cols)[impute_col].transform(regularized_mean)
    
    # Replace NaN values in regularized_means with global_mean
    regularized_means.fillna(global_mean, inplace=True)
    
    # Split the imputed data back into train and test datasets
    imputed_data_train = data.iloc[:len(train_data)][impute_col].fillna(regularized_means)
    imputed_data_test = data.iloc[len(train_data):][impute_col].fillna(regularized_means)
    
    return imputed_data_train, imputed_data_test


def evaluate_regularization(m, non_missing_data, train_idx, test_idx, impute_col, group_by_cols, global_mean) -> float:
    """Evaluate the regularization on a given train-test split."""
    train_data, test_data = non_missing_data.iloc[train_idx], non_missing_data.iloc[test_idx]
    
    # Mask the target column values in the test data to simulate missing values
    masked_test_data = test_data.copy()
    masked_test_data[impute_col] = np.nan
    
    # Append the train and test data
    combined_data = pd.concat(objs=[train_data, masked_test_data])
    
    # Impute the "missing" values
    imputed_values_train, imputed_values_test = impute_with_regularization(m=m, train_data=train_data, test_data=masked_test_data, impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean)
    imputed_values = pd.concat([imputed_values_train, imputed_values_test])
    
    # Calculate the mean squared error between the imputed and actual values
    mse = mean_squared_error(y_true=test_data[impute_col], y_pred=imputed_values.iloc[len(train_data):])
    return mse


def impute_column(train_data, test_data, impute_col, group_by_cols, m_values=[1,2,3,4,5,6,7,8,9,10], n_splits=5):
    """
    Impute missing values in a column using regularized means.
    
    Args:
    - train_data (pd.DataFrame): The training dataset.
    - test_data (pd.DataFrame): The testing dataset.
    - impute_col (str): The column name to impute.
    - group_by_cols (list): List of column names to group by for calculating regularized mean.
    - m_values (list): List of regularization parameters to try.
    - n_splits (int): Number of splits for KFold cross-validation.
    
    Returns:
    - tuple: Tuple of training and testing datasets with imputed values.
    """
    
    # Create an indicator column for imputed values in both datasets
    indicator_col_name = f"{impute_col}_Imputed"
    train_data[indicator_col_name] = train_data[impute_col].isnull().astype(dtype=int)
    test_data[indicator_col_name] = test_data[impute_col].isnull().astype(dtype=int)

    # Compute the global mean of the target column once using the training dataset
    global_mean = train_data[impute_col].mean()

    # Filter rows where target column value is present in the training dataset
    non_missing_data = train_data[train_data[impute_col].notna()]
    
    # Ensure that the number of splits is not greater than the number of samples
    n_splits = min(n_splits, len(non_missing_data))
    kf = KFold(n_splits=n_splits, shuffle=True)
    mean_mse_scores = []

    for m in m_values:
        mse_scores = [evaluate_regularization(m=m, non_missing_data=non_missing_data, train_idx=train_idx, test_idx=test_idx, impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean) for train_idx, test_idx in kf.split(X=non_missing_data)]
        mean_mse_scores.append(np.mean(mse_scores))

    # Get the best regularization parameter
    best_m = m_values[mean_mse_scores.index(min(mean_mse_scores))]
    print(f"Best regularization parameter for {impute_col}: {best_m}")

    # Calculate the imputed values for the training and testing datasets separately using the best regularization parameter
    imputed_values_train, _ = impute_with_regularization(m=best_m, train_data=train_data, test_data=pd.DataFrame(), impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean)
    _, imputed_values_test = impute_with_regularization(m=best_m, train_data=pd.DataFrame(), test_data=test_data, impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean)

    # Fill the missing values in the target column in the training and testing datasets with the imputed values
    train_data[impute_col] = train_data[impute_col].fillna(value=imputed_values_train)
    test_data[impute_col] = test_data[impute_col].fillna(value=imputed_values_test)
    
    return train_data, test_data
