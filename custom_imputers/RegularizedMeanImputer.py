from sklearn.impute._base import _BaseImputer
from sklearn.utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from sklearn.utils._param_validation import Interval
from sklearn.utils._mask import _get_mask
from sklearn.utils import is_scalar_nan
from sklearn.base import _fit_context
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numbers import Integral
import numpy as np
import pandas as pd


def impute_with_regularization(m, data, impute_col, group_by_cols, global_mean) -> pd.DataFrame:
    """Compute regularized mean for imputation."""
    def regularized_mean(group) -> float:
        n = len(group)
        sample_mean = group.mean()
        return (n * sample_mean + m * global_mean) / (n + m)
    
    regularized_means = data.groupby(group_by_cols)[impute_col].transform(regularized_mean)
    
    # Replace NaN values in regularized_means with global_mean
    regularized_means.fillna(global_mean, inplace=True)
    
    return data[impute_col].fillna(regularized_means)


def evaluate_regularization(m, non_missing_data, train_idx, test_idx, impute_col, group_by_cols, global_mean) -> float:
    """Evaluate the regularization on a given train-test split."""
    train_data, test_data = non_missing_data.iloc[train_idx], non_missing_data.iloc[test_idx]
    
    # Mask the target column values in the test data to simulate missing values
    masked_test_data = test_data.copy()
    masked_test_data[impute_col] = np.nan
    
    # Append the train and test data
    combined_data = pd.concat(objs=[train_data, masked_test_data])
    
    # Impute the "missing" values
    imputed_values = impute_with_regularization(m=m, data=combined_data, impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean)
    
    # Calculate the mean squared error between the imputed and actual values
    mse = mean_squared_error(y_true=test_data[impute_col], y_pred=imputed_values.iloc[len(train_data):])
    return mse


def impute_column(data, impute_col, group_by_cols, m_values=[1,2,3,4,5,6,7,8,9,10], n_splits=5):
    """
    Impute missing values in a column using regularized means.
    
    Args:
    - data (pd.DataFrame): The dataset.
    - impute_col (str): The column name to impute.
    - group_by_cols (list): List of column names to group by for calculating regularized mean.
    - m_values (list): List of regularization parameters to try.
    - n_splits (int): Number of splits for KFold cross-validation.
    
    Returns:
    - pd.DataFrame: Dataset with imputed values.
    """
    
    # Create an indicator column for imputed values
    indicator_col_name = f"{impute_col}_Imputed"
    data[indicator_col_name] = data[impute_col].isnull().astype(dtype=int)

    # Compute the global mean of the target column once using the entire dataset
    global_mean = data[impute_col].mean()

    # Filter rows where target column value is present
    non_missing_data = data[data[impute_col].notna()]
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    mean_mse_scores = []

    for m in m_values:
        mse_scores = [evaluate_regularization(m=m, non_missing_data=non_missing_data, train_idx=train_idx, test_idx=test_idx, impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean) for train_idx, test_idx in kf.split(X=non_missing_data)]
        mean_mse_scores.append(np.mean(mse_scores))

    # Get the best regularization parameter
    best_m = m_values[mean_mse_scores.index(min(mean_mse_scores))]
    print(f"Best regularization parameter for {impute_col}: {best_m}")

    # Calculate the imputed values for the entire dataset using the best regularization parameter
    imputed_values = impute_with_regularization(m=best_m, data=data, impute_col=impute_col, group_by_cols=group_by_cols, global_mean=global_mean)

    # Fill the missing values in the target column with the imputed values
    data[impute_col] = data[impute_col].fillna(value=imputed_values)
    
    return data


class RegularizedMeanImputer(_BaseImputer):
    
    _parameter_constraints: dict = {
        **_BaseImputer._parameter_constraints,  # Constraints inherited from the base imputer
        "impute_col": [str],  # The column to impute should be a string (column name)
        "group_by_cols": [list],  # The group_by columns should be a list of strings
        "m_values": [list],  # m_values should be a list of integers
        "n_splits": [Interval(Integral, 2, None, closed="left")],  # n_splits should be an integer greater than or equal to 2
        "copy": ["boolean"],  # copy should be a boolean
    }

    def __init__(
                self,
                impute_col,
                group_by_cols,
                m_values=[1,2,3,4,5,6,7,8,9,10],
                n_splits=5,
                missing_values=np.nan,
                copy=True,
                add_indicator=False,
                keep_empty_features=False,
            ):
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self.impute_col = impute_col
        self.group_by_cols = group_by_cols
        self.m_values = m_values
        self.n_splits = n_splits
        self.copy = copy

    def _groupby_regularized_mean(self, X, group_by_cols, m, global_mean):
        # Group the data by the group_by_cols and compute the regularized mean for each group
        grouped = X.groupby(group_by_cols)
        
        # Compute the regularized mean for each group
        regularized_means = grouped[self.impute_col].transform(lambda x: self._regularized_mean(x, m, global_mean))
        
        # For those values which are NaN in the original data, replace them with the computed regularized means
        X.loc[X[self.impute_col].isna(), self.impute_col] = regularized_means[X[self.impute_col].isna()]
        
        return X[self.impute_col].values

    def _regularized_mean(self, group, m, global_mean):
        n = len(group)
        sample_mean = group.mean()
        return (n * sample_mean + m * global_mean) / (n + m)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # Convert DataFrame to numpy array for compatibility
        # Only converting the target column
        target_data = self._validate_data(
            X[self.impute_col].values.reshape(-1, 1), 
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan" if is_scalar_nan(self.missing_values) else True,
            copy=self.copy,
        )
        
        # Check if the impute_col is numeric
        if not np.issubdtype(target_data.dtype, np.number):
            raise ValueError(f"Column {self.impute_col} must be numeric")

        # Store column indices
        self.impute_col_idx = X.columns.get_loc(self.impute_col)
        self.group_by_cols_idx = [X.columns.get_loc(col) for col in self.group_by_cols]

        # Get the mask for missing data in the target column using _get_mask
        self._mask_fit_X = _get_mask(target_data, self.missing_values)

        # Fit the MissingIndicator using _fit_indicator
        self._fit_indicator(self._mask_fit_X)

        global_mean = np.nanmean(target_data) # Compute mean ignoring NaNs

        non_missing_mask = ~self._mask_fit_X.ravel() # Convert 2D mask to 1D
        non_missing_data = X[non_missing_mask]

        kf = KFold(n_splits=self.n_splits, shuffle=True)
        mean_mse_scores = []

        for m in self.m_values:
            mse_scores = []
            for train_idx, test_idx in kf.split(X=non_missing_data):
                train_data, test_data = non_missing_data.iloc[train_idx], non_missing_data.iloc[test_idx]
                
                masked_test_data = test_data.copy(deep=True)
                masked_test_data.iloc[:, self.impute_col_idx] = np.nan
                combined_data = pd.concat([train_data, masked_test_data], axis=0)
                
                imputed_values = self._groupby_regularized_mean(combined_data, self.group_by_cols, m, global_mean)
                
                nan_mask = combined_data[self.impute_col].isna()
                combined_data.loc[nan_mask, self.impute_col] = imputed_values[nan_mask]
                
                # Check for NaNs in the imputed data and fill them with the global mean
                combined_data[self.impute_col] = combined_data[self.impute_col].fillna(global_mean)

                # Calculate MSE
                mse = mean_squared_error(y_true=test_data.iloc[:, self.impute_col_idx], y_pred=combined_data.iloc[len(train_data):, self.impute_col_idx])

                mse_scores.append(mse)
            
            mean_mse_scores.append(np.mean(mse_scores))
        
        self.best_m_ = self.m_values[mean_mse_scores.index(min(mean_mse_scores))]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        # Generate mask for input data
        mask = X[self.impute_col].isna()

        global_mean = X[self.impute_col].mean()

        regularized_means = self._groupby_regularized_mean(X, self.group_by_cols, self.best_m_, global_mean)

        # Fill NaN values in the regularized_means with the global mean
        regularized_means[np.isnan(regularized_means)] = global_mean

        X_imputed = X.copy(deep=True)
        X_imputed.loc[mask, self.impute_col] = regularized_means[mask]
        
        X_indicator = super()._transform_indicator(mask)
        return super()._concatenate_indicator(X_imputed, X_indicator)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
            used as feature names in. If `feature_names_in_` is not defined,
            then the following input feature names are generated:
            `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
            match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        
        # For RegularizedMeanImputer, the output features are the same as input features
        # unless the indicator is added.
        names = input_features
        return self._concatenate_indicator_feature_names_out(names, input_features)
