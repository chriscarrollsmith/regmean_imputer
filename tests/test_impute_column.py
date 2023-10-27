import pytest
import pandas as pd
import numpy as np
from regmean_imputer.impute_column import impute_with_regularization, impute_column, evaluate_regularization

# Create a mock dataset
@pytest.fixture
def mock_data() -> pd.DataFrame:
    data = {
        'Age': [25, np.nan, 30, 35, 40, np.nan, 45],
        'Title': ['Mr', 'Mrs', 'Mr', 'Miss', 'Miss', 'Mrs', 'Mr'],
        'Pclass': [1, 2, 1, 3, 3, 2, 1]
    }
    return pd.DataFrame(data=data)


def test_impute_with_regularization(mock_data) -> None:
    imputed_data = impute_with_regularization(
            m=5,
            data=mock_data,
            impute_col='Age',
            group_by_cols=['Title', 'Pclass'],
            global_mean=mock_data['Age'].mean()
        )
    assert not imputed_data.isnull().any(), "Imputation failed, NaN values found"
    
def test_evaluate_regularization(mock_data) -> None:
    non_missing_data = mock_data[mock_data['Age'].notna()].reset_index(drop=True)
    mse_score = evaluate_regularization(
            m=5,
            non_missing_data=non_missing_data,
            train_idx=[0, 2, 3],
            test_idx=[1, 4],
            impute_col='Age',
            group_by_cols=['Title', 'Pclass'],
            global_mean=mock_data['Age'].mean()
            )
    assert isinstance(mse_score, float), "Function should return a float MSE value"

def test_impute_column(mock_data) -> None:
    imputed_dataset = impute_column(
            data=mock_data,
            impute_col='Age',
            group_by_cols=['Title', 'Pclass']
        )
    assert not imputed_dataset['Age'].isnull().any(), "Imputation failed, NaN values found in Age column"
    assert 'Age_Imputed' in imputed_dataset.columns, "Indicator column not found in the returned dataframe"
