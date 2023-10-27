import pytest
import pandas as pd
import numpy as np
from regmean_imputer.impute_column import impute_with_regularization, impute_column, evaluate_regularization

# Create a mock dataset
@pytest.fixture
def mock_data() -> pd.DataFrame:
    data = {
        'Age': [25, np.nan, 30, 35, 40, np.nan, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        'Title': ['Mr', 'Mrs', 'Mr', 'Miss', 'Miss', 'Mrs', 'Mr', 'Mr', 'Mrs', 'Mr', 'Miss', 'Miss', 'Mrs', 'Mr', 'Mr', 'Mrs', 'Mr', 'Miss'],
        'Pclass': [1, 2, 1, 3, 3, 2, 1, 1, 2, 1, 3, 3, 2, 1, 1, 2, 1, 3]
    }
    return pd.DataFrame(data=data)


def test_impute_with_regularization(mock_data) -> None:
    train_data = mock_data.iloc[:5]
    test_data = mock_data.iloc[5:]
    imputed_data_train, imputed_data_test = impute_with_regularization(
            m=5,
            train_data=train_data,
            test_data=test_data,
            impute_col='Age',
            group_by_cols=['Title', 'Pclass'],
            global_mean=train_data['Age'].mean()
        )
    assert not imputed_data_train.isnull().any(), "Imputation failed, NaN values found in train data"
    assert not imputed_data_test.isnull().any(), "Imputation failed, NaN values found in test data"

def test_impute_uses_only_train_data(mock_data) -> None:
    # Splitting the data such that 'Mrs' and 'Pclass' 2 is not present in the training data but has NaN values in the test data
    train_data = mock_data.iloc[:5]
    test_data = mock_data.iloc[5:]
    global_mean = train_data['Age'].mean()
    
    _, imputed_data_test = impute_with_regularization(
            m=5,
            train_data=train_data,
            test_data=test_data,
            impute_col='Age',
            group_by_cols=['Title', 'Pclass'],
            global_mean=global_mean
        )
    
    # Check the imputed value for the 'Mrs' title and 'Pclass' 2 in the test data
    test_group_value = imputed_data_test[(test_data['Title'] == 'Mrs') & (test_data['Pclass'] == 2)].values
    if len(test_group_value) == 0:
        raise ValueError("The group ('Mrs', 2) does not exist in the test set.")
    
    assert test_group_value[0] == pytest.approx(global_mean, 0.01), f"Expected imputed value {global_mean} but got {test_group_value[0]}"

def test_evaluate_regularization(mock_data) -> None:
    train_data = mock_data.iloc[:10]
    test_data = mock_data.iloc[10:]
    non_missing_data = train_data[train_data['Age'].notna()].reset_index(drop=True)
    train_idx = list(range(len(non_missing_data)//2))
    test_idx = list(range(len(non_missing_data)//2, len(non_missing_data)))
    mse_score = evaluate_regularization(
            m=5,
            non_missing_data=non_missing_data,
            train_idx=train_idx,
            test_idx=test_idx,
            impute_col='Age',
            group_by_cols=['Title', 'Pclass'],
            global_mean=train_data['Age'].mean()
            )
    assert isinstance(mse_score, float), "Function should return a float MSE value"

def test_impute_column(mock_data) -> None:
    train_data = mock_data.iloc[:5]
    test_data = mock_data.iloc[5:]
    imputed_dataset_train, imputed_dataset_test = impute_column(
            train_data=train_data,
            test_data=test_data,
            impute_col='Age',
            group_by_cols=['Title', 'Pclass']
        )
    assert not imputed_dataset_train['Age'].isnull().any(), "Imputation failed, NaN values found in Age column of train data"
    assert not imputed_dataset_test['Age'].isnull().any(), "Imputation failed, NaN values found in Age column of test data"
    assert 'Age_Imputed' in imputed_dataset_train.columns, "Indicator column not found in the returned train dataframe"
    assert 'Age_Imputed' in imputed_dataset_test.columns, "Indicator column not found in the returned test dataframe"

def test_grouped_mean_imputation() -> None:
    """Test that grouped means are used for imputation."""
    # Create a mock dataset where the means for certain groups are known
    data = {
        'Age': [25, 30, 35, 40, 45, 50, np.nan, np.nan, np.nan],
        'Title': ['Mr', 'Mr', 'Mr', 'Miss', 'Miss', 'Miss', 'Mrs', 'Mrs', 'Mrs'],
        'Pclass': [1, 1, 1, 3, 3, 3, 2, 2, 2]
    }
    mock_data = pd.DataFrame(data=data)

    # Define the known group means
    known_means = {
        ('Mr', 1): 30.0,  # (25 + 30 + 35) / 3
        ('Miss', 3): 45.0,  # (40 + 45 + 50) / 3
        ('Mrs', 2): np.nan  # No non-missing data for this group
    }

    train_data = mock_data.iloc[:6]
    test_data = mock_data.iloc[6:]

    global_mean = train_data['Age'].mean()

    _, imputed_data_test = impute_with_regularization(
        m=5,
        train_data=train_data,
        test_data=test_data,
        impute_col='Age',
        group_by_cols=['Title', 'Pclass'],
        global_mean=global_mean
    )

    # Check the imputed values for each group
    for group, known_mean in known_means.items():
        if not ((test_data['Title'] == group[0]) & (test_data['Pclass'] == group[1])).any():
            continue  # Skip to the next group if this group does not exist in the test data

        test_group_value = imputed_data_test[(test_data['Title'] == group[0]) & (test_data['Pclass'] == group[1])].values
        expected_value = known_mean if not np.isnan(known_mean) else global_mean
        assert test_group_value[0] == pytest.approx(expected_value, 0.01), f"Expected imputed value {expected_value} but got {test_group_value[0]}"
