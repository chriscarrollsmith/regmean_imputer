import pytest
import pandas as pd
import numpy as np
from custom_imputers.RegularizedMeanImputer import impute_with_regularization, impute_column, evaluate_regularization, RegularizedMeanImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

def test_fit_transform(mock_data) -> None:
    imputer = RegularizedMeanImputer(impute_col='Age', group_by_cols=['Title', 'Pclass'])
    transformed_data = imputer.fit_transform(X=mock_data)
    assert 'Age' in transformed_data.columns, "Age column not found in the returned dataframe"
    assert not transformed_data['Age'].isnull().any(), "Imputation failed, NaN values found in Age column"

def test_imputer_within_column_transformer(mock_data) -> None:
    imputer = RegularizedMeanImputer(impute_col='Age', group_by_cols=['Title', 'Pclass'])
    
    # Wrap the imputer in a ColumnTransformer
    transformers = [("impute_age", imputer, ['Age', 'Title', 'Pclass'])]
    column_transform = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    # Now, use this in a pipeline
    pipeline = Pipeline(steps=[('preprocess', column_transform)])
    
    transformed_data_array = pipeline.fit_transform(mock_data)
    
    # Convert the NumPy array back to a DataFrame
    transformed_data = pd.DataFrame(transformed_data_array, columns=mock_data.columns)
    
    assert not transformed_data.isnull().any().any(), "Imputation failed, NaN values found"

    # Making sure the transformed data has the expected shape
    assert transformed_data.shape[0] == mock_data.shape[0], "Row mismatch after transformation"
    assert transformed_data.shape[1] == mock_data.shape[1], "Column mismatch after transformation"

def test_best_m_attribute(mock_data) -> None:
    imputer = RegularizedMeanImputer(impute_col='Age', group_by_cols=['Title', 'Pclass'])
    imputer.fit(mock_data)
    assert hasattr(imputer, "best_m_"), "Attribute best_m_ not found after fit method"
    assert imputer.best_m_ in imputer.m_values, "best_m_ value not in expected range"

def test_get_feature_names_out_with_column_transformer(mock_data) -> None:
    # Test again without the indicator
    imputer_no_indicator = RegularizedMeanImputer(impute_col='Age', group_by_cols=['Title', 'Pclass'], add_indicator=False)
    transformers_no_indicator = [("impute_age", imputer_no_indicator, ['Age'])]
    column_transformer_no_indicator = ColumnTransformer(transformers=transformers_no_indicator, remainder='passthrough')
    column_transformer_no_indicator.fit(mock_data)
    
    feature_names_out_no_indicator = column_transformer_no_indicator.get_feature_names_out()
    expected_feature_names_without_indicator = ['impute_age__Age']
    
    assert set(feature_names_out_no_indicator) == set(expected_feature_names_without_indicator), "Output feature names (without indicator) do not match expected feature names"

    imputer = RegularizedMeanImputer(impute_col='Age', group_by_cols=['Title', 'Pclass'], add_indicator=True)
    
    # Wrap the imputer in a ColumnTransformer
    transformers = [("impute_age", imputer, ['Age'])]
    column_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough')
    column_transformer.fit(mock_data)
    
    # Get the output feature names from the ColumnTransformer
    feature_names_out = column_transformer.get_feature_names_out()
    
    # Expected feature names based on the mock_data columns, the add_indicator flag, and the prefix added by the ColumnTransformer
    expected_feature_names = ['impute_age__Age', 'impute_age__missingindicator_Age']
    
    assert set(feature_names_out) == set(expected_feature_names), "Output feature names do not match expected feature names"