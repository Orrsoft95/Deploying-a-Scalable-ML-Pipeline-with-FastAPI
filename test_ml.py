import pytest
# TODO: add necessary import
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import os
from ml.model import load_model

PRECISION_THRESHOLD = 0.75
RECALL_THRESHOLD = 0.55
F1_THRESHOLD = 0.60

project_path = os.getcwd()
data_path = os.path.join(project_path, "data") #store current working directory
y_path = os.path.join(data_path, "y_test.csv")
preds_path = os.path.join(data_path, "preds.csv")
model_path = os.path.join(project_path, "model", "model.pkl")
census_path = os.path.join(data_path, "census.csv")

@pytest.fixture
def y():
    return pd.read_csv(y_path).squeeze()

@pytest.fixture
def preds():
    return pd.read_csv(preds_path).squeeze()

@pytest.fixture
def model():
    return load_model(model_path)

@pytest.fixture
def census():
    return pd.read_csv(census_path)


# DONE: implement the first test. Change the function name and input as needed
def test_model_classification_metrics(y, preds):
    """
    # Test to confirm that the model's classification metrics are AT LEAST 0.75 precision, 0.55 recall, 0.60 F1-score when run against the test dataset.
    Inputs
    ------
    y: np.array
        Known labels, binarized
    preds: np.array
        Predicted labels, binarized
    """
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    assert precision >= PRECISION_THRESHOLD, (
        f"Calculated precision of {precision} is below required threshold of {PRECISION_THRESHOLD}"
    )

    assert recall >= RECALL_THRESHOLD, (
        f"Calculated recall of {recall} is below required threshold of {RECALL_THRESHOLD}"
    )

    assert f1 >= F1_THRESHOLD, (
        f"Calculated F1-score of {f1} is below required threshold of {F1_THRESHOLD}"
    )


# DONE: implement the second test. Change the function name and input as needed
def test_expected_algorithm(model):
    """
    # Test to confirm that the model uses the expected algorithm: random forest classification.
    Inputs
    ------
    model: the ML model saved in model folder
    """

    assert isinstance(model, RandomForestClassifier), (
        f"Expected Random Forest Classifier model, but got {type(model).__name__} instead."
    )
    


# DONE: implement the third test. Change the function name and input as needed
def test_age_range(census):
    """
    # Test to confirm that the census data contains no ages lower than 14 (min. legal working age in USA) or greater than 122 (oldest recorded person in history).
    Inputs
    ------
    census: pd.DataFrame
        census data loaded into a dataframe.
    """
    min_age = 14
    max_age = 122

    ages_below_min = census[census["age"] < min_age]
    ages_above_max = census[census["age"] > max_age]

    assert len(ages_below_min) == 0, (
        f"Found {len(ages_below_min)} records with age below minimum working age."
        f"Indices: {ages_below_min.index.toList()}"
    )

    assert len(ages_above_max) == 0, (
        f"Found {len(ages_above_max)} records with age above maximum valid age of {max_age}."
        f"Indices: {ages_above_max.index.toList()}"
    )