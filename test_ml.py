import pytest
# TODO: add necessary import
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os

PRECISION_THRESHOLD = 0.75
RECALL_THRESHOLD = 0.55
F1_THRESHOLD = 0.60

project_path = os.getcwd() #store current working directory
y_path = os.path.join(project_path, "data", "y_test.csv")
preds_path = os.path.join(project_path, "data", "preds.csv")

@pytest.fixture
def y():
    return pd.read_csv(y_path).squeeze()

@pytest.fixture
def preds():
    return pd.read_csv(preds_path).squeeze()


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


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
