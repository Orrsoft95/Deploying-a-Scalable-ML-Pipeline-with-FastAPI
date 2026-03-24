import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # DONE: implement the function
    param_grid = {
        'n_estimators': [100, 200, 250, 300],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10]
    }

    #Instantiate a base Random Forest model
    base_model = RandomForestClassifier(random_state=42)
    
    #Set up GridSearch
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        error_score='f1', #optimize for f1 score
        n_jobs=-1,
        verbose=1,
    )

    #fit on training data!
    grid_search.fit(X_train, y_train) 

    #Train a new model using the best parameters that we found
    model = RandomForestClassifier(
        **grid_search.best_params_,
        random_state=42
    )

    return model.fit(X_train, y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # DONE: implement the function
    preds = model.predict(X)
    #probabilities = model.predict_proba(X)
    return preds

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # DONE: implement the function
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    # DONE: implement the function
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : ???
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # DONE: implement the function
    X_slice, y_slice, _, _ = process_data(
        # for input data, use data in column given as "column_name", with the slice_value 
        # use training = False
        X= data[data[column_name] == slice_value],
        categorical_features= categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb

    )
    preds = inference(model=model, X=X_slice) # your code here to get prediction on X_slice using the inference function
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta