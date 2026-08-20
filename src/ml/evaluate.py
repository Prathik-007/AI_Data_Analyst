from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score)

def evaluate_classification(model, X_test, y_test):
    """
    Evaluate a classification model on test data.
    """

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

def evaluate_regression(model, X_test, y_test):
    """
    Evaluate a regression model on test data.
    """

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": r2_score(y_test, predictions)
    }

def evaluate_model(model, X_test, y_test, problem_type):
    """
    Evaluate a trained model based on the selected problem type.
    """

    if problem_type == "classification":
        return evaluate_classification(
            model,
            X_test,
            y_test
        )

    if problem_type == "regression":
        return evaluate_regression(
            model,
            X_test,
            y_test
        )

    raise ValueError(
        "Problem type must be 'classification' or 'regression'."
    )