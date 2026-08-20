from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def get_classification_models():
    """
    Return the classification models supported by the application.
    """

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            random_state=42
        )
    }


def get_regression_models():
    """
    Return the regression models supported by the application.
    """

    return {
        "linear_regression": LinearRegression(),

        "decision_tree": DecisionTreeRegressor(
            random_state=42
        ),

        "random_forest": RandomForestRegressor(
            random_state=42
        )
    }


def get_models(problem_type):
    """
    Return models based on the selected ML problem type.
    """

    if problem_type == "classification":
        return get_classification_models()

    if problem_type == "regression":
        return get_regression_models()

    raise ValueError(
        "Problem type must be 'classification' or 'regression'."
    )

def get_model(problem_type, model_name):
    """
    Return a specific model based on problem type and model name.
    """

    models = get_models(problem_type)

    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}' for {problem_type}."
        )

    return models[model_name]