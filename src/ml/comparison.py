from src.ml.train import train_model
from src.ml.evaluate import evaluate_model

from src.ml.train import train_model
from src.ml.evaluate import evaluate_model


def compare_models( models, X_train, X_test, y_train, y_test, problem_type):
    """
    Train and evaluate multiple machine learning models.
    """

    trained_models = {}
    metrics = {}

    for model_name, model in models.items():

        trained_model = train_model(
            model,
            X_train,
            y_train
        )

        model_metrics = evaluate_model(
            trained_model,
            X_test,
            y_test,
            problem_type
        )

        trained_models[model_name] = trained_model
        metrics[model_name] = model_metrics

    return {
        "models": trained_models,
        "metrics": metrics
    }