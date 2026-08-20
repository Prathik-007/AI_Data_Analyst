from src.ml.target import validate_target, split_features_target
from src.ml.split import split_train_test
from src.ml.preprocessor import preprocess_train_test
from src.ml.model import get_models
from src.ml.comparison import compare_models

def run_ml_pipeline(df,target_column,problem_type,column_types):
    """
    Run the complete machine learning workflow.
    """

    validate_target(df,target_column,problem_type,column_types)

    X, y = split_features_target(df, target_column)

    X_train, X_test, y_train, y_test = split_train_test(X,y,problem_type)

    (X_train_processed,X_test_processed,preprocessor,feature_names,preprocessing_metadata) = preprocess_train_test(X_train,X_test,column_types)

    models = get_models(problem_type)

    comparison = compare_models( models, X_train_processed, X_test_processed, y_train, y_test, problem_type)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "preprocessing_metadata": preprocessing_metadata,
        "models": comparison["models"],
        "metrics": comparison["metrics"]
    }