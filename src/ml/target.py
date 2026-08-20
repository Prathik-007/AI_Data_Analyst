
def validate_target(df, target_column, problem_type, column_types):
    """
    Validate the target column and selected ML problem type.
    """
    if target_column not in df.columns:
        raise ValueError(f"{target_column} does not exists in the dataset")

    if problem_type not in ['classification','regression']:
        raise ValueError("Should be classification or regression")

    target_type = column_types[target_column]

    if target_type == 'all_null':
        raise ValueError(f"Target column '{target_column}' contains only missing values.")
    elif target_type == 'constant':
        raise ValueError(f"Target column '{target_column}' must contain more than one unique value.")
    elif target_type == 'identifier':
        raise ValueError(f"Target column '{target_column}' appears to be an identifier.")

    if problem_type == 'classification':
        if target_type not in ['binary','categorical']:
            raise ValueError(f"{target_column} is of type {target_type} which is not suitable for Classification")

    if problem_type == 'regression':
        if target_type != 'numerical':
            raise ValueError(f"{target_column} is of type {target_type} which is not suitable for Regression")

    return True


def split_features_target(df, target_column):
    """
    Split the dataset into features (X) and target (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X,y