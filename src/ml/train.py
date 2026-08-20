def train_model(model, X_train, y_train):
    """
    Train a machine learning model using training data.
    """
    model.fit(X_train, y_train)

    return model

