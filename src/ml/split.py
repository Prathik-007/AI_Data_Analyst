from sklearn.model_selection import train_test_split

def split_train_test(X, y, problem_type):
    """
    Split features and target into training and testing sets.
    """
    if problem_type == 'classification':
        stratify = y
    else:
        stratify = None
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=stratify)

    return X_train,X_test,y_train,y_test