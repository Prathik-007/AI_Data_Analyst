from sklearn.preprocessing import StandardScaler

def scale_numerical_columns(df, column_types):
    """
    Standardize numerical columns using standardScaler
    """
    df =df.copy()
    numerical_colums = [ col for col in df.columns if column_types[col] == "numerical"]
    if numerical_colums:
        scaler = StandardScaler()
        df[numerical_colums] = scaler.fit_transform(df[numerical_colums])

    return df