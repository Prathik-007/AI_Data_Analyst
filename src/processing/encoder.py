from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_binary_columns(df, column_types):
    """ encodes binary categorical columns to 0 and 1. """
    df =df.copy()
    label = LabelEncoder()
    for col in df.columns:
        if column_types[col] == "binary":
            df[col] = label.fit_transform(df[col].astype(str))
    return df