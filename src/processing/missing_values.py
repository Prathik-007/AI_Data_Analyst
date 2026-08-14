

def fill_missing_values(df, column_types):
    """Fill missing values based on column types."""
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for col in df.columns:
        if column_types[col] == "numerical":
            df[col] = df[col].fillna(df[col].median())
        elif column_types[col] == "categorical":
            df[col] = df[col].fillna(df[col].mode()[0])
    return df