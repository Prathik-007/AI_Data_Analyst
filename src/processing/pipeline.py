from src.processing.cleaner import remove_duplicate_rows, remove_constant_columns, remove_all_null_columns
from src.processing.encoder import encode_binary_columns
from src.processing.missing_values import fill_missing_values
from src.processing.scaler import scale_numerical_columns

def clean_dataset(df):
    """
    Cleans the dataset by removing duplicate rows, constant columns, and columns with all null values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = remove_all_null_columns(df)
    df = remove_duplicate_rows(df)
    df = remove_constant_columns(df)
    
    return df

def preprocess_dataset(df, column_types):
    """
     Run the complete preprocessing pipeline on the dataset.
    """
    df = clean_dataset(df)
    df = fill_missing_values(df, column_types)
    df = encode_binary_columns(df, column_types)
    df = scale_numerical_columns(df,column_types)

    return {
        "data" : df,
        "column_types" : column_types
    }