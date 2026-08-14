""" 
This module contains functions for cleaning and preprocessing text data. Includes,
    remove duplicate rows
    remove all-null columns
    remove constant columns
    handle outliers

    Theoriginal dataset is not modified, a new dataframe is returned.becz we need to keep the original dataset for reference and comparison. 
    The functions are designed to be used in a data preprocessing pipeline, allowing for easy integration with other data processing steps.
"""

from streamlit import columns


def remove_all_null_columns(df):
    """Remove all-null columns from the dataframe."""
    return df.dropna(axis=1, how='all') # axis=1 means drop column and how='all' means drop if all values are null

def remove_duplicate_rows(df):
    """Remove duplicate rows from the dataframe."""
    return df.drop_duplicates()

def remove_constant_columns(df):
    """Remove constant columns from the dataframe."""
    return df.loc[:, df.nunique() > 1] # loc[rows,columns] so row = : means all rows and columns = df.nunique() > 1 means keep only 
                                    # columns with more than 1 unique value so based on the boolean masl created inside the df.loc[] the values are reatined or removed