

def get_basic_info(df):
    """Return rows, columns, memory usage, etc."""
    row_count, column_count = df.shape
    memory_usage = df.memory_usage(deep=True).sum()
    columns_name = df.columns.to_list()   # returns an Index object, convert to list
    columns_types = df.dtypes.to_dict()   # returns a Series object, convert to dict
    return {
        "rows_count": row_count,
        "columns_count": column_count,
        "columns_name": columns_name,
        "columns_types": columns_types,
        "memory_usage_bytes": memory_usage
    }

def get_column_info(df):
    """Return information about each column."""


def get_missing_values(df):
    """Return missing value counts and percentages."""

def get_duplicate_info(df):
    """Return duplicate row information."""

def get_numerical_statistics(df):
    """Return descriptive statistics for numerical columns."""

def get_categorical_summary(df):
    """Return summary of categorical columns."""

def get_outlier_info(df):
    """Placeholder for future outlier detection."""

def generate_profile(df):
    """Generate a complete profile by combining all the above."""