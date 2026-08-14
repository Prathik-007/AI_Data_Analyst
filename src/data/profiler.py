import pandas as pd
import re
NUMERICAL_HINTS = {"age","height","weight","salary","income","price","amount","score","distance","duration","temperature","value"}

CATEGORICAL_HINTS = {"category","type","status","level","grade","class","education"}

def make_json_serializable(value):
     """Recursively convert Pandas/NumPy values to native Python types."""
     if isinstance(value,dict):
         return {key: make_json_serializable(val) for key, val in value.items()}
     elif isinstance(value,list):
         return [make_json_serializable(item) for item in value]
     elif isinstance(value,tuple):
         return tuple(make_json_serializable(item) for item in value)
     else:
        return convert_to_python_types(value)

def _is_all_null(series):
    """Return True if the column contains no non-null values."""
    return series.notna().sum() == 0


def _is_constant(series):
    """Return True if the column has exactly one unique value."""
    return series.nunique() == 1


def _is_datetime(series):
    """Return True if the column contains datetime values."""
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_identifier(series, column_name):
    """Return True if the column appears to be an identifier."""

    identifier_pattern = (
        r"(^id$|^id_|_id$|"
        r"identifier|uuid|"
        r"^key$|^key_|_key$)"
    )

    name_suggests_identifier = bool(
        re.search(identifier_pattern, str(column_name).lower().strip())
    )

    non_null_count = series.notna().sum()
    unique_count = series.nunique()

    high_uniqueness = (
        non_null_count > 0
        and unique_count / non_null_count >= 0.95
    )

    return name_suggests_identifier and high_uniqueness


def _is_text(series):
    """Return True if a string column appears to contain free-form text."""

    if not pd.api.types.is_string_dtype(series):
        return False

    non_null_values = series.dropna().astype(str)

    if len(non_null_values) == 0:
        return False

    average_length = non_null_values.str.len().mean()

    return average_length > 20

def _get_numeric_semantic_type(series, column_name):
    """Classify a numeric column using cardinality and name hints."""

    unique_values = series.nunique()
    column_name = str(column_name).lower().strip()

    if unique_values <= 10:

        if column_name in NUMERICAL_HINTS:
            return "numerical"

        if column_name in CATEGORICAL_HINTS:
            return "categorical"

        return "categorical"

    return "numerical"

def detect_column_types(df):
    """
    Infer semantic types for DataFrame columns.

    Classification priority:
        1. all_null
        2. constant
        3. datetime
        4. identifier
        5. binary
        6. text
        7. categorical
        8. numerical
    """

    column_types = {}

    for col in df.columns:

        series = df[col]
        unique_values = series.nunique()

        # 1. All-null
        if _is_all_null(series):
            column_types[col] = "all_null"
            continue

        # 2. Constant
        if _is_constant(series):
            column_types[col] = "constant"
            continue

        # 3. Datetime
        if _is_datetime(series):
            column_types[col] = "datetime"
            continue

        # 4. Identifier
        if _is_identifier(series, col):
            column_types[col] = "identifier"
            continue

        # 5. Binary
        if unique_values == 2:
            column_types[col] = "binary"
            continue

        # 6. Boolean
        if pd.api.types.is_bool_dtype(series):
            column_types[col] = "binary"
            continue
        # 8. Categorical dtype
        if isinstance(series.dtype, pd.CategoricalDtype):
            column_types[col] = "categorical"
            continue
        if _is_text(series):
            column_types[col] = "text"
            continue

        # 9. Numerical
        if pd.api.types.is_numeric_dtype(series):
            column_types[col] = _get_numeric_semantic_type(series, col)

            continue

        # 10. Fallback
        column_types[col] = "categorical"

    return column_types

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
    column_info = {}
    non_null_count = df.notnull().sum()
    for col in df.columns:
        non_null_values = df[col].dropna()
        null_count = df[col].isnull().sum()
        column_info[col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": non_null_count[col],
            "null_count": null_count,
            "unique_count": df[col].nunique(),
            "null_percentage" : (null_count / len(df)) * 100,
            "sample_values": non_null_values.sample(min(5, non_null_count[col]), random_state=42).tolist()
        }
    return column_info


def get_missing_values(df):
    """Return missing value counts and percentages."""
    missing_info = {}
    missing_counts = df.isnull().sum()
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_info[col] = {
                "missing_count": missing_counts[col],
                "missing_percentage": (missing_counts[col] / len(df)) * 100
            }
    return missing_info

def get_duplicate_info(df):
    """Return duplicate row information."""
    duplicate_count =int(df.duplicated().sum())
    if len(df) == 0:
        duplicate_percentage = 0.0
    else:
        duplicate_percentage = float((duplicate_count / len(df)) * 100)

    return {    
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage
    }

def get_numerical_statistics(df):
    """Return descriptive statistics for numerical columns."""
    column_types = detect_column_types(df)
    numerical_stats = {}
    for col in df.columns:
        if column_types[col] == "numerical":
            numerical_stats[col] = df[col].describe().to_dict()
    return numerical_stats

def get_categorical_summary(df):
    """Return summary of categorical columns."""
    column_types = detect_column_types(df)
    categorical_summary = {}
    for col in df.columns:
        if column_types[col] == "categorical":
            categorical_summary[col] = {
                "unique_count": df[col].nunique(),
                "value_counts": df[col].value_counts().sort_index().to_dict(),
                "percentages": (df[col].value_counts(normalize=True).sort_index() * 100).to_dict()
            }
    return categorical_summary
            
def get_outlier_info(df):
    """Return IQR-based outlier information for numerical columns."""

    outlier_info = {}
    column_types = detect_column_types(df)

    for col in df.columns:

        if column_types[col] != "numerical":
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = int(
            (
                (df[col] < lower_bound)
                | (df[col] > upper_bound)
            ).sum()
        )

        if len(df) == 0:
            outlier_percentage = 0.0
        else:
            outlier_percentage = float(
                (outliers_count / len(df)) * 100
            )

        outlier_info[col] = {
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": outliers_count,
            "outlier_percentage": outlier_percentage
        }

    return outlier_info

def get_binary_summary(df):
    """Return summary of binary columns."""
    column_types = detect_column_types(df)
    binary_summary = {}
    for col in df.columns:
        if column_types[col] == "binary":
            binary_summary[col] = {
                "unique_count": df[col].nunique(),
                "value_counts": df[col].value_counts().sort_index().to_dict(),
                "percentages": (df[col].value_counts(normalize=True).sort_index() * 100).to_dict()
            }
    return binary_summary

def generate_profile(df):
    """Generate a complete profile by combining all the above."""
    profile = {
        "basic_info": get_basic_info(df),
        "column_info": get_column_info(df),
        "column_types": detect_column_types(df),
        "missing_values": get_missing_values(df),
        "duplicate_info": get_duplicate_info(df),
        "numerical_statistics": get_numerical_statistics(df),
        "categorical_summary": get_categorical_summary(df),
        "outlier_info": get_outlier_info(df),
        "binary_summary": get_binary_summary(df)
    }
    return make_json_serializable(profile)

def convert_to_python_types(value):
    """Convert numpy/pandas scalar values to native python types."""

    if hasattr(value, 'item'):  #checks if the value has an item() method, which is common for numpy/pandas scalar types
        return value.item()
    else:
        return value

