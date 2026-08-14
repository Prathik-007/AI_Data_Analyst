import numpy as np
from src.data.loader import load_dataset
from src.data.profiler import generate_profile, get_binary_summary, get_duplicate_info, get_missing_values, get_column_info, get_basic_info, get_numerical_statistics, detect_column_types, get_categorical_summary, get_outlier_info, convert_to_python_types, make_json_serializable

df = load_dataset("data/raw/heart.csv")
"""
print("Basic Info:")
basic_info = get_basic_info(df)
print(basic_info)

print("\nMissing Values Info:")
missing_info = get_missing_values(df)
print(missing_info)

print("\nColumn Info:")
column_info = get_column_info(df)
print(column_info)

print("\nDuplicate Info:")
duplicate_info = get_duplicate_info(df)
print(duplicate_info)

print("\nNumerical Statistics:")
numerical_stats = get_numerical_statistics(df)
print(numerical_stats)

print("\n detected column types:")
column_types = detect_column_types(df)
print(column_types)

print("\n categorical summary:")
categorical_summary = get_categorical_summary(df)
print(categorical_summary)

print(" outlier detection: ")
outlier_info = get_outlier_info(df)
print(outlier_info)

print("Binary column detection:")
binary_summary = get_binary_summary(df)
print(binary_summary)


profile = generate_profile(df)

print(profile.keys())


print(convert_to_python_types(np.int64(10)))
print(type(convert_to_python_types(np.int64(10))))

print(convert_to_python_types(np.float64(10.5)))
print(type(convert_to_python_types(np.float64(10.5))))

test_data = {
    "rows": np.int64(1025),
    "statistics": {
        "mean": np.float64(54.43)
    },
    "values": [
        np.int64(10),
        np.float64(20.5)
    ]
}

converted = make_json_serializable(test_data)

print(converted)
print(type(converted["rows"]))
print(type(converted["statistics"]["mean"]))
print(type(converted["values"][0]))

"""
profile = generate_profile(df)

print(type(profile["basic_info"]["rows_count"]))
print(type(profile["basic_info"]["memory_usage_bytes"]))
print(type(profile["numerical_statistics"]["age"]["mean"]))
