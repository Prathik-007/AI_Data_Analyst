import pytest
import pandas as pd
import numpy as np
from src.data.loader import load_dataset
from src.data.profiler import generate_profile, get_basic_info, get_column_info, get_missing_values, get_duplicate_info, get_numerical_statistics, detect_column_types, get_categorical_summary, get_outlier_info, get_binary_summary, make_json_serializable


def test_get_basic_info():

    df = load_dataset("data/raw/heart.csv")

    result = get_basic_info(df)

    # Check row and column counts
    assert result["rows_count"] == 1025
    assert result["columns_count"] == 14

    # Check column information
    assert len(result["columns_name"]) == 14
    assert len(result["columns_types"]) == 14

    # Check that memory usage is positive
    assert result["memory_usage_bytes"] > 0

def test_detect_column_types():

    df = load_dataset("data/raw/heart.csv")

    result = detect_column_types(df)

    assert result["age"] == "numerical"
    assert result["sex"] == "binary"
    assert result["cp"] == "categorical"
    assert result["target"] == "binary"

def test_get_missing_values():

    test_df = pd.DataFrame({
        "age": [20, 25, None, 30],
        "name": ["A", None, "C", "D"]
    })

    result = get_missing_values(test_df)

    assert result["age"]["missing_count"] == 1
    assert result["name"]["missing_count"] == 1

    assert result["age"]["missing_percentage"] == 25.0
    assert result["name"]["missing_percentage"] == 25.0

def test_get_duplicate_info():

    test_df = pd.DataFrame({
        "age": [20, 25, 20, 30, 25],
        "score": [80, 90, 80, 70, 90]
    })

    result = get_duplicate_info(test_df)

    assert result["duplicate_count"] == 2
    assert result["duplicate_percentage"] == 40.0

def test_get_numerical_statistics():

    test_df = pd.DataFrame({
        "age": list(range(1, 13)),
        "score": list(range(101, 113)),
        "category": ["A", "B"] * 6
    })

    result = get_numerical_statistics(test_df)

    assert "age" in result
    assert "score" in result
    assert "category" not in result

    assert result["age"]["count"] == 12.0
    assert result["age"]["mean"] == 6.5
    assert result["age"]["min"] == 1.0
    assert result["age"]["max"] == 12.0

def test_get_categorical_summary():

    test_df = pd.DataFrame({
        "category": ["A", "B", "A", "C", "A", "B"]
    })

    assert detect_column_types(test_df)["category"] == "categorical"

    result = get_categorical_summary(test_df)

    assert "category" in result

    assert result["category"]["unique_count"] == 3

    assert result["category"]["value_counts"]["A"] == 3
    assert result["category"]["value_counts"]["B"] == 2
    assert result["category"]["value_counts"]["C"] == 1

    assert result["category"]["percentages"]["A"] == pytest.approx(50.0)
    assert result["category"]["percentages"]["B"] == pytest.approx(33.3333333333)
    assert result["category"]["percentages"]["C"] == pytest.approx(16.6666666667)

def test_get_binary_summary():

    test_df = pd.DataFrame({
        "is_active": [0, 1, 1, 0, 1, 1, 0, 1]
    })

    result = get_binary_summary(test_df)

    assert "is_active" in result

    assert result["is_active"]["unique_count"] == 2

    assert result["is_active"]["value_counts"][0] == 3
    assert result["is_active"]["value_counts"][1] == 5

    assert result["is_active"]["percentages"][0] == pytest.approx(37.5)
    assert result["is_active"]["percentages"][1] == pytest.approx(62.5)

def test_get_outlier_info():

    test_df = pd.DataFrame({
        "value": [
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 100
        ]
    })

    result = get_outlier_info(test_df)

    assert "value" in result

    assert result["value"]["Q1"] == pytest.approx(12.25)
    assert result["value"]["Q3"] == pytest.approx(16.75)
    assert result["value"]["IQR"] == pytest.approx(4.5)

    assert result["value"]["lower_bound"] == pytest.approx(5.5)
    assert result["value"]["upper_bound"] == pytest.approx(23.5)

    assert result["value"]["outlier_count"] == 1
    assert result["value"]["outlier_percentage"] == pytest.approx(10.0)

def test_make_json_serializable():

    test_data = {
        "integer": np.int64(10),
        "float": np.float64(10.5),
        "nested": {
            "value": np.int64(20)
        },
        "list": [
            np.int64(30),
            np.float64(40.5)
        ]
    }

    result = make_json_serializable(test_data)

    assert type(result["integer"]) is int
    assert type(result["float"]) is float
    assert type(result["nested"]["value"]) is int
    assert type(result["list"][0]) is int
    assert type(result["list"][1]) is float

def test_generate_profile():

    df = load_dataset("data/raw/heart.csv")

    result = generate_profile(df)

    expected_sections = {
        "basic_info",
        "column_info",
        "column_types",
        "missing_values",
        "duplicate_info",
        "numerical_statistics",
        "categorical_summary",
        "outlier_info",
        "binary_summary"
    }

    assert set(result.keys()) == expected_sections

    assert type(result["basic_info"]["rows_count"]) is int
    assert type(
        result["numerical_statistics"]["age"]["mean"]
    ) is float

def test_empty_dataframe():

    empty_df = pd.DataFrame()

    result = generate_profile(empty_df)

    assert isinstance(result, dict)

def test_all_null_column():

    test_df = pd.DataFrame({
        "age": [None, None, None, None],
        "name": ["A", "B", None, "D"]
    })

    result = generate_profile(test_df)

    assert result["column_info"]["age"]["null_count"] == 4
    assert result["column_info"]["age"]["non_null_count"] == 0

    assert result["column_types"]["age"] == "all_null"

def test_constant_column():

    test_df = pd.DataFrame({
        "age": [25, 25, 25, 25, 25]
    })

    result = generate_profile(test_df)

    assert result["column_info"]["age"]["unique_count"] == 1
    assert result["column_types"]["age"] == "constant"

def test_single_row_dataframe():

    test_df = pd.DataFrame({
        "age": [25],
        "name": ["Alice"]
    })

    result = generate_profile(test_df)

    assert result["basic_info"]["rows_count"] == 1
    assert result["basic_info"]["columns_count"] == 2

def test_single_column_dataframe():

    test_df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40]
    })

    result = generate_profile(test_df)

    assert result["basic_info"]["rows_count"] == 5
    assert result["basic_info"]["columns_count"] == 1
    assert result["basic_info"]["columns_name"] == ["age"]

def test_mixed_data_types():

    test_df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "city": ["Mangalore", "Bangalore", "Mangalore", "Mysore", "Bangalore"],
        "is_active": [0, 1, 1, 0, 1],
        "verified": [True, False, True, False, True],
        "empty": [None, None, None, None, None],
        "constant": ["X", "X", "X", "X", "X"]
    })

    result = generate_profile(test_df)

    assert result["column_types"]["age"] == "numerical"
    assert result["column_types"]["city"] == "categorical"
    assert result["column_types"]["is_active"] == "binary"
    assert result["column_types"]["verified"] == "binary"
    assert result["column_types"]["empty"] == "all_null"
    assert result["column_types"]["constant"] == "constant"

def test_datetime_column():

    test_df = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04"
        ])
    })

    result = generate_profile(test_df)

    assert result["column_types"]["created_at"] == "datetime"

    assert "created_at" not in result["numerical_statistics"]
    assert "created_at" not in result["categorical_summary"]
    assert "created_at" not in result["binary_summary"]
    assert "created_at" not in result["outlier_info"]

def test_identifier_column():

    test_df = pd.DataFrame({
        "user_id": [100001, 100002, 100003, 100004, 100005],
        "age": [25, 30, 35, 40, 45]
    })

    result = generate_profile(test_df)

    assert result["column_types"]["user_id"] == "identifier"

def test_high_cardinality_categorical():

    test_df = pd.DataFrame({
        "category": [f"category_{i}" for i in range(50)]
    })

    result = generate_profile(test_df)

    assert result["column_types"]["category"] == "categorical"

def test_numeric_identifier():

    test_df = pd.DataFrame({
        "customer_id": range(1000, 1050),
        "age": list(range(20, 70))
    })

    result = generate_profile(test_df)

    assert result["column_types"]["customer_id"] == "identifier"
    assert result["column_types"]["age"] == "numerical"

def test_numeric_categorical_column():

    test_df = pd.DataFrame({
        "education": [1, 2, 3, 4, 2, 3, 1, 4]
    })

    result = generate_profile(test_df)

    assert result["column_types"]["education"] == "categorical"

    assert "education" in result["categorical_summary"]

    assert result["categorical_summary"]["education"]["unique_count"] == 4

def test_text_column():

    test_df = pd.DataFrame({
        "description": [
            "The product arrived late and the package was damaged",
            "Excellent product with very good overall quality",
            "Customer requested a replacement because it was broken",
            "The delivery experience was disappointing and slow",
            "Very happy with the product and would recommend it"
        ]
    })

    result = generate_profile(test_df)

    print("\nRESULT:", result["column_types"])

    assert result["column_types"]["description"] == "text"

def test_numeric_semantic_types():

    test_df = pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "education": [1, 2, 3, 4, 2]
    })

    result = generate_profile(test_df)

    assert result["column_types"]["age"] == "numerical"
    assert result["column_types"]["education"] == "categorical"

def test_numeric_name_hints():

    test_df = pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "status": [1, 2, 1, 2, 1]
    })

    result = generate_profile(test_df)

    assert result["column_types"]["age"] == "numerical"
    assert result["column_types"]["status"] == "binary"
