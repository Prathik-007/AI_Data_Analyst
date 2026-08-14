import pandas as pd

from src.preprocessing.cleaner import remove_duplicate_rows


def test_remove_duplicate_rows():
    df = pd.DataFrame({
        "age": [25, 30, 35, 30],
        "city": ["Bangalore", "Mysore", "Bangalore", "Mysore"],
        "salary": [50000, 60000, 70000, 60000]
    })

    result = remove_duplicate_rows(df)

    assert len(result) == 3
    assert result.duplicated().sum() == 0