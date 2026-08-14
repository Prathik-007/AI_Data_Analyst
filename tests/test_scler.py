import pandas as pd
import numpy as np
from src.processing.scaler import scale_numerical_columns


def test_scale_numerical_columns():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [10000, 20000, 30000],
        "city": ["Bangalore", "Mysore", "Mangalore"]
    })

    column_types = {
        "age": "numerical",
        "salary": "numerical",
        "city": "categorical"
    }

    result = scale_numerical_columns(df, column_types)

    assert np.isclose(result["age"].mean(), 0)
    assert np.isclose(result["age"].std(ddof=0), 1)

    assert np.isclose(result["salary"].mean(), 0)
    assert np.isclose(result["salary"].std(ddof=0), 1)