import pandas as pd
import plotly.graph_objects as go

from src.visualization.distributions import plot_numerical_distribution
from src.visualization.categorical import ( plot_categorical_distribution, plot_binary_distribution)
from src.visualization.relationships import ( plot_numerical_relationship, plot_numerical_categorical_relationship)
from src.visualization.correlation import plot_correlation_heatmap
from src.visualization.automatic import generate_univariate_plots

def test_plot_numerical_distribution_returns_figure():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40]
    })

    result = plot_numerical_distribution(df, "age")

    assert isinstance(result, go.Figure)


def test_plot_numerical_distribution_has_correct_title():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40]
    })

    result = plot_numerical_distribution(df, "age")

    assert result.layout.title.text == "Distribution of age"


def test_plot_numerical_distribution_has_correct_xaxis():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40]
    })

    result = plot_numerical_distribution(df, "age")

    assert result.layout.xaxis.title.text == "age"


def test_plot_numerical_distribution_has_correct_yaxis():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40]
    })

    result = plot_numerical_distribution(df, "age")

    assert result.layout.yaxis.title.text == "Count"


def test_plot_numerical_distribution_does_not_modify_original():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "salary": [20000, 30000, 40000, 50000, 60000]
    })

    original = df.copy()

    plot_numerical_distribution(df, "age")

    pd.testing.assert_frame_equal(df, original)

def test_plot_categorical_distribution_returns_figure():
    df = pd.DataFrame({
        "city": ["Bangalore", "Mysore", "Bangalore", "Mangalore"]
    })

    result = plot_categorical_distribution(df, "city")

    assert isinstance(result, go.Figure)


def test_plot_categorical_distribution_has_correct_title():
    df = pd.DataFrame({
        "city": ["Bangalore", "Mysore", "Bangalore"]
    })

    result = plot_categorical_distribution(df, "city")

    assert result.layout.title.text == "Distribution of city"


def test_plot_categorical_distribution_has_correct_xaxis():
    df = pd.DataFrame({
        "city": ["Bangalore", "Mysore", "Bangalore"]
    })

    result = plot_categorical_distribution(df, "city")

    assert result.layout.xaxis.title.text == "city"


def test_plot_categorical_distribution_has_correct_yaxis():
    df = pd.DataFrame({
        "city": ["Bangalore", "Mysore", "Bangalore"]
    })

    result = plot_categorical_distribution(df, "city")

    assert result.layout.yaxis.title.text == "Count"


def test_plot_categorical_distribution_counts_categories_correctly():
    df = pd.DataFrame({
        "city": [
            "Bangalore",
            "Mysore",
            "Bangalore",
            "Mangalore",
            "Bangalore"
        ]
    })

    result = plot_categorical_distribution(df, "city")

    counts = {
        result.data[0].x[i]: result.data[0].y[i]
        for i in range(len(result.data[0].x))
    }

    assert counts["Bangalore"] == 3
    assert counts["Mysore"] == 1
    assert counts["Mangalore"] == 1


def test_plot_categorical_distribution_does_not_modify_original():
    df = pd.DataFrame({
        "city": ["Bangalore", "Mysore", "Bangalore"],
        "age": [25, 30, 35]
    })

    original = df.copy()

    plot_categorical_distribution(df, "city")

    pd.testing.assert_frame_equal(df, original)

def test_plot_binary_distribution_returns_figure():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male", "Female"]
    })

    result = plot_binary_distribution(df, "sex")

    assert isinstance(result, go.Figure)


def test_plot_binary_distribution_has_correct_title():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"]
    })

    result = plot_binary_distribution(df, "sex")

    assert result.layout.title.text == "distribution of sex"


def test_plot_binary_distribution_counts_values_correctly():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male", "Male", "Female"]
    })

    result = plot_binary_distribution(df, "sex")

    counts = {
        result.data[0].x[i]: result.data[0].y[i]
        for i in range(len(result.data[0].x))
    }

    assert counts["Male"] == 3
    assert counts["Female"] == 2


def test_plot_binary_distribution_does_not_modify_original():
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"],
        "age": [25, 30, 35]
    })

    original = df.copy()

    plot_binary_distribution(df, "sex")

    pd.testing.assert_frame_equal(df, original)


def test_plot_numerical_relationship_returns_figure():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35],
        "salary": [20000, 30000, 40000, 50000]
    })

    result = plot_numerical_relationship(df, "age", "salary")

    assert isinstance(result, go.Figure)


def test_plot_numerical_relationship_has_correct_title():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [20000, 30000, 40000]
    })

    result = plot_numerical_relationship(df, "age", "salary")

    assert result.layout.title.text == "age vs salary"


def test_plot_numerical_relationship_has_correct_axes():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [20000, 30000, 40000]
    })

    result = plot_numerical_relationship(df, "age", "salary")

    assert result.layout.xaxis.title.text == "age"
    assert result.layout.yaxis.title.text == "salary"


def test_plot_numerical_relationship_does_not_modify_original():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [20000, 30000, 40000]
    })

    original = df.copy()

    plot_numerical_relationship(df, "age", "salary")

    pd.testing.assert_frame_equal(df, original)

def test_plot_numerical_categorical_relationship_returns_figure():
    df = pd.DataFrame({
        "salary": [20000, 30000, 40000, 50000],
        "city": ["Bangalore", "Mysore", "Bangalore", "Mysore"]
    })

    result = plot_numerical_categorical_relationship(
        df,
        "salary",
        "city"
    )

    assert isinstance(result, go.Figure)


def test_plot_numerical_categorical_relationship_has_correct_title():
    df = pd.DataFrame({
        "salary": [20000, 30000, 40000],
        "city": ["Bangalore", "Mysore", "Bangalore"]
    })

    result = plot_numerical_categorical_relationship(
        df,
        "salary",
        "city"
    )

    assert result.layout.title.text == "salary by city"


def test_plot_numerical_categorical_relationship_does_not_modify_original():
    df = pd.DataFrame({
        "salary": [20000, 30000, 40000],
        "city": ["Bangalore", "Mysore", "Bangalore"]
    })

    original = df.copy()

    plot_numerical_categorical_relationship(
        df,
        "salary",
        "city"
    )

    pd.testing.assert_frame_equal(df, original)

def test_plot_correlation_heatmap_returns_figure():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35],
        "salary": [20000, 30000, 40000, 50000],
        "score": [50, 60, 70, 80]
    })

    result = plot_correlation_heatmap(
        df,
        ["age", "salary", "score"]
    )

    assert isinstance(result, go.Figure)


def test_plot_correlation_heatmap_has_correct_title():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [20000, 30000, 40000]
    })

    result = plot_correlation_heatmap(
        df,
        ["age", "salary"]
    )

    assert result.layout.title.text == "Correlation Heatmap"


def test_plot_correlation_heatmap_does_not_modify_original():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [20000, 30000, 40000],
        "city": ["A", "B", "C"]
    })

    original = df.copy()

    plot_correlation_heatmap(
        df,
        ["age", "salary"]
    )

    pd.testing.assert_frame_equal(df, original)

def test_generate_univariate_plots_returns_dictionary():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Bangalore", "Mysore", "Bangalore"],
        "sex": ["Male", "Female", "Male"]
    })

    column_types = {
        "age": "numerical",
        "city": "categorical",
        "sex": "binary"
    }

    result = generate_univariate_plots(df, column_types)

    assert isinstance(result, dict)


def test_generate_univariate_plots_creates_plot_for_each_supported_column():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Bangalore", "Mysore", "Bangalore"],
        "sex": ["Male", "Female", "Male"]
    })

    column_types = {
        "age": "numerical",
        "city": "categorical",
        "sex": "binary"
    }

    result = generate_univariate_plots(df, column_types)

    assert "age" in result
    assert "city" in result
    assert "sex" in result

    assert len(result) == 3


def test_generate_univariate_plots_ignores_unsupported_types():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "name": ["A", "B", "C"],
        "id": [1, 2, 3]
    })

    column_types = {
        "age": "numerical",
        "name": "text",
        "id": "identifier"
    }

    result = generate_univariate_plots(df, column_types)

    assert "age" in result
    assert "name" not in result
    assert "id" not in result


def test_generate_univariate_plots_does_not_modify_original():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "city": ["Bangalore", "Mysore", "Bangalore"]
    })

    column_types = {
        "age": "numerical",
        "city": "categorical"
    }

    original = df.copy()

    generate_univariate_plots(df, column_types)

    pd.testing.assert_frame_equal(df, original)