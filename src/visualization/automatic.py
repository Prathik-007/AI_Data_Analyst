from src.visualization.distributions import plot_numerical_distribution
from src.visualization.categorical import plot_categorical_distribution
from src.visualization.categorical import plot_binary_distribution


def generate_univariate_plots(df, column_types):
    """
    Generate appropriate univariate plots based on column types.
    """

    plots = {}

    for column in df.columns:

        column_type = column_types[column]

        if column_type == "numerical":
            plots[column] = plot_numerical_distribution(df, column)

        elif column_type == "categorical":
            plots[column] = plot_categorical_distribution(df, column)

        elif column_type == "binary":
            plots[column] = plot_binary_distribution(df, column)

    return plots