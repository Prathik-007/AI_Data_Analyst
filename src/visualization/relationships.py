import plotly.express as px

def plot_numerical_relationship(df, x_column, y_column):
    """
    Create a scatter plot between two numerical columns.
    """
    fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    return fig

def plot_numerical_categorical_relationship(df, numerical_column, categorical_column):
    """
    Create a box plot showing the distribution of a numerical variable across categorical
    """
    fig = px.box( df, x=numerical_column, y=categorical_column, title=f"{numerical_column} by {categorical_column}")

    return fig