import plotly.express as px


def plot_numerical_distribution(df, column):
    """
    Create a histogram for a numerical column.
    """
    fig = px.histogram(df, x=column, nbins=5, title=f"Distribution of {column}")

    fig.update_layout( xaxis_title=column, yaxis_title="Count")

    return fig