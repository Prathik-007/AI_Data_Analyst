import plotly.express as px

def plot_correlation_heatmap(df, numerical_columns):
    """
    Create a correlation heatmap for numerical columns.
    """
    correlation_matrix = df[numerical_columns].corr()

    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
    return fig  

