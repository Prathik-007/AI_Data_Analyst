# categorical.py

import plotly.express as px


def plot_categorical_distribution(df, column):
    """
    Create a bar chart showing the frequency of each category.
    """
    counts = df[column].value_counts().reset_index()   #value_count creates a panda series and reset index reorders the index values

    counts.columns = [column, "Count"]  # renaming the columns 

    fig = px.bar(counts, x=column, y="Count", title=f"Distribution of {column}")

    return fig


def  plot_binary_distribution(df, column):
    """
    Create a bar chart showing the frequency of each binary value.
    """
    counts = df[column].value_counts().reset_index()

    counts.columns = [column,"Counts"]

    fig = px.bar( counts, x=column,y="Counts",title=f"distribution of {column}")

    return fig