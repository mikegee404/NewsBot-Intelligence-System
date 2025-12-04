import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_label_distribution(df: pd.DataFrame, label_col: str = "label") -> None:
    plt.figure(figsize=(6, 4))
    df[label_col].value_counts().plot(kind="bar")
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.show()


def plot_sentiment_distribution(df: pd.DataFrame, col: str = "sent_vader") -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title("Sentiment (VADER) Distribution")
    plt.tight_layout()
    plt.show()
