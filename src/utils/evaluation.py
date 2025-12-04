from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def print_classification_report(y_true, y_pred) -> None:
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, labels=None, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
