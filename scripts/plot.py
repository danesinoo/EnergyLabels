import matplotlib.pyplot as plt
import numpy as np

figNum = 0


def axis(results, title, xlabel, ylabel, symbol="-o", save_path=None):
    """
    Plot training and evaluation losses using seaborn

    Args:
        results: Dictionary {'function name': [array of values]}
        save_path: Optional path to save the plot
    """
    global figNum
    figNum += 1
    colors = ["black", "blue", "purple", "cyan", "yellow", "orange", "red"]

    labels = list(results.keys())
    epochs = [(i + 1) * 50 for i in range(len(results[labels[0]]))]

    plt.figure(figNum, figsize=(10, 6))

    for i, label in enumerate(labels):
        plt.plot(
            epochs,
            results[label],
            symbol,
            markersize=5,
            label=label,
            color=colors[i % len(colors)],
        )

    # Create the plot

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.grid(True, alpha=0.1)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")

    plt.show()

def confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            cm[y_true[i][j], y_pred[i][j]] += 1

    global figNum
    figNum += 1
    plt.figure(figNum, figsize=(10, 8))

    plt.imshow(np.log10(cm + 1), cmap=plt.cm.viridis)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(np.arange(7), labels)
    plt.yticks(np.arange(7), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), 
                    ha='center', va='center', 
                    fontsize=10)

    plt.show()