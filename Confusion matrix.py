import numpy as np

def custom_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.

    Parameters:
        y_true (numpy.ndarray): Ground truth labels (1D array).
        y_pred (numpy.ndarray): Predicted labels (1D array).

    Returns:
        numpy.ndarray: The confusion matrix.
    """
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            true_mask = (y_true == unique_classes[i])
            pred_mask = (y_pred == unique_classes[j])
            confusion_matrix[i, j] = np.sum(true_mask & pred_mask)

    return confusion_matrix

# Sample ground truth labels and predicted labels (replace with your data)
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])

# Compute and print the confusion matrix
conf_matrix = custom_confusion_matrix(y_true, y_pred)
print("Custom Confusion Matrix:")
print(conf_matrix)
