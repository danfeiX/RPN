import numpy as np
from rpn.utils.torch_utils import to_numpy


def binary_accuracy(preds, labels):
    preds = to_numpy(preds)
    labels = to_numpy(labels)
    num_c = preds.shape[-1]
    fp = np.logical_and(preds > 0.5, labels < 0.5).astype(np.float64)
    fn = np.logical_and(preds < 0.5, labels > 0.5).astype(np.float64)
    acc = ((preds > 0.5) == (labels > 0.5)).astype(np.float64).mean()
    return acc, fp.reshape([-1, num_c]).mean(axis=0), fn.reshape([-1, num_c]).mean(axis=0)


def classification_accuracy(preds, labels):
    preds = to_numpy(preds)
    labels = to_numpy(labels)
    pred_labels = np.argmax(preds, axis=-1)
    assert(pred_labels.shape == labels.shape)
    return (pred_labels == labels).astype(np.float64).mean()


def masked_binary_accuracy(preds, labels):
    preds = to_numpy(preds)
    labels = to_numpy(labels)
    assert(np.all(labels <= 2))
    pred_labels = np.argmax(preds, axis=-1)
    masked_acc = (pred_labels == labels)[labels != 2]
    mask_acc = (pred_labels == labels)[labels == 2]
    return masked_acc.astype(np.float64).mean(), mask_acc.astype(np.float64).mean()

