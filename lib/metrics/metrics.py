import torch
import numpy as np


class Accuracy:
    def __init__(self):
        self.reset()

    def update(self, labels, preds):
        correct, labeled = batch_pix_accuracy(preds, labels)
        self.total_correct += correct
        self.total_label += labeled

    def get(self):
        accuracy = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        return accuracy

    def reset(self):
        self.total_correct = 0
        self.total_label = 0


class SegmentationMetric:
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass):
        self.nclass = nclass
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.
        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """
        correct, labeled = batch_pix_accuracy_iou(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter.cpu().numpy()
        self.total_union += union.cpu().numpy()

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(predicted, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary

    predict = predicted.to(torch.int64) + 1
    target = target.to(torch.int64) + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_pix_accuracy_iou(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary

    predict = torch.argmax(output, dim=1).to(torch.int64) + 1
    target = target.to(torch.int64) + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    mini = 1
    maxi = nclass
    nbins = nclass

    predict = torch.argmax(output, dim=1).to(torch.int64) + 1
    target = target.to(torch.int64) + 1

    predict = predict * (target > 0).to(predict.dtype)
    intersection = predict * (predict == target).to(predict.dtype)

    area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)

    area_union = area_pred + area_lab - area_inter

    return area_inter, area_union