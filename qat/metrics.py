"""CPU-testable training metrics; independent of SiMa QAT initialization."""
import torch
from torch import Tensor

OCCUPANCY_CLASSES = 18


def confusion_matrix(labels: Tensor, predictions: Tensor) -> Tensor:
    bins = OCCUPANCY_CLASSES * labels.to(torch.int64) + predictions.to(torch.int64)
    return torch.bincount(
        bins, minlength=OCCUPANCY_CLASSES**2
    ).reshape(OCCUPANCY_CLASSES, OCCUPANCY_CLASSES)


def occupancy_metrics(confusion: Tensor) -> tuple[float, float]:
    diagonal = confusion.diag().to(torch.float64)
    union = confusion.sum(0) + confusion.sum(1) - diagonal
    present_semantic_classes = union[: OCCUPANCY_CLASSES - 1] > 0
    iou = diagonal[: OCCUPANCY_CLASSES - 1] / union[: OCCUPANCY_CLASSES - 1].clamp_min(1)
    mean_iou = iou[present_semantic_classes].mean().item()
    accuracy = diagonal.sum().item() / confusion.sum().clamp_min(1).item()
    return accuracy, mean_iou


