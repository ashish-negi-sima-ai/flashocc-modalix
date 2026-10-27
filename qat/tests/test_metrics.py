import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qat"))

from train import confusion_matrix, occupancy_metrics
from validate_qat_onnx import confusion_matrix as numpy_confusion_matrix
from validate_qat_onnx import occupancy_metrics as numpy_occupancy_metrics


class OccupancyMetricsTest(unittest.TestCase):
    def test_miou_excludes_free_class(self):
        labels = torch.tensor([0, 1, 17, 17])
        predictions = torch.tensor([0, 0, 17, 1])
        confusion = confusion_matrix(labels, predictions)
        accuracy, mean_iou = occupancy_metrics(confusion)
        self.assertEqual(confusion.sum().item(), 4)
        self.assertAlmostEqual(accuracy, 0.5)
        self.assertAlmostEqual(mean_iou, 0.25)

    def test_numpy_confusion_promotes_uint8_before_binning(self):
        labels = np.array([0, 1, 17, 17], dtype=np.uint8).reshape(1, 2, 2)
        predictions = np.array([0, 0, 17, 1], dtype=np.uint8).reshape(1, 2, 2)
        confusion = numpy_confusion_matrix(labels, predictions)
        metrics = numpy_occupancy_metrics(confusion)
        self.assertEqual(confusion[17, 17], 1)
        self.assertEqual(confusion.sum(), 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["miou_excluding_free"], 0.25)


if __name__ == "__main__":
    unittest.main()
