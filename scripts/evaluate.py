import torch
from torch.utils.data import DataLoader
from dataset import VehicleDamageDataset
from model import get_model
import torchvision.transforms as T
import numpy as np
from collections import defaultdict


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_model(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """Evaluate model and calculate metrics."""
    model.eval()

    all_predictions = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            predictions = model(images)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    print(f"Processed {len(all_predictions)} images")

    # Calculate metrics per class
    num_classes = 2  # background + damage (we'll focus on damage class)
    class_metrics = {}

    # For damage class (class_id = 1)
    class_id = 1
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt_boxes = 0

    # Store predictions and ground truths for mAP calculation
    all_scores = []
    all_matches = []

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()

        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        # Filter predictions by confidence threshold and class
        mask = (pred_scores >= conf_threshold) & (pred_labels == class_id)
        pred_boxes_filtered = pred_boxes[mask]
        pred_scores_filtered = pred_scores[mask]

        # Filter ground truth by class
        gt_mask = gt_labels == class_id
        gt_boxes_filtered = gt_boxes[gt_mask]
        total_gt_boxes += len(gt_boxes_filtered)

        # Match predictions to ground truths
        matched_gt = set()

        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-pred_scores_filtered)

        for idx in sorted_indices:
            pred_box = pred_boxes_filtered[idx]
            pred_score = pred_scores_filtered[idx]

            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth box
            for gt_idx, gt_box in enumerate(gt_boxes_filtered):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Record match result
            all_scores.append(pred_score)
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                all_matches.append(1)  # True positive
                matched_gt.add(best_gt_idx)
                true_positives += 1
            else:
                all_matches.append(0)  # False positive
                false_positives += 1

        # Unmatched ground truths are false negatives
        false_negatives += len(gt_boxes_filtered) - len(matched_gt)

    # Calculate precision, recall, F1
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    # Calculate Average Precision (AP)
    if len(all_scores) > 0:
        sorted_indices = np.argsort(-np.array(all_scores))
        sorted_matches = np.array(all_matches)[sorted_indices]

        cumsum_tp = np.cumsum(sorted_matches)
        cumsum_fp = np.cumsum(1 - sorted_matches)

        precisions = cumsum_tp / (cumsum_tp + cumsum_fp)
        recalls = cumsum_tp / \
            total_gt_boxes if total_gt_boxes > 0 else np.zeros_like(cumsum_tp)

        ap = calculate_ap(precisions, recalls)
    else:
        ap = 0.0

    class_metrics[class_id] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ap': ap,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_gt_boxes': total_gt_boxes
    }

    # Calculate mean Average Precision (mAP)
    mAP = ap  # Since we only have one class (damage)

    return class_metrics, mAP


def main():
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - MODEL EVALUATION")
    print("="*60)

    # Setup device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"\nUsing device: {device}")

    # Load validation dataset
    print("\nLoading validation dataset...")
    dataset_val = VehicleDamageDataset(
        'Datasets/coco/val',
        'Datasets/coco/val/COCO_val_annos.json',
        transforms=get_transform()
    )
    print(f"Validation dataset size: {len(dataset_val)}")

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Load model
    print("\nLoading trained model...")
    num_classes = 2  # background + damage
    model = get_model(num_classes)
    model.load_state_dict(torch.load('vehicle_damage_model.pth'))
    model.to(device)
    print("Model loaded successfully!")

    # Evaluate with different thresholds
    iou_thresholds = [0.5, 0.75]
    conf_threshold = 0.5

    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"Confidence Threshold: {conf_threshold}")

    for iou_thresh in iou_thresholds:
        print(f"\n{'-'*60}")
        print(f"IoU Threshold: {iou_thresh}")
        print(f"{'-'*60}")

        class_metrics, mAP = evaluate_model(
            model, data_loader_val, device,
            iou_threshold=iou_thresh,
            conf_threshold=conf_threshold
        )

        # Print results for damage class
        metrics = class_metrics[1]
        print(f"\nClass: Damage (ID: 1)")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1-Score:         {metrics['f1_score']:.4f}")
        print(f"  Average Precision (AP): {metrics['ap']:.4f}")
        print(f"\n  Detection Statistics:")
        print(f"    True Positives:  {metrics['true_positives']}")
        print(f"    False Positives: {metrics['false_positives']}")
        print(f"    False Negatives: {metrics['false_negatives']}")
        print(f"    Total GT Boxes:  {metrics['total_gt_boxes']}")

        print(f"\n  mAP@{iou_thresh}: {mAP:.4f}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
