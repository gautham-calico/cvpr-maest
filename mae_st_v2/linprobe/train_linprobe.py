#!/usr/bin/env python3
"""
train_linprobe.py — Train 15 sklearn LogisticRegression classifiers on
pre-extracted ViT features for the CVPR 2026 CT Foundation Model competition.

For each disease:
  1. Load labels from CSV (case_id, label, split)
  2. Load pre-extracted features (ROI for 11 diseases, global for 4)
  3. L2-normalize features
  4. Sweep C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], pick best by val AUC-ROC
  5. Report val metrics
  6. Save all classifiers to classifiers.pkl

Usage:
  python -m mae_st.linprobe.train_linprobe \
      --feature_dir /path/to/features \
      --label_dir /path/to/labels \
      --output_dir /path/to/output
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize

ROI_DISEASES = [
    "adrenal_hyperplasia",
    "cholecystitis",
    "fatty_liver",
    "gallstone",
    "hydronephrosis",
    "kidney_stone",
    "liver_calcifications",
    "liver_cyst",
    "liver_lesion",
    "renal_cyst",
    "splenomegaly",
]

NON_ROI_DISEASES = [
    "ascites",
    "atherosclerosis",
    "colorectal_cancer",
    "lymphadenopathy",
]

ALL_DISEASES = ROI_DISEASES + NON_ROI_DISEASES

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_labels(label_path, disease):
    """
    Load CSV with columns: case_id, {disease}, split
    The case_id column may include .nii.gz extension — we strip it.
    Returns (train_labels, val_labels) dicts of {case_id: label}.
    """
    import csv

    train_labels = {}
    val_labels = {}

    with open(label_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"].replace(".nii.gz", "")
            label = int(row[disease])
            split = row["split"].strip()
            if split == "train":
                train_labels[case_id] = label
            elif split == "val":
                val_labels[case_id] = label

    return train_labels, val_labels


def load_features_for_disease(feature_dir, disease):
    """Load pre-extracted features for a disease. Returns dict {case_id: numpy array}."""
    import torch

    feature_dir = Path(feature_dir)

    if disease in ROI_DISEASES:
        feat_path = feature_dir / "roi_features" / f"{disease}.pt"
    else:
        feat_path = feature_dir / "global_features.pt"

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    features = torch.load(feat_path, map_location="cpu")
    return {k: v.numpy() for k, v in features.items()}


def prepare_data(features, labels):
    """
    Align features and labels.
    Returns X (N, embed_dim) numpy array and y (N,) numpy array.
    """
    case_ids = sorted(set(features.keys()) & set(labels.keys()))
    if not case_ids:
        return None, None
    X = np.stack([features[cid] for cid in case_ids])
    y = np.array([labels[cid] for cid in case_ids])
    return X, y


def train_single_disease(disease, feature_dir, label_dir):
    """
    Train LogisticRegression for one disease with C sweep.
    Returns best classifier, best C, and val metrics dict.
    """
    # Load labels
    label_path = Path(label_dir) / f"{disease}.csv"
    if not label_path.exists():
        print(f"  Label file not found: {label_path}")
        return None, None, None

    train_labels, val_labels = load_labels(label_path, disease)

    # Load features
    features = load_features_for_disease(feature_dir, disease)

    # Prepare train/val splits
    X_train, y_train = prepare_data(features, train_labels)
    X_val, y_val = prepare_data(features, val_labels)

    if X_train is None or len(X_train) == 0:
        print(f"  No training data for {disease}")
        return None, None, None
    if X_val is None or len(X_val) == 0:
        print(f"  No validation data for {disease}")
        return None, None, None

    # L2 normalize features
    X_train = normalize(X_train, norm="l2")
    X_val = normalize(X_val, norm="l2")

    print(f"  Train: {len(X_train)} samples ({y_train.sum()} pos, {(1-y_train).sum()} neg)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} pos, {(1-y_val).sum()} neg)")

    # Sweep over C values
    best_auc = -1
    best_c = None
    best_clf = None

    for c in C_VALUES:
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # Predict probabilities for AUC
        if len(np.unique(y_val)) < 2:
            # Can't compute AUC with only one class
            y_prob = clf.predict_proba(X_val)[:, 1]
            auc = 0.5
        else:
            y_prob = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)

        if auc > best_auc:
            best_auc = auc
            best_c = c
            best_clf = clf

    # Final metrics with best model
    y_prob = best_clf.predict_proba(X_val)[:, 1]
    y_pred = best_clf.predict(X_val)

    if len(np.unique(y_val)) >= 2:
        val_auc = roc_auc_score(y_val, y_prob)
    else:
        val_auc = 0.5
    val_bacc = balanced_accuracy_score(y_val, y_pred)

    metrics = {
        "auc_roc": val_auc,
        "balanced_accuracy": val_bacc,
        "best_C": best_c,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_val": float(y_val.mean()),
    }

    return best_clf, best_c, metrics


def main():
    parser = argparse.ArgumentParser(description="Train linear probing classifiers")
    parser.add_argument("--feature_dir", required=True, help="Directory with pre-extracted features")
    parser.add_argument("--label_dir", required=True, help="Directory with label CSVs")
    parser.add_argument("--output_dir", required=True, help="Directory to save classifiers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    classifiers = {}
    all_metrics = {}

    print("=" * 60)
    print("Training 15 linear probe classifiers")
    print("=" * 60)

    for disease in ALL_DISEASES:
        print(f"\n--- {disease} ---")
        is_roi = disease in ROI_DISEASES
        print(f"  Type: {'ROI' if is_roi else 'Non-ROI (global)'}")

        clf, best_c, metrics = train_single_disease(
            disease, args.feature_dir, args.label_dir
        )

        if clf is not None:
            classifiers[disease] = clf
            all_metrics[disease] = metrics
            print(f"  Best C={best_c}")
            print(f"  Val AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  Val Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        else:
            print(f"  SKIPPED")

    # Save classifiers
    output_path = Path(args.output_dir) / "classifiers.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(classifiers, f)
    print(f"\nSaved {len(classifiers)} classifiers to {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Disease':<25} {'Type':<8} {'AUC-ROC':<10} {'Bal Acc':<10} {'Best C':<10}")
    print("-" * 80)
    for disease in ALL_DISEASES:
        if disease in all_metrics:
            m = all_metrics[disease]
            dtype = "ROI" if disease in ROI_DISEASES else "Global"
            print(f"{disease:<25} {dtype:<8} {m['auc_roc']:<10.4f} {m['balanced_accuracy']:<10.4f} {m['best_C']:<10}")
    print("=" * 80)

    # Save metrics as well
    metrics_path = Path(args.output_dir) / "metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(all_metrics, f)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
