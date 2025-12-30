#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confusion Analysis
Detailed analysis of species-level classification errors

Author: [Your Name]
Paper: "When Surface Fails, Structure Speaks: Multi-Modal Fusion for Wood Species Identification"
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'blue': '#4DBBD5', 'red': '#E64B35', 'green': '#00A087',
    'purple': '#8B5CF6', 'orange': '#F39B7F', 'gray': '#7E7E7E'
}

BASE_DIR = Path('.')
IMG_FEATURE_DIR = BASE_DIR / 'data' / '2d_features'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'confusion_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Species names (Latin)
SPECIES_NAMES = {
 
}


def get_short_name(class_id):
    """Get abbreviated species name"""
    full = SPECIES_NAMES.get(class_id, f"Class {class_id}")
    # Return genus + species initial
    parts = full.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[1]}"
    return full


def analyze_confusion():
    """Analyze confusion patterns in 2D classification"""
    print("="*70)
    print("Confusion Analysis")
    print("="*70)
    
    # Load data
    X_2d = np.load(IMG_FEATURE_DIR / 'X_2d_features.npy')
    y_2d = np.load(IMG_FEATURE_DIR / 'y_2d_labels.npy')
    print(f"Data: {X_2d.shape}, Classes: {len(np.unique(y_2d))}")
    
    # Standardize and encode
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_2d)
    
    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = SVC(kernel='linear', C=1, probability=True)
    
    all_y_true, all_y_pred = [], []
    for train_idx, test_idx in cv.split(X_scaled, y_encoded):
        clf.fit(X_scaled[train_idx], y_encoded[train_idx])
        y_pred = clf.predict(X_scaled[test_idx])
        all_y_true.extend(y_encoded[test_idx])
        all_y_pred.extend(y_pred)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Build confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    # Analyze per-class errors
    print("\n" + "="*70)
    print("Per-Class Error Analysis")
    print("="*70)
    
    confusion_pairs = []
    class_errors = {}
    
    for i, class_label in enumerate(le.classes_):
        class_id = int(class_label)
        short_name = get_short_name(class_id)
        
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        
        errors = []
        for j, target_label in enumerate(le.classes_):
            if i != j and cm[i, j] > 0:
                target_id = int(target_label)
                target_name = get_short_name(target_id)
                errors.append({
                    'target_id': target_id,
                    'target_name': target_name,
                    'count': int(cm[i, j])
                })
                confusion_pairs.append({
                    'from_id': class_id,
                    'from_name': short_name,
                    'to_id': target_id,
                    'to_name': target_name,
                    'count': int(cm[i, j])
                })
        
        errors.sort(key=lambda x: x['count'], reverse=True)
        
        class_errors[class_id] = {
            'name': short_name,
            'accuracy': accuracy,
            'total': int(total),
            'correct': int(correct),
            'errors': errors
        }
    
    # Sort by accuracy
    sorted_classes = sorted(class_errors.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n[Sorted by Accuracy - Low to High]\n")
    for class_id, info in sorted_classes:
        if info['accuracy'] < 1.0:
            print(f"* {info['name']} (ID:{class_id})")
            print(f"  Accuracy: {info['accuracy']*100:.1f}% ({info['correct']}/{info['total']})")
            if info['errors']:
                print(f"  Misclassified to:")
                for err in info['errors'][:3]:
                    print(f"    -> {err['target_name']}: {err['count']} times")
            print()
    
    # Find mutual confusion pairs
    print("\n" + "="*70)
    print("Mutual Confusion Pairs (Bidirectional)")
    print("="*70)
    
    mutual_confusion = []
    for i, c1 in enumerate(le.classes_):
        for j, c2 in enumerate(le.classes_):
            if i < j:
                c1_to_c2 = cm[i, j]
                c2_to_c1 = cm[j, i]
                if c1_to_c2 > 0 or c2_to_c1 > 0:
                    mutual_confusion.append({
                        'class1_id': int(c1),
                        'class1_name': get_short_name(int(c1)),
                        'class2_id': int(c2),
                        'class2_name': get_short_name(int(c2)),
                        'c1_to_c2': int(c1_to_c2),
                        'c2_to_c1': int(c2_to_c1),
                        'total': int(c1_to_c2 + c2_to_c1)
                    })
    
    mutual_confusion.sort(key=lambda x: x['total'], reverse=True)
    
    print("\n[Top 15 Most Confused Pairs]\n")
    print(f"{'Species 1':<20} {'<->':<5} {'Species 2':<20} {'A->B':<5} {'B->A':<5} {'Total':<5}")
    print("-" * 70)
    for pair in mutual_confusion[:15]:
        print(f"{pair['class1_name']:<20} {'<->':<5} {pair['class2_name']:<20} "
              f"{pair['c1_to_c2']:<5} {pair['c2_to_c1']:<5} {pair['total']:<5}")
    
    return cm, le, confusion_pairs, mutual_confusion, class_errors


def plot_confusion_matrix(cm, le, output_dir):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Normalize
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    
    # Show only off-diagonal (errors)
    cm_display = cm_norm.copy()
    np.fill_diagonal(cm_display, 0)
    
    im = ax.imshow(cm_display, cmap='Reds', vmin=0, vmax=0.3)
    
    class_labels = [get_short_name(int(c)) for c in le.classes_]
    
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90, ha='center', fontsize=7, style='italic')
    ax.set_yticklabels(class_labels, fontsize=7, style='italic')
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('2D Classification Confusion Matrix\n(Off-diagonal elements only)', 
                 fontsize=12, fontweight='bold', pad=15)
    
    plt.colorbar(im, ax=ax, shrink=0.7, label='Proportion')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_full_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: confusion_full_matrix.png")


def plot_confusion_pairs(mutual_confusion, output_dir):
    """Plot top confused species pairs"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top_pairs = mutual_confusion[:15]
    labels = [f"{p['class1_name']} <-> {p['class2_name']}" for p in top_pairs]
    values = [p['total'] for p in top_pairs]
    
    colors = [COLORS['red'] if v >= 2 else COLORS['orange'] for v in values]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.6)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val}', va='center', ha='left', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, style='italic')
    ax.set_xlabel('Number of Misclassifications', fontsize=11)
    ax.set_title('Top 15 Most Confused Species Pairs', fontsize=12, fontweight='bold', pad=10)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_pairs_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: confusion_pairs_bar.png")


def main():
    # Check if data exists
    if not (IMG_FEATURE_DIR / 'X_2d_features.npy').exists():
        print("Error: Feature files not found.")
        print("Please run wood_2d_classification.py first.")
        return
    
    # Analyze
    cm, le, confusion_pairs, mutual_confusion, class_errors = analyze_confusion()
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating Confusion Analysis Figures")
    print("="*70)
    
    plot_confusion_matrix(cm, le, OUTPUT_DIR)
    plot_confusion_pairs(mutual_confusion, OUTPUT_DIR)
    
    # Save detailed report
    print("\n" + "="*70)
    print("Saving Detailed Report")
    print("="*70)
    
    with open(OUTPUT_DIR / 'confusion_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Wood 2D Classification Confusion Report\n")
        f.write("="*70 + "\n\n")
        
        f.write("[Top 20 Most Confused Pairs]\n\n")
        f.write(f"{'Species 1':<25} {'<->':<5} {'Species 2':<25} {'A->B':<5} {'B->A':<5} {'Total':<5}\n")
        f.write("-" * 80 + "\n")
        for pair in mutual_confusion[:20]:
            f.write(f"{pair['class1_name']:<25} {'<->':<5} {pair['class2_name']:<25} "
                   f"{pair['c1_to_c2']:<5} {pair['c2_to_c1']:<5} {pair['total']:<5}\n")
        
        f.write("\n\n[Per-Class Error Details]\n\n")
        sorted_classes = sorted(class_errors.items(), key=lambda x: x[1]['accuracy'])
        for class_id, info in sorted_classes:
            if info['accuracy'] < 1.0:
                f.write(f"\n* {info['name']} (ID:{class_id})\n")
                f.write(f"  Accuracy: {info['accuracy']*100:.1f}% ({info['correct']}/{info['total']})\n")
                if info['errors']:
                    f.write(f"  Misclassified to:\n")
                    for err in info['errors']:
                        f.write(f"    -> {err['target_name']}: {err['count']} times\n")
    
    print("  Saved: confusion_report.txt")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nAll files saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
