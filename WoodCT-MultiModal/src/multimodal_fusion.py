#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal Fusion Analysis
Publication-quality figures (Nature/Science style)

Author: [Your Name]
Paper: "When Surface Fails, Structure Speaks: Multi-Modal Fusion for Wood Species Identification"
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ==================== Publication Style Settings ====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Nature-style color palette
COLORS = {
    'blue': '#4DBBD5',
    'red': '#E64B35', 
    'green': '#00A087',
    'purple': '#8B5CF6',
    'orange': '#F39B7F',
    'yellow': '#F0E442',
    'gray': '#7E7E7E',
    'dark_blue': '#3C5488',
    'light_gray': '#E8E8E8'
}

BASE_DIR = Path('.')
CT_FEATURE_DIR = BASE_DIR / 'data' / 'ct_features'
IMG_FEATURE_DIR = BASE_DIR / 'data' / '2d_features'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load feature data"""
    print("="*70)
    print("Loading Data")
    print("="*70)
    
    X_2d = np.load(IMG_FEATURE_DIR / 'X_2d_features.npy')
    y_2d = np.load(IMG_FEATURE_DIR / 'y_2d_labels.npy')
    print(f"2D Features: {X_2d.shape}, Classes: {len(np.unique(y_2d))}")
    
    X_ct = np.load(CT_FEATURE_DIR / 'X_ct_features.npy')
    y_ct = np.load(CT_FEATURE_DIR / 'y_ct_labels.npy')
    print(f"CT Features: {X_ct.shape}, Classes: {len(np.unique(y_ct))}")
    
    return X_2d, y_2d, X_ct, y_ct


def plot_failure_mode(output_dir):
    """Figure: Failure mode analysis pie chart and accuracy comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # (a) Failure mode distribution
    ax1 = axes[0]
    labels = ['Both Succeeded\n(73%)', '2D Failed,\n3D Rescued\n(27%)']
    sizes = [73, 27]
    colors_pie = [COLORS['green'], COLORS['red']]
    explode = (0, 0.05)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                        autopct='%1.0f%%', startangle=90,
                                        textprops={'fontsize': 9})
    ax1.set_title('(a) Failure Mode Distribution', fontsize=11, fontweight='bold', pad=10)
    
    # (b) Accuracy comparison
    ax2 = axes[1]
    methods = ['2D Only', '3D Only', 'Fusion']
    accuracies = [90.8, 96.7, 98.0]
    colors_bar = [COLORS['blue'], COLORS['red'], COLORS['green']]
    
    bars = ax2.bar(methods, accuracies, color=colors_bar, edgecolor='none', width=0.6)
    
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Classification Accuracy (%)', fontsize=10)
    ax2.set_ylim(85, 102)
    ax2.set_title('(b) Accuracy Comparison', fontsize=11, fontweight='bold', pad=10)
    
    # Add improvement annotation
    ax2.annotate('', xy=(2, 98), xytext=(0, 90.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2))
    ax2.text(1, 94, '+7.2%', fontsize=10, color=COLORS['purple'], fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Fig2_failure_mode.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'Fig2_failure_mode.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: Fig2_failure_mode.png/pdf")


def plot_feature_pca(X_2d, y_2d, X_ct, y_ct, output_dir):
    """Figure: PCA visualization comparing 2D vs 3D features"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D features PCA
    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)
    pca_2d = PCA(n_components=2)
    X_2d_pca = pca_2d.fit_transform(X_2d_scaled)
    
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_2d_pca[:, 0], X_2d_pca[:, 1], c=y_2d, cmap='tab20', 
                           alpha=0.7, s=15, edgecolors='none')
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax1.set_title('(a) 2D Surface Features', fontsize=11, fontweight='bold')
    
    # CT features PCA
    scaler_ct = StandardScaler()
    X_ct_scaled = scaler_ct.fit_transform(X_ct)
    pca_ct = PCA(n_components=2)
    X_ct_pca = pca_ct.fit_transform(X_ct_scaled)
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(X_ct_pca[:, 0], X_ct_pca[:, 1], c=y_ct, cmap='tab20',
                           alpha=0.7, s=15, edgecolors='none')
    ax2.set_xlabel(f'PC1 ({pca_ct.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax2.set_ylabel(f'PC2 ({pca_ct.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax2.set_title('(b) 3D Physical Features', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Fig4_feature_pca.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'Fig4_feature_pca.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: Fig4_feature_pca.png/pdf")


def plot_class_rescue(output_dir):
    """Figure: Species-level rescue effect (dumbbell plot)"""
    # Data from paper Table S6
    species = [
        'D. tuberculatus', 'B. sinica var.', 'A. sympetalum', 
        'T. grandis', 'Gluta sp.', 'D. granadillo',
        'P. macrocarpus', 'R. dumetorum', 'C. obtusa', 'M. racemosum'
    ]
    acc_2d = [0, 60, 79, 81, 85, 85, 86, 87, 88, 88]
    acc_3d = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(species))
    
    # Draw connecting lines
    for i, (a2, a3) in enumerate(zip(acc_2d, acc_3d)):
        ax.plot([a2, a3], [i, i], color=COLORS['gray'], linewidth=2, alpha=0.5, zorder=1)
    
    # Draw points
    ax.scatter(acc_2d, y_pos, c=COLORS['blue'], s=100, zorder=2, label='2D Surface', edgecolors='white')
    ax.scatter(acc_3d, y_pos, c=COLORS['green'], s=100, zorder=2, label='3D CT', marker='s', edgecolors='white')
    
    # Add improvement labels
    for i, (a2, a3) in enumerate(zip(acc_2d, acc_3d)):
        improvement = a3 - a2
        ax.text(a3 + 2, i, f'+{improvement}%', va='center', fontsize=8, color=COLORS['red'], fontweight='bold')
    
    # 90% threshold line
    ax.axvline(x=90, color=COLORS['red'], linestyle='--', alpha=0.7, linewidth=1.5, label='90% Threshold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{s}' for s in species], fontsize=9, style='italic')
    ax.set_xlabel('Classification Accuracy (%)', fontsize=11)
    ax.set_xlim(-5, 115)
    ax.set_title('Species-Level Rescue Effect\n"When Surface Fails, Structure Speaks"', 
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'Fig6_class_rescue.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'Fig6_class_rescue.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: Fig6_class_rescue.png/pdf")


def main():
    print("="*70)
    print("Multi-Modal Fusion Analysis - Publication Figures")
    print("="*70)
    
    # Check if data exists
    if not (IMG_FEATURE_DIR / 'X_2d_features.npy').exists():
        print("\nNote: Feature files not found. Generating demo figures...")
        # Generate demo data
        np.random.seed(42)
        X_2d = np.random.randn(1000, 2048)
        y_2d = np.random.randint(0, 38, 1000)
        X_ct = np.random.randn(500, 20)
        y_ct = np.random.randint(0, 38, 500)
    else:
        X_2d, y_2d, X_ct, y_ct = load_data()
    
    # Generate figures
    print("\n" + "="*70)
    print("Generating Publication-Quality Figures (300 DPI)")
    print("="*70)
    
    plot_failure_mode(OUTPUT_DIR)
    plot_feature_pca(X_2d, y_2d, X_ct, y_ct, OUTPUT_DIR)
    plot_class_rescue(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
