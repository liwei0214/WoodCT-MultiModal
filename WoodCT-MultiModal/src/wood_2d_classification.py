#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wood 2D Image Classification
Feature extraction and classification for 38 wood species using ResNet50/EfficientNet

Author: [Your Name]
Paper: "When Surface Fails, Structure Speaks: Multi-Modal Fusion for Wood Species Identification"
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
DATA_DIR = Path('./data/2d_images')
OUTPUT_DIR = Path('./outputs/2d_classification')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class FeatureExtractor:
    """Deep feature extractor using pretrained CNN"""
    
    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = device
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Data augmentation transforms
        self.aug_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ]),
        ]
    
    def extract_single(self, img_path):
        """Extract features from a single image"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            return None
    
    def extract_with_augmentation(self, img_path, n_aug=4):
        """Extract features with data augmentation"""
        features = []
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Original image
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(img_tensor)
            features.append(feat.cpu().numpy().flatten())
            
            # Augmented images
            for i, aug_transform in enumerate(self.aug_transforms[:n_aug]):
                img_tensor = aug_transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model(img_tensor)
                features.append(feat.cpu().numpy().flatten())
            
            return features
        except Exception as e:
            print(f"  Error: {e}")
            return None


def load_dataset(data_dir, extractor, use_augmentation=True, n_aug=4):
    """Load dataset and extract features"""
    X_list = []
    y_list = []
    filenames = []
    
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()],
                       key=lambda x: int(x.name))
    
    print(f"\nFound {len(class_dirs)} class directories")
    
    for class_dir in tqdm(class_dirs, desc="Extracting features"):
        class_id = int(class_dir.name)
        img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        for img_file in img_files:
            if use_augmentation:
                features = extractor.extract_with_augmentation(img_file, n_aug)
                if features is not None:
                    for feat in features:
                        X_list.append(feat)
                        y_list.append(class_id)
                        filenames.append(str(img_file))
            else:
                feat = extractor.extract_single(img_file)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(class_id)
                    filenames.append(str(img_file))
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, filenames


def evaluate_classifiers(X, y, cv_folds=10):
    """Evaluate multiple classifiers"""
    print("\n" + "="*70)
    print("Classifier Evaluation")
    print("="*70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    classifiers = {
        'KNN-1': KNeighborsClassifier(n_neighbors=1),
        'KNN-3': KNeighborsClassifier(n_neighbors=3),
        'KNN-5': KNeighborsClassifier(n_neighbors=5),
        'SVM-RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        'SVM-Linear': SVC(kernel='linear', C=1, probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"\n{cv_folds}-fold cross-validation:")
    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_scaled, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
            results[name] = {'mean': scores.mean(), 'std': scores.std()}
            print(f"  {name}: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    return results, scaler, le


def evaluate_topk(X, y, cv_folds=10):
    """Evaluate Top-K accuracy"""
    print("\n" + "="*70)
    print("Top-K Accuracy Evaluation")
    print("="*70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    clf = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    
    top1_scores, top3_scores, top5_scores = [], [], []
    
    for train_idx, test_idx in cv.split(X_scaled, y_encoded):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        
        top1, top3, top5 = 0, 0, 0
        for i, true_label in enumerate(y_test):
            sorted_idx = np.argsort(proba[i])[::-1]
            pred_labels = clf.classes_[sorted_idx]
            
            if true_label == pred_labels[0]:
                top1 += 1
            if true_label in pred_labels[:3]:
                top3 += 1
            if true_label in pred_labels[:5]:
                top5 += 1
        
        top1_scores.append(top1 / len(y_test))
        top3_scores.append(top3 / len(y_test))
        top5_scores.append(top5 / len(y_test))
    
    print(f"  Top-1: {np.mean(top1_scores)*100:.2f}% +/- {np.std(top1_scores)*100:.2f}%")
    print(f"  Top-3: {np.mean(top3_scores)*100:.2f}% +/- {np.std(top3_scores)*100:.2f}%")
    print(f"  Top-5: {np.mean(top5_scores)*100:.2f}% +/- {np.std(top5_scores)*100:.2f}%")
    
    return {
        'top1': {'mean': np.mean(top1_scores), 'std': np.std(top1_scores)},
        'top3': {'mean': np.mean(top3_scores), 'std': np.std(top3_scores)},
        'top5': {'mean': np.mean(top5_scores), 'std': np.std(top5_scores)},
    }


def main():
    print("="*70)
    print("Wood 2D Image Classification")
    print("="*70)
    
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please place your 2D images in ./data/2d_images/")
        return
    
    # Initialize feature extractor
    print("\n[Loading pretrained model]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    extractor = FeatureExtractor(model_name='resnet50', device=device)
    print(f"  Model: ResNet50, Feature dim: {extractor.feature_dim}")
    
    # Extract features
    print("\n[Extracting features with augmentation]")
    X_aug, y_aug, _ = load_dataset(DATA_DIR, extractor, use_augmentation=True, n_aug=4)
    print(f"  Feature matrix: {X_aug.shape}")
    print(f"  Class distribution: {Counter(y_aug)}")
    
    # Evaluate classifiers
    results_aug, scaler, le = evaluate_classifiers(X_aug, y_aug, cv_folds=10)
    topk_aug = evaluate_topk(X_aug, y_aug, cv_folds=10)
    
    # Save features
    np.save(OUTPUT_DIR / 'X_2d_features.npy', X_aug)
    np.save(OUTPUT_DIR / 'y_2d_labels.npy', y_aug)
    
    print(f"\nFeatures saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
