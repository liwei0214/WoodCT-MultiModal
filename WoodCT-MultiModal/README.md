# WoodCT-MultiModal

Multi-modal wood species identification using CT imaging and 2D surface features.

## Paper

**"When Surface Fails, Structure Speaks: Multi-Modal Fusion Resolves Systematic Failures in Wood Species Identification"**

## Key Results

| Method | Accuracy | Description |
|--------|----------|-------------|
| 2D Surface (ResNet50) | 91.54% | Baseline using surface images |
| 3D CT (CT-PINN) | **99.18%** | Physical features from CT scans |
| Late Fusion | 99.09% | Combined 2D + 3D features |

### Rescue Effect
- 10 species with <90% 2D accuracy all achieve **100%** with 3D CT features
- Average improvement: +26.1%

## Dataset

- **38 precious wood species** for cultural heritage and furniture authentication
- 2D surface images: 772 samples with 5x augmentation
- 3D CT scans: ~300 slices per species at 29.77 μm resolution

## Project Structure

```
WoodCT-MultiModal/
├── src/
│   ├── wood_2d_classification.py    # 2D feature extraction (ResNet50)
│   ├── ablation_study.py            # Fusion strategy comparison
│   ├── multimodal_fusion.py         # Multi-modal analysis
│   └── confusion_analysis.py        # Detailed error analysis
├── data/
│   ├── 2d_images/                   # Surface images (by class)
│   └── ct_features/                 # Extracted CT features
├── outputs/
│   └── figures/                     # Generated figures
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract 2D Features
```bash
python src/wood_2d_classification.py
```

### 2. Run Ablation Study
```bash
python src/ablation_study.py
```

### 3. Generate Analysis Figures
```bash
python src/multimodal_fusion.py
```

## Citation

```bibtex
@article{wood2025multimodal,
  title={When Surface Fails, Structure Speaks: Multi-Modal Fusion Resolves Systematic Failures in Wood Species Identification},
  author={Li, Wei and Zhu, Meng and Ren, Honge and Ma, Yan and Geng, Shiran},
  journal={Wood Science and Technology},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- Northeast Forestry University Museum
- Dalian Natural History Museum  
- Grand Canal Zitan Museum
