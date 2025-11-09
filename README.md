# Surfing Maneuver Recognition using S3D Transfer Learning

Automated surfing maneuver recognition using transfer learning with S3D (Separable 3D Convolutional Neural Networks) architecture.

## Overview

This project applies transfer learning from the Kinetics-400 dataset to classify surfing maneuvers in video clips. By freezing pretrained S3D layers and fine-tuning only the classification head, we achieve **75.36% test accuracy** with just **4,100 trainable parameters** out of 7.9M total parameters.

![Sample Surfing Frames](Papers/Final_Paper/figures/surfing_frames.png)
*Sample frames from a surfing maneuver video showing water spray and wave interaction patterns*

## Dataset

- **Source**: [Surfing Maneuver Classification Dataset](https://www.kaggle.com/datasets/twonzii/surfing-maneuver-classification)
- **Total Clips**: 1,062 videos
- **Classes**: 4 maneuvers
  - Cutback-Frontside (sharp turn toward breaking wave)
  - Take-off (catching and standing up)
  - 360 (full rotation)
  - Roller (riding on top of breaking wave)
- **Split**: 741 train / 110 validation / 211 test (70%/10%/20%)

## Key Features

- **Transfer Learning**: Leverages S3D model pretrained on Kinetics-400
- **Efficient Training**: Only 4,100 trainable parameters
- **Hyperparameter Optimization**: Optuna-based TPE search for optimal learning rate, dropout, and weight decay
- **Balanced Performance**: 70-79% per-class accuracy despite 4.36:1 class imbalance

## Results

### Test Set Performance
- **Overall Accuracy**: 75.36% (159/211 correct)
- **Test Loss**: 0.6350

### Per-Class Accuracy
| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Cutback-Frontside | 70.21% | 33/47 |
| Take-off | 79.17% | 19/24 |
| 360 | 72.22% | 26/36 |
| Roller | 77.88% | 81/104 |

### Training History

![Training History](Papers/Final_Paper/figures/training_history.png)
*Training and validation loss, accuracy, and learning rate over 25 epochs*

### Confusion Matrix

![Confusion Matrix](Papers/Final_Paper/figures/confusion_matrix.png)
*Confusion matrix showing most errors occur between similar turning maneuvers*

## Model Architecture

- **Base Model**: S3D (Separable 3D CNN)
- **Pretrained on**: Kinetics-400
- **Total Parameters**: 7.9M (7.896M frozen backbone + 4.1K trainable classifier)
- **Input Shape**: (3, 64, 224, 224) - 64 frames per video at 224×224 resolution

## Training Configuration

- **Epochs**: 25
- **Batch Size**: 6
- **Optimizer**: RMSprop with cosine annealing
- **Learning Rate**: 7.921 × 10⁻⁴ (optimized)
- **Dropout**: 0.0036 (optimized)
- **Weight Decay**: 5.284 × 10⁻⁴ (optimized)
- **Loss Function**: Cross-entropy (unweighted)

## Key Findings

1. Transfer learning from general action recognition datasets generalizes well to specialized sports video analysis
2. Frozen pretrained features provide strong regularization and prevent overfitting
3. **Weighted loss and oversampling degraded performance** - pretrained features already provide sufficient class discrimination
4. Minority class (take-off) achieved highest accuracy (79.17%), showing no bias toward majority class

## Installation & Usage

See the Jupyter notebook `s3d_surfing_model.ipynb` for complete implementation and training details.

## Citation

If you use this work, please cite:
```
Kidis Sako. "Surfing Maneuver Recognition using Transfer Learning with S3D Convolutional Neural Networks."
University of Applied Sciences Ulm, Advanced Machine Learning, 2025.
```

## Contact

For questions or collaboration inquiries: sakoki01@thu.de

## License

This project is part of academic coursework at University of Applied Sciences Ulm.

