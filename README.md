# Lung Cancer Detection using ResNet50

## Overview
This project implements a deep learning model for lung cancer detection using a pretrained ResNet50 model. The dataset used is the **IQ-OTHNCCD Lung Cancer Dataset (Augmented)**, which is available on Kaggle.

## Dataset
- **Source:** [IQ-OTHNCCD Lung Cancer Dataset (Augmented)](https://www.kaggle.com/datasets/subhajeetdas/iq-othnccd-lung-cancer-dataset-augmented)
- The dataset contains augmented images of lung cancer samples, categorized into different classes.

## Model Architecture
- **Base Model:** ResNet50 (Pretrained on ImageNet)
- **Transfer Learning:** The fully connected layers of ResNet50 are modified for binary/multi-class classification.
- **Training:** Fine-tuned on the lung cancer dataset using a suitable optimizer and loss function.

## Implementation Details
- **Frameworks:** PyTorch
- **Data Augmentation:** Applied transformations to improve generalization
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/parsamhb82/lung-cancer-detection.git
   ```
2. Open the Jupyter Notebook (`lung_cancer_detection.ipynb`) in your preferred environment (Jupyter Notebook, VS Code, etc.).
3. Download the dataset from Kaggle and place it in the appropriate directory.
4. Run the notebook cells to train and evaluate the model.

## Results
- The trained ResNet50 model achieved high accuracy in detecting lung cancer.
- Visualizations of loss and accuracy curves are included in the notebook.

## Repository Structure
```
📂 lung-cancer-detection
├── 📜 README.md
├── 📜 lung_cancer_detection.ipynb
├── 📂 dataset (optional: not included due to size constraints)
└── 📜 requirements.txt
```

## Future Improvements
- Experiment with other deep learning architectures such as EfficientNet and DenseNet.
- Implement a web-based interface for real-time detection.
- Enhance dataset quality with further augmentation techniques.

## References
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet50 Documentation](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet50)

## License
This project is open-source and available under the MIT License.


