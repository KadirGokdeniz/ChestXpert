# ChestXpert: Detection of Lung Diseases from Chest X-rays

<div align="center">
  <img src="assets/chestxpert_logo.png" alt="ChestXpert Logo" width="300"/>
  <br>
  <p>
    <b>Detection of lung diseases from chest X-rays using deep learning</b>
  </p>
  <br>
  
  [![GitHub license](https://img.shields.io/github/license/KadirGokdeniz/ChestXpert)](https://github.com/KadirGokdeniz/ChestXpert/blob/main/LICENSE)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
</div>

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Demo](#-demo)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🔍 About the Project

ChestXpert is a deep learning project developed for detecting 14 different lung diseases from chest X-rays. This project compares the performance of various modern CNN architectures (ResNet, Inception, MobileNet, VGG, AlexNet) and enhances diagnostic accuracy by integrating patient metadata (age, gender).

The project has localization capabilities by utilizing not only visual features but also bounding box information of disease regions. Models are equipped with explainable AI techniques, making it possible to visualize the reasoning behind the model's decisions.

## ✨ Features

- **Multiple CNN Architecture Support**: Implementation and comparison of ResNet, Inception, MobileNet, VGG, and AlexNet architectures
- **Multi-Modal Learning**: Integration of image data and patient metadata (age, gender)
- **Bounding Box Support**: Utilization of bounding box data for disease region localization
- **Explainable AI**: Visualization of model decisions with techniques like Grad-CAM and SHAP
- **Ensemble Learning**: Ensemble models combining the strengths of different architectures
- **Model Optimization**: Optimized models with techniques like quantization and pruning
- **Web Demo**: Interactive web interface for testing trained models

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

### Steps

1. Clone this repository:
```bash
git clone https://github.com/[KadirGokdeniz]/Chestxpert.git
cd chestxpert
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Run with Docker:
```bash
docker build -t chestxpert .
docker run -p 8080:8080 chestxpert
```

## 📈 Usage

### Data Preprocessing

```bash
python scripts/preprocess_data.py --data_dir /path/to/data --output_dir data/processed
```

### Model Training

```bash
# Train ResNet50 model
python scripts/train.py --config configs/resnet_config.yaml

# For other architectures
python scripts/train.py --config configs/inception_config.yaml
python scripts/train.py --config configs/mobilenet_config.yaml
python scripts/train.py --config configs/vgg_config.yaml
python scripts/train.py --config configs/alexnet_config.yaml

# Train ensemble model
python scripts/train_ensemble.py --config configs/ensemble_config.yaml
```

### Model Evaluation

```bash
python scripts/evaluate.py --model_path experiments/models/best_model.pth --test_data data/processed/test
```

### Demo Application

```bash
python app/app.py --model_path experiments/models/best_model.pth --port 8080
```

You can access the demo application at `http://localhost:8080`.

## 📊 Dataset

The ChestXpert project is developed on 84,999 chest X-rays. This dataset:

- Contains 14 different lung disease classes
- Has metadata CSV files with patient age and gender information
- Includes bounding box data marking disease regions for some images

> **Note**: The dataset used in this README may not be real and might have been created for educational purposes. When using a real dataset, make sure to follow the relevant dataset license and citation requirements.

## 🧠 Model Architectures

The ChestXpert project supports the following CNN architectures:

1. **ResNet50**: Deep residual networks that use skip connections to enhance classification performance
2. **InceptionV3**: Features parallel convolution blocks for multi-scale processing and dimension reduction techniques
3. **MobileNetV2**: A lightweight architecture optimized for mobile and embedded devices
4. **VGG16**: A classic deep CNN architecture known for its simple structure
5. **AlexNet**: A pioneer of modern CNNs, included for historical comparison

Additionally, custom hybrid models that combine and extend these base architectures are supported:

- **MetadataNet**: Custom architecture that combines CNN features with patient metadata
- **AttentionNet**: Model using attention mechanisms for disease localization
- **EnsembleNet**: Ensemble model combining the strengths of different architectures

## 📊 Results

| Model | AUC | F1 Score | Sensitivity | Specificity | Processing Time (ms) |
|-------|-----|----------|-------------|-------------|-------------------|
| ResNet50 | * | * | * | * | * |
| InceptionV3 | * | * | * | * | * |
| MobileNetV2 | * | * | * | * | * |
| VGG16 | * | * | * | * | * |
| AlexNet | * | * | * | * | * |
| Ensemble | * | * | * | * | * |

> **Note**: These results belong to the current version of the project and may change with new experiments or dataset updates.

## 🎮 Demo

The ChestXpert web demo has the following features:

- Upload chest X-rays and predict diseases
- Add patient metadata (age, gender)
- View disease probabilities
- Visualize disease localization with GradCAM
- Switch between different models

<div align="center">
  <img src="assets/demo_screenshot.png" alt="ChestXpert Demo" width="700"/>
</div>

## 🤝 Contributing

If you want to contribute, please follow these steps:

1. Fork this repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

[Kadir Gokdeniz] - [kadirqokdeniz@hotmail.com] - [LinkedIn/Twitter profile links]

Project Link: [https://github.com/KadirGokdeniz/chestxpert](https://github.com/KadirGokdeniz/Chestxpert)

---

<div align="center">
  Developed with ❤️ for advancing medical imaging diagnostics
</div>
