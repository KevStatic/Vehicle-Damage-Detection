# 🚗 Multi-Class Vehicle Damage Severity Classification with Grad-CAM Explainability

This repository contains an **explainable deep learning-based vehicle damage severity classification system** that categorizes input images into **three severity levels**:
- **Minor Damage**
- **Moderate Damage**
- **Severe Damage**

The project uses **MobileNetV2 (Transfer Learning)** for classification and **Grad-CAM** for visual explanation to highlight the regions that influenced the model’s decision — improving **trust**, **interpretability**, and **real-world usability**.

---

## 📌 Key Features

✔ Multi-class vehicle damage severity classification  
✔ Lightweight model — CPU friendly (MobileNetV2)  
✔ Grad-CAM based explainability & heatmap visualization  
✔ Clean modular pipeline (train → evaluate → infer → explain)  
✔ Works with custom user-uploaded images  

---

## 🧪 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/KevStatic/Multi-Class-Vehicle-Damage-Severity-Classification-with-Grad-CAM-Explainability.git
cd Multi-Class-Vehicle-Damage-Severity-Classification-with-Grad-CAM-Explainability
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏋️ Training the Model
```bash
python src/train.py
```

## 📈 Model Evaluation
```bash
python src/evaluation.py
```
Outputs include:
- Accuracy & Loss Curves
- Confusion Matrix
- Classification Report

## 🔍 Run Inference + Grad-CAM Visualization
```bash
python src/inference.py --image path/to/image.jpg
```
Outputs:
- Predicted class label
- Grad-CAM heatmap overlay in results/gradcam/

## 🧠 Model Used
- MobileNetV2 (Pretrained on ImageNet)
- Modified final classification head
- Optimized using Adam + Cross-Entropy Loss
- Designed for low-compute environments

## 📊 Results Summary

| Metric              | Score  |
| ------------------- | ------ |
| Train Accuracy      | 69.20% |
| Validation Accuracy | 63.71% |
| Test Accuracy       | 70.37% |

Grad-CAM visualizations confirm that predictions are based on actual damaged regions, not background elements.

## 🚀 Future Enhancements

- Add No-Damage class
- Expand dataset & improve Moderate class balance
- Try EfficientNet / ViT / ConvNeXt
- Add YOLO-based localization
- Deploy using Streamlit / FastAPI

## 📜 License

This project is intended for academic and research use.
Please credit the repository if used in publications or derivative work.
