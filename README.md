# 🚗 Car Price Prediction with PyTorch

## 📖 Overview
This project predicts **student exam outcomes (Pass/Fail)** using a simple neural network built with **PyTorch**.  
It demonstrates a full machine learning pipeline from data loading to inference, including:

- 🧠 **Simple Neural Network** with one hidden layer using ReLU activation function  
- ⚖️ **Binary Cross-Entropy (BCE) Loss** for training  
- 🔀 **Mini-batch training** with `DataLoader`  
- 📊 **Train/Validation/Test split** for robust evaluation  
- 📈 **Live training & validation loss monitoring**  
- ✅ **Sigmoid activation on the output** to produce probabilities, with a threshold for Pass/Fail decision

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling  
- **matplotlib** – loss visualization  

---

## ⚙️ Requirements

- Python **3.13+**
- Recommended editor: **VS Code**

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/Exam-Pass-Fail-Classifier.git
```

- Navigate to the `Exam-Pass-Fail-Classifier` directory
```bash
cd Car_Price_Predictor
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `Exam-Pass-Fail-Classifier/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── student_exam_data.csv     # Raw dataset

model/
└── exam_classifier.pth       # Trained model (after training)

src/
├── config.py                 # Paths, hyperparameters, splits, threshold
├── dataset.py                # Dataset class & DataLoader creation
├── model.py                  # Neural network class with save/load
├── train.py                  # Training loop with optional validation
├── test.py                   # Test/evaluation loop
├── inference.py              # Inference utilities for real-world input
├── main_train.py             # Script to train and evaluate the model
├── main_inference.py         # Script to run predictions on new data
├── visualize.py              # Loss monitoring class

requirements.txt             # Python dependencies
```
---

## 📂 Train the Model
```bash
python main_train.py
```
or
```bash
python3 main_train.py
```

---

## 📂 Run Predictions on Real Data
```bash
python main_inference.py
```
or
```bash
python3 main_inference.py
```
