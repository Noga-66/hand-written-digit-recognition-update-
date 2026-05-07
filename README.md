# Handwritten Digit Recognizer

A real-time handwritten digit recognition web app built with **Streamlit** and a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

---
<img width="1852" height="856" alt="Screenshot 2026-05-07 204029" src="https://github.com/user-attachments/assets/8d0322a7-476c-4244-835b-a2e0865bcfe4" />

---
<img width="1852" height="860" alt="Screenshot 2026-05-07 204012" src="https://github.com/user-attachments/assets/441c3158-897f-4f87-8139-b1c923c7c6a0" />

---
<img width="1863" height="861" alt="Screenshot 2026-05-07 203951" src="https://github.com/user-attachments/assets/b6314723-5829-47e5-a8f5-cf260829d0a3" />

---
<img width="1853" height="866" alt="Screenshot 2026-05-07 204504" src="https://github.com/user-attachments/assets/90a9b9c3-e0b8-4970-925f-6e880bfe6706" />

--


##  Preview

Draw any digit from 0–9 on the canvas and the model predicts it instantly with confidence scores.

---

##  Features

-  Interactive drawing canvas
-  Real-time digit prediction
-  Confidence score for each digit (0–9)
-  CNN model with 99%+ accuracy on MNIST test set

---

##  Model

- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Dataset:** MNIST (70,000 handwritten digit images)
- **Accuracy:** 99%+ on test set
- **Input:** 28×28 grayscale image

---

##  Project Structure

```
├── app.py                  # Main Streamlit app
├── mnist_model.keras       # Trained CNN model
├── requirements.txt        # Python dependencies
├── .python-version         # Python version pin (3.11)
└── README.md
```

---

##  Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```


> **Note:** The `.python-version` file pins Python to `3.11` which is required for TensorFlow compatibility on Streamlit Cloud.

---

##  Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `tensorflow-cpu` | Model inference |
| `numpy` | Array processing |
| `Pillow` | Image preprocessing |
| `streamlit-drawable-canvas` | Drawing canvas widget |

---

## 📄 License

MIT License — feel free to use and modify.
