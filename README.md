# Emotion-Classification-with-LSTM---TextCNN

DeepEmotion is a deep-learning project that performs **emotion classification** on text using state-of-the-art NLP models (LSTM and TextCNN). The system is trained to recognize six emotions:

* Sadness
* Joy
* Love
* Anger
* Fear
* Surprise

This project includes a complete end-to-end machine learning pipeline: data loading, preprocessing, balancing, tokenization, model training, evaluation, and deployment through a Streamlit app.

---

## 🚀 Features

### **End-to-End NLP Pipeline**

From raw text data to a fully deployed model.

### **Two Deep Learning Models**

* **Bidirectional LSTM (BiLSTM)**
* **TextCNN**

### **Balanced Dataset**

The dataset is oversampled to ensure equal representation for each emotion.

### **Streamlit Deployment**

Interactive UI for real-time emotion prediction.

---

## 🧠 Streamlit App Overview

The Streamlit app (`streamlit_app.py`) performs:

* Model loading
* Tokenizer loading
* Text preprocessing
* Emotion prediction
* UI rendering

It uses clear error handling if model or tokenizer files are missing.

---

## 📂 Project Structure

```
Emotion-Classification-with-LSTM-TextCNN/
│
├── data/
│   └── emotions.csv
│
├── models/
│   ├── emotion_lstm_model.h5
│   └── tokenizer.pkl
│
├── notebooks/
│   └── train_lstm_textcnn.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── lstm_model.py
│   ├── textcnn_model.py
│   ├── train_models.py
│   └── evaluate.py
│
├── streamlit_app.py
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository

```
git clone https:[//github.com/<your-username>/Emotion-Classification-with-LSTM--TextCNN.git](https://github.com/AbdelrahmanElshatlawy/Emotion-Classification-with-LSTM---TextCNN.git)
cd Emotion-Classification-with-LSTM--TextCNN
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Training the Models

Train both LSTM and TextCNN models using:

```
notebooks/train_lstm_textcnn.ipynb
```

This notebook handles data preprocessing, tokenization, model training, evaluation, and saving output files.

---

## 🌐 Running the Streamlit App

Ensure the model and tokenizer exist at the expected paths, then run:

```
streamlit run streamlit_app.py
```

Enter text into the app to receive a predicted emotion.

---

## 💾 Model Saving & Loading

The trained model is saved as:

```
emotion_lstm_model.h5
```

and loaded via TensorFlow.

Tokenizer is saved as:

```
tokenizer.pkl
```

and loaded with pickle.

---

## 📈 Future Improvements

* Integrate BERT or DistilBERT
* Add confusion matrix visualization
* Implement hyperparameter tuning
* Deploy to the cloud

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome. Fork the project and open a pull request to contribute.
