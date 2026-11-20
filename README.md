
# SMS Spam Detection System Using NLP and Machine Learning

## Project Overview

The SMS Spam Detection System is an intelligent text classification application developed using Natural Language Processing (NLP) and Machine Learning.

The system analyses SMS messages and classifies them into two categories:

- **Spam:** Unwanted or suspicious messages.
- **Ham:** Normal and legitimate messages.

The application uses TF-IDF for text feature extraction and Multinomial Naive Bayes for classification.

## Project Domain

**Domain 6 – NLP Applications**

## Objectives

- To develop an intelligent SMS spam detection system.
- To preprocess and analyse text messages.
- To extract important text features using TF-IDF.
- To train a machine learning classification model.
- To classify messages as Spam or Ham.
- To develop an interactive web application.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Dataset Handling |
| Scikit-learn | Machine Learning |
| TF-IDF | Text Feature Extraction |
| Multinomial Naive Bayes | Classification Algorithm |
| Streamlit | Web Application |
| Matplotlib | Data Visualization |
| Seaborn | Confusion Matrix |

## Dataset

**UCI SMS Spam Collection Dataset**

The dataset contains 5,574 labelled SMS messages.

- Ham: Legitimate messages
- Spam: Unwanted messages

Dataset Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## System Workflow

1. Dataset Collection
2. Data Loading and Preprocessing
3. Text Feature Extraction using TF-IDF
4. Training and Testing Data Split
5. Model Training using Multinomial Naive Bayes
6. Model Evaluation
7. SMS Classification through Streamlit

## Project Features

- User-friendly interface
- SMS message input
- Spam and Ham classification
- Machine Learning-based prediction
- Prediction probability display
- Model evaluation using performance metrics

## Project Structure

```text
SMS-Spam-Detection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── SMSSpamCollection
├── spam_classifier.pkl
├── confusion_matrix.png
├── .gitignore
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/ayaan-alig/SMS-Spam-Detection.git
```

### 2. Open the project folder

```bash
cd SMS-Spam-Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning model

```bash
python train_model.py
```

### 5. Run the application

```bash
streamlit run app.py
```

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Actual evaluation results are generated during model training.

## Limitations

- Predictions depend on patterns learned from the training dataset.
- Unseen or unusual messages may be misclassified.
- The model may not identify every new type of spam message.

## Future Scope

- Support for multiple languages.
- Improved text preprocessing.
- Comparison with additional classification algorithms.
- Integration with real-time SMS filtering systems.

## Academic Information

**Course:** Introduction to Artificial Intelligence

**Course Code:** AIC2022

**Assignment:** Intelligent Software Development

**Project:** SMS Spam Detection System Using NLP and Machine Learning

**Academic Year:** 2026–2027

---

Developed as part of the AIC2022 Course Assignment.
