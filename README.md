
# SMS Spam Detection System Using NLP and Machine Learning

## Live Application

**Try the application here:**

https://ayaan-sms-spam-detector.streamlit.app

## Project Overview

The SMS Spam Detection System is an intelligent text classification application developed using Natural Language Processing (NLP) and Machine Learning.

It analyses SMS messages and classifies them into two categories:

- Spam: Unwanted or suspicious messages.
- Ham: Normal and legitimate messages.

The system uses TF-IDF for text feature extraction and Multinomial Naive Bayes for classification.

## Project Domain

Domain 6 – NLP Applications

## Objectives

- Develop an intelligent SMS spam detection system.
- Process and analyse text messages.
- Extract text features using TF-IDF.
- Train a Machine Learning classification model.
- Classify SMS messages as Spam or Ham.
- Build an interactive web application.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Dataset Handling |
| Scikit-learn | Machine Learning |
| TF-IDF | Feature Extraction |
| Multinomial Naive Bayes | Classification |
| Streamlit | Web Application |
| Matplotlib | Visualization |
| Seaborn | Confusion Matrix |

## Dataset

UCI SMS Spam Collection Dataset

Dataset Source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains 5,574 labelled SMS messages.

## System Workflow

1. Dataset Collection
2. Data Loading
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. Training and Testing Split
6. Model Training
7. Model Evaluation
8. SMS Classification

## Application Features

- Interactive web interface.
- SMS message input.
- Spam and Ham classification.
- Machine Learning-based predictions.
- Example SMS messages.
- Prediction result display.

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

### 1. Clone Repository

```bash
git clone https://github.com/ayaan-alig/SMS-Spam-Detection.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

### 4. Run the Application

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

## Limitations

- Unseen messages may be misclassified.
- Predictions depend on the training dataset.
- New spam patterns may require model retraining.

## Future Scope

- Support for multiple languages.
- Improved text preprocessing.
- Comparison with additional ML algorithms.
- Integration with real-time SMS filtering.

## Academic Information

**Course:** Introduction to Artificial Intelligence

**Course Code:** AIC2022

**Assignment:** Intelligent Software Development

**Project:** SMS Spam Detection System

**Academic Year:** 2026–2027

---

Developed as part of the AIC2022 Course Assignment.
