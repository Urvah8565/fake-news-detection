# Fake News Detection using Machine Learning

## Project Overview
This project implements a Fake News Detection system using Python and 
classical Machine Learning techniques. The system allows users to input 
a news article and predicts whether it is REAL or FAKE. It also displays 
prediction confidence and an explanation of the most influential words 
used in the decision.

The application uses Streamlit for the frontend and Logistic Regression 
for classification.

## Live Demo
https://fake-news-detection-h6yg8dcyttoyorj4whs8kg.streamlit.app/

## Dataset
- Source: WELFake Dataset (Kaggle)
- Size: 20,000 news articles (sampled from 72,134 total)
- Balanced: ~10,167 Fake, ~9,833 Real

### Labels:
- 0 – Fake News
- 1 – Real News

The dataset is used only during model training and is not included 
in this repository.

## Machine Learning Pipeline
- Load WELFake dataset
- Combine article title and text
- Clean and preprocess text
- Convert text to numerical features using TF-IDF
- Split data into training and testing sets (80/20)
- Train Logistic Regression model
- Save trained model and vectorizer as .pkl files
- Load saved model in Streamlit for prediction on new user input

## Features
- Fake / Real news prediction
- Confidence score with low-confidence warning
- Article-specific explanation showing top influencing keywords
- Streamlit-based web interface

## Model Performance
- Training Accuracy: 99.09%
- Testing Accuracy: 95.80%
- Evaluated on 4,000 held-out test samples

### Confusion Matrix:
|                  | Predicted Fake | Predicted Real |
|------------------|----------------|----------------|
| **Actual Fake**  | 1886           | 81             |
| **Actual Real**  | 87             | 1946           |

## Application Preview

### Home Screen
![Home](./app_home.png)

### Prediction Result
![Result](./app_result.png)

## How to Run the Project

### Install dependencies:
pip install -r requirements.txt

### Train the model:
python read_data.py

### Run the Streamlit application:
streamlit run app.py

## Limitations
- Confidence varies on short or vague headlines due to limited 
  vocabulary signal
- Model may misclassify real news written in formal journalistic 
  style if similar patterns exist in fake training articles
- Logistic Regression does not capture deep semantic meaning — 
  a transformer-based model (e.g. BERT) would perform better

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorizer
- Logistic Regression

## Project Developer
[Urvah Mansuri]
