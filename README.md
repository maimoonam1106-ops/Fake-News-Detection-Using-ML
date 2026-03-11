# Fake News Detection using Machine Learning

This project detects whether a news article is **Fake or Real** using Machine Learning and Natural Language Processing (NLP) techniques.  
The model analyzes the textual content of news articles and predicts the authenticity of the news.

---

## Project Overview
Fake news has become a major issue in the digital world. This project uses machine learning techniques to classify news articles as fake or real based on their textual content. The system processes news text using NLP techniques and applies a classification algorithm to make predictions.

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Streamlit (for web interface)

---

## Dataset
The dataset used for this project contains labeled news articles categorized as **Fake** and **Real**.

Dataset files:
- `Fake.csv`
- `True.csv`

You can download the dataset from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Project Structure

```
Fake-News-Detection
│
├── dataset
│   ├── Fake.csv
│   └── True.csv
│
├── fake_news_detection.py
├── streamlit_app.py
└── README.md
```

---

## How the Model Works
1. Load the dataset containing fake and real news articles.
2. Assign labels to the dataset.
3. Combine both datasets into one.
4. Convert text data into numerical features using **TF-IDF Vectorizer**.
5. Split the dataset into training and testing sets.
6. Train a **Logistic Regression model**.
7. Predict whether a news article is fake or real.

---

## How to Run the Project

### 1 Install Required Libraries

```
pip install pandas numpy scikit-learn streamlit
```

### 2 Run the Machine Learning Model

```
python fake_news_detection.py
```

### 3 Run the Streamlit Web App

```
streamlit run streamlit_app.py
```

---

## Features
- Detects fake news using machine learning
- Uses NLP techniques for text processing
- Interactive web interface using Streamlit
- High accuracy prediction model

---

## Future Improvements
- Add deep learning models for better accuracy
- Deploy the web application online
- Improve UI for better user experience

---

## Author
**Maimoona Anbara**  
B.Tech – Artificial Intelligence and Data Science
