# Sentiment Analysis using NLP Techniques

## Overview
This project implements sentiment analysis using Natural Language Processing (NLP) techniques. It processes textual data, extracts features, and applies a Naive Bayes classifier to predict sentiment labels.

## Features
- Preprocessing of text data (lowercasing, stopword removal, tokenization, and lemmatization)
- Feature extraction using TF-IDF vectorization
- Training and evaluation of a Naive Bayes model
- Visualization of results with a confusion matrix

## Installation
To run this project, install the required dependencies:
```bash
pip install pandas numpy nltk scikit-learn seaborn matplotlib
```

## Dataset
The project uses a Twitter sentiment analysis dataset, which is loaded from a CSV file:
```python
dataset_path = 'twitter_training.csv'
df = pd.read_csv(dataset_path, header=None)
```

## Project Structure
- **Data Preprocessing**: Cleans and transforms text data.
- **Feature Extraction**: Uses TF-IDF for numerical representation.
- **Model Training**: Trains a Naive Bayes classifier.
- **Evaluation**: Computes accuracy and visualizes the confusion matrix.

## Code Overview

### 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
```

### 2. Download Necessary NLTK Resources
```python
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 3. Load Dataset
```python
df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['ID', 'Entity', 'Sentiment', 'Text']
df = df[['Sentiment', 'Text']]
```

### 4. Data Preprocessing
```python
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    else:
        return ""

df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
```

### 5. Encode Sentiment Labels
```python
df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})
df.dropna(subset=['Sentiment'], inplace=True)
```

### 6. Split Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Text'], df['Sentiment'], test_size=0.2, random_state=42)
```

### 7. Feature Extraction using TF-IDF
```python
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### 8. Train a Naive Bayes Model
```python
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

### 9. Make Predictions
```python
y_pred = model.predict(X_test_tfidf)
```

### 10. Evaluate Model Performance
```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
```

### 11. Confusion Matrix Visualization
```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive', 'Neutral'], yticklabels=['Negative', 'Positive', 'Neutral'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Running the Project
To execute the project, run the script:
```bash
python sentiment_analysis.py
```

## Output :- Screenshot 2025-03-04 233939.png
- **Accuracy**: Displays the classification accuracy.
- **Classification Report**: Shows precision, recall, and F1-score.
- **Confusion Matrix**: Visualizes the model's performance.
