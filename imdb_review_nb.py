# %% [markdown]
# # Step 1: Import Libraries

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T01:38:19.858939Z","iopub.execute_input":"2024-10-24T01:38:19.859385Z","iopub.status.idle":"2024-10-24T01:38:22.837762Z","shell.execute_reply.started":"2024-10-24T01:38:19.859338Z","shell.execute_reply":"2024-10-24T01:38:22.836176Z"}}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# %% [markdown]
# # Step 2: Load and Inspect Data

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T01:38:29.306226Z","iopub.execute_input":"2024-10-24T01:38:29.306962Z","iopub.status.idle":"2024-10-24T01:38:31.016145Z","shell.execute_reply.started":"2024-10-24T01:38:29.306906Z","shell.execute_reply":"2024-10-24T01:38:31.014920Z"}}
data = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

print(data.head())
print(data['sentiment'].value_counts())


# %% [markdown]
# # Step 3: Clean Up Data

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T01:40:11.253698Z","iopub.execute_input":"2024-10-24T01:40:11.254278Z","iopub.status.idle":"2024-10-24T02:07:32.289385Z","shell.execute_reply.started":"2024-10-24T01:40:11.254230Z","shell.execute_reply":"2024-10-24T02:07:32.287776Z"}}
import re
from nltk.corpus import stopwords

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text


data['cleaned_review'] = data['review'].apply(clean_text)

# %% [markdown]
# # Step 4: Feature Extraction

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T02:43:37.398470Z","iopub.execute_input":"2024-10-24T02:43:37.399186Z","iopub.status.idle":"2024-10-24T02:43:47.362597Z","shell.execute_reply.started":"2024-10-24T02:43:37.399102Z","shell.execute_reply":"2024-10-24T02:43:47.361142Z"}}
X = data['cleaned_review']
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0) 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)


# %% [markdown]
# # Step 5: Training

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T02:43:53.349472Z","iopub.execute_input":"2024-10-24T02:43:53.349981Z","iopub.status.idle":"2024-10-24T02:43:53.402573Z","shell.execute_reply.started":"2024-10-24T02:43:53.349935Z","shell.execute_reply":"2024-10-24T02:43:53.401306Z"}}
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# %% [markdown]
# # Step 6: Model Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T02:43:57.427049Z","iopub.execute_input":"2024-10-24T02:43:57.427560Z","iopub.status.idle":"2024-10-24T02:43:57.475359Z","shell.execute_reply.started":"2024-10-24T02:43:57.427516Z","shell.execute_reply":"2024-10-24T02:43:57.473921Z"}}
y_pred = nb_model.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

print(classification_report(y_val, y_pred))

conf_matrix = confusion_matrix(y_val, y_pred)
print('Confusion Matrix:\n', conf_matrix)


# %% [markdown]
# # Step 7: Predictions on New Data 

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T02:44:37.809072Z","iopub.execute_input":"2024-10-24T02:44:37.810654Z","iopub.status.idle":"2024-10-24T02:44:37.825685Z","shell.execute_reply.started":"2024-10-24T02:44:37.810560Z","shell.execute_reply":"2024-10-24T02:44:37.823884Z"}}
new_reviews = ["This movie was fantastic! I really enjoyed it.", "The plot was terrible and the acting was worse."]
cleaned_reviews = [clean_text(review) for review in new_reviews]
new_reviews_tfidf = tfidf_vectorizer.transform(cleaned_reviews)
predictions = nb_model.predict(new_reviews_tfidf)

for review, sentiment in zip(new_reviews, predictions):
    print(f'Review: "{review}"\nPredicted Sentiment: {"positive" if sentiment == 1 else "negative"}\n')


# %% [markdown]
# 
