# 1. Imports
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 2. Read CSV
data = pd.read_csv("news.csv", encoding="latin-1")

print("LABEL VALUES:")
print(data['labels'].value_counts())


# 3. Clean labels
data = data.dropna(subset=['labels'])

data['labels'] = (
    data['labels']
    .astype(str)
    .str.strip()
    .str.upper()
)

# Map labels
y = data['labels'].astype(int)

# Remove rows where mapping failed
print("Valid samples:", len(y))

# Reset index
data = data.reset_index(drop=True)
y = y.reset_index(drop=True)

print("Valid samples:", len(y))
print("Any NaN in y?", y.isna().sum())


# 4. Combine title + content
data['article_title'] = data['article_title'].fillna('').astype(str)
data['article_content'] = data['article_content'].fillna('').astype(str)

data['text'] = (data['article_title'] + " " + data['article_content']).str.lower()
data['text'] = data['text'].str.replace(r"[^a-z\s]", "", regex=True)


# REMOVE empty rows
# data = data[data['text'] != ""]

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=50000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 3),
    sublinear_tf=True
)

X = vectorizer.fit_transform(data['text'])
print("TF-IDF Matrix Shape:", X.shape)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 7. Train model
model = LogisticRegression(
    class_weight="balanced",
    C=4.0,
    max_iter=4000
)

model.fit(X_train, y_train)

# Save trained model
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and Vectorizer saved successfully")


# 8. Evaluation
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))

# Save test data for Streamlit evaluation
with open("test_data.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)
print(data.columns)
