import streamlit as st
from model_loader import load_model
from ui import render_ui


# Load model & vectorizer
model, vectorizer = load_model()

# Render UI
news_input = render_ui()

# Prediction button
if st.button("🔍 Predict", use_container_width=True):

    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")

    elif len(news_input.strip()) < 30:
        st.warning("⚠️ Please enter a longer article (min 30 characters).")

   
    elif "key words influencing" in news_input.lower():
        st.warning("⚠️ Please enter a real news article, not UI text.")

    else:
        news_vector = vectorizer.transform([news_input])
        proba = model.predict_proba(news_vector)[0]
        result = model.predict(news_vector)[0]



        fake_proba = proba[0] * 100
        real_proba = proba[1] * 100
        confidence = fake_proba if result == 0 else real_proba


        st.divider()

        if result == 1:
            st.success("✅ REAL Newss")
        else:
            st.error("❌ FAKE News")

        st.metric("Confidence", f"{confidence:.2f}%")

        if confidence < 60:
            st.info("Low confidence prediction. Result may be unreliable.")

        # Article-specific explanation
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]

        input_vec = news_vector.toarray()[0]

        word_scores = []

        for idx, value in enumerate(input_vec):
            if value > 0:
                word_scores.append((feature_names[idx], coefs[idx] * value))

        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)

        st.subheader("Key Words Influencing the Prediction")

        for word, score in word_scores[:8]:
            st.write(f"- {word}")

st.divider()
st.markdown(
    "<p style='text-align:center; font-size:13px; color:gray;'>"
    "TF-IDF + Logistic Regression | Developed by Urvah"
    "</p>",
    unsafe_allow_html=True
)


# 1. Imports
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 2. Read CSV
data = pd.read_csv("WELFake_Dataset.csv", encoding="utf-8")
data = data.sample(n=20000, random_state=42)
print("LABEL VALUES:")
print(data['label'].value_counts())
print(data['label'].unique())


# 3. Clean labels
data = data.dropna(subset=['label'])
y = data['label'].astype(int)
data = data.reset_index(drop=True)
y = y.reset_index(drop=True)

print("Valid samples:", len(y))

# 4. Combine title + text
data['title'] = data['title'].fillna('').astype(str)
data['text'] = data['text'].fillna('').astype(str)

data['combined'] = (data['title'] + " " + data['text']).str.lower()
data['combined'] = data['combined'].str.replace(r"[^a-z\s]", "", regex=True)

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(data['combined'])
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
    C=1.0,
    max_iter=1000,
    solver='saga',
    fit_intercept=False
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

with open("test_data.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)
