import pickle
import streamlit as st

@st.cache_resource
def load_model():
    with open("logreg_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer
