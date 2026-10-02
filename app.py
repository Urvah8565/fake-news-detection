import streamlit as st
from model_loader import load_model
from ui import render_ui


# Load model and vectorizer
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
        # Convert news text into TF-IDF features
        news_vector = vectorizer.transform([news_input])

        # Prediction
        proba = model.predict_proba(news_vector)[0]
        result = model.predict(news_vector)[0]

        # Probabilities
        fake_proba = proba[0] * 100
        real_proba = proba[1] * 100

        # Confidence
        confidence = fake_proba if result == 0 else real_proba

        st.divider()

        # Result
        if result == 1:
            st.success("✅ REAL News")
        else:
            st.error("❌ FAKE News")

        # Confidence
        st.metric("Confidence", f"{confidence:.2f}%")

        if confidence < 60:
            st.info(
                "Low confidence prediction. Result may be unreliable."
            )

        # Article-specific explanation
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]

        input_vec = news_vector.toarray()[0]

        word_scores = []

        for idx, value in enumerate(input_vec):
            if value > 0:
                word_scores.append(
                    (feature_names[idx], coefs[idx] * value)
                )

        # Sort by strongest influence
        word_scores = sorted(
            word_scores,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        st.subheader("Key Words Influencing the Prediction")

        for word, score in word_scores[:8]:
            st.write(f"- {word}")


# Footer
st.divider()

st.markdown(
    "<p style='text-align:center; font-size:13px; color:gray;'>"
    "TF-IDF + Logistic Regression | Developed by Urvah"
    "</p>",
    unsafe_allow_html=True
)