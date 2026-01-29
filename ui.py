import streamlit as st

def render_ui():
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="📰",
        layout="centered"
    )

    st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "AI-powered Fake News Detection System"
        "</p>",
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("📝 Enter News Text")

    news_input = st.text_area(
        "",
        height=200,
        placeholder="Paste the news article text here..."
    )

    return news_input
