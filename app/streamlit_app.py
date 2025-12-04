import streamlit as st
import requests

# Backend URL
API_URL = "http://localhost:8000"

# Page Config
st.set_page_config(
    page_title="Godfather AI",
    page_icon="🌹",
    layout="centered",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0d0d0d 0%, #330000 100%);
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.result-item {
    padding: 8px 12px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #b30000;
    border-radius: 6px;
}

</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<h1 style='text-align:center; color:white;'>🌹 The Godfather Word Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#cccccc;'>Discover the hidden relationships between words inside The Godfather universe.</p>", unsafe_allow_html=True)

st.write("")  # spacing

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔍 Find Similar Words", "⚖️ Compare Two Words"])

# ---------------- TAB 1: FIND SIMILAR WORDS -----------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    st.subheader("🔎 Word Similarity Search")
    word_input = st.text_input("Enter a word:", "godfather")

    if st.button("Search Similar", use_container_width=True):
        if word_input:
            try:
                response = requests.get(f"{API_URL}/similar/{word_input}")

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Words similar to **'{word_input}'**:")

                    for item in data:
                        st.markdown(
                            f"<div class='result-item'><b>{item['word']}</b> — confidence: {item['Score']}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.error(f"Error: {response.json().get('detail')}")

            except requests.exceptions.ConnectionError:
                st.error("🚨 Backend API is down! Is FastAPI running?")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 2: COMPARE TWO WORDS -----------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("⚖️ Compare Words")

    col1, col2 = st.columns(2)
    with col1:
        w1 = st.text_input("Word 1:", "michael")
    with col2:
        w2 = st.text_input("Word 2:", "son")

    if st.button("Calculate Similarity", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/similarity?w1={w1}&w2={w2}")

            if response.status_code == 200:
                data = response.json()
                score = data['Similarity']

                st.metric(label="Cosine Similarity Score", value=f"{score:.4f}")

                if score > 0.5:
                    st.success("🔥 Strong Relationship! These words appear closely related.")
                elif score > 0.2:
                    st.warning("⚠️ Moderate Connection.")
                else:
                    st.info("❄️ Low Similarity. The words rarely appear in similar contexts.")

            else:
                st.error(f"Error: {response.json().get('detail')}")

        except requests.exceptions.ConnectionError:
            st.error("🚨 Backend API is down!")

    st.markdown("</div>", unsafe_allow_html=True)
