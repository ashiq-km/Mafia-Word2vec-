from pathlib import Path

import requests
import streamlit as st
from gensim.models import Word2Vec

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🌹 Godfather AI",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌹 The Godfather: Word Embeddings")
st.markdown(
    """
    Explore semantic relationships in the Godfather novel using AI.
    🔍 Find similar words, perform analogies, and visualize relationships.
    """
)

# --- MODEL DOWNLOAD CONFIG ---
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "godfather_w2v.model"

# Google Drive direct download URL
FILE_ID = "1S_nDsZgciriwEOYgcyvXcdZ3_1MyAYAv"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# --- DOWNLOAD MODEL IF MISSING ---
@st.cache_resource
def download_and_load_model():
    if not MODEL_PATH.exists():
        st.info("📥 Downloading pre-trained Word2Vec model...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    # Load the model
    try:
        model = Word2Vec.load(str(MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# --- LOAD MODEL ---
model = download_and_load_model()

if model is None:
    st.stop()

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs(
    ["🔍 Find Similar Words", "➗ Word Math (Analogies)", "📊 Vocabulary Stats"]
)

# --- TAB 1: SIMILARITY ---
with tab1:
    st.subheader("Find Synonyms & Context")
    col1, col2 = st.columns([1, 2])

    with col1:
        word_input = (
            st.text_input("Enter a word (e.g., michael, gun):", "michael")
            .strip()
            .lower()
        )

    with col2:
        if word_input:
            if word_input in model.wv:
                similar = model.wv.most_similar(word_input, topn=10)
                st.success(f"Words closest to **'{word_input}'**:")

                for w, score in similar:
                    st.progress(score, text=f"{w} ({score:.2f})")
            else:
                st.warning(f"⚠️ The word '{word_input}' \
                           is not in the vocabulary.")

# --- TAB 2: ANALOGIES ---
with tab2:
    st.subheader("Semantic Analogies")
    st.markdown("Equation: `Positive 1 - Negative + Positive 2 = Result`")

    c1, c2, c3 = st.columns(3)
    with c1:
        pos1 = st.text_input("Positive 1 (e.g., vito):", "vito").lower()
    with c2:
        neg = st.text_input("Negative (e.g., father):", "father").lower()
    with c3:
        pos2 = st.text_input("Positive 2 (e.g., son):", "son").lower()

    if st.button("Calculate Analogy Result"):
        try:
            result = model.wv.most_similar(
                positive=[pos1, pos2], negative=[neg], topn=1
            )
            prediction, confidence = result[0]
            st.balloons()
            st.success(f"✨ **Result:** {prediction} ({confidence:.2f})")
        except KeyError as e:
            st.error(f"Word not found in vocabulary: {e}")

# --- TAB 3: VOCAB STATS ---
with tab3:
    st.subheader("Vocabulary Overview")
    st.info(f"Total words in vocabulary: **{len(model.wv)}**")

    st.markdown("**Top 20 words (by frequency)**")
    # gensim doesn't provide raw frequency easily; using key_to_index as proxy
    top_words = list(model.wv.key_to_index.keys())[:20]
    for i, w in enumerate(top_words, start=1):
        st.text(f"{i}. {w}")

# --- SIDEBAR ---
st.sidebar.header("Godfather AI Controls")
st.sidebar.markdown(
    """
    - Use tabs to explore embeddings
    - Input words for similarity or analogies
    - Model automatically downloads if missing
    - Works offline after first run
    """
)
