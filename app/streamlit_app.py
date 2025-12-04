import streamlit as st
import os
from pathlib import Path
from gensim.models import Word2Vec

# --- FIX PATH ISSUE ---
# Ensure project root is in sys.path so we can import config
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.py"

# Dynamically load MODEL_FILE from src/config.py
import importlib.util
spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
MODEL_FILE = config.MODEL_FILE

# --- PAGE SETUP ---
st.set_page_config(
    page_title="🌹 Godfather AI",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TITLE & INFO ---
st.title("🌹 The Godfather: Word Embeddings")
st.markdown(
    """
    Explore semantic relationships in the Godfather novel using AI.  
    🔍 Find similar words, perform analogies, and visualize relationships.
    """
)

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    try:
        model = Word2Vec.load(str(MODEL_FILE))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error(f"❌ Model not found at `{MODEL_FILE}`.")
    st.warning("👉 Please run `python src/train.py` first to generate the model.")
    st.stop()

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Find Similar Words", 
    "➗ Word Math (Analogies)",
    "📊 Vocabulary Stats"
])

# --- TAB 1: SIMILARITY ---
with tab1:
    st.subheader("Find Synonyms & Context")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        word_input = st.text_input(
            "Enter a word (e.g., michael, gun):", 
            "michael"
        ).strip().lower()
    
    with col2:
        if word_input:
            if word_input in model.wv:
                similar = model.wv.most_similar(word_input, topn=10)
                st.success(f"Words closest to **'{word_input}'**:")
                
                # Better display using columns
                for w, score in similar:
                    st.progress(score, text=f"{w} ({score:.2f})")
            else:
                st.warning(f"⚠️ The word '{word_input}' is not in the vocabulary.")

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
                positive=[pos1, pos2],
                negative=[neg],
                topn=1
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
    
    # Show top 20 most frequent words
    st.markdown("**Top 20 words (by frequency)**")
    freqs = [(w, v) for w, v in model.wv.key_to_index.items()]
    freqs = sorted(freqs, key=lambda x: x[1])[:20]
    
    for w, idx in freqs:
        st.text(f"{w} → index: {idx}")
    
    st.markdown("---")
    st.markdown("💡 Tip: Use the first tab to explore semantic similarity.")

# --- SIDEBAR ---
st.sidebar.header("Godfather AI Controls")
st.sidebar.markdown(
    """
    - Use tabs to explore embeddings  
    - Input words for similarity or analogies  
    - Make sure to train the model first  
    - This app works offline using local Word2Vec model
    """
)


