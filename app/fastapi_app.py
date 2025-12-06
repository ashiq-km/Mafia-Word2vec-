# We will create an API that has one job:
# take a word. look up the math, and return the similar words.


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from gensim.models import Word2Vec

import src.config as config

# Global variable to hold the model

ml_models = {}

# Load the model on start up (So, we don't load it for every request)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # --- Startup Logic ---
    print("Loading model...")
    if not config.MODEL_FILE.exists():

        # In Production, you might want to
        # download the model from S3 / DVC here
        raise FileNotFoundError("Model file not found. Run training first.")

    # Load the full model

    full_model = Word2Vec.load(str(config.MODEL_FILE))

    # Store only the KeyedVectors (wv) in our dictionary for speed
    ml_models["wv"] = full_model.wv

    print("Model loaded successfully!")

    yield  # Control is passed to the application

    # --- shutdown logic ---

    print("Cleaning up resources...")
    ml_models.clear()


# Initialize the App with the lifespan manager

app = FastAPI(
    title="Godfather Word2vec API", lifespan=lifespan
)  # This is the framework for the uvicorn server.


@app.get("/")
def home():
    return {"message": "Welcome to the Godfather API. \
            Go to the /docs for testing."}


@app.get("/similar/{word}")
def get_similar_words(word: str):
    """Returns Top 10 similar words."""

    clean_word = word.lower().strip()

    # Access the model from our Global dictionary.

    model_wv = ml_models.get("wv")

    if clean_word not in model_wv:
        raise HTTPException(
            status_code=404, detail=f"Word {word} not found in Vocabulary."
        )

    # Get similarities
    results = model_wv.most_similar(clean_word, topn=10)

    return [{"word": w, "Score": f"{s:.2f}"} for w, s in results]


@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    """
    Docstring for get_similarity

    :param w1: Description
    :type w1: str
    :param w2: Description
    :type w2: str

    >> Compare two words directly.
    """

    w1, w2 = w1.lower(), w2.lower()

    model_wv = ml_models.get("wv")

    if w1 not in model_wv or w2 not in model_wv:
        raise HTTPException(
            status_code=404, detail="One of the words is \
                missing from the vocabulary."
        )

    score = model_wv.similarity(w1, w2)
    return {"Word 1": w1, "Word 2": w2, "Similarity": float(score)}
