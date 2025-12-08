# # We will create an API that has one job:
# # take a word. look up the math, and return the similar words.


# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException
# from gensim.models import Word2Vec

# import src.config as config

# # Global variable to hold the model

# ml_models = {}

# # Load the model on start up (So, we don't load it for every request)


# @asynccontextmanager
# async def lifespan(app: FastAPI):

#     # --- Startup Logic ---
#     print("Loading model...")
#     if not config.MODEL_FILE.exists():

#         # In Production, you might want to
#         # download the model from S3 / DVC here
#         raise FileNotFoundError("Model file not found. Run training first.")

#     # Load the full model

#     full_model = Word2Vec.load(str(config.MODEL_FILE))

#     # Store only the KeyedVectors (wv) in our dictionary for speed
#     ml_models["wv"] = full_model.wv

#     print("Model loaded successfully!")

#     yield  # Control is passed to the application

#     # --- shutdown logic ---

#     print("Cleaning up resources...")
#     ml_models.clear()


# # Initialize the App with the lifespan manager

# app = FastAPI(
#     title="Godfather Word2vec API", lifespan=lifespan
# )  # This is the framework for the uvicorn server.


# @app.get("/")
# def home():
#     return {
#         "message": "Welcome to the Godfather API. \
#             Go to the /docs for testing."
#     }


# @app.get("/similar/{word}")
# def get_similar_words(word: str):
#     """Returns Top 10 similar words."""

#     clean_word = word.lower().strip()

#     # Access the model from our Global dictionary.

#     model_wv = ml_models.get("wv")

#     if clean_word not in model_wv:
#         raise HTTPException(
#             status_code=404, detail=f"Word {word} not found in Vocabulary."
#         )

#     # Get similarities
#     results = model_wv.most_similar(clean_word, topn=10)

#     return [{"word": w, "Score": f"{s:.2f}"} for w, s in results]


# @app.get("/similarity")
# def get_similarity(w1: str, w2: str):
#     """
#     Docstring for get_similarity

#     :param w1: Description
#     :type w1: str
#     :param w2: Description
#     :type w2: str

#     >> Compare two words directly.
#     """

#     w1, w2 = w1.lower(), w2.lower()

#     model_wv = ml_models.get("wv")

#     if w1 not in model_wv or w2 not in model_wv:
#         raise HTTPException(
#             status_code=404,
#             detail="One of the words is \
#                 missing from the vocabulary.",
#         )

#     score = model_wv.similarity(w1, w2)
#     return {"Word 1": w1, "Word 2": w2, "Similarity": float(score)}

###############################################################################


"""
Enhanced FastAPI application for Word2Vec similarity API
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from gensim.models import Word2Vec


import src.config as config

# --- Response Models ---


class SimilarWordResponse(BaseModel):
    word: str = Field(..., description="The similar word")
    score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)


class SimilarityResponse(BaseModel):
    word1: str = Field(..., description="First word")
    word2: str = Field(..., description="Second word")
    similarity: float = Field(..., description="Similarity score", ge=0.0, le=1.0)


class VocabResponse(BaseModel):
    total_words: int
    sample_words: List[str]


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")

    status: str
    model_loaded: bool
    vocabulary_size: Optional[int] = None


# --- Global Variables to hold the model ---

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup, cleanup on shutdown"""

    # --- Startup Logic ---
    print("Loading model...")
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_FILE}. Run training first."
        )

    # Load the full model
    full_model = Word2Vec.load(str(config.MODEL_FILE))

    # Store only the KeyedVectors (wv) in our dictionary for speed
    ml_models["wv"] = full_model.wv

    print(f"Model loaded successfully! Vocabulary size: {len(ml_models['wv'])}")

    yield  # Control is passed to the application

    # --- Shutdown logic ---
    print("Cleaning up resources...")
    ml_models.clear()

    # Initialize the App with the lifespan manager


app = FastAPI(
    title="Godfather Word2Vec API",
    description="API for finding similar words using Word2Vec trained on The Godfather",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Helper Functions ---


def get_model():
    """Get the loaded model, raise error if not available"""
    model_wv = ml_models.get("wv")
    if model_wv is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )
    return model_wv


# --- API Endpoints ---


@app.get("/", tags=["General"])
def home():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to the Godfather Word2Vec API",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "similar_words": "/similar/{word}",
            "similarity": "/similarity?w1=word1&w2=word2",
            "vocabulary": "/vocabulary",
            "analogy": "/analogy?positive=king,woman&negative=man",
        },
    }


@app.get(
    "/similar/{word}",
    response_model=List[SimilarWordResponse],
    tags=["Word Similarity"],
)
def get_similar_words(
    word: str, topn: int = Query(10, ge=1, le=50, description="Number of similar words")
):
    """
    Returns top N similar words to the given word.

    - **word**: The word to find similarities for
    - **topn**: Number of similar words to return (1-50)
    """
    clean_word = word.lower().strip()
    model_wv = get_model()

    if clean_word not in model_wv:
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary."
        )

    # Get similarities
    results = model_wv.most_similar(clean_word, topn=topn)

    return [{"word": w, "score": float(s)} for w, s in results]


@app.get(
    "/similarity",
    response_model=SimilarityResponse,
    tags=["Word Similarity"],
)
def get_similarity(w1: str, w2: str):
    """
    Compare similarity between two words.

    - **w1**: First word
    - **w2**: Second word

    Returns a similarity score between -1 and 1.
    """
    w1_clean, w2_clean = w1.lower().strip(), w2.lower().strip()
    model_wv = get_model()

    if w1_clean not in model_wv:
        raise HTTPException(status_code=404, detail=f"Word '{w1}' not in vocabulary.")

    if w2_clean not in model_wv:
        raise HTTPException(status_code=404, detail=f"Word '{w2}' not in vocabulary.")

    score = model_wv.similarity(w1_clean, w2_clean)
    return {"word1": w1_clean, "word2": w2_clean, "similarity": float(score)}


@app.get("/vocabulary", response_model=VocabResponse, tags=["General"])
def get_vocabulary(sample_size: int = Query(20, ge=1, le=100)):
    """
    Get vocabulary information.

    - **sample_size**: Number of sample words to return (1-100)
    """
    model_wv = get_model()

    vocab = list(model_wv.index_to_key)
    sample = vocab[:sample_size]

    return {"total_words": len(vocab), "sample_words": sample}


@app.get("/analogy", response_model=List[SimilarWordResponse], tags=["Word Similarity"])
def get_analogy(
    positive: str = Query(..., description="Comma-separated positive words"),
    negative: str = Query(..., description="Comma-separated negative words"),
    topn: int = Query(5, ge=1, le=20),
):
    """
    Solve word analogies. Example: king - man + woman = queen

    - **positive**: Words to add (comma-separated, e.g., "king,woman")
    - **negative**: Words to subtract (comma-separated, e.g., "man")
    - **topn**: Number of results to return
    """
    model_wv = get_model()

    # Parse inputs
    pos_words = [w.strip().lower() for w in positive.split(",") if w.strip()]
    neg_words = [w.strip().lower() for w in negative.split(",") if w.strip()]

    if not pos_words:
        raise HTTPException(
            status_code=400, detail="At least one positive word required."
        )

    # Check all words exist
    all_words = pos_words + neg_words
    for word in all_words:
        if word not in model_wv:
            raise HTTPException(
                status_code=404, detail=f"Word '{word}' not in vocabulary."
            )

    # Compute analogy
    try:
        results = model_wv.most_similar(
            positive=pos_words, negative=neg_words, topn=topn
        )
        return [{"word": w, "score": float(s)} for w, s in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analogy computation failed: {e}")


@app.get("/word-exists/{word}", tags=["General"])
def check_word_exists(word: str):
    """Check if a word exists in the vocabulary"""
    clean_word = word.lower().strip()
    model_wv = get_model()

    return {"word": word, "exists": clean_word in model_wv}


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Health check endpoint — verifies API and model status.
    """
    model_wv = ml_models.get("wv")
    is_loaded = model_wv is not None

    return {
        "status": "ok" if is_loaded else "model_not_loaded",
        "model_loaded": is_loaded,
        "vocabulary_size": len(model_wv) if is_loaded else None,
    }

