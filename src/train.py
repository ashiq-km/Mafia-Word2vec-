from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import src.config as config

def train_model():
    print("Initialize training...")
    
    # Check if data exists
    if not config.PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(f"Processed data not found at {config.PROCESSED_DATA_FILE}. Run preprocess.py first.")

    # Load sentences using LineSentence (memory efficient)
    sentences = LineSentence(str(config.PROCESSED_DATA_FILE))

    print("Training Word2Vec model (this may take a moment)...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )
    
    print("Training finished.")

    # Save the model
    print(f"Saving model to {config.MODEL_FILE}...")
    model.save(str(config.MODEL_FILE))
    
    # Quick Test
    print("Sanity Check:")
    test_word = "godfather"
    if test_word in model.wv:
        print(f"Top 3 similar words to '{test_word}':")
        print(model.wv.most_similar(test_word, topn=3))
    
    print("Done!")

if __name__ == "__main__":
    train_model()