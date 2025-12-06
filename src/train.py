import mlflow
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import src.config as config


def train_model():
    print("Initialize training...")

    # Set up mlflow experiment

    mlflow.set_experiment("Godfather_Word2Vec")

    # Define Hyperparamters (Moving them to variables makes them easier to log)

    params = {
        "vector_size": 200,
        "window": 7,
        "min_count": 2,
        "workers": 4,
        "epochs": 10,
    }

    # Check if data exists
    if not config.PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found at {config.PROCESSED_DATA_FILE}. Run preprocess.py first."
        )

    # Load sentences using LineSentence (memory efficient)
    sentences = LineSentence(str(config.PROCESSED_DATA_FILE))

    with mlflow.start_run():

        print("Logging paramters to MLflow...")

        mlflow.log_params(params)

        print("Training Word2Vec model...")

        model = Word2Vec(
            sentences=sentences,
            vector_size=params["vector_size"],
            window=params["window"],
            min_count=params["min_count"],
            workers=params["workers"],
            epochs=params["epochs"],
        )

        print("Training finished.")

        # Save the model
        print(f"Saving model to {config.MODEL_FILE}...")
        model.save(str(config.MODEL_FILE))

        # Log the model file to MLflow
        # This saves a copy of model to MLflow, so you never lose it.

        print("Uploading model to MLflow artifacts...")

        mlflow.log_artifact(str(config.MODEL_FILE))

        # Log a sample metric (optional, e.g., vocabulary size)
        vocab_size = len(model.wv)
        mlflow.log_metric("vocab_size", vocab_size)
        print(f"Logged vocab_size: {vocab_size}")

        # Quick Test
        print("Sanity Check:")
        test_word = "godfather"
        if test_word in model.wv:
            print(f"Top 3 similar words to '{test_word}':")
            print(model.wv.most_similar(test_word, topn=3))

        print("Done!")


if __name__ == "__main__":
    train_model()
