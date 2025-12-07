# ✨ Mafia Word2Vec – Custom NLP Embeddings Project


![god_father_rd](https://github.com/user-attachments/assets/867d20a6-ef4d-4911-bad6-e39e93bc3a32)


This project trains a custom Word2Vec model on The Godfather / Mafia-themed text corpus and provides tools to:

preprocess raw text

train a Word2Vec model

visualize embeddings (2D + 3D PCA)

track experiments using MLflow

log artifacts (model, visualizations)

load + test model interactively

run analysis scripts

This repository is built with a clean, scalable project structure following good ML engineering practices.

``` 🗂️ Project Structure

Mafia-Word2Vec/
│
├── analyze/
│   ├── visualize_pca.py       # PCA 2D/3D visualization
│
├── data/
│   ├── raw/                   # Original text files
│   ├── processed/             # Cleaned / tokenized text
│   ├── models/                # Saved Word2Vec model
│
├── src/
│   ├── config.py              # Paths and constants
│   ├── preprocess.py          # Clean → tokenize → save text
│   ├── train.py               # Train + log model to MLflow
│   ├── test_model.py          # Manual testing of learned vectors
│
├── mlruns/                    # MLflow experiment storage
│
├── requirements.txt           
├── README.md

```


### 🚀 Features
✔ Word2Vec Training

Train a custom embedding model using gensim.

✔ MLflow Integration

Track:

hyperparameters

metrics

artifacts (saved model + PCA visualizations)

✔ PCA Visualization

2D Matplotlib plot

3D Plotly interactive plot (HTML)



✔ Config-based Paths

All file paths are controlled from src/config.py.

✔ Modular Pipeline

Each stage can be executed independently:

preprocess

train

analyze

test



### 🛠️ Installation
Clone the repo


git clone https://github.com/ashiq-km/Mafia-Word2vec-.git
cd Mafia-Word2vec-



Install dependencies


pip install -r requirements.txt



## 📦 1. Preprocess the Raw Text

Place your raw .txt file inside:

data/raw/


Then run:

python -m src.preprocess



This generates:

data/processed/godfather_cleaned.txt


## 🤖 2. Train Word2Vec + Log to MLflow

Start MLflow UI:

mlflow ui


Then in a new terminal:

python -m src.train



This will:

✔ train the model
✔ save godfather_w2v.model
✔ log hyperparameters
✔ log metrics (vocab size)
✔ generate a 3D PCA plot
✔ upload artifacts to MLflow

Artifacts stored at:


mlruns/<experiment_id>/<run_id>/artifacts/


## 📊 3. Visualize Word Embeddings

Run PCA script independently:

python -m analyze.visualize_pca


It produces:

analyze/pca_visual.html


A fully interactive 3D scatter plot.

## 🧪 4. Test the Model


python -m src.test_model


python -m src.test_model

Or load interactively:

from gensim.models import Word2Vec
model = Word2Vec.load("data/models/godfather_w2v.model")
model.wv.most_similar("godfather")



## 📁 Configuration (src/config.py)


All paths are centrally stored:


BASE_DIR
RAW_DATA_FILE
PROCESSED_DATA_FILE
MODEL_FILE



## 🐳 Docker Support (optional)

Build the image:


docker build -t mafia-w2v .


Run the container:

docker run -it mafia-w2v


## 🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you’d like to change.




