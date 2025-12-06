import plotly.graph_objects as go
import plotly.offline as pyo
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from src.config import MODEL_FILE

# --- Load Word2Vec model ---
print("Loading model:", MODEL_FILE)
model = Word2Vec.load(str(MODEL_FILE))  # Ensure Path is converted to string
print("Model loaded!")

# --- Get words and vectors ---
words = list(model.wv.index_to_key)
vectors = model.wv[words]

# --- PCA to 3D ---
pca = PCA(n_components=3)
result = pca.fit_transform(vectors)

# --- Prepare Plotly 3D Scatter ---
fig = go.Figure()

# Scatter points
fig.add_trace(
    go.Scatter3d(
        x=result[:, 0],
        y=result[:, 1],
        z=result[:, 2],
        mode="markers+text",
        text=words,  # Shows the word when hovering
        textposition="top center",
        marker=dict(size=5, color="blue", opacity=0.8),
    )
)

# --- Layout ---
fig.update_layout(
    title="Word2Vec PCA 3D Visualization",
    scene=dict(xaxis_title="PCA 1", yaxis_title="PCA 2", zaxis_title="PCA 3"),
    margin=dict(l=0, r=0, b=0, t=50),
)

# --- Show Interactive Plot ---
pyo.plot(fig, filename="analyze/pca_visual.html")
