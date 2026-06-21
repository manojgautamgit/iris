"""
app.py — Iris Flower Classifier
================================
Streamlit UI for the custom MLP neural network trained on the Iris dataset.
The model is a hand-built NeuralNetwork class (not sklearn), so we
reconstruct the class and forward pass here.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import pickle
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# Model definition
# The pickle was saved with __main__.NeuralNetwork so we must define the
# class before loading.  The forward pass is implemented here so the
# app has no dependency on the original training script.
# ══════════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Stub that matches the pickled object's attribute layout.
    Actual forward-pass logic is reimplemented below so the
    app works without the original training code.
    """
    pass


@st.cache_resource
def load_model():
    """Load and cache the pickle model. Called once per session."""
    with open("iris_mlp_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict(model, sepal_length: float, sepal_width: float,
            petal_length: float, petal_width: float
            ) -> tuple[str, np.ndarray]:
    """
    Run the forward pass and return (predicted_class, probability_array).

    Architecture:  input(4) → ReLU hidden(10) → softmax output(3)
    """
    X      = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    a1     = _relu(X @ model.W1 + model.b1)
    probs  = _softmax(a1 @ model.W2 + model.b2)[0]
    idx    = int(np.argmax(probs))
    labels = ["Setosa", "Versicolor", "Virginica"]
    return labels[idx], probs


# ── CSS ───────────────────────────────────────────────────────────────────────
# Aesthetic: botanical illustration meets scientific data terminal.
# Palette: deep forest green ground, cream paper panels, blush/coral accents.
# Type: Cormorant Garamond (display) + DM Mono (data/labels).

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --forest:  #1a2e1a;
  --leaf:    #2d5a2d;
  --sage:    #7a9e7a;
  --cream:   #f7f3ec;
  --paper:   #fffdf9;
  --petal:   #e8847a;
  --petal2:  #c4614f;
  --gold:    #c9a84c;
  --ink:     #1a1a1a;
  --muted:   #8a8a7a;
  --border:  #ddd8cc;
  --serif:   'Cormorant Garamond', Georgia, serif;
  --mono:    'DM Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--mono) !important;
  background: var(--forest) !important;
  color: var(--cream) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 2.5rem 1.5rem 4rem !important;
  max-width: 680px !important;
}

/* ── Header ── */
.iris-header {
  text-align: center;
  padding: 2rem 0 1.5rem;
  border-bottom: 1px solid rgba(122,158,122,.3);
  margin-bottom: 2rem;
}
.iris-title {
  font-family: var(--serif);
  font-size: 3rem;
  font-weight: 600;
  font-style: italic;
  color: var(--cream);
  line-height: 1;
  letter-spacing: .01em;
}
.iris-subtitle {
  font-family: var(--mono);
  font-size: .6rem;
  letter-spacing: .25em;
  text-transform: uppercase;
  color: var(--sage);
  margin-top: .5rem;
}

/* ── Input panel ── */
.panel {
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 1rem;
  color: var(--ink) !important;
}
.panel-title {
  font-family: var(--mono);
  font-size: .58rem;
  letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  padding-bottom: .5rem;
  margin-bottom: 1.1rem;
}

/* ── Slider overrides ── */
div.stSlider > label {
  font-family: var(--mono) !important;
  font-size: .65rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
div.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--leaf) !important;
  border-color: var(--leaf) !important;
}
div.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
  font-family: var(--mono) !important;
  background: var(--leaf) !important;
  color: var(--cream) !important;
  font-size: .65rem !important;
}
div.stSlider [class*="StyledSliderBar"] {
  background: var(--border) !important;
}
div.stSlider [class*="StyledSliderBarFill"] {
  background: var(--leaf) !important;
}

/* ── Button ── */
div.stButton > button {
  font-family: var(--mono) !important;
  font-size: .72rem !important;
  font-weight: 500 !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
  background: var(--leaf) !important;
  color: var(--cream) !important;
  border: none !important;
  border-radius: 2px !important;
  padding: .65rem 2rem !important;
  width: 100% !important;
  transition: background .2s !important;
}
div.stButton > button:hover {
  background: var(--forest) !important;
  border: 1px solid var(--sage) !important;
}

/* ── Result card ── */
.result-card {
  border-radius: 3px;
  padding: 1.5rem 1.75rem;
  margin-top: 1rem;
  text-align: center;
}
.result-card.setosa     { background: #eef5ee; border: 1.5px solid var(--leaf); }
.result-card.versicolor { background: #fdf0ee; border: 1.5px solid var(--petal2); }
.result-card.virginica  { background: #fdf8ee; border: 1.5px solid var(--gold); }

.result-label {
  font-family: var(--mono);
  font-size: .55rem;
  letter-spacing: .25em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .3rem;
}
.result-name {
  font-family: var(--serif);
  font-size: 2.6rem;
  font-weight: 600;
  font-style: italic;
  line-height: 1;
}
.result-name.setosa     { color: var(--leaf); }
.result-name.versicolor { color: var(--petal2); }
.result-name.virginica  { color: #8a6a20; }

.result-scientific {
  font-family: var(--serif);
  font-style: italic;
  font-size: .95rem;
  color: var(--muted);
  margin-top: .2rem;
}

/* ── Probability bars ── */
.prob-section {
  margin-top: 1.2rem;
  text-align: left;
}
.prob-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 6px 0;
}
.prob-lbl {
  font-family: var(--mono);
  font-size: .6rem;
  width: 90px;
  color: var(--ink);
  letter-spacing: .04em;
}
.prob-track {
  flex: 1;
  height: 7px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.prob-fill-0 { height:100%; background:var(--leaf);   border-radius:2px; }
.prob-fill-1 { height:100%; background:var(--petal2); border-radius:2px; }
.prob-fill-2 { height:100%; background:var(--gold);   border-radius:2px; }
.prob-pct {
  font-family: var(--mono);
  font-size: .6rem;
  color: var(--muted);
  width: 38px;
  text-align: right;
}

/* ── Reference table ── */
.ref-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--mono);
  font-size: .62rem;
  margin-top: .5rem;
}
.ref-table th {
  color: var(--muted);
  letter-spacing: .15em;
  text-transform: uppercase;
  font-weight: 400;
  padding: .3rem .5rem;
  border-bottom: 1px solid rgba(122,158,122,.3);
  text-align: left;
}
.ref-table td {
  padding: .3rem .5rem;
  color: var(--sage);
  border-bottom: 1px solid rgba(122,158,122,.1);
}
.ref-table td:first-child { color: var(--cream); }

/* ── Warning note ── */
.note {
  background: rgba(201,168,76,.08);
  border-left: 3px solid var(--gold);
  padding: .6rem .9rem;
  border-radius: 0 2px 2px 0;
  font-size: .62rem;
  color: var(--gold);
  margin-top: 1rem;
  line-height: 1.7;
}

hr { border-color: rgba(122,158,122,.2) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="iris-header">
  <div class="iris-title">Iris Classifier</div>
  <div class="iris-subtitle">MLP Neural Network · 4 features → 3 species</div>
</div>
""", unsafe_allow_html=True)

# ── Input sliders ─────────────────────────────────────────────────────────────

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Flower measurements (cm)</div>',
            unsafe_allow_html=True)

# Typical Iris measurement ranges
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length", 4.0, 8.0, 5.8, 0.1, key="sl")
    sepal_width  = st.slider("Sepal width",  2.0, 4.5, 3.0, 0.1, key="sw")
with col2:
    petal_length = st.slider("Petal length", 1.0, 7.0, 3.7, 0.1, key="pl")
    petal_width  = st.slider("Petal width",  0.1, 2.5, 1.2, 0.1, key="pw")

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────

predict_btn = st.button("Classify flower →")

# ── Result ────────────────────────────────────────────────────────────────────

if predict_btn or True:   # show live as sliders change
    label, probs = predict(model, sepal_length, sepal_width,
                           petal_length, petal_width)

    label_lower = label.lower()
    scientific  = {
        "Setosa":     "Iris setosa",
        "Versicolor": "Iris versicolor",
        "Virginica":  "Iris virginica",
    }
    species_labels = ["Setosa", "Versicolor", "Virginica"]
    colors         = ["0", "1", "2"]

    # ── Result card ──────────────────────────────────────────────────────────
    prob_bars = ""
    for i, (sp, p) in enumerate(zip(species_labels, probs)):
        width    = f"{p * 100:.1f}%"
        is_pred  = "font-weight:500;" if sp == label else ""
        prob_bars += f"""
        <div class="prob-row">
          <span class="prob-lbl" style="{is_pred}">{sp}</span>
          <div class="prob-track">
            <div class="prob-fill-{i}" style="width:{width}"></div>
          </div>
          <span class="prob-pct">{p*100:.1f}%</span>
        </div>"""

    st.markdown(f"""
    <div class="result-card {label_lower}">
      <div class="result-label">Predicted species</div>
      <div class="result-name {label_lower}">{label}</div>
      <div class="result-scientific">{scientific[label]}</div>
      <div class="prob-section">{prob_bars}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Reference ranges ──────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("📋  Typical measurement ranges by species"):
    st.markdown("""
    <table class="ref-table">
      <tr>
        <th>Species</th>
        <th>Sepal L</th>
        <th>Sepal W</th>
        <th>Petal L</th>
        <th>Petal W</th>
      </tr>
      <tr><td>Setosa</td>
          <td>4.3 – 5.8</td><td>2.3 – 4.4</td>
          <td>1.0 – 1.9</td><td>0.1 – 0.6</td></tr>
      <tr><td>Versicolor</td>
          <td>4.9 – 7.0</td><td>2.0 – 3.4</td>
          <td>3.0 – 5.1</td><td>1.0 – 1.8</td></tr>
      <tr><td>Virginica</td>
          <td>4.9 – 7.9</td><td>2.2 – 3.8</td>
          <td>4.5 – 6.9</td><td>1.4 – 2.5</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ── Model info ────────────────────────────────────────────────────────────────

with st.expander("⚙  Model details"):
    st.markdown(f"""
    <table class="ref-table">
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Architecture</td><td>4 → 10 (ReLU) → 3 (Softmax)</td></tr>
      <tr><td>Input features</td><td>Sepal length, Sepal width, Petal length, Petal width</td></tr>
      <tr><td>Output classes</td><td>Setosa, Versicolor, Virginica</td></tr>
      <tr><td>Learning rate</td><td>{model.learning_rate}</td></tr>
      <tr><td>Epochs trained</td><td>{model.epochs}</td></tr>
      <tr><td>Final train accuracy</td><td>{model.accuracy_history[-1]:.1%}</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note">
      ⚠  This model reached ~36% training accuracy — just above random chance (33%).
      It was trained for only 200 epochs at lr=0.01 and may not have fully converged.
      Predictions are shown as-is from the original pickle file.
      For better accuracy, retrain with more epochs or a higher learning rate.
    </div>
    """, unsafe_allow_html=True)
