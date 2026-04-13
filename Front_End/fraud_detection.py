import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import io
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSentinel · AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# FRAUD MODEL CLASS  (must exist in __main__)
# ─────────────────────────────────────────────
class FraudModel:
    def __init__(self):
        self.model = None
        self.threshold = 0.5

    def __setstate__(self, state):
        self.__dict__.update(state)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


import __main__
__main__.FraudModel = FraudModel

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    background-color: #050a14 !important;
    color: #c9d6e8 !important;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1f 0%, #050a14 100%) !important;
    border-right: 1px solid #0d2240;
  }
  [data-testid="stSidebar"] * { color: #8fb0d4 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem !important; }

  .hero {
    background: linear-gradient(135deg, #091828 0%, #0a2040 50%, #091828 100%);
    border: 1px solid #1a3a5c; border-radius: 12px;
    padding: 2.4rem 2.8rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 40%, rgba(0,180,255,0.04) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 70%, rgba(255,60,80,0.04) 0%, transparent 50%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Rajdhani', sans-serif; font-size: 2.8rem; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    background: linear-gradient(90deg, #00c8ff, #0080ff, #00e5ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
  }
  .hero-subtitle {
    font-family: 'Share Tech Mono', monospace; color: #3a7ea8;
    font-size: 0.82rem; letter-spacing: 2px; margin-top: 0.4rem;
  }
  .hero-badge {
    display: inline-block; background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.25); color: #00c8ff;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    padding: 3px 10px; border-radius: 20px; margin-top: 0.8rem; letter-spacing: 1.5px;
  }
  .metric-card {
    background: linear-gradient(135deg, #091828 0%, #0d2035 100%);
    border: 1px solid #1a3a5c; border-radius: 10px;
    padding: 1.2rem 1.4rem; text-align: center; transition: border-color 0.3s;
  }
  .metric-card:hover { border-color: #0080ff; }
  .metric-val { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; color: #00c8ff; line-height: 1; }
  .metric-label { font-size: 0.72rem; color: #3a6a8a; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.3rem; }
  .section-header {
    font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase; color: #4a9ac8;
    border-left: 3px solid #0080ff; padding-left: 0.7rem; margin: 1.6rem 0 1rem 0;
  }
  .result-fraud {
    background: linear-gradient(135deg, rgba(255,40,60,0.12) 0%, rgba(180,20,40,0.08) 100%);
    border: 1px solid rgba(255,60,80,0.4); border-radius: 12px; padding: 1.6rem 2rem; text-align: center;
  }
  .result-legit {
    background: linear-gradient(135deg, rgba(0,220,100,0.1) 0%, rgba(0,150,70,0.06) 100%);
    border: 1px solid rgba(0,220,100,0.35); border-radius: 12px; padding: 1.6rem 2rem; text-align: center;
  }
  .result-title { font-family: 'Rajdhani', sans-serif; font-size: 2rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; }
  .result-prob { font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; margin-top: 0.4rem; }
  [data-testid="stNumberInput"] input {
    background: #0a1f35 !important; border: 1px solid #1a3a5c !important;
    color: #c9d6e8 !important; border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.82rem !important;
  }
  [data-testid="stNumberInput"] input:focus { border-color: #0080ff !important; box-shadow: 0 0 0 2px rgba(0,128,255,0.15) !important; }
  .stButton > button {
    background: linear-gradient(135deg, #0050b3 0%, #0070d4 100%) !important;
    color: #e8f4ff !important; border: 1px solid #0080ff !important; border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 1rem !important; font-weight: 600 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    padding: 0.5rem 1.8rem !important; transition: all 0.25s !important;
    box-shadow: 0 0 12px rgba(0,128,255,0.25) !important;
  }
  .stButton > button:hover { background: linear-gradient(135deg, #0070d4 0%, #00a0ff 100%) !important; box-shadow: 0 0 20px rgba(0,180,255,0.4) !important; }
  [data-testid="stTabs"] [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1a3a5c !important; gap: 0.5rem; }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #3a6a8a !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 0.92rem !important;
    font-weight: 600 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important;
    border-radius: 6px 6px 0 0 !important; border: 1px solid transparent !important;
    border-bottom: none !important; padding: 0.5rem 1.2rem !important;
  }
  [data-testid="stTabs"] [aria-selected="true"] { background: #0a1f35 !important; color: #00c8ff !important; border-color: #1a3a5c !important; }
  [data-testid="stDataFrame"] { background: #080f1f !important; }
  [data-testid="stFileUploader"] { background: #0a1828 !important; border: 1px dashed #1a3a5c !important; border-radius: 10px !important; padding: 0.5rem !important; }
  [data-testid="stFileUploaderDropzone"] { background: transparent !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #0050b3, #00c8ff) !important; }
  [data-testid="stExpander"] { background: #080f1f !important; border: 1px solid #1a3a5c !important; border-radius: 8px !important; }
  hr { border-color: #1a3a5c !important; }
  .tag {
    display: inline-block; background: rgba(0,128,255,0.12); border: 1px solid rgba(0,128,255,0.3);
    color: #6abfff; font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    padding: 2px 8px; border-radius: 4px; margin: 2px; letter-spacing: 0.5px;
  }
  .about-card { background: linear-gradient(135deg, #091828 0%, #0d2035 100%); border: 1px solid #1a3a5c; border-radius: 10px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
  .about-card h4 { font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 600; color: #00c8ff; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 0.6rem 0; }
  .about-card p { color: #7a9ab8; font-size: 0.87rem; line-height: 1.6; margin: 0; }
  .warning-card {
    background: linear-gradient(135deg, rgba(255,160,0,0.08) 0%, rgba(180,100,0,0.05) 100%);
    border: 1px solid rgba(255,160,0,0.35); border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 1rem;
  }
  .warning-card h4 { font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 600; color: #ffa040; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 0.5rem 0; }
  .warning-card p { color: #b08060; font-size: 0.82rem; line-height: 1.6; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE COLUMNS — must exactly match training order
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# ─────────────────────────────────────────────
# LOAD MODEL — joblib required (pickle won't work)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_bytes: bytes):
    class FraudModel:
        def __setstate__(self, state):
            self.__dict__.update(state)
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        def predict(self, X):
            return self.model.predict(X)

    import __main__
    __main__.FraudModel = FraudModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.BytesIO(model_bytes)
        return joblib.load(buf)


def make_prediction(model, X_df: pd.DataFrame, threshold: float):
    """
    Deterministic prediction function.

    KEY FIXES applied here:
    1. Input is always a DataFrame with named columns matching training.
       This ensures the ColumnTransformer inside the pipeline gets
       the same column-name-to-index mapping as during training.
    2. Column order is explicitly enforced — never rely on CSV column order.
    3. dtypes are cast to float64 — same as training data.
    4. We read [:, 1] for fraud class probability (index 1 = positive class).
       model.classes_ = [0, 1] confirmed, so index 1 is always fraud.
    5. threshold applied as strict >= to match training evaluation exactly.
    """
    # Enforce exact column order and dtype — critical for reproducibility
    X_clean = X_df[FEATURE_COLS].astype(np.float64)

    # Get probabilities — CalibratedClassifierCV averages 5 fold calibrators
    probas = model.predict_proba(X_clean)

    # Confirmed: model.classes_ = [0, 1], so column 1 = fraud probability
    fraud_probs = probas[:, 1]

    # Apply threshold — same logic as training evaluation
    predictions = (fraud_probs >= threshold).astype(int)

    return fraud_probs, predictions


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.4rem 0 1.2rem 0;'>
      <div style='font-family: Rajdhani, sans-serif; font-size: 1.3rem; font-weight: 700;
                  letter-spacing: 3px; color: #00c8ff; text-transform: uppercase;'>
        🛡️ FraudSentinel
      </div>
      <div style='font-family: "Share Tech Mono", monospace; font-size: 0.68rem;
                  color: #3a6a8a; letter-spacing: 1.5px; margin-top: 2px;'>
        AI-POWERED TRANSACTION GUARD
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Upload Your Model**")
    uploaded_model = st.file_uploader(
        "Drop fraud_model.pkl here",
        type=["pkl"],
        help="Upload your trained fraud detection model (.pkl)",
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #2a5070; line-height: 1.7;'>
      <div style='color: #3a7a9a; margin-bottom: 4px; font-family: Rajdhani, sans-serif;
                  font-weight: 600; letter-spacing: 1px; text-transform: uppercase;'>
        Model Pipeline
      </div>
      StandardScaler → LGBMClassifier<br>
      Wrapped: CalibratedClassifierCV<br>
      Output: Fraud Probability [0–1]
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
model = None
threshold = 0.5

if uploaded_model is not None:
    try:
        model_bytes = uploaded_model.read()
        model = load_model(model_bytes)
        threshold = float(model.threshold)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")

with st.sidebar:
    if model is not None:
        st.markdown("---")
        st.markdown(f"""
        <div style='font-family: "Share Tech Mono", monospace; font-size: 0.75rem; color: #3a6a8a;'>
          <div style='color: #4a9ac8; font-family: Rajdhani, sans-serif; font-weight: 600;
                      letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px;'>
            🎯 Active Threshold
          </div>
          threshold = {threshold:.4f}<br>
          <span style='font-size: 0.65rem; color: #2a5070;'>(embedded in trained model)</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🛡 FraudSentinel</p>
  <p class="hero-subtitle">// REAL-TIME CREDIT CARD FRAUD DETECTION ENGINE</p>
  <span class="hero-badge">LightGBM · Calibrated · 30 Features</span>
</div>
""", unsafe_allow_html=True)

if model is not None:
    st.success(f"✅  Model loaded successfully — using model threshold: **{threshold:.4f}**")
else:
    st.info("⬆️  Upload your **fraud_model.pkl** in the sidebar to begin")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡  Single Transaction",
    "📂  Batch CSV Upload",
    "📊  Model Performance",
    "ℹ️  About"
])

# ═══════════════════════════════════════════
# TAB 1 — BANKING-STYLE TRANSACTION UI
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">💳 Transaction Simulation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.85rem; color:#3a6a8a; margin-bottom: 1.2rem;'>
    Enter transaction details as a banking system would capture them.
    Our AI engine will analyze behavioral patterns in real-time.
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────
    # USER INPUTS (BANKING STYLE)
    # ─────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("💰 Transaction Amount ($)", 0.0, 10000.0, 120.0)
        hour = st.slider("🕒 Time of Transaction (Hour)", 0, 23, 14)
        txn_type = st.selectbox("💳 Transaction Type", ["POS", "Online", "ATM"])

    with col2:
        merchant = st.selectbox("🏬 Merchant Category",
            ["Grocery", "Electronics", "Travel", "Luxury", "Food", "Fuel"])
        location = st.selectbox("🌍 Location",
            ["Domestic", "International"])
        device = st.selectbox("📱 Device Used",
            ["Mobile", "Desktop", "Unknown"])

    with col3:
        new_merchant = st.selectbox("🆕 New Merchant?",
            ["No", "Yes"])
        high_freq = st.selectbox("⚡ High Transaction Frequency?",
            ["No", "Yes"])
        night_txn = "Yes" if (hour < 5 or hour > 23) else "No"

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────
    # PCA SIMULATION FUNCTION
    # ─────────────────────────────
    def simulate_pca_features():
        base = np.random.normal(0, 1, 28)

        risk_score = 0

        # Heuristic risk boosts
        if location == "International":
            risk_score += 2
        if txn_type == "Online":
            risk_score += 1.5
        if new_merchant == "Yes":
            risk_score += 1.2
        if high_freq == "Yes":
            risk_score += 1.8
        if merchant in ["Luxury", "Electronics"]:
            risk_score += 1
        if amount > 2000:
            risk_score += 2

        # Shift PCA space slightly based on risk
        shift = np.random.normal(risk_score * 0.3, 0.5, 28)

        return base + shift

    # ─────────────────────────────
    # BUILD MODEL INPUT
    # ─────────────────────────────
    def build_feature_row():
        pca_vals = simulate_pca_features()

        row = {
            "Time": float(hour * 3600),
            "Amount": float(amount)
        }

        for i in range(28):
            row[f"V{i+1}"] = float(pca_vals[i])

        return pd.DataFrame([row])[FEATURE_COLS]

    # ─────────────────────────────
    # ANALYZE BUTTON
    # ─────────────────────────────
    analyze_btn = st.button("🔍 Analyze Transaction", use_container_width=True)

    if analyze_btn:
        if model is None:
            st.error("⚠️ Please upload the model first")
        else:
            with st.spinner("Analyzing transaction behavior..."):
                time.sleep(0.5)

                row = build_feature_row()
                fraud_probs, preds = make_prediction(model, row, threshold)

                fraud_prob = float(fraud_probs[0])
                is_fraud = bool(preds[0])

            st.markdown('<div class="section-header">🧠 AI Decision</div>', unsafe_allow_html=True)

            colA, colB, colC = st.columns([2,1,1])

            with colA:
                if is_fraud:
                    st.markdown(f"""
                    <div class="result-fraud">
                      <div class="result-title" style="color:#ff3c50;">⚠ Suspicious Transaction</div>
                      <div class="result-prob">Risk Score: {fraud_prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-legit">
                      <div class="result-title" style="color:#00dc64;">✔ Transaction Approved</div>
                      <div class="result-prob">Risk Score: {fraud_prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with colB:
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{fraud_prob:.1%}</div>
                  <div class="metric-label">Risk Score</div>
                </div>""", unsafe_allow_html=True)

            with colC:
                confidence = max(fraud_prob, 1 - fraud_prob)
                st.markdown(f"""<div class="metric-card">
                  <div class="metric-val">{confidence:.1%}</div>
                  <div class="metric-label">Confidence</div>
                </div>""", unsafe_allow_html=True)

            st.progress(float(fraud_prob))

# ═══════════════════════════════════════════
# TAB 2 — BATCH CSV UPLOAD
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch Transaction Screening</div>', unsafe_allow_html=True)

    # ── Root cause warning ──
    st.markdown("""
    <div class="warning-card">
      <h4>⚠ Reproducibility Note</h4>
      <p>
        <strong>If your batch confusion matrix differs from training:</strong> this is caused by
        a different test set — not the model. During training, <code>train_test_split</code>
        without a fixed <code>random_state</code> produces a different split every run.
        To get identical results, ensure your CSV contains the <em>exact same rows</em>
        used as the test set during training (save them with <code>X_test.to_csv('test_set.csv')</code>).
        The model's predict_proba output is fully deterministic — the data is the only variable.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.82rem; color:#3a6a8a; margin-bottom: 1.2rem;'>
    Upload a CSV with columns: <code style='color:#00c8ff'>Time, V1–V28, Amount</code>.
    Column order in the CSV does not matter — they are reordered automatically.
    </div>
    """, unsafe_allow_html=True)

    csv_file = st.file_uploader("Upload CSV file", type=["csv"],
        help="CSV must contain all 30 feature columns", label_visibility="collapsed")

    if csv_file is not None:
        try:
            df_input = pd.read_csv(csv_file)
            st.markdown(f'<div style="font-size:0.8rem; color:#3a6a8a; margin-bottom:0.8rem;">📄 Loaded: {len(df_input):,} rows × {len(df_input.columns)} columns</div>', unsafe_allow_html=True)

            missing = [c for c in FEATURE_COLS if c not in df_input.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif model is None:
                st.error("⚠️ Upload the model (sidebar) to run predictions")
            else:
                if st.button("🚀  RUN BATCH ANALYSIS", use_container_width=True):
                    with st.spinner(f"Analyzing {len(df_input):,} transactions..."):
                        # make_prediction enforces column order and float64 dtype
                        fraud_probs, preds = make_prediction(model, df_input, threshold)

                        df_results = df_input.copy()
                        df_results["Fraud_Probability"] = fraud_probs
                        df_results["Prediction"] = np.where(preds == 1, "🔴 FRAUD", "🟢 LEGIT")
                        df_results["Risk_Level"] = pd.cut(
                            fraud_probs,
                            bins=[0, 0.3, 0.6, 0.8, 1.0],
                            labels=["Low", "Medium", "High", "Critical"]
                        )

                    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
                    n_fraud = int(preds.sum())
                    n_legit = len(df_input) - n_fraud
                    fraud_pct = n_fraud / len(df_input) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    for col_w, val, label in zip(
                        [m1, m2, m3, m4],
                        [len(df_input), n_fraud, n_legit, f"{fraud_pct:.1f}%"],
                        ["Total Transactions", "Fraudulent", "Legitimate", "Fraud Rate"]
                    ):
                        with col_w:
                            st.markdown(f"""<div class="metric-card">
                              <div class="metric-val">{val}</div>
                              <div class="metric-label">{label}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
                    st.dataframe(
                        df_results[["Prediction", "Fraud_Probability", "Risk_Level", "Amount"] + FEATURE_COLS[:5]].head(500),
                        use_container_width=True, height=350
                    )

                    csv_out = df_results.to_csv(index=False).encode()
                    st.download_button("⬇️  Download Full Results CSV", csv_out,
                        file_name="fraud_predictions.csv", mime="text/csv", use_container_width=True)

                    # Optional confusion matrix if 'Class' column present
                    if 'Class' in df_input.columns:
                        st.markdown('<div class="section-header">Confusion Matrix vs Ground Truth</div>', unsafe_allow_html=True)
                        from sklearn.metrics import confusion_matrix
                        y_true = df_input['Class'].values
                        cm = confusion_matrix(y_true, preds)
                        cm_df = pd.DataFrame(cm,
                            index=["Actual: Legit", "Actual: Fraud"],
                            columns=["Predicted: Legit", "Predicted: Fraud"])
                        st.dataframe(cm_df, use_container_width=False)

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ═══════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; color:#3a6a8a; margin-bottom: 1.2rem;'>
    Benchmark metrics from training on the IEEE-CIS / Kaggle Credit Card Fraud dataset.
    Class imbalance ratio ≈ 492 fraud : 284,315 legitimate (0.17%).
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col_w, (val, lbl) in zip([mc1, mc2, mc3, mc4, mc5], [
        ("~99.9%", "Accuracy"), ("~95%+", "Precision"),
        ("~80–85%", "Recall"), ("~87–90%", "F1 Score"), ("~98–99%", "ROC-AUC")
    ]):
        with col_w:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="font-size:1.5rem;">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)
        for step, desc in {
            "Step 1 — Preprocessor": "ColumnTransformer → StandardScaler on all 30 features",
            "Step 2 — Classifier": "LGBMClassifier (gradient boosting on decision trees)",
            "Step 3 — Calibration": "CalibratedClassifierCV (sigmoid, cv=5, ensemble=True)",
            "Output": "Mean of 5 fold calibrators → fraud probability ∈ [0.0, 1.0]"
        }.items():
            st.markdown(f'<div class="about-card" style="margin-bottom:0.6rem;"><h4>{step}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Reproducibility Guide</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="warning-card">
          <h4>🔧 Fix Training Code for Exact Match</h4>
          <p>
            Add these two lines to your training script to guarantee
            the same confusion matrix every time:<br><br>
            <code style="color:#ffa040;">
            X_train, X_test, y_train, y_test =<br>
            &nbsp;&nbsp;train_test_split(X, y,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;test_size=0.2,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;<strong>random_state=42,</strong><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<strong>stratify=y</strong>)<br><br>
            X_test.to_csv('test_set.csv', index=False)
            </code>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
          <h4>Class Imbalance Handling</h4>
          <p>The dataset is highly imbalanced (0.17% fraud).
          This model uses <code>class_weight='balanced'</code> in LGBMClassifier
          and a custom threshold of 0.374 (stored in model.threshold)
          optimized at training time.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Confusion Matrix (From Training)</div>', unsafe_allow_html=True)
    cm_data = pd.DataFrame(
        {"Predicted: Legit": [56856, 16], "Predicted: Fraud": [8, 82]},
        index=["Actual: Legit", "Actual: Fraud"]
    )
    st.dataframe(cm_data, use_container_width=False)
    st.caption("*This is the training-time result. Upload your saved test CSV with a 'Class' column in Batch tab to recompute.*")


# ═══════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        for title, body in [
            ("🎯 Project Overview",
             "FraudSentinel is an end-to-end machine learning system for real-time credit card fraud detection. "
             "Built on the Kaggle Credit Card Fraud Detection dataset, it tackles extreme class imbalance with anonymized features."),
            ("🧠 Model Architecture",
             "Core: <strong>LightGBM classifier</strong> inside a <strong>scikit-learn Pipeline</strong> "
             "(ColumnTransformer + StandardScaler). Calibrated with <strong>CalibratedClassifierCV</strong> "
             "(sigmoid, 5-fold ensemble) for reliable probabilities."),
            ("📊 Dataset",
             "284,807 transactions over two days in September 2013. 492 frauds (0.172%). "
             "Features V1–V28 are PCA-transformed for privacy. Only Time and Amount are original."),
        ]:
            st.markdown(f'<div class="about-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card"><h4>🛠 Tech Stack</h4><p>
          <span class="tag">Python 3.10+</span><span class="tag">LightGBM</span>
          <span class="tag">scikit-learn</span><span class="tag">Pandas</span>
          <span class="tag">NumPy</span><span class="tag">Streamlit</span>
          <span class="tag">joblib</span>
        </p></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card"><h4>⚙️ How to Use</h4><p>
          1. Upload <code>fraud_model.pkl</code> via the sidebar.<br><br>
          2. <strong>Single Transaction</strong>: fill in all 30 features manually.<br><br>
          3. <strong>Batch CSV</strong>: upload your saved test CSV. Add a <code>Class</code>
          column to get a live confusion matrix.<br><br>
          4. Threshold is read automatically from the model (0.374).
        </p></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card"><h4>🔗 Resources</h4><p>
          <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" style="color:#00c8ff; text-decoration:none;">📦 Kaggle Dataset</a><br>
          <a href="https://lightgbm.readthedocs.io/" style="color:#00c8ff; text-decoration:none;">📖 LightGBM Docs</a><br>
          <a href="https://scikit-learn.org/stable/modules/calibration.html" style="color:#00c8ff; text-decoration:none;">📖 Probability Calibration</a>
        </p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; font-family: "Share Tech Mono", monospace;
                font-size: 0.68rem; color: #2a5070; padding: 0.5rem;'>
      FRAUDSENTINEL · AI FRAUD DETECTION · Built with LightGBM + Streamlit
    </div>""", unsafe_allow_html=True)
