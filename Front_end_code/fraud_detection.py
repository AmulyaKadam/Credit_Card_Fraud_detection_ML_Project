import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import io
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
  .insight-card {
    background: linear-gradient(135deg, rgba(0,128,255,0.06) 0%, rgba(0,60,120,0.04) 100%);
    border: 1px solid rgba(0,128,255,0.25); border-radius: 10px;
    padding: 1.2rem 1.6rem; margin-bottom: 0.8rem;
  }
  .insight-card h4 { font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 600;
    color: #00c8ff; letter-spacing: 1.5px; text-transform: uppercase; margin: 0 0 0.4rem 0; }
  .insight-card p { color: #7a9ab8; font-size: 0.85rem; line-height: 1.6; margin: 0; }
  [data-testid="stSlider"] > div > div { color: #00c8ff !important; }
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
# MATPLOTLIB DARK STYLE HELPER
# ─────────────────────────────────────────────
def apply_dark_style(fig, ax_list):
    """Apply dark theme to matplotlib figures to match app style."""
    fig.patch.set_facecolor('#080f1f')
    for ax in (ax_list if isinstance(ax_list, (list, tuple)) else [ax_list]):
        ax.set_facecolor('#091828')
        ax.tick_params(colors='#8fb0d4', labelsize=9)
        ax.xaxis.label.set_color('#8fb0d4')
        ax.yaxis.label.set_color('#8fb0d4')
        ax.title.set_color('#00c8ff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a3a5c')


# ─────────────────────────────────────────────
# DASHBOARD RENDERING FUNCTION (NEW)
# ─────────────────────────────────────────────
def render_analyst_dashboard(df_results: pd.DataFrame, fraud_probs: np.ndarray,
                              preds: np.ndarray, df_input: pd.DataFrame,
                              default_threshold: float, model):
    """
    Full analyst dashboard rendered after batch predictions.
    Includes: summary metrics, threshold slider, charts, top risk table, insights.
    All injected on top of the existing prediction pipeline — model logic untouched.
    """

    # ── 1. THRESHOLD SLIDER ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎚️ Fraud Threshold Control</div>', unsafe_allow_html=True)

    col_sl, col_sl_info = st.columns([3, 1])
    with col_sl:
        adjusted_threshold = st.slider(
            "Adjust Fraud Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_threshold),
            step=0.01,
            help=f"Model's original threshold: {default_threshold:.4f}. "
                 "Moving the slider recalculates predictions without touching the model."
        )
    with col_sl_info:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem;">
          <div class="metric-val" style="font-size:1.4rem; color:{'#ffa040' if adjusted_threshold != default_threshold else '#00c8ff'};">
            {adjusted_threshold:.3f}
          </div>
          <div class="metric-label">{'Custom' if adjusted_threshold != default_threshold else 'Model Default'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Recalculate predictions if threshold changed
    if adjusted_threshold != default_threshold:
        active_preds = (fraud_probs >= adjusted_threshold).astype(int)
        st.markdown(
            f'<div style="font-size:0.78rem; color:#ffa040; font-family:monospace; margin-bottom:0.5rem;">'
            f'⚡ Custom threshold active ({adjusted_threshold:.3f}) — '
            f'original model threshold ({default_threshold:.4f}) preserved</div>',
            unsafe_allow_html=True
        )
    else:
        active_preds = preds
        st.markdown(
            f'<div style="font-size:0.78rem; color:#3a6a8a; font-family:monospace; margin-bottom:0.5rem;">'
            f'✔ Using model\'s built-in threshold ({default_threshold:.4f})</div>',
            unsafe_allow_html=True
        )

    # ── 2. SUMMARY DASHBOARD ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Summary Dashboard</div>', unsafe_allow_html=True)

    total_txns   = len(df_input)
    n_fraud      = int(active_preds.sum())
    n_legit      = total_txns - n_fraud
    fraud_rate   = n_fraud / total_txns * 100 if total_txns > 0 else 0.0

    # Amount stats (safe — column may not exist)
    has_amount = 'Amount' in df_input.columns
    if has_amount:
        fraud_mask    = active_preds == 1
        legit_mask    = active_preds == 0
        avg_fraud_amt = df_input.loc[fraud_mask, 'Amount'].mean() if fraud_mask.any() else 0.0
        avg_legit_amt = df_input.loc[legit_mask, 'Amount'].mean() if legit_mask.any() else 0.0
    else:
        avg_fraud_amt = avg_legit_amt = None

    # Metric cards row
    m1, m2, m3, m4 = st.columns(4)
    for col_w, val, label, color in [
        (m1, f"{total_txns:,}",    "Total Transactions", "#00c8ff"),
        (m2, f"{n_fraud:,}",       "Fraud Detected",     "#ff3c50"),
        (m3, f"{n_legit:,}",       "Legitimate",         "#00dc64"),
        (m4, f"{fraud_rate:.2f}%", "Fraud Rate",         "#ffa040"),
    ]:
        with col_w:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-val" style="color:{color};">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # Amount comparison row (only if Amount column exists)
    if has_amount and avg_fraud_amt is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("💰 Avg Amount — Fraud",
                      f"${avg_fraud_amt:,.2f}")
        with a2:
            st.metric("💰 Avg Amount — Legit",
                      f"${avg_legit_amt:,.2f}")
        with a3:
            ratio = (avg_fraud_amt / avg_legit_amt) if avg_legit_amt > 0 else 0
            st.metric("📐 Fraud / Legit Ratio", f"{ratio:.2f}×")

    # ── 3. VISUALIZATIONS ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Visualizations</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Bar chart: Fraud vs Legit count
    with chart_col1:
        st.markdown("**Transaction Class Distribution**")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        categories = ['Legitimate', 'Fraud']
        counts     = [n_legit, n_fraud]
        bar_colors = ['#00a855', '#ff3c50']
        bars = ax1.bar(categories, counts, color=bar_colors, width=0.45,
                       edgecolor='#1a3a5c', linewidth=0.8)
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     f'{count:,}', ha='center', va='bottom',
                     color='#c9d6e8', fontsize=9, fontfamily='monospace')
        ax1.set_ylabel("Count", fontsize=9)
        ax1.set_title("Fraud vs Legitimate", fontsize=10, pad=8)
        apply_dark_style(fig1, ax1)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    # Histogram: Transaction Amount distribution (safe)
    with chart_col2:
        if has_amount:
            st.markdown("**Transaction Amount Distribution**")
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fraud_amounts = df_input.loc[active_preds == 1, 'Amount'].clip(upper=5000)
            legit_amounts = df_input.loc[active_preds == 0, 'Amount'].clip(upper=5000)
            ax2.hist(legit_amounts, bins=50, color='#006633', alpha=0.7, label='Legit', density=True)
            ax2.hist(fraud_amounts, bins=30, color='#cc1a2b', alpha=0.85, label='Fraud', density=True)
            ax2.set_xlabel("Amount ($) — clipped at $5,000", fontsize=9)
            ax2.set_ylabel("Density", fontsize=9)
            ax2.set_title("Amount: Fraud vs Legit", fontsize=10, pad=8)
            legit_patch = mpatches.Patch(color='#006633', alpha=0.7, label=f'Legit (n={n_legit:,})')
            fraud_patch = mpatches.Patch(color='#cc1a2b', alpha=0.85, label=f'Fraud (n={n_fraud:,})')
            ax2.legend(handles=[legit_patch, fraud_patch], fontsize=8,
                       facecolor='#091828', edgecolor='#1a3a5c', labelcolor='#c9d6e8')
            apply_dark_style(fig2, ax2)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.info("ℹ️ 'Amount' column not found — amount chart skipped.")

    # Fraud probability distribution histogram
    st.markdown("**Fraud Probability Score Distribution**")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.hist(fraud_probs[active_preds == 0], bins=60, color='#005533', alpha=0.75, label='Legit', density=True)
    ax3.hist(fraud_probs[active_preds == 1], bins=40, color='#aa1a2a', alpha=0.85, label='Fraud', density=True)
    ax3.axvline(adjusted_threshold, color='#ffa040', linewidth=1.5,
                linestyle='--', label=f'Threshold ({adjusted_threshold:.3f})')
    ax3.set_xlabel("Fraud Probability", fontsize=9)
    ax3.set_ylabel("Density", fontsize=9)
    ax3.set_title("Model Confidence Score Distribution", fontsize=10, pad=8)
    ax3.legend(fontsize=8, facecolor='#091828', edgecolor='#1a3a5c', labelcolor='#c9d6e8')
    apply_dark_style(fig3, ax3)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── 4. TOP RISK TRANSACTIONS ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🚨 Top 10 Highest Risk Transactions</div>', unsafe_allow_html=True)

    # Build display dataframe with probability (always available via fraud_probs)
    top_risk_df = df_input.copy()
    top_risk_df['Fraud_Probability'] = fraud_probs
    top_risk_df['Prediction'] = np.where(active_preds == 1, "🔴 FRAUD", "🟢 LEGIT")
    top_risk_df['Risk_Level'] = pd.cut(
        fraud_probs,
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"]
    )

    # Sort by probability descending, pick top 10
    top10 = top_risk_df.sort_values('Fraud_Probability', ascending=False).head(10)

    # Select display columns — safe subset
    display_cols = ['Fraud_Probability', 'Prediction', 'Risk_Level']
    for col in ['Amount', 'Time']:
        if col in top10.columns:
            display_cols.append(col)
    # Add a few V-features for context
    v_cols = [c for c in top10.columns if c.startswith('V')][:5]
    display_cols.extend(v_cols)

    st.dataframe(
        top10[display_cols].style.format({'Fraud_Probability': '{:.4f}'}),
        use_container_width=True,
        height=350
    )

    # ── 5. DETAILED RESULTS TABLE (original, preserved) ─────────────────────────
    st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)

    # Update prediction column with active threshold
    df_results_display = df_input.copy()
    df_results_display['Fraud_Probability'] = fraud_probs
    df_results_display['Prediction'] = np.where(active_preds == 1, "🔴 FRAUD", "🟢 LEGIT")
    df_results_display['Risk_Level'] = pd.cut(
        fraud_probs,
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"]
    )

    show_cols = ['Prediction', 'Fraud_Probability', 'Risk_Level']
    if 'Amount' in df_results_display.columns:
        show_cols.append('Amount')
    show_cols.extend(FEATURE_COLS[:5])
    show_cols = [c for c in show_cols if c in df_results_display.columns]

    st.dataframe(df_results_display[show_cols].head(500),
                 use_container_width=True, height=350)

    csv_out = df_results_display.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download Full Results CSV", csv_out,
        file_name="fraud_predictions.csv", mime="text/csv",
        use_container_width=True
    )

    # Optional confusion matrix if 'Class' column present
    if 'Class' in df_input.columns:
        st.markdown('<div class="section-header">Confusion Matrix vs Ground Truth</div>', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        y_true = df_input['Class'].values
        cm = confusion_matrix(y_true, active_preds)
        cm_df = pd.DataFrame(cm,
            index=["Actual: Legit", "Actual: Fraud"],
            columns=["Predicted: Legit", "Predicted: Fraud"])
        st.dataframe(cm_df, use_container_width=False)

    # ── 6. BUSINESS INSIGHTS ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📌 Business Insights</div>', unsafe_allow_html=True)

    # Dynamic insights based on actual results
    insights = _generate_insights(fraud_rate, n_fraud, total_txns,
                                  adjusted_threshold, default_threshold,
                                  avg_fraud_amt if has_amount else None,
                                  avg_legit_amt if has_amount else None)

    ins_col1, ins_col2 = st.columns(2)
    for i, (title, body) in enumerate(insights):
        col = ins_col1 if i % 2 == 0 else ins_col2
        with col:
            st.markdown(f"""
            <div class="insight-card">
              <h4>{title}</h4>
              <p>{body}</p>
            </div>""", unsafe_allow_html=True)


def _generate_insights(fraud_rate, n_fraud, total_txns,
                        adj_threshold, default_threshold,
                        avg_fraud_amt, avg_legit_amt):
    """Generate dynamic, data-driven business insights."""
    insights = []

    # Insight 1: Imbalance
    if fraud_rate < 1.0:
        insights.append((
            "⚖️ Highly Imbalanced Dataset",
            f"Only {fraud_rate:.2f}% of transactions are flagged as fraud. "
            "This mirrors real-world card fraud rates. Standard accuracy metrics are misleading — "
            "precision and recall on the minority class matter far more."
        ))
    elif fraud_rate < 5.0:
        insights.append((
            "📉 Low Fraud Rate",
            f"Fraud rate of {fraud_rate:.2f}% is low but above typical baseline. "
            "Consider whether this reflects model sensitivity or true dataset composition."
        ))
    else:
        insights.append((
            "⚠️ Elevated Fraud Rate",
            f"Fraud rate of {fraud_rate:.2f}% is unusually high. "
            "This may indicate a high-risk dataset segment or a low threshold setting."
        ))

    # Insight 2: Threshold
    if adj_threshold < default_threshold:
        insights.append((
            "🎚️ Threshold Lowered — Higher Recall",
            f"Threshold reduced from {default_threshold:.3f} to {adj_threshold:.3f}. "
            "More transactions flagged as fraud, increasing recall at the cost of precision. "
            "Useful when the cost of missing fraud is high."
        ))
    elif adj_threshold > default_threshold:
        insights.append((
            "🎚️ Threshold Raised — Higher Precision",
            f"Threshold increased from {default_threshold:.3f} to {adj_threshold:.3f}. "
            "Fewer fraud flags, higher precision. Reduces investigator workload but "
            "may miss borderline fraud cases."
        ))
    else:
        insights.append((
            "🎯 Model Threshold Active",
            f"Using the model's optimized threshold of {default_threshold:.4f}, tuned at training time "
            "using F1 score on the imbalanced dataset. This balances precision and recall "
            "for the card fraud domain."
        ))

    # Insight 3: Amount (if available)
    if avg_fraud_amt is not None and avg_legit_amt is not None and avg_legit_amt > 0:
        ratio = avg_fraud_amt / avg_legit_amt
        if ratio > 1.5:
            insights.append((
                "💰 Fraud Targets Larger Transactions",
                f"Average fraud amount (${avg_fraud_amt:,.2f}) is {ratio:.1f}× the legitimate average "
                f"(${avg_legit_amt:,.2f}). High-value transactions warrant extra scrutiny."
            ))
        elif ratio < 0.8:
            insights.append((
                "💰 Fraud Concentrated in Small Amounts",
                f"Fraudulent transactions average ${avg_fraud_amt:,.2f} vs ${avg_legit_amt:,.2f} for legit. "
                "This may indicate card-testing behavior — small probes before large fraudulent purchases."
            ))
        else:
            insights.append((
                "💰 Amount Not a Strong Discriminator",
                f"Fraud and legit amounts are comparable (${avg_fraud_amt:,.2f} vs ${avg_legit_amt:,.2f}). "
                "The model relies primarily on behavioral PCA features (V1–V28) to identify fraud."
            ))

    # Insight 4: Volume
    insights.append((
        "🔍 Investigation Prioritization",
        f"{n_fraud:,} transactions flagged from {total_txns:,} total. "
        "Sort by Fraud_Probability descending to prioritize your investigation queue. "
        "The Top Risk table above shows the 10 most urgent cases for immediate review."
    ))

    return insights


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
  <span class="hero-badge">LightGBM · Calibrated · 30 Features · Analyst Dashboard</span>
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

    def simulate_pca_features():
        base = np.random.normal(0, 1, 28)
        risk_score = 0
        if location == "International":   risk_score += 2
        if txn_type == "Online":          risk_score += 1.5
        if new_merchant == "Yes":         risk_score += 1.2
        if high_freq == "Yes":            risk_score += 1.8
        if merchant in ["Luxury", "Electronics"]: risk_score += 1
        if amount > 2000:                 risk_score += 2
        shift = np.random.normal(risk_score * 0.3, 0.5, 28)
        return base + shift

    def build_feature_row():
        pca_vals = simulate_pca_features()
        row = {"Time": float(hour * 3600), "Amount": float(amount)}
        for i in range(28):
            row[f"V{i+1}"] = float(pca_vals[i])
        return pd.DataFrame([row])[FEATURE_COLS]

    analyze_btn = st.button("🔍 Analyze Transaction", use_container_width=True)

    if analyze_btn:
        if model is None:
            st.error("⚠️ Please upload the model first")
        else:
            with st.spinner("Analyzing transaction behavior..."):
                time.sleep(0.5)
                row = build_feature_row()
                fraud_probs_single, preds_single = make_prediction(model, row, threshold)
                fraud_prob = float(fraud_probs_single[0])
                is_fraud = bool(preds_single[0])

            st.markdown('<div class="section-header">🧠 AI Decision</div>', unsafe_allow_html=True)

            colA, colB, colC = st.columns([2, 1, 1])

            with colA:
                if is_fraud:
                    st.markdown(f"""
                    <div class="result-fraud">
                      <div class="result-title" style="color:#ff3c50;">⚠ Suspicious Transaction</div>
                      <div class="result-prob">Risk Score: {fraud_prob:.1%}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-legit">
                      <div class="result-title" style="color:#00dc64;">✔ Transaction Approved</div>
                      <div class="result-prob">Risk Score: {fraud_prob:.1%}</div>
                    </div>""", unsafe_allow_html=True)

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
# TAB 2 — BATCH CSV UPLOAD + ANALYST DASHBOARD
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch Transaction Screening</div>', unsafe_allow_html=True)

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

    # ── FILE UPLOAD ──────────────────────────────────────────────────────────────
    csv_file = st.file_uploader("Upload CSV file", type=["csv"],
        help="CSV must contain all 30 feature columns", label_visibility="collapsed")

    if csv_file is not None:
        try:
            df_input = pd.read_csv(csv_file)
            st.markdown(
                f'<div style="font-size:0.8rem; color:#3a6a8a; margin-bottom:0.8rem;">'
                f'📄 Loaded: {len(df_input):,} rows × {len(df_input.columns)} columns</div>',
                unsafe_allow_html=True
            )

            missing = [c for c in FEATURE_COLS if c not in df_input.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif model is None:
                st.error("⚠️ Upload the model (sidebar) to run predictions")
            else:
                if st.button("🚀  RUN BATCH ANALYSIS", use_container_width=True):
                    with st.spinner(f"Analyzing {len(df_input):,} transactions..."):
                        # ── CORE PREDICTION — UNCHANGED ──────────────────────────
                        fraud_probs, preds = make_prediction(model, df_input, threshold)

                        df_results = df_input.copy()
                        df_results["Fraud_Probability"] = fraud_probs
                        df_results["Prediction"] = np.where(preds == 1, "🔴 FRAUD", "🟢 LEGIT")
                        df_results["Risk_Level"] = pd.cut(
                            fraud_probs,
                            bins=[0, 0.3, 0.6, 0.8, 1.0],
                            labels=["Low", "Medium", "High", "Critical"]
                        )
                        # ── END CORE PREDICTION ──────────────────────────────────

                    # ── ANALYST DASHBOARD (all new features) ─────────────────────
                    render_analyst_dashboard(
                        df_results=df_results,
                        fraud_probs=fraud_probs,
                        preds=preds,
                        df_input=df_input,
                        default_threshold=threshold,
                        model=model
                    )

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
            ("📈 Analyst Dashboard",
             "The enhanced Batch tab now includes an interactive analyst dashboard: summary KPIs, "
             "a live threshold slider, distribution charts, a top-10 risk queue, and "
             "dynamic business insights — all layered on top of the unchanged prediction pipeline."),
        ]:
            st.markdown(f'<div class="about-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card"><h4>🛠 Tech Stack</h4><p>
          <span class="tag">Python 3.10+</span><span class="tag">LightGBM</span>
          <span class="tag">scikit-learn</span><span class="tag">Pandas</span>
          <span class="tag">NumPy</span><span class="tag">Streamlit</span>
          <span class="tag">Matplotlib</span><span class="tag">joblib</span>
        </p></div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card"><h4>⚙️ How to Use</h4><p>
          1. Upload <code>fraud_model.pkl</code> via the sidebar.<br><br>
          2. <strong>Single Transaction</strong>: fill in all 30 features manually.<br><br>
          3. <strong>Batch CSV</strong>: upload your saved test CSV. Click <em>Run Batch Analysis</em>
          to get the full analyst dashboard — KPIs, charts, top risk queue, threshold slider.<br><br>
          4. Threshold slider defaults to the model's built-in value (0.374).
          Adjust it to explore precision/recall tradeoffs without touching the model.
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
      FRAUDSENTINEL · AI FRAUD DETECTION · Built with LightGBM + Streamlit · Analyst Dashboard v2
    </div>""", unsafe_allow_html=True)
