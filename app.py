# app.py -- Patched, defensive Streamlit app for EMIPredict
# Author: generated patch for Ponnala Rohith
# Requirements: streamlit, pandas, numpy, scikit-learn, joblib, plotly

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import io
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Paths & folders
# -------------------------
BASE = Path.cwd() / "EMIPredict_Project"  # not mandatory; runtime safe
# fallback to repo root if present
if not BASE.exists():
    BASE = Path.cwd()
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
EDA_DIR = BASE / "eda_outputs"
OUT_DIR = BASE / "outputs"

for d in (DATA_DIR, MODELS_DIR, EDA_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utilities: normalize columns, unique names
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and make duplicates unique by suffixing _1, _2..."""
    cols = [str(c).strip() for c in df.columns]
    from collections import Counter
    cnt = Counter(cols)
    seen = Counter()
    new = []
    for c in cols:
        if cnt[c] > 1:
            seen[c] += 1
            new.append(f"{c}_{seen[c]}")
        else:
            new.append(c)
    df.columns = new
    return df

# -------------------------
# Safe plotting helpers
# -------------------------
def safe_sample_df(df, n=5000, random_state=1):
    n = min(max(1, int(n)), max(1, len(df)))
    return df.sample(n=n, random_state=random_state).reset_index(drop=True).copy()

def safe_scatter(df, xcol, ycol, color_col=None, sample_n=5000, height=450, title=None):
    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"Columns not found: {xcol}, {ycol}. Available: {list(df.columns)[:20]}")
    d = safe_sample_df(df, sample_n)
    # coerce numeric for x,y if possible
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[xcol, ycol]).reset_index(drop=True)
    if d.shape[0] == 0:
        raise ValueError("No rows left to plot after cleaning.")
    if color_col and color_col in d.columns:
        d[color_col] = d[color_col].astype(str).fillna("missing")
        fig = px.scatter(d, x=xcol, y=ycol, color=color_col, title=title or f"{xcol} vs {ycol}", height=height)
    else:
        fig = px.scatter(d, x=xcol, y=ycol, title=title or f"{xcol} vs {ycol}", height=height)
    return fig

def safe_bar_counts(df, col, top_k=10, height=450, title=None):
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")
    s = df[col].astype(str).value_counts().nlargest(top_k)
    fig = go.Figure([go.Bar(x=s.index.astype(str), y=s.values)])
    fig.update_layout(title=title or f"Counts of {col}", xaxis_title=col, yaxis_title="count", height=height)
    return fig

def save_fig(fig, name):
    file = EDA_DIR / name
    try:
        fig.write_image(str(file))
    except Exception:
        # fallback: write html
        (EDA_DIR / (name.replace(".png", ".html"))).write_text(fig.to_html(full_html=False))
    return file

# -------------------------
# Model loading / selection
# -------------------------
def find_model_file(kind="clf"):
    """
    Look for best_classifier.joblib / best_regressor.joblib or any *_clf/joblib.
    Preference order: best_*, xgb_*, rf_*, logreg_* etc.
    """
    if kind == "clf":
        pref = ["best_classifier", "xgb_clf", "xgb_clf.joblib", "xgb_clf.joblib", "xgb_clf.joblib", "logreg_clf", "rf_clf", "clf"]
        suffixes = ["joblib", "pkl"]
    else:
        pref = ["best_regressor", "xgb_reg", "rf_reg", "lin_reg", "reg"]
        suffixes = ["joblib", "pkl"]
    # exact names first
    for p in pref:
        for suf in suffixes:
            f = MODELS_DIR / f"{p}.{suf}"
            if f.exists():
                return f
    # any matches
    for f in MODELS_DIR.glob(f"*{kind}*.joblib"):
        return f
    for f in MODELS_DIR.glob(f"*{kind}*.pkl"):
        return f
    # try any joblib
    candidates = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.pkl"))
    return candidates[0] if candidates else None

def load_label_encoder():
    le_file = MODELS_DIR / "label_encoder.joblib"
    if le_file.exists():
        try:
            return joblib.load(le_file)
        except Exception:
            return None
    # fallback try json mapping
    mapf = MODELS_DIR / "label_mapping.json"
    if mapf.exists():
        try:
            mapping = json.loads(mapf.read_text())
            # create dummy mapping object
            class Dummy:
                def __init__(self, m):
                    self._m = m
                    self.classes_ = [m[str(i)] for i in sorted(map(int, m.keys()))]
                def inverse_transform(self, arr):
                    return [self.classes_[int(x)] for x in arr]
            return Dummy(mapping)
        except Exception:
            return None
    return None

def load_model(kind="clf"):
    f = find_model_file(kind= "clf" if kind=="clf" else "reg")
    if not f:
        return None, None
    try:
        model = joblib.load(f)
        return model, f.name
    except Exception as e:
        st.warning(f"Failed to load model {f.name}: {e}")
        return None, None

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="EMIPredict", layout="wide")
st.title("EMIPredict — EMI Eligibility & Max EMI Prediction")

# sidebar nav
page = st.sidebar.radio("Go to", ["Home", "Predict", "EDA", "Model Monitor", "Admin"])

# Try load models at startup (non-blocking)
clf_model, clf_name = load_model("clf")
reg_model, reg_name = load_model("reg")
label_enc = load_label_encoder()

# -------------------------
# Home
# -------------------------
if page == "Home":
    st.header("What this app does")
    st.markdown("""
    - Predict EMI eligibility (classification) and maximum monthly EMI (regression).  
    - Interactive EDA and model monitoring pages.  
    - Admin page to upload / replace models and dataset.
    """)
    st.divider()

    st.subheader("Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classifier loaded", "Yes" if clf_model is not None else "No")
        if clf_model:
            st.caption(f"file: {clf_name}")
    with col2:
        st.metric("Regressor loaded", "Yes" if reg_model is not None else "No")
        if reg_model:
            st.caption(f"file: {reg_name}")
    with col3:
        # check processed sample
        sample = DATA_DIR / "processed_final.csv"
        st.metric("Data available", "Yes" if sample.exists() else "No")
        if sample.exists():
            st.caption(f"{sample.name}")

    st.markdown("Quick start: go to **Predict** to test, **EDA** to explore data, **Model Monitor** to see results.")

# -------------------------
# Predict
# -------------------------
elif page == "Predict":
    st.header("Predict — test single records")
    st.write("Upload a processed sample (processed_final.csv) or use the form below to input features.")
    sample_file = DATA_DIR / "processed_final.csv"
    uploaded = st.file_uploader("Optional: upload processed_final.csv (used to auto-fill fields)", type=["csv"])
    df_sample = None
    if uploaded:
        try:
            df_sample = pd.read_csv(uploaded)
            df_sample = normalize_columns(df_sample)
            st.success("Sample loaded")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    if df_sample is not None:
        cols = df_sample.columns.tolist()
    else:
        # fallback sample columns (best-effort)
        cols = ["age", "monthly_salary", "bank_balance", "credit_score", "current_emi_amount", "employment_years"]

    with st.form("predict_form"):
        st.write("Provide input values (leave as default if unknown):")
        inputs = {}
        for c in cols:
            # numeric guess
            if any(k in c.lower() for k in ["age","salary","balance","emi","credit","amount","years"]):
                inputs[c] = st.number_input(c, value=float(df_sample[c].iloc[0]) if (df_sample is not None and c in df_sample.columns and pd.api.types.is_numeric_dtype(df_sample[c])) else 0.0)
            else:
                inputs[c] = st.text_input(c, value=str(df_sample[c].iloc[0]) if (df_sample is not None and c in df_sample.columns) else "")
        submitted = st.form_submit_button("Predict")
    if submitted:
        X = pd.DataFrame([inputs])
        # Normalize names as app expects
        X = normalize_columns(X)
        # ensure types
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype(str)
        # Classification
        if clf_model is not None:
            try:
                pred = clf_model.predict(X)
                if label_enc is not None:
                    try:
                        pred_label = label_enc.inverse_transform(pred if hasattr(pred, "__iter__") else [pred])[0]
                    except Exception:
                        # if label_enc expects strings
                        pred_label = str(pred[0]) if hasattr(pred, "__iter__") else str(pred)
                else:
                    pred_label = str(pred[0]) if hasattr(pred, "__iter__") else str(pred)
                st.success(f"Classifier prediction: {pred_label}")
            except Exception as e:
                st.error(f"Classifier predict error: {e}\n{traceback.format_exc()}")
        else:
            st.info("No classifier found. Upload on Admin page.")
        # Regression
        if reg_model is not None:
            try:
                rpred = reg_model.predict(X)
                st.success(f"Predicted max monthly EMI: {float(rpred[0]):.2f}")
            except Exception as e:
                st.error(f"Regressor predict error: {e}\n{traceback.format_exc()}")
        else:
            st.info("No regressor found. Upload on Admin page.")

# -------------------------
# EDA
# -------------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis (5 charts)")
    # Load processed sample (either uploaded previously or existing data)
    proc = DATA_DIR / "processed_final.csv"
    if not proc.exists():
        st.warning("No processed_final.csv found in data/. Upload a sample via Admin or push to repo.")
    else:
        try:
            df = pd.read_csv(proc, low_memory=False)
            df = normalize_columns(df)
        except Exception as e:
            st.error(f"Failed to load processed_final.csv: {e}")
            df = None

    if df is not None:
        # 1) Eligibility distribution (if present)
        try:
            if "emi_eligibility" in df.columns:
                fig1 = safe_bar_counts(df, "emi_eligibility", top_k=10, title="EMI eligibility distribution")
                st.plotly_chart(fig1, use_container_width=True)
                save_fig(fig1, "eda_eligibility_distribution.png")
            else:
                st.info("Column 'emi_eligibility' not in dataset.")
        except Exception as e:
            st.error(f"Chart 1 error: {e}")

        # 2) Correlation heatmap of top numeric features
        try:
            num = df.select_dtypes(include=[np.number]).copy()
            if num.shape[1] > 1:
                corr = num.corr().abs()
                # reduce to top 12 variables
                top_vars = corr.var().sort_values(ascending=False).head(12).index.tolist()
                fig2 = px.imshow(num[top_vars].corr(), title="Top numeric correlation (abs)", height=600)
                st.plotly_chart(fig2, use_container_width=True)
                save_fig(fig2, "eda_correlation.png")
            else:
                st.info("Not enough numeric columns for correlation.")
        except Exception as e:
            st.error(f"Chart 2 error: {e}")

        # 3) Age KDE / distribution by eligibility (if present)
        try:
            if "age" in df.columns:
                d = df.copy()
                d["age"] = pd.to_numeric(d["age"], errors="coerce").fillna(-1)
                # use histogram with color
                if "emi_eligibility" in df.columns:
                    fig3 = px.histogram(d[d["age"]>=0], x="age", color="emi_eligibility", nbins=40, title="Age distribution by EMI eligibility")
                else:
                    fig3 = px.histogram(d[d["age"]>=0], x="age", nbins=40, title="Age distribution")
                st.plotly_chart(fig3, use_container_width=True)
                save_fig(fig3, "eda_age_dist.png")
            else:
                st.info("Column 'age' not present")
        except Exception as e:
            st.error(f"Chart 3 error: {e}")

        # 4) Credit score vs max_monthly_emi scatter (if columns present)
        try:
            if set(["credit_score","max_monthly_emi"]).issubset(df.columns):
                fig4 = safe_scatter(df, xcol="credit_score", ycol="max_monthly_emi", color_col="emi_eligibility" if "emi_eligibility" in df.columns else None, sample_n=5000, title="Credit score vs predicted max EMI")
                st.plotly_chart(fig4, use_container_width=True)
                save_fig(fig4, "eda_credit_vs_emi.png")
            else:
                st.info("Need 'credit_score' and 'max_monthly_emi' for chart 4")
        except Exception as e:
            st.error(f"Chart 4 error: {e}")

        # 5) Monthly salary vs bank_balance scatter
        try:
            if set(["monthly_salary","bank_balance"]).issubset(df.columns):
                fig5 = safe_scatter(df, xcol="monthly_salary", ycol="bank_balance", sample_n=5000, title="Salary vs bank balance")
                st.plotly_chart(fig5, use_container_width=True)
                save_fig(fig5, "eda_salary_balance.png")
            else:
                st.info("Need 'monthly_salary' and 'bank_balance' for chart 5")
        except Exception as e:
            st.error(f"Chart 5 error: {e}")

        st.success(f"EDA charts saved to {EDA_DIR.resolve()} (PNG or HTML fallback).")

# -------------------------
# Model Monitor
# -------------------------
elif page == "Model Monitor":
    st.header("Model performance & MLflow (outputs/)")
    # show classification_results.json
    cr = OUT_DIR / "classification_results.json"
    rr = OUT_DIR / "regression_results_fast.json"
    if cr.exists():
        try:
            dfc = pd.read_json(cr)
            st.subheader("Classification summary:")
            st.dataframe(dfc)
            # small bar chart of accuracy/f1
            if not dfc.empty:
                mdf = dfc.melt(id_vars=["name"], value_vars=[c for c in ["accuracy","f1"] if c in dfc.columns], var_name="metric", value_name="value")
                fig = px.bar(mdf, x="name", y="value", color="metric", barmode="group", title="Classifier metrics")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read classification_results.json: {e}")
    else:
        st.info("No classification_results.json found (run training notebook and upload outputs/)")

    if rr.exists():
        try:
            dfr = pd.read_json(rr)
            st.subheader("Regression summary:")
            st.dataframe(dfr)
        except Exception as e:
            st.error(f"Failed to read regression_results_fast.json: {e}")
    else:
        st.info("No regression_results_fast.json found (run training notebook and upload outputs/)")

    # mlruns presence
    if (BASE / "mlruns").exists():
        st.success("mlruns/ present. Use MLflow UI to explore.")
    else:
        st.info("mlruns/ not found. MLflow logs are saved in the notebook folder when training. To see MLflow UI, run the notebook and start mlflow ui with backend-store pointing at mlruns/")

# -------------------------
# Admin
# -------------------------
elif page == "Admin":
    st.header("Admin: upload models / dataset / housekeeping")
    st.write("Use this page to upload models (joblib), processed samples (processed_final.csv) or to cleanup older models.")

    st.subheader("Upload models (joblib)")
    uploaded_models = st.file_uploader("Upload joblib model files (classifier/regressor/encoder/json)", accept_multiple_files=True)
    if uploaded_models:
        for f in uploaded_models:
            try:
                dest = MODELS_DIR / f.name
                bytes_data = f.read()
                dest.write_bytes(bytes_data)
                st.success(f"Saved {f.name} to {MODELS_DIR}")
            except Exception as e:
                st.error(f"Failed to save {f.name}: {e}")

    st.subheader("Upload processed sample (processed_final.csv)")
    uploaded_data = st.file_uploader("Upload processed_final.csv (used by EDA & predict)", type=["csv"], key="data_upload")
    if uploaded_data:
        try:
            dest = DATA_DIR / "processed_final.csv"
            dest.write_bytes(uploaded_data.read())
            st.success("Saved processed_final.csv")
        except Exception as e:
            st.error(f"Failed to save processed_final.csv: {e}")

    st.subheader("Cleanup old models")
    if st.button("Keep only best_{classifier,regressor,label_encoder,label_mapping}"):
        removed = []
        keep_names = {"best_classifier.joblib", "best_regressor.joblib", "label_encoder.joblib", "label_mapping.json"}
        for f in MODELS_DIR.iterdir():
            if f.is_file() and f.name not in keep_names:
                try:
                    f.unlink()
                    removed.append(f.name)
                except Exception:
                    pass
        st.write("Removed:", removed or "Nothing to remove")

    st.markdown("Tip: For production, store models in S3 or MLflow registry instead of committing large joblib files to GitHub.")

# End of app
