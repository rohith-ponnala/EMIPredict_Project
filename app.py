# app.py — EMIPredict (multi-page streamlit app)
import streamlit as st
import pandas as pd, numpy as np, joblib, json, os, webbrowser
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Setup folders (assume repo root on Streamlit Cloud)
BASE = Path.cwd()
MODELS = BASE / "models"
DATA = BASE / "data"
EDA_DIR = BASE / "eda_outputs"
OUT = BASE / "outputs"
MLRUNS = BASE / "mlruns"

# Page config
st.set_page_config(page_title="EMIPredict", layout="wide", initial_sidebar_state="expanded")

# Helpers
def load_joblib_safe(p: Path):
    try:
        return joblib.load(p)
    except Exception as e:
        return None

def load_json_safe(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def format_number(n):
    try:
        return f"{float(n):,.2f}"
    except Exception:
        return str(n)

# Load models & label mapping when app starts
clf_model = load_joblib_safe(MODELS / "best_classifier.joblib")
reg_model = load_joblib_safe(MODELS / "best_regressor.joblib")
label_map = load_json_safe(MODELS / "label_mapping.json") or {}
label_encoder = None
if (MODELS / "label_encoder.joblib").exists():
    try:
        label_encoder = joblib.load(MODELS / "label_encoder.joblib")
    except Exception:
        label_encoder = None

# Sidebar navigation
st.sidebar.title("EMIPredict")
page = st.sidebar.radio("Go to", ["Home", "Predict", "EDA", "Model Monitor", "Admin"])

# -----------------------
# Home
# -----------------------
if page == "Home":
    st.title("EMIPredict — EMI Eligibility & Max EMI Prediction")
    st.markdown("""
    **What this app does**
    - Predict EMI eligibility (classification) and predicted max monthly EMI (regression).
    - Interactive EDA and model monitoring pages.
    - Admin page to upload / replace models and dataset.
    """)
    st.write("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Classifier loaded", "Yes" if clf_model else "No")
    c2.metric("Regressor loaded", "Yes" if reg_model else "No")
    c3.metric("Data available", "Yes" if (DATA / "processed_final.csv").exists() else "No")
    st.markdown("**Quick start**: go to `Predict` to test, `EDA` to explore data, `Model Monitor` to see results.")

# -----------------------
# Predict (real-time)
# -----------------------
if page == "Predict":
    st.header("Real-time prediction")
    st.write("Enter applicant values below and hit Predict. The app will run the same preprocessing pipeline that the saved models expect (pipeline is embedded in joblib if saved with sklearn pipeline).")

    # Load sample or processed features to auto-create inputs
    proc = DATA / "processed_final.csv"
    sample_df = None
    if proc.exists():
        try:
            sample_df = pd.read_csv(proc, nrows=5)
        except Exception:
            sample_df = None

    # Determine numeric features to show (fallback set)
    if sample_df is not None:
        numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()
        default_features = numeric_cols[:8] if numeric_cols else ["monthly_salary","age","credit_score","current_emi_amount","requested_amount"]
    else:
        default_features = ["monthly_salary","age","credit_score","current_emi_amount","requested_amount"]

    st.subheader("Input features")
    cols = st.columns(4)
    inputs = {}
    for i, f in enumerate(default_features):
        label = f.replace("_"," ").title()
        default = float(sample_df[f].median()) if (sample_df is not None and f in sample_df.columns) else 0.0
        # use sensible ranges for some known fields
        if "age" in f.lower():
            val = cols[i%4].number_input(label, min_value=18, max_value=100, value=int(default) if default else 30)
        else:
            val = cols[i%4].number_input(label, value=float(default))
        inputs[f] = val

    st.write("Optional: Paste a JSON row to predict many at once (one JSON object per line).")
    multi_text = st.text_area("Batch JSON (optional)", height=80)
    do_batch = bool(multi_text.strip())

    if st.button("Predict"):
        if clf_model is None or reg_model is None:
            st.error("Models not loaded — place `best_classifier.joblib` and `best_regressor.joblib` in models/")
        else:
            try:
                if do_batch:
                    # parse lines into dataframe
                    rows = [json.loads(line) for line in multi_text.strip().splitlines() if line.strip()]
                    X = pd.DataFrame(rows)
                else:
                    X = pd.DataFrame([inputs])
                # predictions
                clf_preds = clf_model.predict(X)
                reg_preds = reg_model.predict(X).astype(float)

                # map classification back to labels
                def decode_label(v):
                    s = str(int(v)) if (isinstance(v,(int,np.integer)) or (isinstance(v,str) and v.isdigit())) else str(v)
                    if s in label_map:
                        return label_map[s]
                    try:
                        if label_encoder is not None:
                            return label_encoder.inverse_transform([v])[0]
                    except Exception:
                        pass
                    return str(v)

                if len(X) == 1:
                    st.success(f"EMI Eligibility → **{decode_label(clf_preds[0])}**")
                    st.info(f"Predicted max monthly EMI → **{format_number(reg_preds[0])}**")
                else:
                    out = X.copy()
                    out["pred_eligibility"] = [decode_label(p) for p in clf_preds]
                    out["pred_max_monthly_emi"] = reg_preds
                    st.write(out)
            except Exception as e:
                st.error("Prediction failed — likely feature mismatch. Error: " + str(e))
                st.write("Tip: ensure the model pipeline's feature names/types match. Use Admin to upload a compatible processed_final.csv sample.")

# -----------------------
# EDA (interactive visualizations)
# -----------------------
if page == "EDA":
    st.header("Interactive EDA")
    proc = DATA / "processed_final.csv"
    if not proc.exists():
        st.warning("processed_final.csv not found in data/. Run notebook preprocessing.")
    else:
        df = pd.read_csv(proc, low_memory=False)
        st.markdown(f"Dataset: {proc.name} — rows: {len(df):,}, columns: {df.shape[1]}")
        # Quick summary
        with st.expander("Show data sample"):
            st.dataframe(df.sample(min(200, len(df))).reset_index(drop=True))

        # Interactive widgets for EDA
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("1. Numeric scatter / correlation explorer")
        colx, coly = st.columns(2)
        xcol = colx.selectbox("X axis", numeric_cols, index=0)
        ycol = coly.selectbox("Y axis", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
        fig = px.scatter(df.sample(min(5000,len(df))), x=xcol, y=ycol, color=df['emi_eligibility'] if 'emi_eligibility' in df.columns else None,
                         title=f"{xcol} vs {ycol}", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("2. Correlation heatmap (top numeric columns)")
        if len(numeric_cols) > 1:
            topn = st.slider("Top N numeric columns", min_value=5, max_value=min(30,len(numeric_cols)), value=min(12,len(numeric_cols)))
            top_vars = df[numeric_cols].var().sort_values(ascending=False).head(topn).index.tolist()
            corr = df[top_vars].corr()
            fig2 = px.imshow(corr, text_auto=False, title="Correlation heatmap")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("3. Categorical counts by target")
        if cat_cols and 'emi_eligibility' in df.columns:
            cat = st.selectbox("Categorical column", [c for c in cat_cols if df[c].nunique() < 50], index=0)
            tab = pd.crosstab(df[cat], df['emi_eligibility'], normalize='index')
            fig3 = px.bar(df.sample(min(20000,len(df))).groupby([cat,'emi_eligibility']).size().reset_index(name='count'),
                          x=cat, y='count', color='emi_eligibility', title=f"{cat} by EMI eligibility")
            st.plotly_chart(fig3, use_container_width=True)

        # show saved EDA PNGs as fallback/documents
        st.subheader("Saved EDA images")
        if EDA_DIR.exists():
            images = sorted(list(EDA_DIR.glob("*.png")))
            if images:
                for im in images:
                    st.image(str(im), caption=im.name, use_column_width=True)
            else:
                st.info("No saved EDA PNGs found in eda_outputs/")

# -----------------------
# Model Monitor & MLflow
# -----------------------
if page == "Model Monitor":
    st.header("Model performance & MLflow")
    # show outputs JSON
    st.subheader("Model results (outputs/)")
    clf_out = OUT / "classification_results.json"
    reg_out = OUT / "regression_results.json"
    if clf_out.exists():
        try:
            cr = load_json_safe(clf_out)
            dfc = pd.DataFrame(cr)
            st.write("Classification summary:")
            st.dataframe(dfc)
            # small bar chart f1 & accuracy
            st.plotly_chart(px.bar(dfc, x='name', y=['accuracy','f1'], barmode='group', title="Classifier metrics"), use_container_width=True)
        except Exception as e:
            st.write("Failed to read classification_results.json:", e)
    else:
        st.info("No classification_results.json found (run notebook training).")

    if reg_out.exists():
        try:
            rr = load_json_safe(reg_out)
            dfr = pd.DataFrame(rr)
            st.write("Regression summary:")
            st.dataframe(dfr)
            st.plotly_chart(px.bar(dfr, x='name', y=['RMSE','MAE'], barmode='group', title="Regressor metrics"), use_container_width=True)
        except Exception as e:
            st.write("Failed to read regression_results.json:", e)
    else:
        st.info("No regression_results.json found (run notebook training).")

    st.subheader("MLflow runs (if mlruns/ present)")
    if MLRUNS.exists():
        # list experiments by reading mlruns dir (simple)
        exps = [p.name for p in MLRUNS.iterdir() if p.is_dir()]
        st.write("Detected mlruns experiments:", exps)
        st.markdown("**Open MLflow UI (if you have an MLflow server running)**")
        mlflow_ui_url = st.text_input("MLflow UI URL (e.g., http://<host>:5000)", value="")
        if mlflow_ui_url:
            if st.button("Open MLflow UI"):
                try:
                    webbrowser.open(mlflow_ui_url)
                    st.success("Opening MLflow UI in new tab (if allowed by environment).")
                except Exception:
                    st.write("Please open the URL manually:", mlflow_ui_url)
    else:
        st.info("mlruns/ not found. MLflow logs are saved in the notebook folder when training. To see MLflow UI, run the notebook and start `mlflow ui` with backend-store pointing at mlruns/")

# -----------------------
# Admin
# -----------------------
if page == "Admin":
    st.header("Admin: upload models / dataset / housekeeping")
    st.markdown("Use this page to upload models or a processed sample dataset, or to remove older models to save space.")
    st.subheader("Upload models (joblib)")
    uploaded = st.file_uploader("Upload one or more joblib files (classifier/regressor/encoder/json)", accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            dest = MODELS / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            st.success(f"Saved {uf.name} → models/")
    st.write("---")
    st.subheader("Upload processed sample (processed_final.csv)")
    uploaded_data = st.file_uploader("Upload processed_final.csv", type=["csv"])
    if uploaded_data:
        dest = DATA / "processed_final.csv"
        with open(dest, "wb") as f:
            f.write(uploaded_data.getbuffer())
        st.success("Saved processed_final.csv to data/")
    st.write("---")
    st.subheader("Cleanup old models (keep only best files)")
    if st.button("Cleanup models folder (keep best_classifier, best_regressor, label_encoder, label_mapping)"):
        keep = {"best_classifier.joblib","best_regressor.joblib","label_encoder.joblib","label_mapping.json"}
        removed = []
        for f in MODELS.glob("*"):
            if f.name not in keep:
                try:
                    f.unlink()
                    removed.append(f.name)
                except Exception:
                    pass
        st.write("Removed files:", removed)
        st.success("Cleanup done.")
    st.write("---")
    st.info("Tip: For production, store models in S3 or MLflow registry instead of committing large joblib to GitHub.")
