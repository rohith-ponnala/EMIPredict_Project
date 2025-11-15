# app.py
# Patched EMIPredict Streamlit app (robust input handling + missing-column fixes)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import traceback
from pathlib import Path

# --- Config
BASE = Path.cwd()
MODELS_DIR = BASE / "models"
OUT_DIR = BASE / "outputs"
DATA_DIR = BASE / "data"
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "best_regressor.joblib"
LABEL_MAP = MODELS_DIR / "label_mapping.json"
FEATURES_JSON = MODELS_DIR / "feature_names.json"  # optional helper file if present

st.set_page_config(page_title="EMIPredict", layout="wide")


# --- Utility helpers -------------------------------------------------------
def safe_load_joblib(p):
    try:
        return joblib.load(p)
    except Exception as e:
        st.warning(f"Failed to load model {p}: {e}")
        st.exception(e)
        return None


def get_expected_columns_from_pipeline(pipe):
    """
    Attempt several ways to extract the expected input feature names/order
    from a fitted sklearn pipeline/column transformer. Returns list or None.
    """
    if pipe is None:
        return None
    try:
        # If pipeline has named_steps with a ColumnTransformer
        if hasattr(pipe, "named_steps"):
            for name, step in pipe.named_steps.items():
                # ColumnTransformer common name 'columntransformer' or 'preprocessor'
                if step.__class__.__name__.lower().startswith("columntransf"):
                    # try feature_names_in_
                    if hasattr(step, "feature_names_in_"):
                        return list(getattr(step, "feature_names_in_"))
                    # try get_feature_names_out (ColumnTransformer + sklearn>=1.0)
                    try:
                        out = step.get_feature_names_out()
                        if isinstance(out, (list, np.ndarray)):
                            return list(out)
                    except Exception:
                        pass
                    # else try transformer's feature_names_in_
                    # fall back to concatenating transformer feature input names
                    cols = []
                    try:
                        for transformer_name, transformer, columns in step.transformers_:
                            # columns can be list of names or slice
                            if isinstance(columns, (list, tuple, np.ndarray)):
                                cols.extend(list(columns))
                            elif isinstance(columns, str):
                                cols.append(columns)
                    except Exception:
                        pass
                    if cols:
                        return cols
        # otherwise try pipeline.feature_names_in_ (scikit-learn 1.0+)
        if hasattr(pipe, "feature_names_in_"):
            return list(getattr(pipe, "feature_names_in_"))
    except Exception:
        pass
    return None


def load_feature_list_from_json(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return None


def prepare_input_dataframe(user_df, expected_cols, column_types_hint=None):
    """
    Ensure the DataFrame has all expected_cols. For missing columns add defaults:
    - numeric -> 0
    - categorical -> 'missing'
    If we have a hint of which columns are numeric/categorical (column_types_hint dict),
    use it. Otherwise we try to infer from current user_df / expected patterns.
    Returns a DataFrame with columns in expected order and safe dtypes.
    """
    df = user_df.copy(deep=True)
    # Coerce everything to consistent representation first:
    # For any column present, coerce numeric-like to numeric
    for c in df.columns:
        # if column_types_hint says numeric, coerce numeric
        try:
            if column_types_hint and column_types_hint.get(c) == "numeric":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            else:
                # Try to coerce numeric to numeric if mostly numeric
                temp = pd.to_numeric(df[c], errors="coerce")
                non_na_ratio = temp.notna().sum() / max(1, len(temp))
                if non_na_ratio >= 0.7:
                    df[c] = temp.fillna(0)
                else:
                    # treat as categorical: convert to string and fill missing
                    df[c] = df[c].astype(str).fillna("missing")
        except Exception:
            df[c] = df[c].astype(str).fillna("missing")

    # For missing columns, add defaults
    added = []
    for c in expected_cols:
        if c not in df.columns:
            # default numeric? try to guess by name heuristics
            if column_types_hint and column_types_hint.get(c) == "numeric":
                df[c] = 0
            elif any(k in c.lower() for k in ("amount", "salary", "balance", "score", "emi", "rent", "fees", "age", "years", "depend", "size", "tenure")):
                df[c] = 0
            else:
                df[c] = "missing"
            added.append(c)

    # Reorder columns exactly as expected
    df = df[[c for c in expected_cols]]

    # Final safety pass: numeric columns -> numeric
    # If we have hints, enforce numeric where indicated
    if column_types_hint:
        for c, tp in column_types_hint.items():
            if c in df.columns and tp == "numeric":
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Convert object dtype categorical columns to string to avoid isnan errors
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).fillna("missing")

    return df, added


# --- Load models & get expected features ----------------------------------
st.title("EMIPredict — EMI Eligibility & Max EMI Prediction (patched)")

clf_model = safe_load_joblib(CLASSIFIER_PATH) if CLASSIFIER_PATH.exists() else None
reg_model = safe_load_joblib(REGRESSOR_PATH) if REGRESSOR_PATH.exists() else None

# Try to get expected columns from pipeline / fallback to JSON file
expected_cols_clf = get_expected_columns_from_pipeline(clf_model) if clf_model else None
expected_cols_reg = get_expected_columns_from_pipeline(reg_model) if reg_model else None

# If not found, try JSON file
if expected_cols_clf is None:
    expected_cols_clf = load_feature_list_from_json(MODELS_DIR / "clf_feature_names.json") or load_feature_list_from_json(FEATURES_JSON)
if expected_cols_reg is None:
    expected_cols_reg = load_feature_list_from_json(MODELS_DIR / "reg_feature_names.json") or load_feature_list_from_json(FEATURES_JSON)

# If both still None but one model exists, try unify
if expected_cols_clf is None and expected_cols_reg is not None:
    expected_cols_clf = expected_cols_reg
if expected_cols_reg is None and expected_cols_clf is not None:
    expected_cols_reg = expected_cols_clf

# Minimal UI and dataset input
st.markdown("### Predict — test single records")
st.info("Enter values below to predict (no file required). If you want exact inference matching training, upload processed_final.csv used during training on Admin page.")

# Build a simple form automatically from expected_cols_clf/reg or fallback to a compact default set.
# If we don't have expected cols at all, present a basic manual form for numeric core features.
DEFAULT_FIELDS = [
    "age", "gender", "monthly_salary", "bank_balance", "credit_score",
    "requested_amount", "requested_tenure", "current_emi_amount", "employment_type"
]

form = st.form("predict_form")
# choose field list:
field_list = expected_cols_clf if expected_cols_clf is not None else (expected_cols_reg if expected_cols_reg is not None else DEFAULT_FIELDS)

# Limit fields shown to a reasonable number (if huge), but allow user to upload processed CSV for full
if len(field_list) > 80:
    st.warning("Detected a large number of features. The form will show a compact set; upload processed_final.csv on Admin for full automatic input.")
    # choose top defaults instead:
    visible_fields = DEFAULT_FIELDS
else:
    visible_fields = field_list

# Prepare a dict to collect inputs
input_values = {}
col1, col2 = st.columns([1, 1])
with form:
    # render each visible field with best widget selection heuristics
    for f in visible_fields:
        key = f"input_{f}"
        fname = f.replace("_", " ").capitalize()
        if any(k in f for k in ("gender", "marital", "company", "education", "employment", "house_type", "emi_scenario")):
            input_values[f] = st.selectbox(fname, options=["missing", "Male", "Female", "Other", "Private", "Public", "Self-employed", "Married", "Single"], key=key)
        elif any(k in f for k in ("depend", "family_size", "years", "tenure", "age")):
            input_values[f] = st.number_input(fname, value=0.0, step=1.0, key=key)
        else:
            # default numeric widget
            input_values[f] = st.number_input(fname, value=0.0, step=1.0, key=key)

    submit = st.form_submit_button("Predict")

# Build DataFrame from form inputs (single row)
if submit:
    # create a single-row df from input_values
    user_row = pd.DataFrame([{k: v for k, v in input_values.items()}])
    # If expected columns available, prepare types hint by simple heuristics
    type_hint = {}
    for c in (expected_cols_clf or expected_cols_reg or []):
        if any(k in c.lower() for k in ("amount", "salary", "balance", "score", "emi", "rent", "fees", "age", "years", "depend", "size", "tenure")):
            type_hint[c] = "numeric"
        else:
            type_hint[c] = "categorical"

    # If we lack expected_cols, attempt to use the user's fields only for predictions (best-effort)
    if expected_cols_clf is None:
        st.warning("Expected classifier feature list not found. The app will try to predict using only the fields you entered (may fail if pipeline expects more columns).")
        Xclf = user_row
    else:
        Xclf, added = prepare_input_dataframe(user_row, expected_cols_clf, type_hint)
        if added:
            st.info(f"Added missing classifier columns with defaults: {added}")

    if expected_cols_reg is None:
        Xreg = user_row
    else:
        Xreg, added2 = prepare_input_dataframe(user_row, expected_cols_reg, type_hint)
        if added2:
            st.info(f"Added missing regressor columns with defaults: {added2}")

    # Now run predictions with robust try/except and helpful debug output
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classifier prediction")
        if clf_model is None:
            st.error("Classifier model not loaded.")
        else:
            try:
                # Ensure categorical columns are strings to avoid OneHotEncoder isnan errors
                for c in Xclf.select_dtypes(include=["object"]).columns:
                    Xclf[c] = Xclf[c].astype(str).fillna("missing")
                for c in Xclf.columns:
                    if Xclf[c].dtype == "O":
                        Xclf[c] = Xclf[c].astype(str)
                preds = clf_model.predict(Xclf)
                # Try probabilities
                probs = None
                if hasattr(clf_model, "predict_proba"):
                    try:
                        probs = clf_model.predict_proba(Xclf)
                    except Exception:
                        probs = None
                st.success("Classifier predicted successfully.")
                st.write("Prediction:", preds.tolist())
                if probs is not None:
                    st.write("Probabilities:", probs.tolist())
            except Exception as e:
                st.error("Classifier predict error: see details below.")
                st.exception(e)
                # brief actionable hint
                st.info("Hint: If you get 'columns are missing' or dtype errors, upload the processed_final.csv used during training on Admin page or re-export models from training notebook.")
    with col2:
        st.subheader("Regressor prediction")
        if reg_model is None:
            st.error("Regressor model not loaded.")
        else:
            try:
                for c in Xreg.select_dtypes(include=["object"]).columns:
                    Xreg[c] = Xreg[c].astype(str).fillna("missing")
                rpred = reg_model.predict(Xreg)
                st.success("Regressor predicted successfully.")
                st.write("Prediction (max monthly EMI):", rpred.tolist())
            except Exception as e:
                st.error("Regressor predict error: see details below.")
                st.exception(e)
                st.info("Hint: If you get 'columns are missing' or dtype errors, upload the processed_final.csv used during training on Admin page or re-export models from training notebook.")

# Admin / debug panel (upload processed_final.csv / upload models)
st.markdown("---")
st.header("Admin / debug")
st.markdown("Upload models (joblib) or `processed_final.csv` used during training to ensure consistent features/dtypes.")

# File upload area for processed_final.csv (recommended to upload)
uploaded = st.file_uploader("Upload processed_final.csv (optional, will be used to infer expected features/dtypes)", type=["csv"])
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.success(f"Uploaded processed sample with {len(df_uploaded.columns)} columns.")
        # save a copy for app to use
        save_path = DATA_DIR if DATA_DIR.exists() else BASE
        outp = Path(save_path) / "processed_final.csv"
        df_uploaded.to_csv(outp, index=False)
        st.write("Saved processed_final.csv to", str(outp))
        # Offer to generate features json
        if st.button("Export detected columns as models/feature_names.json"):
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
                json.dump(list(df_uploaded.columns), f, indent=2)
            st.success("Saved feature list to models/feature_names.json. Re-run the app.")
    except Exception as e:
        st.error("Failed to read uploaded CSV.")
        st.exception(e)

# Small debug info box with model & pipeline details
with st.expander("Debug: model info"):
    st.write("Classifier path:", str(CLASSIFIER_PATH))
    st.write("Regressor path:", str(REGRESSOR_PATH))
    st.write("Loaded classifier:", type(clf_model).__name__ if clf_model else None)
    st.write("Loaded regressor:", type(reg_model).__name__ if reg_model else None)
    st.write("Detected classifier expected cols:", expected_cols_clf)
    st.write("Detected regressor expected cols:", expected_cols_reg)

st.markdown("---")
st.write("If prediction still fails: re-run the training notebook and re-export 'best_classifier.joblib' and 'best_regressor.joblib' in the same environment or upload the `processed_final.csv` used for training.")
