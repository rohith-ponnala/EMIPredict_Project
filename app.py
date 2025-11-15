# app.py
"""
Patched EMIPredict app.py - robust input harmonization + safe prediction.

Instructions:
 - Put this file in your project root (where models/ and data/ folders live).
 - Ensure the joblib model files are present:
     models/best_classifier.joblib
     models/best_regressor.joblib
     models/label_encoder.joblib  (optional)
 - Run locally: `streamlit run app.py`
"""

import sys
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# suppress sklearn version compatibility noisy warnings in UI (they're safe here)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Tweak these ---
BASE = Path(".")
MODELS_DIR = BASE / "models"
DATA_DIR = BASE / "data"
CLASSIFIER_FNAME = MODELS_DIR / "best_classifier.joblib"
REGRESSOR_FNAME = MODELS_DIR / "best_regressor.joblib"
LABEL_ENCODER_FNAME = MODELS_DIR / "label_encoder.joblib"  # optional
# List of columns the trained pipeline expects (use the one you trained with)
# This list should reflect your processed_final.csv column names exactly.
EXPECTED_COLUMNS = [
    "school_fees", "house_type", "years_of_employment", "requested_amount", "dependents",
    "groceries_utilities", "existing_loans", "gender", "employment_type", "college_fees",
    "other_monthly_expenses", "max_monthly_emi", "emergency_fund", "emi_scenario",
    "travel_expenses", "education", "monthly_rent", "marital_status", "family_size",
    "requested_tenure", "company_type", "age", "monthly_salary", "bank_balance",
    "credit_score", "current_emi_amount", "employment_years", "requested_tenure"  # duplicate ok
]
# Default values to fill missing features for single-record prediction (safe defaults)
DEFAULT_INPUTS = {
    "age": 30, "monthly_salary": 50000, "bank_balance": 300000, "credit_score": 700,
    "current_emi_amount": 0, "employment_years": 1, "requested_tenure": 12,
    # categorical defaults
    "gender": "Male", "education": "Graduate", "employment_type": "Salaried",
    "marital_status": "Single", "company_type": "Private", "house_type": "Owned",
    # numeric extras
    "requested_amount": 100000, "max_monthly_emi": 0, "dependents": 0,
    "groceries_utilities": 0, "other_monthly_expenses": 0, "monthly_rent": 0,
    "school_fees": 0, "college_fees": 0, "existing_loans": 0, "emergency_fund": 0,
    "travel_expenses": 0, "family_size": 1, "requested_tenure": 12, "emi_scenario": "Eligible"
}
# --------------------

st.set_page_config(page_title="EMIPredict — EMI Eligibility & Max EMI", layout="wide")

# --- Utilities ----------------------------------------------------------------

def safe_load_joblib(path: Path):
    if not path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(path)
        return obj
    except Exception as e:
        st.warning(f"Failed to load {path.name}: {e}")
        return None

def _get_column_transformer_steps(model):
    """
    Recursively find ColumnTransformer.transformers_ inside pipelines and return list of
    (name, transformer, columns).
    """
    from sklearn.compose import ColumnTransformer
    out = []
    def search(obj):
        # pipeline-like objects:
        if hasattr(obj, "named_steps"):
            for step in obj.named_steps.values():
                search(step)
        if hasattr(obj, "steps"):
            for _, step in obj.steps:
                search(step)
        if isinstance(obj, ColumnTransformer):
            # transformers_ is a list of (name, transformer, columns)
            out.extend(obj.transformers_)
    search(model)
    return out

def harmonize_input_dtypes(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Cast columns of X to types compatible with fitted transformers in model.
    - For OneHotEncoder categories that are strings, cast incoming column to str
      (and fillna with empty string).
    - For numeric categories, cast to numeric (coerce -> NaN).
    - Also attempt safe numeric conversions for columns that look numeric.
    """
    import numbers
    from sklearn.preprocessing import OneHotEncoder

    if model is None:
        return X.copy()
    X = X.copy()

    try:
        transformers = _get_column_transformer_steps(model)
    except Exception:
        return X

    # Process each transformer set
    for name, transformer, cols in transformers:
        if transformer is None or transformer == "drop" or transformer == "passthrough":
            continue
        # Get the actual encoder step if pipeline
        t = transformer
        try:
            # pipeline -> get last estimator
            if hasattr(t, "named_steps"):
                vals = list(t.named_steps.values())
                if vals:
                    t = vals[-1]
            elif hasattr(t, "steps"):
                vals = [s for _, s in t.steps]
                if vals:
                    t = vals[-1]
        except Exception:
            pass

        # Handle OneHotEncoder-like
        if hasattr(t, "categories_") or t.__class__.__name__ == "OneHotEncoder":
            cats = getattr(t, "categories_", None)
            # ensure cols is list-like
            if not isinstance(cols, (list, tuple)):
                continue
            for i, col in enumerate(cols):
                if col not in X.columns:
                    continue
                # if categories found, inspect dtype
                if cats is not None and i < len(cats):
                    cat_vals = cats[i]
                    # detect if categories contain strings or numbers
                    has_str = any(isinstance(v, str) for v in cat_vals)
                    has_num = all(isinstance(v, (numbers.Number, np.integer, np.floating)) or v is None or (isinstance(v, float) and np.isnan(v)) for v in cat_vals)
                    if has_str and not has_num:
                        # cast to string
                        X[col] = X[col].astype(str).fillna("")
                    elif has_num and not has_str:
                        X[col] = pd.to_numeric(X[col], errors="coerce")
                    else:
                        # mixed or unknown, cast to str as safe default
                        X[col] = X[col].astype(str).fillna("")
                else:
                    # No categories info - make conservative choices:
                    # If column currently numeric-like, coerce numeric, else cast str
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X[col] = pd.to_numeric(X[col], errors="coerce")
                    else:
                        X[col] = X[col].astype(str).fillna("")
    # small heuristic: coerce columns with numeric-looking names to numeric
    maybe_numeric_suffix = ("age","salary","amount","emi","balance","score","years","tenure","rent")
    for c in X.columns:
        if any(c.lower().endswith(suf) for suf in maybe_numeric_suffix) or any(suf in c.lower() for suf in ("monthly","requested","current","max_")):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def ensure_columns_for_predict(X: pd.DataFrame, expected_columns, defaults=None) -> pd.DataFrame:
    """
    Ensure X contains all expected_columns. If columns are missing, add them
    with default values from defaults dict or NaN.
    Returns DataFrame with columns in expected_columns order (plus extras).
    """
    X = X.copy()
    defaults = defaults or {}
    for col in expected_columns:
        if col not in X.columns:
            if col in defaults:
                X[col] = defaults[col]
            else:
                X[col] = np.nan
    # keep extra columns if present
    return X

# --- Load models --------------------------------------------------------------

clf_model = safe_load_joblib(CLASSIFIER_FNAME)
reg_model = safe_load_joblib(REGRESSOR_FNAME)
label_encoder = safe_load_joblib(LABEL_ENCODER_FNAME)

# --- UI ----------------------------------------------------------------------

st.title("EMIPredict — EMI Eligibility & Max EMI Prediction")

col1, col2 = st.columns([1, 3])
with col1:
    st.header("Quick actions")
    st.markdown("- Upload `processed_final.csv` (optional).")
    st.markdown("- Fill the form and click Predict.")
    st.markdown("- If models missing, upload them to `models/` folder.")

with col2:
    if clf_model is None:
        st.warning("Classifier not loaded. Place models/best_classifier.joblib in repo.")
    else:
        st.success("Classifier loaded")
    if reg_model is None:
        st.warning("Regressor not loaded. Place models/best_regressor.joblib in repo.")
    else:
        st.success("Regressor loaded")
    if label_encoder is None:
        st.info("Label encoder not found (optional).")

st.markdown("---")
st.header("Predict — test single or upload sample")

uploaded = st.file_uploader("Upload processed_final.csv (optional, will prefill inputs)", type=["csv"])
if uploaded is not None:
    sample_df = pd.read_csv(uploaded)
    # Take first row as default
    if not sample_df.empty:
        sample_row = sample_df.iloc[0].to_dict()
    else:
        sample_row = {}
else:
    sample_row = {}

# build input form
st.subheader("Provide input values (leave as default if unknown)")
input_data = {}
# use expected columns but show subset: numeric primary fields + some categorical
form = st.form("input_form")
# numeric fields
num_fields = ["age", "monthly_salary", "bank_balance", "credit_score", "current_emi_amount", "employment_years", "requested_amount", "requested_tenure"]
for f in num_fields:
    default = sample_row.get(f, DEFAULT_INPUTS.get(f, np.nan))
    # ensure numeric default
    try:
        default = float(default) if pd.notna(default) else DEFAULT_INPUTS.get(f, np.nan)
    except Exception:
        default = DEFAULT_INPUTS.get(f, np.nan)
    input_data[f] = form.number_input(f, value=float(default) if not pd.isna(default) else 0.0, format="%.2f")

# categorical quick fields
cat_fields = ["gender", "education", "employment_type", "marital_status", "company_type", "house_type", "emi_scenario"]
for f in cat_fields:
    defv = sample_row.get(f, DEFAULT_INPUTS.get(f, "Unknown"))
    input_data[f] = form.text_input(f, value=str(defv))

submit = form.form_submit_button("Predict")

# When user submits
if submit:
    # convert input dict to DataFrame single-row
    X_user = pd.DataFrame([input_data])
    # ensure all expected columns exist
    X_user = ensure_columns_for_predict(X_user, EXPECTED_COLUMNS, defaults=DEFAULT_INPUTS)

    # Harmonize dtypes for classifier/regressor
    Xclf = X_user.copy()
    Xreg = X_user.copy()
    try:
        Xclf = harmonize_input_dtypes(Xclf, clf_model)
    except Exception as e:
        st.warning(f"Warning while harmonizing classifier input dtypes: {e}")
    try:
        Xreg = harmonize_input_dtypes(Xreg, reg_model)
    except Exception as e:
        st.warning(f"Warning while harmonizing regressor input dtypes: {e}")

    # Select only expected columns (pipeline expects these columns)
    # Keep order - sklearn ColumnTransformer expects same names
    # If pipeline accepts more columns, extras are kept by ensure step.
    Xclf = Xclf.reindex(columns=[c for c in EXPECTED_COLUMNS if c in Xclf.columns])
    Xreg = Xreg.reindex(columns=[c for c in EXPECTED_COLUMNS if c in Xreg.columns])

    # Prediction
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classifier prediction")
        if clf_model is None:
            st.error("Classifier model not loaded.")
        else:
            try:
                # Make sure we pass a DataFrame to pipeline
                preds = clf_model.predict(Xclf)
                if hasattr(clf_model, "predict_proba"):
                    probs = clf_model.predict_proba(Xclf)
                else:
                    probs = None
                st.success("Prediction OK")
                st.write("Predicted label:", preds[0])
                if probs is not None:
                    # show top probabilities if available
                    probs_df = pd.DataFrame(probs, columns=getattr(clf_model, "classes_", list(range(probs.shape[1]))))
                    st.write("Predicted probabilities:")
                    st.dataframe(probs_df.T.rename(columns={0: "prob"}))
            except Exception as e:
                st.error("Classifier predict error: see details below.")
                tb = traceback.format_exc()
                st.code(tb)
                # try a best-effort simplified error message
                st.write("Brief error:", str(e))

    with col2:
        st.subheader("Regressor prediction")
        if reg_model is None:
            st.error("Regressor model not loaded.")
        else:
            try:
                rpred = reg_model.predict(Xreg)
                st.success("Regression OK")
                st.write("Predicted max monthly EMI:", float(rpred[0]))
            except Exception as e:
                st.error("Regressor predict error: see details below.")
                tb = traceback.format_exc()
                st.code(tb)
                st.write("Brief error:", str(e))

st.markdown("---")
st.info("If you still get dtype/encoder errors: ensure models were trained with the same feature set and dtypes. "
        "You can re-run the training notebook to re-export models or provide `processed_final.csv` used for training.")

# Developer helper: quick model status dump (collapsed)
with st.expander("Debug: model info"):
    st.write("Classifier path:", CLASSIFIER_FNAME)
    st.write("Regressor path:", REGRESSOR_FNAME)
    st.write("Loaded classifier:", type(clf_model).__name__ if clf_model is not None else None)
    st.write("Loaded regressor:", type(reg_model).__name__ if reg_model is not None else None)
    try:
        if clf_model is not None and hasattr(clf_model, "named_steps"):
            st.write("Classifier pipeline steps:", list(clf_model.named_steps.keys()))
    except Exception:
        pass

