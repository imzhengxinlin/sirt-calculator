"""
Y-90 SIRT HCC Lesion Response Calculator
Model F: Radiomics + DVH (Random Forest)
Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import io
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIRT Response Calculator",
    page_icon="🏥",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────
DVH_FEATURES = [
    {"key": "Dmean (Gy)",   "label": "Dmean — Mean absorbed dose (Gy)",           "min": 0.0, "max": 2000.0, "default": 156.16, "help": "Mean dose to the lesion volume"},
    {"key": "D10",          "label": "D10 — Dose covering 10% of volume (Gy)",     "min": 0.0, "max": 5000.0, "default": 257.31, "help": "Dose received by the highest 10% of volume"},
    {"key": "V100 (%)",     "label": "V100 — Volume fraction ≥100 Gy (%)",         "min": 0.0, "max": 100.0,  "default": 49.22,  "help": "Percentage of lesion volume receiving ≥100 Gy"},
    {"key": "V120 (%)",     "label": "V120 — Volume fraction ≥120 Gy (%)",         "min": 0.0, "max": 100.0,  "default": 42.86,  "help": "Percentage of lesion volume receiving ≥120 Gy"},
    {"key": "CoV",          "label": "CoV — Coefficient of variation (SD/Dmean)",  "min": 0.0, "max": 10.0,   "default": 0.48,   "help": "Relative dose heterogeneity; CoV = SD / Dmean"},
]

# DVH-only特征（与DVH_FEATURES相同，单独定义方便管理）
DVH_FEATURES_A = DVH_FEATURES  # 5个DVH参数（同DVH_FEATURES）

RADIO_FEATURES = [
    {"key": "delayed__original__shape__Elongation",
     "label": "Delayed phase · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.8054,
     "help": "Ratio of minor to major axis (equilibrium phase)"},
    {"key": "pre__original__shape__Elongation",
     "label": "Pre-contrast phase · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.8047,
     "help": "Ratio of minor to major axis (pre-contrast)"},
    {"key": "venous__original__shape__Elongation",
     "label": "Portal venous phase · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.8054,
     "help": "Ratio of minor to major axis (portal venous phase)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__LowGrayLevelZoneEmphasis",
     "label": "Arterial phase · LoG σ=1mm · GLSZM · LGLZE",
     "min": 0.0, "max": 1.0, "default": 0.1719,
     "help": "Low gray-level zone emphasis (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__SmallAreaEmphasis",
     "label": "Arterial phase · LoG σ=1mm · GLSZM · Small Area Emphasis",
     "min": 0.0, "max": 1.0, "default": 0.2806,
     "help": "Emphasis of small zone sizes (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__log-sigma-3-0-mm-3D__glszm__SmallAreaLowGrayLevelEmphasis",
     "label": "Arterial phase · LoG σ=3mm · GLSZM · SALGLE",
     "min": 0.0, "max": 1.0, "default": 0.014,
     "help": "Small area low gray-level emphasis (arterial phase, LoG σ=3mm)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__GrayLevelNonUniformityNormalized",
     "label": "Arterial phase · LoG σ=1mm · GLSZM · GLNUN",
     "min": 0.0, "max": 1.0, "default": 0.2511,
     "help": "Gray-level non-uniformity normalized (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__original__glszm__SmallAreaLowGrayLevelEmphasis",
     "label": "Arterial phase · Original · GLSZM · SALGLE",
     "min": 0.0, "max": 1.0, "default": 0.0296,
     "help": "Small area low gray-level emphasis (arterial phase, original)"},
]

ALL_FEATURE_KEYS = [f["key"] for f in DVH_FEATURES] + \
                   [f["key"] for f in RADIO_FEATURES]

# Thresholds — loaded from json files at runtime, fallback to hardcoded
def _load_threshold(fname, fallback):
    try:
        model_dir = Path("models")
        for candidate in [Path("models"),
                          Path("sirt-calculator/models"),
                          Path(__file__).parent / "models"]:
            if candidate.exists():
                model_dir = candidate
                break
        with open(model_dir / fname) as f:
            return json.load(f)["threshold"]
    except Exception:
        return fallback

THRESHOLD_F = _load_threshold("threshold_F.json", 0.5110)
THRESHOLD_A = _load_threshold("threshold_A.json", 0.6460)
THRESHOLD_G = _load_threshold("threshold_G.json", 0.5120)

# ─────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource

def scale_X(scaler_params, X):
    mean  = np.array(scaler_params["mean_"])
    scale = np.array(scaler_params["scale_"])
    return (X - mean) / scale



@st.cache_resource
def load_models():
    """Load both Model F (RF) and Model A (XGB) artifacts."""
    model_dir = Path("models")
    for candidate in [Path("models"),
                      Path("sirt-calculator/models"),
                      Path(__file__).parent / "models"]:
        if candidate.exists():
            model_dir = candidate
            break

    results = {}

    # Model F: Radiomics + DVH (RF, ONNX)
    try:
        import onnxruntime as rt
        sess_f = rt.InferenceSession(str(model_dir / "model_F_RF.onnx"))
        with open(model_dir / "scaler_params.json") as f:
            scaler_f = json.load(f)
        with open(model_dir / "features_F.json") as f:
            features_f = json.load(f)
        results["F"] = {"sess": sess_f, "scaler": scaler_f,
                        "features": features_f, "error": None}
    except Exception as e:
        results["F"] = {"sess": None, "scaler": None,
                        "features": None, "error": str(e)}

    # Model A: DVH only (LR, ONNX)
    try:
        import onnxruntime as rt
        sess_a = rt.InferenceSession(str(model_dir / "model_A_LR.onnx"))
        with open(model_dir / "scaler_params_A.json") as f:
            scaler_a = json.load(f)
        with open(model_dir / "features_A.json") as f:
            features_a = json.load(f)
        results["A"] = {"sess": sess_a, "scaler": scaler_a,
                        "features": features_a, "error": None}
    except Exception as e:
        results["A"] = {"sess": None, "scaler": None,
                        "features": None, "error": str(e)}

    # Model G: Combined (RF, ONNX)
    try:
        import onnxruntime as rt
        sess_g = rt.InferenceSession(str(model_dir / "model_G_RF.onnx"))
        with open(model_dir / "scaler_params_G.json") as f:
            scaler_g = json.load(f)
        with open(model_dir / "features_G.json") as f:
            features_g = json.load(f)
        results["G"] = {"sess": sess_g, "scaler": scaler_g,
                        "features": features_g, "error": None}
    except Exception as e:
        results["G"] = {"sess": None, "scaler": None,
                        "features": None, "error": str(e)}

    return results

def predict_F(sess, scaler_params, features_f, threshold, feature_values):
    """Model F: ONNX RF"""
    missing = [k for k in features_f if k not in feature_values]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    x = np.array([[feature_values[k] for k in features_f]])
    x_scaled = scale_X(scaler_params, x)
    inp  = sess.get_inputs()[0].name
    prob = float(sess.run(["probabilities"],
                          {inp: x_scaled.astype(np.float32)})[0][0, 1])
    return prob, "Responder" if prob >= threshold else "Non-responder"


def predict_A(booster, scaler_params, features_f, threshold, feature_values):
    """Model A: XGBoost Booster JSON"""
    import xgboost as xgb
    missing = [k for k in features_f if k not in feature_values]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    x = np.array([[feature_values[k] for k in features_f]])
    x_scaled = scale_X(scaler_params, x)
    dmat = xgb.DMatrix(x_scaled, feature_names=features_f)
    prob = float(booster.predict(dmat)[0])
    return prob, "Responder" if prob >= threshold else "Non-responder"


def predict_batch_F(sess, scaler_params, features_f, threshold, df):
    missing = [k for k in features_f if k not in df.columns]
    if missing: return None, missing
    X = scale_X(scaler_params, df[features_f].values.astype(float))
    inp   = sess.get_inputs()[0].name
    probs = sess.run(["probabilities"], {inp: X.astype(np.float32)})[0][:, 1]
    out   = df.copy()
    out["Predicted_probability"] = probs.round(4)
    out["Predicted_decision"] = ["Responder" if p >= threshold else "Non-responder"
                                  for p in probs]
    return out, []


def predict_batch_A(booster, scaler_params, features_f, threshold, df):
    import xgboost as xgb
    missing = [k for k in features_f if k not in df.columns]
    if missing: return None, missing
    X = scale_X(scaler_params, df[features_f].values.astype(float))
    dmat  = xgb.DMatrix(X, feature_names=features_f)
    probs = booster.predict(dmat)
    out   = df.copy()
    out["Predicted_probability"] = probs.round(4)
    out["Predicted_decision"] = ["Responder" if p >= threshold else "Non-responder"
                                  for p in probs]
    return out, []

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("Y-90 SIRT — HCC Lesion Response Calculator")
st.markdown("Single-center (UMCG) · 4-month mRECIST response probability · For research use only")
st.divider()

models = load_models()
mF = models["F"]
mA = models["A"]
mG = models["G"]

model_choice = st.radio(
    "**Select prediction model:**",
    ["Model F: Radiomics + DVH (RF, Val-AUC=0.874)",
     "Model A: DVH only (LR, Val-AUC=0.862)",
     "Model G: Combined (RF, Val-AUC=0.866)"],
    horizontal=True,
    help="Model F and A show no significant AUC difference (DeLong p=0.610). "
         "Model A requires only DVH parameters. "
         "Model G adds clinical variables (Portal hypertension, PIVKA-II) to Model F."
)

if "Model A" in model_choice:
    active_thresh = THRESHOLD_A
    active_label  = "Model A: DVH only (LR)"
    model_ready   = mA["error"] is None
    load_err      = mA["error"]
elif "Model G" in model_choice:
    active_thresh = THRESHOLD_G
    active_label  = "Model G: Combined (RF)"
    model_ready   = mG["error"] is None
    load_err      = mG["error"]
else:
    active_thresh = THRESHOLD_F
    active_label  = "Model F: Radiomics + DVH (RF)"
    model_ready   = mF["error"] is None
    load_err      = mF["error"]

st.caption(f"**{active_label}** · Threshold (Youden index, training set) = {active_thresh:.2f}")

if load_err:
    st.warning(f"⚠️ Model files not found: {load_err}")
else:
    st.success("✓ Model loaded successfully", icon="✅")

tab_manual, tab_batch, tab_info = st.tabs(
    ["🖊 Manual input", "📂 Batch CSV", "ℹ️ About"])

# ─── Manual input ────────────────────────────────────────────
with tab_manual:
    st.markdown("#### DVH parameters")
    st.caption(
        "Enter voxel-based dosimetric values from post-treatment Y-90 PET/CT dosimetry. "
        "All values should be entered as original (unstandardized) units. "
        "CoV = SD / Dmean (calculate before entering)."
    )

    dvh_vals = {}
    cols = st.columns(2)
    for i, feat in enumerate(DVH_FEATURES):
        with cols[i % 2]:
            dvh_vals[feat["key"]] = st.number_input(
                feat["label"],
                min_value=float(feat["min"]),
                max_value=float(feat["max"]),
                value=float(feat["default"]),
                step=0.01,
                help=feat["help"],
                key="dvh_" + str(i),
            )

    radio_vals = {}
    if "Model A" not in model_choice:
        st.markdown("#### MRI radiomics features")
        st.caption(
            "Enter original (unstandardized) feature values extracted from "
            "pre-treatment MRI using PyRadiomics (v3.1.0) with identical "
            "preprocessing settings (isotropic resampling 1×1×1 mm³, "
            "z-score normalization within ROI, fixed bin width = 25). "
            "Standardization is applied automatically by the calculator."
        )
        for i, feat in enumerate(RADIO_FEATURES):
            radio_vals[feat["key"]] = st.number_input(
                feat["label"],
                min_value=float(feat["min"]),
                max_value=float(feat["max"]),
                value=float(feat["default"]),
                step=0.001,
                format="%.4f",
                help=feat["help"],
                key=f"radio_{i}",
            )

    # Clinical inputs for Model G
    clin_vals = {}
    if "Model G" in model_choice:
        st.markdown("#### Clinical variables")
        st.caption("Required for Model G (Combined).")
        col1, col2 = st.columns(2)
        with col1:
            clin_vals["Portal hypertension"] = st.selectbox(
                "Portal hypertension",
                options=[0, 1],
                format_func=lambda x: "Yes" if x else "No",
                key="clin_portal"
            )
        with col2:
            clin_vals["PIVKA-II"] = st.number_input(
                "PIVKA-II (mAU/mL)",
                min_value=0.0, max_value=100000.0,
                value=40.0, step=1.0, format="%.1f",
                help="Enter original PIVKA-II value in mAU/mL. Standardization applied automatically.",
                key="clin_pivka"
            )

    st.divider()

    if st.button("▶ Predict response probability",
                 type="primary", use_container_width=True,
                 disabled=not model_ready):
        all_vals = {**dvh_vals, **radio_vals, **clin_vals}
        try:
            if "Model A" in model_choice:
                prob, decision = predict_A(
                    mA["sess"], mA["scaler"], mA["features"],
                    THRESHOLD_A, all_vals)
            elif "Model G" in model_choice:
                prob, decision = predict_F(
                    mG["sess"], mG["scaler"], mG["features"],
                    THRESHOLD_G, all_vals)
            else:
                prob, decision = predict_F(
                    mF["sess"], mF["scaler"], mF["features"],
                    THRESHOLD_F, all_vals)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
        pct = prob * 100

        col1, col2, col3 = st.columns(3)
        thresh_pct = active_thresh * 100
        col1.metric("Probability", f"{pct:.1f}%")
        col2.metric("Threshold",   f"{thresh_pct:.0f}%")
        col3.metric("Decision", "✅ Yes" if decision=="Responder" else "❌ No")

        if decision == "Responder":
            st.success(
                f"🟢 **Responder** — Predicted probability {pct:.1f}% "
                f"exceeds threshold ({thresh_pct:.0f}%). "
                "Lesion is predicted to achieve CR or PR at 4 months post-SIRT.",
                icon="✅"
            )
        else:
            st.error(
                f"🔴 **Non-responder** — Predicted probability {pct:.1f}% "
                f"is below threshold ({thresh_pct:.0f}%). "
                "Lesion is predicted to have SD or PD at 4 months post-SIRT.",
                icon="❌"
            )

        st.progress(prob, text=f"Response probability: {pct:.1f}%")
        st.caption(
            "⚠️ For research purposes only. Not validated for clinical decision-making."
        )

# ─── Batch CSV ───────────────────────────────────────────────
with tab_batch:
    st.markdown(
        "Upload a CSV file with **one row per lesion**. "
        "Column names must exactly match the feature names below."
    )

    with st.expander("Required column names (copy for CSV template)"):
        st.code("\n".join(ALL_FEATURE_KEYS))
        template_df = pd.DataFrame(
            {k: [f["default"]] for f in DVH_FEATURES   for k in [f["key"]]} |
            {k: [f["default"]] for f in RADIO_FEATURES  for k in [f["key"]]}
        )
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            "⬇ Download CSV template",
            csv_template,
            file_name="sirt_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.write(f"Loaded: **{len(df_in)} lesion(s)** · {len(df_in.columns)} columns")
            st.dataframe(df_in.head(3), use_container_width=True)

            if st.button("▶ Run batch prediction",
                         type="primary", use_container_width=True,
                         disabled=not model_ready):
                df_out, missing = predict_batch(model, scaler, features_f, threshold, df_in)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    n_resp = (df_out["Predicted_decision"] == "Responder").sum()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total lesions",  len(df_out))
                    col2.metric("Responders",     n_resp)
                    col3.metric("Response rate",  f"{n_resp/len(df_out)*100:.1f}%")

                    def color_decision(val):
                        color = "#d4edda" if val == "Responder" else "#f8d7da"
                        return f"background-color: {color}"

                    st.dataframe(
                        df_out[["Predicted_probability",
                                "Predicted_decision"]].style.applymap(
                            color_decision, subset=["Predicted_decision"]
                        ),
                        use_container_width=True,
                    )

                    csv_out = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download results CSV",
                        csv_out,
                        file_name="sirt_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ─── About ───────────────────────────────────────────────────
with tab_info:
    st.markdown("""
#### About this calculator

This tool implements three lesion-level predictive models for early tumor response 
after Y-90 selective internal radiation therapy (SIRT) in hepatocellular carcinoma, 
developed and validated at the University Medical Center Groningen (UMCG).

| | Model F | Model A | Model G |
|---|---|---|---|
| Algorithm | Random Forest | Logistic Regression | Random Forest |
| Features | Radiomics + DVH | DVH only | Radiomics + DVH + Clinical |
| n features | 13 | 5 | 15 |
| Val-AUC | 0.874 (0.770–0.956) | 0.862 (0.759–0.947) | 0.866 (0.759–0.954) |
| Threshold | 0.511 | 0.646 | 0.512 |

**DVH features (all models):** Dmean, D10, V100, V120, CoV  
**Radiomics features (Model F & G):** 8 MRI radiomics features (shape + arterial GLSZM)  
**Clinical features (Model G only):** Portal hypertension, PIVKA-II  
**Outcome:** 4-month mRECIST response (CR + PR = responder)

#### Citation
> *Please cite the original manuscript when using this tool.*  
> [Citation will appear here upon publication]

#### Disclaimer
For **research purposes only**. Not validated for clinical decision-making.

#### Contact
Department of Nuclear Medicine & Molecular Imaging, UMCG, Groningen, Netherlands
""")
st.divider()
st.caption(
    "Y-90 SIRT HCC Response Calculator · UMCG · "
    "For research use only · Not for clinical decision-making"
)
