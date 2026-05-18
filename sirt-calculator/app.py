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
    page_icon="⚕️",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────
DVH_FEATURES = [
    {"key": "Dmean (Gy)",   "label": "Dmean — Mean absorbed dose (Gy)",          "min": 0.0,  "max": 2000.0,     "default": 156.16,      "help": "Mean dose to the lesion volume"},
    {"key": "Dmin (Gy)",    "label": "Dmin — Minimum dose (Gy)",                  "min": 0.0,  "max": 1000.0,     "default": 31.62,       "help": "Minimum dose within lesion"},
    {"key": "SD",           "label": "SD — Dose standard deviation (Gy)",         "min": 0.0,  "max": 2000.0,     "default": 74.84,       "help": "Standard deviation of dose distribution"},
    {"key": "D10",          "label": "D10 — Dose covering 10% of volume (Gy)",    "min": 0.0,  "max": 5000.0,     "default": 257.31,      "help": "Dose received by the highest 10% of volume"},
    {"key": "V100 (%)",     "label": "V100 — Volume fraction ≥100 Gy (%)",        "min": 0.0,  "max": 100.0,      "default": 49.22,       "help": "Percentage of lesion volume receiving ≥100 Gy"},
    {"key": "V120 (%)",     "label": "V120 — Volume fraction ≥120 Gy (%)",        "min": 0.0,  "max": 100.0,      "default": 42.86,       "help": "Percentage of lesion volume receiving ≥120 Gy"},
    {"key": "nCI",          "label": "nCI — Normalized coverage index",            "min": 0.0,  "max": 100000000.0,"default": 2002440.0,   "help": "nCI value from DVH"},
]

# DVH-only特征（与DVH_FEATURES相同，单独定义方便管理）
DVH_FEATURES_A = DVH_FEATURES  # 7个DVH参数完全相同

RADIO_FEATURES = [
    {"key": "delayed__original__shape__Elongation",
     "label": "Equilibrium · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.8054,
     "help": "Ratio of minor to major axis (equilibrium phase)"},
    {"key": "pre__original__shape__Elongation",
     "label": "Pre-contrast · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.8047,
     "help": "Ratio of minor to major axis (pre-contrast)"},
    {"key": "venous__original__shape__Elongation",
     "label": "Portal venous · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.8054,
     "help": "Ratio of minor to major axis (portal venous phase)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__LowGrayLevelZoneEmphasis",
     "label": "Arterial · LoG σ=1mm · GLSZM · LGLZE",
     "min": 0.0, "max": 1.0, "default": 0.1719,
     "help": "Low gray-level zone emphasis (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__SmallAreaEmphasis",
     "label": "Arterial · LoG σ=1mm · GLSZM · Small Area Emphasis",
     "min": 0.0, "max": 1.0, "default": 0.2806,
     "help": "Emphasis of small zone sizes (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__log-sigma-3-0-mm-3D__glszm__SmallAreaLowGrayLevelEmphasis",
     "label": "Arterial · LoG σ=3mm · GLSZM · SALGLE",
     "min": 0.0, "max": 1.0, "default": 0.014,
     "help": "Small area low gray-level emphasis (arterial phase, LoG σ=3mm)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__GrayLevelNonUniformityNormalized",
     "label": "Arterial · LoG σ=1mm · GLSZM · GLNUN",
     "min": 0.0, "max": 1.0, "default": 0.2511,
     "help": "Gray-level non-uniformity normalized (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__original__glszm__SmallAreaLowGrayLevelEmphasis",
     "label": "Arterial · Original · GLSZM · SALGLE",
     "min": 0.0, "max": 1.0, "default": 0.0296,
     "help": "Small area low gray-level emphasis (arterial phase, original)"},
]

ALL_FEATURE_KEYS = [f["key"] for f in DVH_FEATURES] + \
                   [f["key"] for f in RADIO_FEATURES]

THRESHOLD   = 0.5233  # Model F: Radiomics+DVH RF threshold
THRESHOLD_A = 0.3394  # Model A: DVH only XGB threshold

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

    # Model A: DVH only (XGB, JSON)
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(model_dir / "model_A_XGB.json"))
        with open(model_dir / "scaler_params_A.json") as f:
            scaler_a = json.load(f)
        with open(model_dir / "features_A.json") as f:
            features_a = json.load(f)
        results["A"] = {"booster": booster, "scaler": scaler_a,
                        "features": features_a, "error": None}
    except Exception as e:
        results["A"] = {"booster": None, "scaler": None,
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
st.title("⚕️ Y-90 SIRT — HCC Lesion Response Calculator")
st.markdown("Single-center (UMCG) · 4-month mRECIST response probability · For research use only")
st.divider()

models = load_models()
mF = models["F"]
mA = models["A"]

model_choice = st.radio(
    "**Select prediction model:**",
    ["🔬 Model F: Radiomics + DVH (RF, Val-AUC=0.899)",
     "💊 Model A: DVH only (XGB, Val-AUC=0.895)"],
    horizontal=True,
    help="Both models show no significant AUC difference (DeLong p=0.953). "
         "Model A requires only dosimetric parameters."
)
use_model_A = "Model A" in model_choice

if use_model_A:
    active_thresh = THRESHOLD_A
    active_label  = "Model A: DVH only (XGB)"
    model_ready   = mA["error"] is None
    load_err      = mA["error"]
else:
    active_thresh = THRESHOLD
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
    st.caption("Enter voxel-based dosimetric values from post-treatment Y-90 PET/CT dosimetry.")

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

    st.markdown("#### MRI radiomics features")
    st.caption(
        "Extracted from pre-treatment MRI using PyRadiomics (v3.1.0), "
        "z-score standardization applied per training set parameters."
    )

    radio_vals = {}
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

    st.divider()

    if st.button("▶ Predict response probability",
                 type="primary", use_container_width=True,
                 disabled=not model_ready):
        all_vals = {**dvh_vals, **radio_vals}
        # 检查features_f里有哪些key不在all_vals里
        missing_keys = [k for k in features_f if k not in all_vals]
        if missing_keys:
            st.error(f"Feature mismatch: {missing_keys}")
            st.stop()
        try:
            if use_model_A:
                prob, decision = predict_A(
                    mA["booster"], mA["scaler"], mA["features"],
                    THRESHOLD_A, all_vals)
            else:
                prob, decision = predict_F(
                    mF["sess"], mF["scaler"], mF["features"],
                    THRESHOLD, all_vals)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
        pct = prob * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability", f"{pct:.1f}%")
        col2.metric("Threshold",   f"{THRESHOLD*100:.0f}%")
        col3.metric("Decision", "✅ Yes" if decision=="Responder" else "❌ No")

        if decision == "Responder":
            st.success(
                f"🟢 **Responder** — Predicted probability {pct:.1f}% "
                f"exceeds threshold ({THRESHOLD*100:.0f}%). "
                "Lesion is predicted to achieve CR or PR at 4 months post-SIRT.",
                icon="✅"
            )
        else:
            st.error(
                f"🔴 **Non-responder** — Predicted probability {pct:.1f}% "
                f"is below threshold ({THRESHOLD*100:.0f}%). "
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
    #### Model information
    | Item | Details |
    |---|---|
    | Study | Single-center retrospective (UMCG, 2016–2025) |
    | Algorithm | Random Forest (Model F: Radiomics + DVH) |
    | Training set | 134 lesions / 70 patients |
    | Validation set | 78 lesions / 31 patients |
    | Validation AUC | 0.899 (95% CI: reported in manuscript) |
    | Threshold | Youden index on training set |
    | Outcome | 4-month mRECIST (CR+PR = responder) |
    | Radiomics | PyRadiomics v3.1.0, IBSI-compliant |
    | Dosimetry | Y-90 PET/CT voxel-based DVH (MIM Software v7.3.4) |

    #### Feature extraction requirements
    - **MRI**: 3.0T, gadoxetate disodium, 4-phase (pre-contrast, arterial, portal venous, equilibrium)
    - **Segmentation**: 3D Slicer v5.10.0, 3D whole-lesion ROI on all 4 phases
    - **Radiomics**: PyRadiomics v3.1.0, isotropic resampling 1×1×1 mm³, z-score normalization
    - **Dosimetry**: Post-treatment Y-90 PET/CT within 3 days, voxel-based DVH

    #### Citation
    > *Please cite the original manuscript when using this tool.*
    > [Citation will appear here upon publication]

    #### Disclaimer
    This calculator is intended for **research purposes only** and has not been
    validated for clinical decision-making. Predictions should not replace
    clinical judgment or institutional treatment protocols.

    #### Contact
    Department of Nuclear Medicine & Molecular Imaging, UMCG, Groningen, Netherlands
    """)

st.divider()
st.caption(
    "Y-90 SIRT HCC Response Calculator · UMCG · "
    "For research use only · Not for clinical decision-making"
)
