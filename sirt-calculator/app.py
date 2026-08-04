"""
Y-90 SIRT HCC Lesion Response Calculator
Models: A (DVH only, LR), F (Radiomics+DVH, RF), G (Combined, RF)
Deploy: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

st.set_page_config(
    page_title="SIRT Response Calculator",
    page_icon="🏥",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────
DVH_FEATURES = [
    {"key": "Dmean (Gy)", "label": "Dmean — Mean absorbed dose (Gy)",
     "min": 0.0, "max": 2000.0, "default": 156.16,
     "help": "Mean dose to the lesion volume"},
    {"key": "D10", "label": "D10 — Dose covering 10% of volume (Gy)",
     "min": 0.0, "max": 5000.0, "default": 257.31,
     "help": "Dose received by the highest 10% of volume"},
    {"key": "V120 (%)", "label": "V120 — Volume fraction ≥120 Gy (%)",
     "min": 0.0, "max": 100.0, "default": 42.86,
     "help": "Percentage of lesion volume receiving ≥120 Gy"},
    {"key": "CoV", "label": "CoV — Coefficient of variation (SD/Dmean)",
     "min": 0.0, "max": 10.0, "default": 0.48,
     "help": "Relative dose heterogeneity; CoV = SD / Dmean"},
]

CLIN_FEATURES = [
    {"key": "Portal hypertension", "label": "Portal hypertension",
     "type": "binary", "default": 0,
     "help": "Presence of portal hypertension (0=No, 1=Yes)"},
    {"key": "PIVKA-II", "label": "PIVKA-II (mAU/mL)",
     "min": 0.0, "max": 100000.0, "default": 40.0,
     "help": "Protein induced by vitamin K absence (raw value)"},
]

RADIO_FEATURES = [
    {"key": "pre__original__shape__Elongation",
     "label": "Pre-contrast · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.805,
     "help": "Ratio of minor to major axis (pre-contrast phase)"},
    {"key": "pre__log-sigma-5-0-mm-3D__gldm__LargeDependenceHighGrayLevelEmphasis",
     "label": "Pre-contrast · LoG σ=5mm · GLDM · LDHGLE",
     "min": 0.0, "max": 1000.0, "default": 50.0,
     "help": "Large dependence high gray-level emphasis (pre-contrast, LoG σ=5mm)"},
    {"key": "pre__log-sigma-2-0-mm-3D__ngtdm__Busyness",
     "label": "Pre-contrast · LoG σ=2mm · NGTDM · Busyness",
     "min": 0.0, "max": 100.0, "default": 1.0,
     "help": "Neighborhood gray-tone difference busyness (pre-contrast, LoG σ=2mm)"},
    {"key": "pre__log-sigma-2-0-mm-3D__glcm__Imc1",
     "label": "Pre-contrast · LoG σ=2mm · GLCM · Imc1",
     "min": -1.0, "max": 1.0, "default": -0.15,
     "help": "Informational measure of correlation 1 (pre-contrast, LoG σ=2mm)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__SmallAreaEmphasis",
     "label": "Arterial · LoG σ=1mm · GLSZM · Small Area Emphasis",
     "min": 0.0, "max": 1.0, "default": 0.281,
     "help": "Emphasis of small zone sizes (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__log-sigma-1-0-mm-3D__glszm__LowGrayLevelZoneEmphasis",
     "label": "Arterial · LoG σ=1mm · GLSZM · LGLZE",
     "min": 0.0, "max": 1.0, "default": 0.172,
     "help": "Low gray-level zone emphasis (arterial phase, LoG σ=1mm)"},
    {"key": "arterial__original__glszm__SmallAreaLowGrayLevelEmphasis",
     "label": "Arterial · Original · GLSZM · SALGLE",
     "min": 0.0, "max": 1.0, "default": 0.030,
     "help": "Small area low gray-level emphasis (arterial phase, original)"},
    {"key": "venous__original__shape__Elongation",
     "label": "Portal venous · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.805,
     "help": "Ratio of minor to major axis (portal venous phase)"},
    {"key": "venous__original__gldm__DependenceVariance",
     "label": "Portal venous · Original · GLDM · Dependence Variance",
     "min": 0.0, "max": 100.0, "default": 10.0,
     "help": "Variance of dependence values (portal venous phase, original)"},
    {"key": "venous__log-sigma-5-0-mm-3D__glcm__MCC",
     "label": "Portal venous · LoG σ=5mm · GLCM · MCC",
     "min": 0.0, "max": 1.0, "default": 0.5,
     "help": "Maximal correlation coefficient (portal venous phase, LoG σ=5mm)"},
    {"key": "delayed__original__shape__Elongation",
     "label": "Delayed · Original · Shape · Elongation",
     "min": 0.0, "max": 1.0, "default": 0.805,
     "help": "Ratio of minor to major axis (delayed/equilibrium phase)"},
    {"key": "delayed__log-sigma-2-0-mm-3D__glcm__Imc1",
     "label": "Delayed · LoG σ=2mm · GLCM · Imc1",
     "min": -1.0, "max": 1.0, "default": -0.15,
     "help": "Informational measure of correlation 1 (delayed phase, LoG σ=2mm)"},
]

# Thresholds (from model development)
THRESHOLDS = {"A": 0.645, "F": 0.533, "G": 0.564}

# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_dir = Path(__file__).parent / "models"
    models = {}
    for model_id in ["A", "F", "G"]:
        try:
            with open(model_dir / f"model_{model_id}.pkl", "rb") as f:
                data = pickle.load(f)
            models[model_id] = {
                "model":    data["model"],
                "features": data["features"],
                "algo":     data["algo"],
                "scaler":   data.get("scaler", None),
                "error":    None
            }
        except Exception as e:
            models[model_id] = {
                "model": None, "features": None,
                "algo": None, "error": str(e)
            }
    return models

def predict_model(model_data, feature_values, threshold):
    if model_data["model"] is None:
        return None, None, f"Model load error: {model_data['error']}"
    model    = model_data["model"]
    features = model_data["features"]
    scaler   = model_data.get("scaler", None)
    missing  = [k for k in features if k not in feature_values]
    if missing:
        return None, None, f"Missing features: {missing[:3]}..."
    X = np.array([[feature_values[k] for k in features]], dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    return prob, pred, None

# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
def main():
    st.title("Y-90 SIRT HCC Response Calculator")
    st.markdown(
        "Lesion-level mRECIST response prediction after Y-90 selective "
        "internal radiation therapy for hepatocellular carcinoma."
    )

    models = load_models()

    # Model selection
    st.subheader("Model selection")
    model_choice = st.radio(
        "Select prediction model:",
        options=["A: DVH only (LR)",
                 "F: Radiomics + DVH (RF)",
                 "G: Combined — Radiomics + DVH + Clinical (RF)"],
        index=1,
        horizontal=True,
    )
    model_id = model_choice[0]  # "A", "F", or "G"

    if models[model_id]["error"]:
        st.error(f"Model {model_id} failed to load: {models[model_id]['error']}")
        return

    threshold = THRESHOLDS[model_id]

    st.markdown("---")

    # ── DVH inputs (always shown) ──────────────────────────────
    st.subheader("Dosimetry parameters (DVH)")
    feature_values = {}
    cols = st.columns(2)
    for i, feat in enumerate(DVH_FEATURES):
        with cols[i % 2]:
            feature_values[feat["key"]] = st.number_input(
                feat["label"],
                min_value=float(feat["min"]),
                max_value=float(feat["max"]),
                value=float(feat["default"]),
                help=feat["help"],
                key=f"dvh_{feat['key']}",
            )

    # ── Clinical inputs (G only) ───────────────────────────────
    if model_id == "G":
        st.markdown("---")
        st.subheader("Clinical parameters")
        cols_c = st.columns(2)
        for i, feat in enumerate(CLIN_FEATURES):
            with cols_c[i % 2]:
                if feat.get("type") == "binary":
                    feature_values[feat["key"]] = int(st.selectbox(
                        feat["label"],
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x else "No",
                        index=int(feat["default"]),
                        help=feat["help"],
                        key=f"clin_{feat['key']}",
                    ))
                else:
                    feature_values[feat["key"]] = st.number_input(
                        feat["label"],
                        min_value=float(feat["min"]),
                        max_value=float(feat["max"]),
                        value=float(feat["default"]),
                        help=feat["help"],
                        key=f"clin_{feat['key']}",
                    )

    # ── Radiomics inputs (F and G) ─────────────────────────────
    if model_id in ("F", "G"):
        st.markdown("---")
        st.subheader("Radiomics features")
        st.caption("Values should be standardized (z-score) as output by PyRadiomics.")
        cols_r = st.columns(2)
        for i, feat in enumerate(RADIO_FEATURES):
            with cols_r[i % 2]:
                feature_values[feat["key"]] = st.number_input(
                    feat["label"],
                    min_value=float(feat["min"]),
                    max_value=float(feat["max"]),
                    value=float(feat["default"]),
                    help=feat["help"],
                    key=f"radio_{feat['key']}",
                )

    st.markdown("---")

    # ── Predict ───────────────────────────────────────────────
    if st.button("Predict response", type="primary", use_container_width=True):
        prob, pred, err = predict_model(models[model_id], feature_values, threshold)

        if err:
            st.error(err)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted probability", f"{prob:.1%}")
            with col2:
                label = "Responder (CR/PR)" if pred == 1 else "Non-responder (SD/PD)"
                color = "green" if pred == 1 else "red"
                st.markdown(
                    f"**Prediction:** <span style='color:{color};font-size:1.1em'>"
                    f"{label}</span>",
                    unsafe_allow_html=True,
                )

            # Gauge bar
            bar_color = "#2ecc71" if prob >= threshold else "#e74c3c"
            st.markdown(
                f"""
                <div style='background:#eee;border-radius:8px;height:20px;margin:8px 0'>
                  <div style='background:{bar_color};width:{prob*100:.1f}%;
                              height:20px;border-radius:8px;'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:0.8em'>
                  <span>0%</span>
                  <span>Threshold: {threshold:.1%}</span>
                  <span>100%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                f"Model {model_id} | Algorithm: {models[model_id]['algo']} | "
                f"Threshold: {threshold:.3f} (Youden index, training set) | "
                f"Validation AUC: {'0.866' if model_id=='A' else '0.877'}"
            )

    st.markdown("---")
    st.caption(
        "⚠️ For research use only. Not validated for clinical decision-making. "
        "UMCG, Groningen, Netherlands."
    )

if __name__ == "__main__":
    main()
