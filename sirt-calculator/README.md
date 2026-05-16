# Y-90 SIRT HCC Response Calculator

Online calculator for predicting 4-month mRECIST response after Y-90 SIRT in HCC.
Model F: Radiomics + DVH (Random Forest).

## Deploy to Streamlit Cloud

1. Fork this repository
2. Add model files to `models/` folder:
   - `models/model_F_RF.pkl`
   - `models/scaler_F.pkl`
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repo → Deploy
5. Share the generated URL in your manuscript

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model files

Generate with `03_modeling.py` — add this after model training:

```python
import pickle
pickle.dump(best_models["F_radiomics_dvh"]["RF"],
            open("models/model_F_RF.pkl", "wb"))
pickle.dump(scalers["radio"],
            open("models/scaler_F.pkl", "wb"))
```
