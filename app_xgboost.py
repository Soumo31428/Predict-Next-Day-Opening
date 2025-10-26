import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np

st.set_page_config(page_title="Nifty50 Gap Predictor - XGBoost", page_icon="📈", layout="wide")

@st.cache_resource
def load_xgboost_model():
    try:
        model = joblib.load('xgboost_gap_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        return None

@st.cache_resource
def load_feature_columns():
    try:
        with open('feature_columns.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Feature columns could not be loaded: {e}")
        return []

def load_latest_features_from_db():
    try:
        conn = sqlite3.connect('nifty_gap.db')
        df = pd.read_sql('SELECT * FROM features_advanced ORDER BY Date DESC LIMIT 1', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Could not load features from DB: {e}")
        return None

def predict_with_xgboost(model, feature_cols, latest_features_df):
    try:
        if latest_features_df is None or len(latest_features_df) == 0:
            return None, None

        X_pred = latest_features_df[feature_cols].copy()
        X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)

        prediction = model.predict(X_pred)[0]
        confidence = model.predict_proba(X_pred)[0][int(prediction)] * 100
        gap_text = "Gap Up ⬆️" if prediction == 1 else "Gap Down ⬇️"
        return gap_text, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    st.title("📈 Nifty50 Gap Predictor - XGBoost Edition")
    st.caption("ML-Powered Next Day Opening Gap Prediction")
    st.markdown("---")

    # Load model and features
    model = load_xgboost_model()
    feature_cols = load_feature_columns()

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🔄 Refresh Data", key="refresh_data", help="Reload the latest market features", type="primary"):
            st.rerun()
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        if model is not None:
            st.success("✅ XGBoost model loaded")
            st.info(f"Features used: {len(feature_cols)}\n\nAdvanced ML-based prediction using 70+ engineered features")
        else:
            st.error("❌ Model Not Found")
            st.warning("Train using `python train_xgboost.py`")

    st.header("🔮 Tomorrow's Prediction")
    latest_features_df = load_latest_features_from_db()
    if (model is None) or (not feature_cols):
        st.error("Model/Features not loaded. Please train the model first with `python train_xgboost.py`.")
    else:
        prediction, confidence = predict_with_xgboost(model, feature_cols, latest_features_df)
        if prediction is not None:
            if "Up" in prediction:
                st.success(f"## {prediction}")
            else:
                st.error(f"## {prediction}")
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence:.0f}%")
            st.info("Model uses advanced technical, volatility, and price action features.")

    # Historical data section
    st.markdown("---")
    st.header("📜 Recent Prediction Inputs")
    try:
        conn = sqlite3.connect('nifty_gap.db')
        display = pd.read_sql('SELECT Date, Close, Gap, Gap_Percent, Gap_Direction FROM features_advanced ORDER BY Date DESC LIMIT 10', conn)
        st.dataframe(display, width='stretch', hide_index=True)
        conn.close()
    except Exception as e:
        st.info("No historical feature data available.")

if __name__ == "__main__":
    main()
