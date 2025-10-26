import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib

def load_advanced_features():
    """Load advanced features from database"""
    conn = sqlite3.connect('nifty_gap.db')
    df = pd.read_sql('SELECT * FROM features_advanced', conn)
    conn.close()
    return df

def prepare_xgboost_data(df):
    """Prepare data for XGBoost"""
    print("\n📊 Preparing data for XGBoost...")
    
    # Target variable
    y = df['Target'].astype(int)
    
    # Feature columns (exclude non-predictive ones)
    exclude_cols = ['Date', 'Gap', 'Gap_Percent', 'Gap_Direction', 'Target', 
                    'Prediction', 'Bullish_Signals', 'Open', 'High', 'Low', 
                    'Close', 'Volume', 'Previous_Close']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Fill any remaining NaN with 0
    X = X.fillna(0)
    
    # Replace inf with 0
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"   ✅ Features selected: {len(feature_cols)}")
    print(f"   ✅ Total samples: {len(X)}")
    print(f"   ✅ Gap Up ratio: {(y == 1).sum() / len(y):.1%}")
    
    return X, y, feature_cols

def time_series_split_data(X, y, test_ratio=0.2):
    """Split data chronologically for time series"""
    split_idx = int(len(X) * (1 - test_ratio))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n🔀 Train-Test Split:")
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Gap Up train: {(y_train == 1).sum() / len(y_train):.1%}")
    print(f"   Gap Up test: {(y_test == 1).sum() / len(y_test):.1%}")
    
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model with good default parameters"""
    print("\n🚀 Training XGBoost model...")
    
    # Solid default parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Train the model
    model.fit(
        X_train, y_train,
        verbose=False
    )
    
    print(f"   ✅ Training complete!")
    
    return model

def evaluate_xgboost(model, X_test, y_test, feature_cols):
    """Evaluate XGBoost model performance"""
    print("\n" + "="*60)
    print("XGBOOST MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   AUC-ROC: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(f"   True Gap Down: {cm[0][0]} | False Gap Up: {cm[0][1]}")
    print(f"   False Gap Down: {cm[1][0]} | True Gap Up: {cm[1][1]}")
    
    # Classification Report
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Gap Down', 'Gap Up'], digits=3))
    
    # Separate accuracies
    gap_down_mask = y_test == 0
    gap_up_mask = y_test == 1
    
    acc_down = accuracy_score(y_test[gap_down_mask], y_pred[gap_down_mask]) if gap_down_mask.sum() > 0 else 0
    acc_up = accuracy_score(y_test[gap_up_mask], y_pred[gap_up_mask]) if gap_up_mask.sum() > 0 else 0
    
    print(f"\n✅ Gap Down Accuracy: {acc_down:.2%}")
    print(f"✅ Gap Up Accuracy: {acc_up:.2%}")
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🏆 Top 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    return accuracy, auc, importance_df

def save_model_and_features(model, feature_cols, importance_df):
    """Save model and feature information"""
    print(f"\n💾 Saving model...")
    
    joblib.dump(model, 'xgboost_gap_model.pkl')
    importance_df.to_csv('xgboost_feature_importance.csv', index=False)
    
    # Save feature columns for later use
    with open('feature_columns.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print(f"   ✅ Model saved: xgboost_gap_model.pkl")
    print(f"   ✅ Feature importance saved: xgboost_feature_importance.csv")
    print(f"   ✅ Feature columns saved: feature_columns.txt")

def main():
    print("="*60)
    print("XGBOOST TRAINING PIPELINE (Simplified)")
    print("="*60)
    
    # Step 1: Load advanced features
    print("\n1️⃣  Loading advanced features...")
    df = load_advanced_features()
    print(f"   ✅ Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Check if we have enough data
    if len(df) < 50:
        print(f"\n⚠️  WARNING: Only {len(df)} records found!")
        print(f"   You need at least 100+ records for good model training.")
        response = input(f"\n   Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("   Aborting...")
            return None, None, None
    
    # Step 2: Prepare data
    print("\n2️⃣  Preparing data for XGBoost...")
    X, y, feature_cols = prepare_xgboost_data(df)
    
    # Step 3: Time series split
    print("\n3️⃣  Splitting data (chronologically)...")
    X_train, X_test, y_train, y_test = time_series_split_data(X, y)
    
    # Check if splits are valid
    if len(X_train) < 20 or len(X_test) < 5:
        print(f"\n❌ ERROR: Not enough data after split!")
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"   Need at least 20 train samples and 5 test samples.")
        return None, None, None
    
    # Step 4: Train model (skip optimization for XGBoost compatibility)
    print("\n4️⃣  Training XGBoost model...")
    print("   Using optimized default parameters")
    model = train_xgboost(X_train, X_test, y_train, y_test)
    
    # Step 5: Evaluate
    print("\n5️⃣  Evaluating model...")
    accuracy, auc, importance_df = evaluate_xgboost(model, X_test, y_test, feature_cols)
    
    # Step 6: Save
    print("\n6️⃣  Saving model and artifacts...")
    save_model_and_features(model, feature_cols, importance_df)
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   AUC-ROC: {auc:.4f}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"\n📁 Saved files:")
    print(f"   - xgboost_gap_model.pkl (the trained model)")
    print(f"   - xgboost_feature_importance.csv (which features matter most)")
    print(f"   - feature_columns.txt (feature names for inference)")
    print(f"\n🚀 Next step: Run 'streamlit run app_xgboost.py'!")
    print("="*60)
    
    return model, accuracy, auc

if __name__ == "__main__":
    model, accuracy, auc = main()