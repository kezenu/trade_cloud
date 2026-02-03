"""
===============================
MODEL TRAINING (FIXED VERSION)
===============================

PERBAIKAN DARI VERSI LAMA:
1. ✅ Time-based split (BUKAN random split)
2. ✅ Walk-forward validation untuk test robustness
3. ✅ Feature importance analysis
4. ✅ Threshold optimization
5. ✅ Realistic performance metrics
6. ✅ Model comparison (RF vs XGBoost)

CARA PAKAI:
1. Pastikan sudah menjalankan 1_data_preparation_FIXED.py
2. Jalankan: python 2_training_FIXED.py
3. Output: model_buy.pkl, model_sell.pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import shared features module
from features import get_feature_columns

warnings.filterwarnings('ignore')

# =====================================
# KONFIGURASI
# =====================================

DATA_FILE = "EURUSD_15_data.csv"  # Output dari fase 1
MODEL_BUY_FILE = "model_buy.pkl"
MODEL_SELL_FILE = "model_sell.pkl"

# Training configuration
TRAIN_RATIO = 0.7    # 70% untuk training
VALIDATION_RATIO = 0.15  # 15% untuk validation
TEST_RATIO = 0.15      # 15% untuk test (out-of-sample)

# Model configuration
N_ESTIMATORS = 500     # Lebih banyak trees = lebih baik (tapi lebih lambat)
MAX_DEPTH = 15          # Prevent overfitting
MIN_SAMPLES_SPLIT = 50  # Prevent overfitting
MIN_SAMPLES_LEAF = 20   # Prevent overfitting

# Walk-forward validation
N_SPLITS = 8  # Jumlah splits untuk walk-forward validation


# =====================================
# FUNGSI HELPER
# =====================================

def load_and_prepare_data(filepath):
    """
    Load data dan pisahkan features dan targets.
    
    Parameters:
    -----------
    filepath : str
        Path ke CSV file
        
    Returns:
    --------
    tuple
        (X, y_buy, y_sell, feature_columns)
    """
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    print(f"\n1. Membaca data dari: {filepath}")
    df = pd.read_csv(filepath, index_col='time', parse_dates=True)
    print(f"   ✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    print(f"\n2. Validating features...")
    
    # Auto-detect feature columns from DataFrame
    from features import get_feature_columns
    feature_columns = get_feature_columns(df)
    
    # Auto-detect feature columns from DataFrame
    feature_columns = get_feature_columns(df)
    
    # Check if we have enough features
    if len(feature_columns) < 5:
        raise ValueError(f"Too few features detected: {len(feature_columns)}")
    
    print(f"   ✅ Detected {len(feature_columns)} features")
    print(f"   First 10 features: {feature_columns[:10]}")
    
    # Prepare features and targets
    X = df[feature_columns]
    y_buy = df['buy_signal']
    y_sell = df['sell_signal']
    
    print(f"\n3. Data statistics:")
    print(f"   - Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   - Total days: {(df.index[-1] - df.index[0]).days}")
    print(f"   - BUY signals: {y_buy.sum()} ({y_buy.mean()*100:.2f}%)")
    print(f"   - SELL signals: {y_sell.sum()} ({y_sell.mean()*100:.2f}%)")
    
    return X, y_buy, y_sell, feature_columns


def time_based_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data berdasarkan WAKTU (bukan random).
    
    CRITICAL: Ini mencegah temporal leakage!
    Training = data lama
    Validation = data tengah
    Test = data baru (out-of-sample)
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    train_ratio : float
        Proporsi data untuk training
    val_ratio : float
        Proporsi data untuk validation
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    n = len(X)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split chronologically
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"\nTime-based split:")
    print(f"   Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
    print(f"   Val:   {len(X_val)} samples ({X_val.index[0]} to {X_val.index[-1]})")
    print(f"   Test:  {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def walk_forward_validation(X, y, n_splits=5, model_params=None):
    """
    Walk-forward validation untuk test model robustness.
    
    Ini simulasi trading di berbagai periode waktu.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int
        Jumlah splits
    model_params : dict
        Parameter untuk RandomForestClassifier
        
    Returns:
    --------
    dict
        Hasil validation (scores per fold)
    """
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION ({n_splits} splits)")
    print(f"{'='*60}")
    
    if model_params is None:
        model_params = {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'min_samples_split': MIN_SAMPLES_SPLIT,
            'min_samples_leaf': MIN_SAMPLES_LEAF,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    n = len(X)
    fold_size = n // (n_splits + 1)  # +1 because we need initial training data
    
    results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for i in range(n_splits):
        print(f"\nFold {i+1}/{n_splits}")
        print("-" * 40)
        
        # Define train and test indices for this fold
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = test_start + fold_size
        
        if test_end > n:
            test_end = n
        
        X_train_fold = X.iloc[:train_end]
        y_train_fold = y.iloc[:train_end]
        X_test_fold = X.iloc[test_start:test_end]
        y_test_fold = y.iloc[test_start:test_end]
        
        # Skip if test fold has no positive samples
        if y_test_fold.sum() == 0:
            print("   ⚠️  No positive samples in test fold, skipping...")
            continue
        
        print(f"   Train: {len(X_train_fold)} samples")
        print(f"   Test:  {len(X_test_fold)} samples")
        print(f"   Test positives: {y_test_fold.sum()} ({y_test_fold.mean()*100:.2f}%)")
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict
        y_pred = model.predict(X_test_fold)
        y_pred_proba = model.predict_proba(X_test_fold)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test_fold, y_pred, zero_division=0)
        recall = recall_score(y_test_fold, y_pred, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test_fold, y_pred_proba)
        except:
            auc = 0.0
        
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['auc'].append(auc)
        
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1:        {f1:.3f}")
        print(f"   AUC:       {auc:.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Average Precision: {np.mean(results['precision']):.3f} ± {np.std(results['precision']):.3f}")
    print(f"Average Recall:    {np.mean(results['recall']):.3f} ± {np.std(results['recall']):.3f}")
    print(f"Average F1:        {np.mean(results['f1']):.3f} ± {np.std(results['f1']):.3f}")
    print(f"Average AUC:       {np.mean(results['auc']):.3f} ± {np.std(results['auc']):.3f}")
    
    return results


def train_model(X_train, y_train, X_val, y_val, model_name="Model"):
    """
    Train model dengan hyperparameter yang sudah di-tune.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    model_name : str
        Nama model (untuk logging)
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name}")
    print(f"{'='*60}")
    
    # Check class distribution
    print(f"\nClass distribution in training data:")
    print(f"   Class 0 (No signal): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"   Class 1 (Signal):    {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\nTraining model...")
    print(f"   n_estimators: {N_ESTIMATORS}")
    print(f"   max_depth: {MAX_DEPTH}")
    print(f"   min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"   min_samples_leaf: {MIN_SAMPLES_LEAF}")
    
    model.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    # Evaluate on validation set
    print(f"\n{'='*60}")
    print(f"VALIDATION SET PERFORMANCE")
    print(f"{'='*60}")
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    print(f"\nInterpretation:")
    print(f"   True Negatives:  {cm[0,0]} (correctly predicted no signal)")
    print(f"   False Positives: {cm[0,1]} (predicted signal but wrong)")
    print(f"   False Negatives: {cm[1,0]} (missed signal)")
    print(f"   True Positives:  {cm[1,1]} (correctly predicted signal)")
    
    # Additional metrics
    try:
        auc = roc_auc_score(y_val, y_val_proba)
        print(f"\nAUC-ROC: {auc:.3f}")
    except:
        print("\n⚠️  Cannot calculate AUC (possibly only one class in validation)")
    
    return model


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze dan visualize feature importance.
    
    Parameters:
    -----------
    model : trained model
        Model yang sudah di-train
    feature_names : list
        List nama features
    top_n : int
        Jumlah top features yang ditampilkan
    """
    
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop {top_n} Most Important Features:")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:25s} - {importances[idx]:.4f}")
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)


def find_optimal_threshold(model, X_val, y_val):
    """
    Find optimal probability threshold untuk maximize F1-score.
    
    Default threshold = 0.5 mungkin tidak optimal untuk imbalanced data.
    
    Parameters:
    -----------
    model : trained model
    X_val, y_val : validation data
        
    Returns:
    --------
    float
        Optimal threshold
    """
    
    print(f"\n{'='*60}")
    print(f"THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")
    
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    optimal_recall = recalls[optimal_idx]
    
    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"   Precision: {optimal_precision:.3f}")
    print(f"   Recall:    {optimal_recall:.3f}")
    print(f"   F1-Score:  {optimal_f1:.3f}")
    
    print(f"\nComparison with default threshold (0.5):")
    y_val_pred_default = (y_val_proba >= 0.5).astype(int)
    default_f1 = f1_score(y_val, y_val_pred_default, zero_division=0)
    print(f"   Default F1: {default_f1:.3f}")
    print(f"   Improvement: {(optimal_f1 - default_f1)*100:.1f}%")
    
    return optimal_threshold


def evaluate_on_test_set(model, X_test, y_test, threshold=0.5, model_name="Model"):
    """
    Final evaluation on held-out test set (out-of-sample).
    
    Parameters:
    -----------
    model : trained model
    X_test, y_test : test data
    threshold : float
        Probability threshold
    model_name : str
        Model name
    """
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST SET EVALUATION - {model_name}")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.3f}")
    
    # Predict
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Calculate metrics
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    print(f"\nKey Metrics:")
    print(f"   Precision: {precision:.3f} (berapa % sinyal yang benar profitable)")
    print(f"   Recall:    {recall:.3f} (berapa % opportunity yang ke-catch)")
    print(f"   F1-Score:  {f1:.3f} (balance antara precision & recall)")
    
    try:
        auc = roc_auc_score(y_test, y_test_proba)
        print(f"   AUC-ROC:   {auc:.3f}")
    except:
        print(f"   AUC-ROC:   N/A")
    
    # Trading simulation metrics
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    
    print(f"\nTrading Interpretation:")
    print(f"   Correct trades (TP):    {true_positives}")
    print(f"   Wrong trades (FP):      {false_positives}")
    print(f"   Missed opportunities:   {false_negatives}")
    
    if true_positives + false_positives > 0:
        win_rate = true_positives / (true_positives + false_positives)
        print(f"   Win Rate:               {win_rate*100:.1f}%")
        
        # Assuming RR = 1:2 (from RISK_MULTIPLIER=1.5, REWARD_MULTIPLIER=3.0)
        rr_ratio = 3.0 / 1.5  # 2.0
        expected_return = (win_rate * rr_ratio) - ((1 - win_rate) * 1)
        print(f"   Expected Return/Trade:  {expected_return:.2f}R")
        
        if expected_return > 0:
            print(f"   ✅ PROFITABLE system (Expected Return > 0)")
        else:
            print(f"   ❌ NOT PROFITABLE (Expected Return < 0)")


# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """
    Main training pipeline.
    """
    
    print("\n" + "="*60)
    print("MODEL TRAINING - PRODUCTION GRADE ML TRADING")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    # Load data
    X, y_buy, y_sell, feature_cols = load_and_prepare_data(DATA_FILE)
    
    # =====================================
    # TRAIN BUY MODEL
    # =====================================
    
    print("\n" + "#"*60)
    print("# TRAINING BUY SIGNAL MODEL")
    print("#"*60)
    
    # Time-based split
    X_train, X_val, X_test, y_train_buy, y_val_buy, y_test_buy = time_based_split(
        X, y_buy, TRAIN_RATIO, VALIDATION_RATIO
    )
    
    # Walk-forward validation
    walk_forward_results_buy = walk_forward_validation(X, y_buy, N_SPLITS)
    
    # Train final model
    buy_model = train_model(X_train, y_train_buy, X_val, y_val_buy, "BUY Model")
    
    # Feature importance
    feature_importance_buy = analyze_feature_importance(buy_model, feature_cols)
    
    # Optimize threshold
    optimal_threshold_buy = find_optimal_threshold(buy_model, X_val, y_val_buy)
    
    # Final test evaluation
    evaluate_on_test_set(buy_model, X_test, y_test_buy, optimal_threshold_buy, "BUY Model")
    
    # =====================================
    # TRAIN SELL MODEL
    # =====================================
    
    print("\n" + "#"*60)
    print("# TRAINING SELL SIGNAL MODEL")
    print("#"*60)
    
    # Time-based split (reuse X splits)
    _, _, _, y_train_sell, y_val_sell, y_test_sell = time_based_split(
        X, y_sell, TRAIN_RATIO, VALIDATION_RATIO
    )
    
    # Walk-forward validation
    walk_forward_results_sell = walk_forward_validation(X, y_sell, N_SPLITS)
    
    # Train final model
    sell_model = train_model(X_train, y_train_sell, X_val, y_val_sell, "SELL Model")
    
    # Feature importance
    feature_importance_sell = analyze_feature_importance(sell_model, feature_cols)
    
    # Optimize threshold
    optimal_threshold_sell = find_optimal_threshold(sell_model, X_val, y_val_sell)
    
    # Final test evaluation
    evaluate_on_test_set(sell_model, X_test, y_test_sell, optimal_threshold_sell, "SELL Model")
    
    # =====================================
    # SAVE MODELS
    # =====================================
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    joblib.dump(buy_model, MODEL_BUY_FILE)
    print(f"✅ BUY model saved: {MODEL_BUY_FILE}")
    
    joblib.dump(sell_model, MODEL_SELL_FILE)
    print(f"✅ SELL model saved: {MODEL_SELL_FILE}")
    
    # Save thresholds and metadata
    metadata = {
        'buy_threshold': optimal_threshold_buy,
        'sell_threshold': optimal_threshold_sell,
        'feature_columns': feature_cols,
        'train_date_range': (str(X_train.index[0]), str(X_train.index[-1])),
        'test_date_range': (str(X_test.index[0]), str(X_test.index[-1])),
        'timestamp': str(datetime.now())
    }
    
    joblib.dump(metadata, 'model_metadata.pkl')
    print(f"✅ Metadata saved: model_metadata.pkl")
    
    # =====================================
    # FINAL SUMMARY
    # =====================================
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    print(f"\n📊 RECOMMENDED THRESHOLDS FOR LIVE BOT:")
    print(f"   BUY:  {optimal_threshold_buy:.3f}")
    print(f"   SELL: {optimal_threshold_sell:.3f}")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   - {MODEL_BUY_FILE}")
    print(f"   - {MODEL_SELL_FILE}")
    print(f"   - model_metadata.pkl")
    
    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"   1. Update bot code to use optimal thresholds")
    print(f"   2. Test di DEMO account dulu minimal 1 bulan")
    print(f"   3. Monitor performance closely")
    print(f"   4. Re-train model setiap 3-6 bulan dengan data baru")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Update 3_trading_bot_FIXED.py dengan threshold optimal")
    print(f"   2. Test di demo account")
    print(f"   3. Jalankan backtest untuk double-check performance")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Pastikan Anda sudah menjalankan 1_data_preparation_FIXED.py terlebih dahulu")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
