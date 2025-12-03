"""
Water Price Prediction Model with Ensemble Methods and Feature Selection
This script processes hydro data, selects features, trains models, and predicts prices across scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb

from calfews_src.util import *
warnings.filterwarnings('ignore')


def aggregate_hydro_features(df):

    df = df.copy()
    cols = [str(c) for c in df.columns]
    
    suffix_map = {
        '_S': 'total_S',
        '_R': 'total_R',
        '_SNPK': 'total_SNPK',
        '_fnf': 'total_fnf',
        '_Q': 'total_Q',
    }
    
    for suf, outcol in suffix_map.items():
        matched = [c for c in cols if c.endswith(suf)]
        if matched:
            df[outcol] = df[matched].sum(axis=1)
    
    wd_matched = [c for c in cols if ('wonderful' in c.lower() and 'delivery' in c.lower())]
    if wd_matched:
        df['total_wonderful_delivery'] = df[wd_matched].sum(axis=1)
    
    return df


def create_monthly_features(df, lags=[1, 2, 3,6], roll_specs={'rmean3': 3, 'rstd3': 3}):
    """
    Convert daily hydro data to monthly aggregated features with lags and rolling statistics. lags, and rolling statistics
    """
    _df = df.copy()
    _df.index = pd.to_datetime(_df.index)
    
    agg_mean_cols = ['total_S', 'total_R', 'total_Q', 'total_fnf']
    agg_last_cols = ['total_SNPK', 'total_wonderful_delivery']
    
    agg_mean_cols = [c for c in agg_mean_cols if c in _df.columns]
    agg_last_cols = [c for c in agg_last_cols if c in _df.columns]
    
    agg_map = {}
    agg_map.update({c: 'mean' for c in agg_mean_cols})
    agg_map.update({c: 'last' for c in agg_last_cols})
    
    monthly_df = _df[list(agg_map.keys())].resample('M').agg(agg_map)
    monthly_df.index = pd.to_datetime(monthly_df.index)
    monthly_df['month'] = monthly_df.index.to_period('M').astype(str)
    monthly_df = monthly_df.reset_index(drop=False)
    monthly_df = monthly_df.rename(columns={'index': 'month_end'}) if 'index' in monthly_df.columns else monthly_df
    monthly_df = monthly_df.set_index('month')
    
    hydro_cols = [c for c in (agg_mean_cols + agg_last_cols) if c in monthly_df.columns]
    
    for col in hydro_cols:
        for L in lags:
            monthly_df[f'{col}_lag{L}'] = monthly_df[col].shift(L)
        
        for suffix, win in roll_specs.items():
            if 'mean' in suffix:
                monthly_df[f'{col}_{suffix}'] = monthly_df[col].rolling(win, min_periods=win).mean()
            else:
                monthly_df[f'{col}_{suffix}'] = monthly_df[col].rolling(win, min_periods=win).std()
    
    monthly_df = monthly_df.drop('month_end', axis=1, errors='ignore')
    
    return monthly_df


def load_and_process_hydro_data(path):
    hydro_data = get_results_sensitivity_number_outside_model(path, '')
    if hydro_data is None:
        return None
    
    df_agg = aggregate_hydro_features(hydro_data)
    
    monthly_df = create_monthly_features(df_agg)
    
    return monthly_df


def load_price_data(csv_path='calfews_src/data/price_index/Data_NQH2O.csv'):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    
    s = df.set_index('Date')['NQH2O'].dropna().sort_index()
    monthly = s.resample('M').mean()
    monthly = monthly.dropna()
    log_price = np.log(monthly)
    
    log_price.index = pd.PeriodIndex(log_price.index, freq='M')
    
    return log_price



def remove_colinear_features(X, threshold=0.95, method='correlation'):
    """
    Remove highly correlated features to reduce multicollinearity.
    """
    X_clean = X.copy()
    X_clean = X_clean.ffill().bfill()
    
    corr_matrix = X_clean.corr().abs()
    
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_remove = [column for column in upper_triangle.columns 
                 if any(upper_triangle[column] > threshold)]
    
    X_reduced = X_clean.drop(columns=to_remove, errors='ignore')
    
    print(f"\n{'='*60}")
    print("COLINEARITY PRUNING")
    print(f"{'='*60}")
    print(f"Original features: {len(X.columns)}")
    print(f"Removed {len(to_remove)} highly correlated features (threshold={threshold})")
    print(f"Remaining features: {len(X_reduced.columns)}")
    
    return X_reduced, to_remove


def analyze_seasonal_patterns(X, y, model, scaler, selected_features):
    """
    Analyze which features cause winter peaks (months 12-3) in predictions.
    Returns features that show winter-biased patterns.
    """
    print(f"\n{'='*60}")
    print("SEASONAL PATTERN ANALYSIS - IDENTIFYING WINTER PEAK FEATURES")
    print(f"{'='*60}")
    
    X_clean = X.ffill().bfill()
    X_scaled = scaler.transform(X_clean)
    
    months = []
    if isinstance(X.index, pd.PeriodIndex):
        months = [idx.month for idx in X.index]
    elif isinstance(X.index, pd.DatetimeIndex):
        months = [idx.month for idx in X.index]
    elif hasattr(X.index, 'month'):
        months = [idx.month for idx in X.index]
    else:
        if 'month' in X.columns:
            try:
                months = [pd.Period(idx).month if isinstance(idx, str) else idx.month 
                         for idx in X.index]
            except:
                months = [(9 + (i % 12)) % 12 + 1 for i in range(len(X))]  # Oct=10, Nov=11, Dec=12, Jan=1, etc.
        else:
            months = [(9 + (i % 12)) % 12 + 1 for i in range(len(X))]  # Oct=10, Nov=11, Dec=12, Jan=1, etc.
    
    winter_months = [12, 1, 2, 3]
    summer_months = [6, 7, 8, 9]  # Jun, Jul, Aug, Sep
    
    y_pred_full = model.predict(X_scaled)
    
    winter_indices = [i for i, m in enumerate(months) if m in winter_months]
    summer_indices = [i for i, m in enumerate(months) if m in summer_months]
    
    if len(winter_indices) > 0 and len(summer_indices) > 0:
        avg_winter_pred = np.mean(y_pred_full[winter_indices])
        avg_summer_pred = np.mean(y_pred_full[summer_indices])
        
        print(f"Average predicted log price - Winter (Dec-Mar): {avg_winter_pred:.4f}")
        print(f"Average predicted log price - Summer (Jun-Sep): {avg_summer_pred:.4f}")
        print(f"Difference (Winter - Summer): {avg_winter_pred - avg_summer_pred:.4f}")
        
        if avg_winter_pred > avg_summer_pred:
            print(f"WARNING: Winter predictions are HIGHER than summer - this is unrealistic!")
            print("Analyzing which features contribute to winter peaks...")
        else:
            print(f"Winter predictions are lower than summer (expected pattern)")
    
    feature_winter_bias = {}
    
    for feature in selected_features:
        if feature not in X.columns:
            continue
            
        feature_idx = X.columns.get_loc(feature)
        feature_values = X_clean[feature].values
        
        winter_feat_vals = [feature_values[i] for i in winter_indices if i < len(feature_values)]
        summer_feat_vals = [feature_values[i] for i in summer_indices if i < len(feature_values)]
        
        if len(winter_feat_vals) > 0 and len(summer_feat_vals) > 0:
            avg_winter_feat = np.mean(winter_feat_vals)
            avg_summer_feat = np.mean(summer_feat_vals)
            
            winter_feat_higher = avg_winter_feat > avg_summer_feat
            
            if hasattr(model, 'feature_importances_'):
                feat_importance = model.feature_importances_[feature_idx]
            elif hasattr(model, 'coef_'):
                feat_importance = abs(model.coef_[feature_idx]) if model.coef_.ndim == 1 else abs(model.coef_[0][feature_idx])
            else:
                feat_importance = 0
            
            if len(winter_feat_vals) > 1:
                winter_corr = np.corrcoef(winter_feat_vals, y_pred_full[winter_indices[:len(winter_feat_vals)]])[0, 1]
            else:
                winter_corr = 0
            
            winter_bias_score = 0
            if winter_feat_higher and winter_corr > 0:
                winter_bias_score = feat_importance * winter_corr * (avg_winter_feat / (avg_summer_feat + 1e-6))
            
            feature_winter_bias[feature] = {
                'winter_avg': avg_winter_feat,
                'summer_avg': avg_summer_feat,
                'winter_higher': winter_feat_higher,
                'winter_corr': winter_corr,
                'importance': feat_importance,
                'winter_bias_score': winter_bias_score
            }
    
    bias_df = pd.DataFrame(feature_winter_bias).T
    bias_df = bias_df.sort_values('winter_bias_score', ascending=False)
    
    print(f"\nFeatures ranked by contribution to winter peaks:")
    print(bias_df.head(10))
    
    if len(bias_df) > 0:
        threshold = bias_df['winter_bias_score'].quantile(0.9)  # Top 25% of winter-biased features
        problematic_features = bias_df[bias_df['winter_bias_score'] > threshold].index.tolist()
        
        print(f"\n⚠️  Identified {len(problematic_features)} potentially problematic features:")
        for feat in problematic_features:
            info = bias_df.loc[feat]
            print(f"  - {feat}: winter_bias_score={info['winter_bias_score']:.4f}, "
                  f"winter_avg={info['winter_avg']:.4f}, summer_avg={info['summer_avg']:.4f}")
    else:
        problematic_features = []
    
    return problematic_features, bias_df


def select_features_permutation(X, y, model, n_repeats=10, random_state=42, top_k=None, exclude_winter_features=True):

    X_clean = X.ffill().bfill()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    model.fit(X_scaled, y)
    
    perm_importance = permutation_importance(
        model, X_scaled, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='neg_mean_squared_error',
        n_jobs=1
    )
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    if top_k is not None:
        selected_features = importance_df.head(top_k)['feature'].tolist()
    else:
        threshold = importance_df['importance_mean'].quantile(0.25)  # Top 75%
        selected_features = importance_df[importance_df['importance_mean'] > threshold]['feature'].tolist()
    
    problematic_features = []
    if exclude_winter_features and len(selected_features) > 0:
        problematic_features, bias_df = analyze_seasonal_patterns(X, y, model, scaler, selected_features)
        
        if problematic_features:
            original_count = len(selected_features)
            selected_features = [f for f in selected_features if f not in problematic_features]
            print(f"\n{'='*60}")
            print(f"REMOVED {len(problematic_features)} WINTER-BIASED FEATURES")
            print(f"{'='*60}")
            print(f"Original feature count: {original_count}")
            print(f"After removing winter-biased features: {len(selected_features)}")
            print(f"Removed features: {problematic_features}")
            
            if top_k is not None and len(selected_features) < top_k:
                remaining_features = [f for f in importance_df['feature'].tolist() 
                                    if f not in problematic_features and f not in selected_features]
                needed = top_k - len(selected_features)
                selected_features.extend(remaining_features[:needed])
                print(f"Added {needed} additional features to reach top_k={top_k}")
    
    print(f"\n{'='*60}")
    print("PERMUTATION IMPORTANCE FEATURE SELECTION")
    print(f"{'='*60}")
    print(f"Selected {len(selected_features)} features")
    print("\nTop features by permutation importance:")
    print(importance_df.head(min(15, len(importance_df))))
    
    return selected_features, importance_df



def train_test_split_time(X, y, test_size=24):

    split_idx = len(X) - test_size
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    for obj in [y_train, y_test]:
        if isinstance(obj.index, pd.PeriodIndex):
            obj.index = obj.index.to_timestamp(how="end")
    
    return X_train, X_test, y_train, y_test


def fit_evaluate(X_train, y_train, X_test=None, y_test=None, drop_cols=None, 
                 n_splits=5, random_state=42, plot=True):
    """
    Train and evaluate multiple models with cross-validation.

    """
    if drop_cols is None:
        drop_cols = []
    
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols, errors='ignore')
        if X_test is not None:
            X_test = X_test.drop(columns=drop_cols, errors='ignore')
    
    X_train = X_train.ffill().bfill()
    if X_test is not None:
        X_test = X_test.ffill().bfill()
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler_X.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = {}
    cv_scores = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        cv_scores[name] = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_squared_error')
        
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'cv_mean': -cv_scores[name].mean(),
                'cv_std': cv_scores[name].std(),
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'model': model,
                'scaler': scaler_X
            }
        else:
            results[name] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'cv_mean': -cv_scores[name].mean(),
                'cv_std': cv_scores[name].std(),
                'y_train_pred': y_train_pred,
                'model': model,
                'scaler': scaler_X
            }
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    if X_test is not None and y_test is not None:
        print(f"{'Model':<20} {'Train R²':<10} {'Test R²':<10} {'Train RMSE':<12} {'Test RMSE':<12} {'CV RMSE':<12}")
        print("-" * 80)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['train_r2']:<10.4f} {metrics['test_r2']:<10.4f} "
                  f"{metrics['train_rmse']:<12.4f} {metrics['test_rmse']:<12.4f} {metrics['cv_mean']:<12.4f}")
    else:
        print(f"{'Model':<20} {'Train R²':<10} {'Train RMSE':<12} {'CV RMSE':<12}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['train_r2']:<10.4f} {metrics['train_rmse']:<12.4f} {metrics['cv_mean']:<12.4f}")
    
    if plot and X_test is not None and y_test is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        axes[0, 0].scatter(y_test, results[best_model_name]['y_test_pred'], alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title(f'{best_model_name}: Test Set Predictions vs Actual')
        
        # Plot 2: Time series comparison
        axes[0, 1].plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
        axes[0, 1].plot(y_test.index, results[best_model_name]['y_test_pred'], label='Predicted', alpha=0.7)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Log Price')
        axes[0, 1].set_title(f'{best_model_name}: Time Series Comparison')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Model comparison (R²)
        model_names = list(results.keys())
        train_r2_scores = [results[name]['train_r2'] for name in model_names]
        test_r2_scores = [results[name]['test_r2'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, train_r2_scores, width, label='Train R²', alpha=0.7)
        axes[1, 0].bar(x + width/2, test_r2_scores, width, label='Test R²', alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Model Comparison: R² Scores')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # Plot 4: Model comparison (RMSE)
        train_rmse_scores = [results[name]['train_rmse'] for name in model_names]
        test_rmse_scores = [results[name]['test_rmse'] for name in model_names]
        
        axes[1, 1].bar(x - width/2, train_rmse_scores, width, label='Train RMSE', alpha=0.7)
        axes[1, 1].bar(x + width/2, test_rmse_scores, width, label='Test RMSE', alpha=0.7)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Model Comparison: RMSE Scores')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('hybrid_result/model_comparison_ensemble_water_price.png', dpi=300, bbox_inches='tight')
    
    return results



def load_scenario_data(scenario_path_base, scenario_num):
    """
    Load hydro data for a specific scenario.
    """
    path = f"{scenario_path_base}/2024_{scenario_num}/results.hdf5"
    hydro_data = get_results_sensitivity_number_outside_model(path, '')
    return hydro_data


def process_scenario_data(hydro_data):
    """
    Process hydro data for a scenario using the same structure as training data.
    """
    if hydro_data is None:
        return None
    
    # Use the same processing functions as training data
    df_agg = aggregate_hydro_features(hydro_data)
    monthly_df = create_monthly_features(df_agg)
    
    return monthly_df


def predict_scenario_prices(scenario_data, trained_model, scaler, selected_features):
    """
    Predict prices for a scenario using trained model.

    """
    if scenario_data is None:
        return None
    
    X_scenario = scenario_data[selected_features].copy()
    
    X_scenario = X_scenario.ffill().bfill()
    
    X_scenario_scaled = scaler.transform(X_scenario)
    
    predictions = trained_model.predict(X_scenario_scaled)
    
    return predictions



if __name__ == "__main__":
    print("="*80)
    print("LOADING AND PROCESSING TRAINING DATA")
    print("="*80)
    
    log_price = load_price_data()
    
    training_path = "results/short_test/results.hdf5"
    monthly_df = load_and_process_hydro_data(training_path)
    
    if monthly_df is None:
        raise ValueError("Failed to load training hydro data")
    
    print(f"Monthly result shape: {monthly_df.shape}")
    
    monthly_df.index = pd.PeriodIndex(monthly_df.index, freq='M')
    monthly_combined = monthly_df.join(log_price.rename('log_price'), how='right')
    monthly_combined.dropna(inplace=True)
    
    y = monthly_combined["log_price"]
    X = monthly_combined.drop(columns=["log_price"])
    
    print(f"\nData shape: {X.shape}")
    print(f"Features: {len(X.columns)}")
    print(f"Samples: {len(X)}")
    
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    
    X_no_colinear, removed_features = remove_colinear_features(X, threshold=0.9)
    
    # Step 2: Use permutation importance for final selection
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    selected_features, importance_df = select_features_permutation(
        X_no_colinear, y, base_model, n_repeats=10, random_state=42, top_k=8, 
        exclude_winter_features=True  # Remove features causing winter peaks
    )
    
    X_selected = X_no_colinear[selected_features]
    
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split_time(X_selected, y, test_size=24)
    print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    results = fit_evaluate(
        X_train, y_train,
        X_test=X_test, y_test=y_test,
        drop_cols=[],
        n_splits=5,
        random_state=42,
        plot=True
    )
    
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
    best_model = results[best_model_name]['model']
    best_scaler = results[best_model_name]['scaler']
    
    print(f"\nBest model: {best_model_name}")
    
    print("\n" + "="*80)
    print("SCENARIO ANALYSIS")
    print("="*80)
    
    X_full_clean = X_selected.ffill().bfill()
    X_full_scaled = best_scaler.fit_transform(X_full_clean)
    best_model.fit(X_full_scaled, y)
    
    print(f"Trained best model on {X_full_clean.shape[0]} samples with {X_full_clean.shape[1]} features")
    
    train_mae_log = results[best_model_name]['train_mae']  # MAE in log space
    print(f"\nTraining MAE (log space): {train_mae_log:.4f}")
    print(f"This will be used to create uncertainty bands around predictions")
    
    scenario_results = {}
    scenario_predictions = {}  # Will store dict with 'base', 'upper', 'lower' for each scenario
    scenario_path_base = "results/startyear_4_1"
    
    print("\nAnalyzing scenarios with uncertainty bands...")
    for scenario_num in range(1, 101):
        if scenario_num % 20 == 0:
            print(f"Processing scenario {scenario_num}...")
        
        hydro_data = load_scenario_data(scenario_path_base, scenario_num)
        
        if hydro_data is not None:
            scenario_monthly = process_scenario_data(hydro_data)
            
            if scenario_monthly is not None:
                scenario_12m = scenario_monthly.head(36)
                
                predictions = predict_scenario_prices(
                    scenario_12m, best_model, best_scaler, selected_features
                )
                
                if predictions is not None:
                    log_pred = np.asarray(predictions).ravel()
 
                    log_pred_upper = log_pred + train_mae_log
                    log_pred_lower = log_pred - train_mae_log
                    
                    price_pred_base = np.exp(log_pred)
                    price_pred_upper = np.exp(log_pred_upper)
                    price_pred_lower = np.exp(log_pred_lower)
                    
                    scenario_results[scenario_num] = {
                        'mean_log_price': np.mean(log_pred),
                        'std_log_price': np.std(log_pred),
                        'min_log_price': np.min(log_pred),
                        'max_log_price': np.max(log_pred),
                        'mean_price': np.mean(price_pred_base),
                        'std_price': np.std(price_pred_base),
                        'min_price': np.min(price_pred_base),
                        'max_price': np.max(price_pred_base),
                    }
                    
                    scenario_predictions[scenario_num] = {
                        'base': price_pred_base,
                        'upper': price_pred_upper,
                        'lower': price_pred_lower
                    }
    
    print(f"\nSuccessfully processed {len(scenario_results)} scenarios")
    
    print("\n" + "="*60)
    print("VALIDATING SCENARIO PREDICTIONS FOR WINTER PEAKS")
    print("="*60)

    
    scenarios_with_winter_peaks = []
    for scenario_num, pred_dict in scenario_predictions.items():
        preds = np.asarray(pred_dict['base']).ravel()
        
        winter_indices = [2, 3, 4, 5]  # Dec, Jan, Feb, Mar
        summer_indices = [8, 9, 10, 11]  # Jun, Jul, Aug, Sep
        
        if len(preds) > max(winter_indices + summer_indices):
            winter_avg = np.mean([preds[i] for i in winter_indices if i < len(preds)])
            summer_avg = np.mean([preds[i] for i in summer_indices if i < len(preds)])
            
            if winter_avg > summer_avg:
                scenarios_with_winter_peaks.append({
                    'scenario': scenario_num,
                    'winter_avg': winter_avg,
                    'summer_avg': summer_avg,
                    'difference': winter_avg - summer_avg
                })
    
    if scenarios_with_winter_peaks:
        print(f"\n⚠️  WARNING: {len(scenarios_with_winter_peaks)} scenarios still show winter peaks (winter > summer):")
        peak_df = pd.DataFrame(scenarios_with_winter_peaks).sort_values('difference', ascending=False)
        print(peak_df.head(10))
        print(f"\nThis suggests some winter-biased features may still be present.")
        print("Consider reviewing feature selection or increasing the winter bias threshold.")
    else:
        print(f"\n✓ All scenarios show realistic patterns (summer prices >= winter prices)")
    
    if scenario_results:
        print("\n" + "="*60)
        print("SCENARIO ANALYSIS RESULTS")
        print("="*60)
        
        results_df = pd.DataFrame(scenario_results).T
        
        print(f"\nPrice Statistics Across {len(scenario_results)} Scenarios:")
        print(f"Mean log price: {results_df['mean_log_price'].mean():.4f} ± {results_df['mean_log_price'].std():.4f}")
        print(f"Min mean price: {results_df['mean_price'].min():.4f}")
        print(f"Max mean price: {results_df['mean_price'].max():.4f}")
        
        scen_ids = list(scenario_predictions.keys())
        S = len(scen_ids)

        first_scenario = scenario_predictions[scen_ids[0]]
        T = len(first_scenario['base'])
        t = np.arange(T)

        start_date = pd.Timestamp('2024-10-01')
        date_range = pd.date_range(start=start_date, periods=T, freq='M')
        x_labels = [d.strftime('%Y-%m') for d in date_range]

        Y_base = np.array([scenario_predictions[s]['base'] for s in scen_ids])
        Y_upper = np.array([scenario_predictions[s]['upper'] for s in scen_ids])
        Y_lower = np.array([scenario_predictions[s]['lower'] for s in scen_ids])

        med = np.median(Y_base, axis=0)
        p10 = np.percentile(Y_base, 10, axis=0)
        p90 = np.percentile(Y_base, 90, axis=0)
        p5 = np.percentile(Y_base, 5, axis=0)
        p95 = np.percentile(Y_base, 95, axis=0)
        std_t_base = np.std(Y_base, axis=0)  # Std across base predictions

        std_t = np.zeros(T)
        for i in range(T):
            all_vals_t = np.concatenate([Y_base[:, i], Y_upper[:, i], Y_lower[:, i]])
            std_t[i] = np.std(all_vals_t)

        all_vals = np.concatenate([Y_base.ravel(), Y_upper.ravel(), Y_lower.ravel()])


        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif', 'serif']
        
        fig = plt.figure(figsize=(10, 7), dpi=300)
        gs = fig.add_gridspec(nrows=2, ncols=2,
                            height_ratios=[1, 4], width_ratios=[4, 1],
                            hspace=0.09, wspace=0.09)

        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # === MAIN: plot uncertainty bands for each scenario ===
        for i, scen_id in enumerate(scen_ids):
            pred_data = scenario_predictions[scen_id]
            if i == 0:
                ax_main.fill_between(t, pred_data['lower'], pred_data['upper'], 
                                     alpha=0.15, color='gray', linewidth=0,
                                     label=f'Individual scenario uncertainty range')
                ax_main.plot(t, pred_data['base'], lw=0.5, alpha=0.4, color='gray',
                            label='Individual scenario base predictions')
            else:
                ax_main.fill_between(t, pred_data['lower'], pred_data['upper'], 
                                     alpha=0.15, color='gray', linewidth=0)
                ax_main.plot(t, pred_data['base'], lw=0.5, alpha=0.4, color='gray')

        ax_main.fill_between(t, p5, p95, color='skyblue', alpha=0.6, 
                            label='Ensemble 5-95th percentile')
        ax_main.plot(t, med, lw=2.0, color='navy', 
                    label='Ensemble median')

        tick_positions = np.linspace(0, T - 1, 7, dtype=int)
        ax_main.set_xticks(tick_positions)
        ax_main.set_xticklabels([x_labels[i] for i in tick_positions])
        ax_main.set_ylabel('NQH2O Price ($/AF)', fontsize=14)
        ax_main.grid(True, axis='y', alpha=0.2)
        ax_main.legend(frameon=True, loc='upper left')

        # === TOP: time-varying dispersion (std across all values including uncertainty bounds) ===
        ax_top.plot(t, std_t, lw=1.2, color='darkorange', label='Std. (with uncertainty)')
        ax_top.plot(t, std_t_base, lw=1.0, color='gray', linestyle='--', alpha=0.6, label='Std. (base only)')
        ax_top.fill_between(t, 0, std_t, color='orange', alpha=0.2)
        ax_top.set_ylabel('Std. across\nscenarios')
        ax_top.legend(loc='upper left', fontsize=8, framealpha=0.8)
        ax_top.grid(True, axis='y', alpha=0.2)
        ax_top.tick_params(axis='x', labelbottom=False)

        # === RIGHT: marginal density along price axis ===
        ax_right.hist(all_vals, bins=40, orientation='horizontal', density=True, 
                      color='gray', edgecolor='black', linewidth=0.4)
        ax_right.set_xlabel('Density')
        ax_right.tick_params(axis='y', labelleft=False)

        plt.tight_layout()
        plt.savefig('hybrid_result/scenario_predictions_marginals.png', bbox_inches='tight')
        plt.show()
        
        print(f"\nExtreme Scenarios:")
        print(f"Highest mean price: Scenario {results_df['mean_price'].idxmax()} ({results_df['mean_price'].max():.4f})")
        print(f"Lowest mean price: Scenario {results_df['mean_price'].idxmin()} ({results_df['mean_price'].min():.4f})")
        print(f"Highest volatility: Scenario {results_df['std_price'].idxmax()} ({results_df['std_price'].max():.4f})")
        print(f"Lowest volatility: Scenario {results_df['std_price'].idxmin()} ({results_df['std_price'].min():.4f})")
    
    if scenario_predictions:
        scenario_predictions_df = pd.DataFrame({
            f'scenario_{scen_id}': scenario_predictions[scen_id]['base'] 
            for scen_id in scenario_predictions.keys()
        }).T
        scenario_predictions_df.to_csv('hybrid_result/scenario_predictions.csv')
        
        bounds_data = {}
        for scen_id in scenario_predictions.keys():
            bounds_data[f'scenario_{scen_id}_upper'] = scenario_predictions[scen_id]['upper']
            bounds_data[f'scenario_{scen_id}_lower'] = scenario_predictions[scen_id]['lower']
        
        scenario_bounds_df = pd.DataFrame(bounds_data).T
        scenario_bounds_df.to_csv('hybrid_result/scenario_predictions_bounds.csv')
        
        print("\nSaved scenario predictions to 'hybrid_result/scenario_predictions.csv'")
        print("Saved scenario bounds to 'hybrid_result/scenario_predictions_bounds.csv'")
    

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBest model: {best_model_name}")
    if 'results' in locals():
        print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
        print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
    print(f"\nSelected {len(selected_features)} features from {len(X.columns)} original features")
    print(f"Selected features: {selected_features}")
    if scenario_results:
        print(f"\nAnalyzed {len(scenario_results)} scenarios")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)