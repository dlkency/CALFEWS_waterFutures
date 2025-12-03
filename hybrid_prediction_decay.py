"""
 Hybrid Prediction Model for Water Futures
===============================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pandas.api.types import CategoricalDtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LinearRegression
import xgboost as xgb
import warnings
import textwrap
import pickle
import os
from pathlib import Path

warnings.filterwarnings("ignore")

np.random.seed(42)

def load_and_process_data():
    """Load and process all data sources following the original script logic."""
    print("Loading and processing data...")
    
    hydro_df = pd.read_csv('merged_df_short_test.csv', index_col='Date')
    hydro_df.index = pd.to_datetime(hydro_df.index)
    
    suffix_mapping = {
        '_S': 'total_S',
        '_R': 'total_R', 
        '_SNPK': 'total_SNPK',
        '_fnf': 'total_fnf',
        '_Q': 'total_Q',
    }
    
    for suffix, output_col in suffix_mapping.items():
        matching_cols = [col for col in hydro_df.columns if col.endswith(suffix)]
        if matching_cols:
            hydro_df[output_col] = hydro_df[matching_cols].sum(axis=1)
            print(f"Created {output_col} from {len(matching_cols)} columns")
    
    wd_cols = [col for col in hydro_df.columns if 'wonderful' in col.lower() and 'delivery' in col.lower()]
    if wd_cols:
        hydro_df['total_wonderful_delivery'] = hydro_df[wd_cols].sum(axis=1)
        print(f"Created total_wonderful_delivery from {len(wd_cols)} columns")
    
    hro_pump_cols = [col for col in hydro_df.columns if 'HRO_pump' in col]
    trp_pump_cols = [col for col in hydro_df.columns if 'TRP_pump' in col]
    
    if hro_pump_cols:
        hydro_df['delta_HRO_pump'] = hydro_df[hro_pump_cols].sum(axis=1)
        print(f"Created delta_HRO_pump from {len(hro_pump_cols)} columns")
    
    if trp_pump_cols:
        hydro_df['delta_TRP_pump'] = hydro_df[trp_pump_cols].sum(axis=1)
        print(f"Created delta_TRP_pump from {len(trp_pump_cols)} columns")
    
    agg_mean_cols = ['total_S', 'total_R', 'total_Q', 'total_fnf', 'delta_HRO_pump', 'delta_TRP_pump']
    agg_last_cols = ['total_SNPK', 'total_wonderful_delivery']
    
    agg_mean_cols = [c for c in agg_mean_cols if c in hydro_df.columns]
    agg_last_cols = [c for c in agg_last_cols if c in hydro_df.columns]
    
    agg_mapping = {}
    agg_mapping.update({c: 'mean' for c in agg_mean_cols})
    agg_mapping.update({c: 'last' for c in agg_last_cols})
    
    monthly_hydro = hydro_df[list(agg_mapping.keys())].resample('M').agg(agg_mapping)
    monthly_hydro.index = pd.to_datetime(monthly_hydro.index)
    monthly_hydro['month'] = monthly_hydro.index.to_period('M').astype(str)
    monthly_hydro = monthly_hydro.reset_index(drop=False)
    monthly_hydro = monthly_hydro.rename(columns={'index': 'month_end'})
    monthly_hydro = monthly_hydro.set_index('month')
    
    price_df = pd.read_csv('calfews_src/data/price_index/Data_NQH2O.csv', parse_dates=['Date'])
    price_series = price_df.set_index('Date')['NQH2O'].dropna().sort_index()
    monthly_prices = price_series.resample('M').mean().dropna()
    log_prices = np.log(monthly_prices)
    
    drought_df = pd.read_csv('calfews_src/data/price_index/USDM-california.csv')
    drought_df['Date'] = drought_df['Date'].astype(str).str.strip()
    drought_df['date'] = pd.to_datetime(drought_df['Date'], format='%Y%m%d', errors='coerce')
    drought_df = drought_df.loc[~drought_df['date'].isna()].copy()
    drought_df = drought_df.sort_values('date').reset_index(drop=True)
    
    drought_columns = ['None', 'D0', 'D1', 'D2', 'D3', 'D4']
    present_columns = [col for col in drought_columns if col in drought_df.columns]
    
    for col in present_columns:
        drought_df[col] = pd.to_numeric(drought_df[col], errors='coerce').fillna(0.0)
    
    if 'None' in drought_df.columns:
        drought_df['pct_drought'] = 100.0 - drought_df['None']
    
    severity_weights = {'None': 0.0, 'D0': 1.0, 'D1': 2.0, 'D2': 3.0, 'D3': 4.0, 'D4': 5.0}
    severity_score = np.zeros(len(drought_df), dtype=float)
    
    for category, weight in severity_weights.items():
        if category in drought_df.columns:
            severity_score += drought_df[category].astype(float).fillna(0.0) * float(weight)
    
    drought_df['severity_weighted'] = severity_score / 100.0
    drought_df['month'] = drought_df['date'].dt.to_period('M')
    
    monthly_drought = drought_df.groupby('month').agg({
        'severity_weighted': 'mean'
    }).rename(columns={'severity_weighted': 'ca_drought_severity_mean'})
    
    monthly_period_index = pd.to_datetime(monthly_hydro.index, format='%Y-%m', errors='coerce').to_period('M')
    monthly_tmp = monthly_hydro.copy()
    monthly_tmp.index = monthly_period_index
    
    merged = monthly_tmp
    
    price_df_for_merge = log_prices.to_frame('log_nqh2o')
    price_df_for_merge.index = price_df_for_merge.index.to_period('M')
    print(f"Price data shape: {price_df_for_merge.shape}")
    print(f"Price data index sample: {price_df_for_merge.index[:5]}")
    print(f"Merged data index sample: {merged.index[:5]}")
    
    std_resid = merged.join(price_df_for_merge, how='inner')
    print(f"Data shape after merging: {std_resid.shape}")
    
    if std_resid.empty:
        print("Warning: No common periods found between hydro/drought data and price data")
        print("Trying outer join instead...")
        std_resid = merged.join(price_df_for_merge, how='outer')
        print(f"Data shape after outer join: {std_resid.shape}")
        std_resid = std_resid.dropna(subset=['log_nqh2o'])
        print(f"Data shape after dropping NaN: {std_resid.shape}")
    
    supervised = std_resid
    base_cols = ['total_S', 'total_R', 'total_Q', 'total_fnf', 'total_SNPK', 
                 'total_wonderful_delivery', 'log_nqh2o']
    pump_cols = ['delta_HRO_pump', 'delta_TRP_pump']
    available_pump_cols = [col for col in pump_cols if col in supervised.columns]
    supervised_new = supervised[base_cols + available_pump_cols]
    
    print(f"Final data shape: {supervised_new.shape}")
    print(f"Data sample:\n{supervised_new.head()}")
    print(f"Data info:\n{supervised_new.info()}")
    return supervised_new

def create_lag_features(df, target='log_nqh2o', hydro_cols=[], lag_start=3, lag_end=1, roll_window=3):
    """Create lag features and rolling statistics - NO ORIGINAL FEATURES TO PREVENT DATA LEAKAGE."""
    df_features = df.copy()
    
    for lag in range(lag_start, lag_end - 1, -1):
        df_features[f'{target}_lag{lag}'] = df_features[target].shift(lag)

    for col in hydro_cols:
        for lag in range(lag_start, lag_end - 1, -1):
            df_features[f'{col}_lag{lag}'] = df_features[col].shift(lag)
            
        if lag_start >= 2:
            df_features[f'{col}_momentum_{lag_start}'] = df_features[col].shift(lag_start) - df_features[col].shift(lag_start + 1)
        if lag_start >= 3:
            df_features[f'{col}_acceleration_{lag_start}'] = (df_features[col].shift(lag_start) - 
                                               2 * df_features[col].shift(lag_start + 1) + 
                                               df_features[col].shift(lag_start + 2))
        
        df_features[f'{col}_3M_roll_mean{lag_start}'] = df_features[col].shift(lag_start).rolling(roll_window).mean()
        df_features[f'{col}_3M_roll_std{lag_start}'] = df_features[col].shift(lag_start).rolling(roll_window).std()
        
        rolling_window = 3
        df_features[f'{col}_is_peak_{lag_start}'] = (df_features[col].shift(lag_start) > df_features[col].shift(lag_start).rolling(rolling_window, center=True).max()).astype(int)
        
        df_features[f'{col}_consecutive_up_{lag_start}'] = ((df_features[col].shift(lag_start) > df_features[col].shift(lag_start + 1)) & 
                                                           (df_features[col].shift(lag_start + 1) > df_features[col].shift(lag_start + 2))).astype(int)
        
        df_features[f'{col}_price_acceleration_{lag_start}'] = (df_features[col].shift(lag_start) - 2*df_features[col].shift(lag_start + 1) + 
                                                              df_features[col].shift(lag_start + 2))
    
    lag_features = [col for col in df_features.columns if 'lag' in col or 'momentum' in col or 'acceleration' in col or 'roll' in col or 'trend' in col or 'volatility' in col or 'peak' in col or 'consecutive' in col or col == target]
    df_features = df_features[lag_features]
    
    print(f"Created {len(lag_features)} features (removed original features to prevent data leakage)")
    print(f"Sample lag features: {[col for col in lag_features if 'lag' in col][:5]}")
    
    return df_features.dropna()


def permutation_feature_selection(X, y, model, n_features=10, always_keep='log_nqh2o_lag1'):
    """Select features using permutation importance."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for i in range(len(upper.columns)):
        for j in range(i+1, len(upper.columns)):
            if upper.iloc[i, j] > 0.8:
                high_corr_pairs.append((upper.columns[i], upper.columns[j]))
    
    features_to_remove = set()
    for feat1, feat2 in high_corr_pairs:
        corr1 = abs(X[feat1].corr(y))
        corr2 = abs(X[feat2].corr(y))
        if corr1 > corr2:
            features_to_remove.add(feat2)
        else:
            features_to_remove.add(feat1)
    
    X_filtered = X.drop(columns=features_to_remove)
    
    if always_keep not in X_filtered.columns:
        if always_keep in X.columns:
            X_filtered[always_keep] = X[always_keep]
        else:
            print(f"Warning: {always_keep} not found in original features")
    
    model.fit(X_filtered, y)
    perm_importance = permutation_importance(model, X_filtered, y, n_repeats=10, random_state=42)
    
    feature_scores = pd.DataFrame({
        'feature': X_filtered.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    top_features = feature_scores.head(n_features)['feature'].tolist()
    if always_keep in X_filtered.columns and always_keep not in top_features:
        top_features = top_features[:-1] + [always_keep]
    
    X_final = X_filtered[top_features]
    
    print(f"Selected {X_final.shape[1]} features using permutation importance")
    print(f"Final selected features: {top_features}")
    print(f"Top 5 features: {top_features[:5]}")
    
    return X_final, feature_scores


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }




def get_coarse_param_grid(model_type, horizon_length=None):
    """
    Get coarse parameter grid for initial hyperparameter search (Stage 1).
    """
    if model_type == 'XGBoost':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.1, 1],
            'reg_lambda': [0.1, 1]
        }
    elif model_type == 'RandomForest':
        if horizon_length and horizon_length >= 9:
            return {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 8],
                'max_features': ['sqrt'],
                'bootstrap': [True]
            }
        else:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
    elif model_type == 'SVR':
        return {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['rbf', 'linear']
        }
    return {}

def get_refined_param_grid(model_type, best_params, fast_mode=True):
    """
    Get refined parameter grid around best parameters from Stage 1 (Stage 2).

    """
    if model_type == 'XGBoost':
        best_n_est = best_params.get('n_estimators', 100)
        best_max_depth = best_params.get('max_depth', 3)
        best_lr = best_params.get('learning_rate', 0.1)
        best_subsample = best_params.get('subsample', 1.0)
        best_colsample = best_params.get('colsample_bytree', 1.0)
        best_alpha = best_params.get('reg_alpha', 0.1)
        best_lambda = best_params.get('reg_lambda', 0.1)
        
        if fast_mode:
            return {
                'n_estimators': [best_n_est, min(300, best_n_est + 50)],
                'max_depth': [best_max_depth, min(10, best_max_depth + 1)],
                'learning_rate': [best_lr, max(0.005, best_lr - 0.05)],
                'subsample': [best_subsample, max(0.7, best_subsample - 0.1)],
                'colsample_bytree': [best_colsample, max(0.7, best_colsample - 0.1)],
                'reg_alpha': [best_alpha, min(2, best_alpha + 0.1)],
                'reg_lambda': [best_lambda, min(2, best_lambda + 0.1)]
            }
        else:
            return {
                'n_estimators': sorted(list(set([max(50, best_n_est - 50), best_n_est, min(300, best_n_est + 50)]))),
                'max_depth': sorted(list(set([max(2, best_max_depth - 1), best_max_depth, min(10, best_max_depth + 1)]))),
                'learning_rate': sorted(list(set([max(0.005, best_lr - 0.05), best_lr, min(0.3, best_lr + 0.05)]))),
                'subsample': sorted(list(set([max(0.7, best_subsample - 0.1), best_subsample, min(1.0, best_subsample + 0.1)]))),
                'colsample_bytree': sorted(list(set([max(0.7, best_colsample - 0.1), best_colsample, min(1.0, best_colsample + 0.1)]))),
                'reg_alpha': sorted(list(set([max(0.01, best_alpha - 0.1), best_alpha, min(2, best_alpha + 0.1)]))),
                'reg_lambda': sorted(list(set([max(0.01, best_lambda - 0.1), best_lambda, min(2, best_lambda + 0.1)])))
            }
    
    elif model_type == 'RandomForest':
        best_n_est = best_params.get('n_estimators', 100)
        best_max_depth = best_params.get('max_depth', 5)
        best_min_split = best_params.get('min_samples_split', 2)
        best_min_leaf = best_params.get('min_samples_leaf', 1)
        best_max_feat = best_params.get('max_features', 'sqrt')
        best_bootstrap = best_params.get('bootstrap', True)
        
        refined_grid = {
            'bootstrap': [best_bootstrap],
            'max_features': [best_max_feat]
        }
        
        if fast_mode:
            refined_grid['n_estimators'] = [best_n_est, min(300, best_n_est + 50)]
            
            if best_max_depth is None:
                refined_grid['max_depth'] = [None, 10]
            else:
                refined_grid['max_depth'] = [best_max_depth, min(15, best_max_depth + 2)]
            
            refined_grid['min_samples_split'] = [best_min_split, min(10, best_min_split + 2)]
            refined_grid['min_samples_leaf'] = [best_min_leaf, min(5, best_min_leaf + 1)]
        else:
            n_est_options = [max(50, best_n_est - 50), best_n_est, min(300, best_n_est + 50)]
            refined_grid['n_estimators'] = sorted(list(set(n_est_options)))
            
            if best_max_depth is None:
                refined_grid['max_depth'] = [None, 10, 15]
            else:
                depth_options = [max(3, best_max_depth - 2), best_max_depth, min(15, best_max_depth + 2)]
                refined_grid['max_depth'] = sorted(list(set(depth_options)))
            
            split_options = [max(2, best_min_split - 1), best_min_split, min(10, best_min_split + 2)]
            refined_grid['min_samples_split'] = sorted([s for s in split_options if s >= 2])
            
            leaf_options = [max(1, best_min_leaf - 1), best_min_leaf, min(5, best_min_leaf + 1)]
            refined_grid['min_samples_leaf'] = sorted([l for l in leaf_options if l >= 1])
        
        return refined_grid
    
    elif model_type == 'SVR':
        best_C = best_params.get('C', 10)
        best_gamma = best_params.get('gamma', 'scale')
        best_epsilon = best_params.get('epsilon', 0.1)
        best_kernel = best_params.get('kernel', 'rbf')
        
        if fast_mode:
            if isinstance(best_C, (int, float)):
                c_options = [best_C, min(1000, best_C * 2)]
            else:
                c_options = [best_C, 10]
            
            if isinstance(best_gamma, str):
                gamma_options = [best_gamma, 0.001]
            else:
                gamma_options = [best_gamma, min(1, best_gamma * 2)]
            
            epsilon_options = [best_epsilon, min(0.5, best_epsilon + 0.05)]
        else:
            if isinstance(best_C, (int, float)):
                c_options = [max(0.01, best_C * 0.5), best_C, min(1000, best_C * 2)]
            else:
                c_options = [0.1, 1, 10, 100]
            
            if isinstance(best_gamma, str):
                gamma_options = [best_gamma, 0.001, 0.01, 0.1]
            else:
                gamma_options = [max(0.0001, best_gamma * 0.5), best_gamma, min(1, best_gamma * 2)]
            
            epsilon_options = [max(0.001, best_epsilon - 0.05), best_epsilon, min(0.5, best_epsilon + 0.05)]
        
        return {
            'C': c_options,
            'gamma': gamma_options,
            'epsilon': epsilon_options,
            'kernel': [best_kernel]
        }
    
    return {}

def optimize_hyperparameters(X_train, y_train, model_type='XGBoost', cv_folds=3, tuning_stages=2, fast_refined=True, horizon_length=None):
    """
    Optimize hyperparameters using multi-stage grid search with time series cross-validation.
    """
    print(f"\n{'='*60}")
    print(f"Optimizing hyperparameters for {model_type} (Multi-stage tuning)")
    if horizon_length:
        print(f"Horizon length: {horizon_length} months (applying regularization constraints)")
    print(f"{'='*60}")
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    if model_type == 'XGBoost':
        base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
    elif model_type == 'RandomForest':
        base_model = RandomForestRegressor(random_state=42, n_jobs=1)
    elif model_type == 'SVR':
        base_model = SVR()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    best_params = None
    best_score = float('-inf')
    best_estimator = None
    
    print(f"\n--- Stage 1: Coarse Grid Search ---")
    param_grid_coarse = get_coarse_param_grid(model_type, horizon_length=horizon_length)
    
    grid_size = 1
    for v in param_grid_coarse.values():
        grid_size *= len(v)
    print(f"Grid size: {grid_size} combinations")
    
    grid_search_coarse = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid_coarse,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1
    )
    
    grid_search_coarse.fit(X_train, y_train.ravel() if model_type == 'SVR' else y_train)
    
    best_params = grid_search_coarse.best_params_
    best_score = grid_search_coarse.best_score_
    best_estimator = grid_search_coarse.best_estimator_
    
    print(f"Stage 1 Best Parameters: {best_params}")
    print(f"Stage 1 Best CV Score (MSE): {-best_score:.6f}")
    
    if tuning_stages >= 2:
        print(f"\n--- Stage 2: Refined Grid Search (fast_mode={fast_refined}) ---")
        param_grid_refined = get_refined_param_grid(model_type, best_params, fast_mode=fast_refined)
        
        grid_size_refined = 1
        for v in param_grid_refined.values():
            grid_size_refined *= len(v)
        print(f"Refined grid size: {grid_size_refined} combinations")
        print(f"Refined grid: {param_grid_refined}")
        
        grid_search_refined = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid_refined,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=1
        )
        
        grid_search_refined.fit(X_train, y_train.ravel() if model_type == 'SVR' else y_train)
        
        if grid_search_refined.best_score_ > best_score:
            print(f"✓ Refined search found better parameters!")
            print(f"  Improvement: {-(grid_search_refined.best_score_ - best_score):.6f} MSE")
            best_params = grid_search_refined.best_params_
            best_score = grid_search_refined.best_score_
            best_estimator = grid_search_refined.best_estimator_
        else:
            print(f"Refined search did not improve (best from Stage 1 retained)")
        
        print(f"Stage 2 Best Parameters: {best_params}")
        print(f"Stage 2 Best CV Score (MSE): {-best_score:.6f}")
    
    if tuning_stages >= 3:
        print(f"\n--- Stage 3: Fine-tuning ---")
        param_grid_fine = {}
        
        if model_type == 'XGBoost':
            for key, value in best_params.items():
                if isinstance(value, (int, float)):
                    if key == 'learning_rate':
                        param_grid_fine[key] = [max(0.001, value - 0.01), value, min(0.3, value + 0.01)]
                    elif key in ['n_estimators']:
                        param_grid_fine[key] = [max(50, value - 25), value, min(300, value + 25)]
                    elif key in ['reg_alpha', 'reg_lambda']:
                        param_grid_fine[key] = [max(0.01, value - 0.05), value, min(2, value + 0.05)]
                    elif key in ['subsample', 'colsample_bytree']:
                        param_grid_fine[key] = [max(0.7, value - 0.05), value, min(1.0, value + 0.05)]
                    else:
                        param_grid_fine[key] = [value]
                else:
                    param_grid_fine[key] = [value]
        
        elif model_type == 'RandomForest':
            for key, value in best_params.items():
                if key == 'n_estimators':
                    param_grid_fine[key] = [max(50, value - 25), value, min(300, value + 25)]
                elif key == 'max_depth' and value is not None:
                    param_grid_fine[key] = [max(3, value - 1), value, min(15, value + 1)]
                else:
                    param_grid_fine[key] = [value]
        
        elif model_type == 'SVR':
            for key, value in best_params.items():
                if key == 'C' and isinstance(value, (int, float)):
                    param_grid_fine[key] = [value * 0.8, value, value * 1.2]
                elif key == 'gamma' and isinstance(value, (int, float)):
                    param_grid_fine[key] = [max(0.0001, value * 0.8), value, min(1, value * 1.2)]
                elif key == 'epsilon' and isinstance(value, (int, float)):
                    param_grid_fine[key] = [max(0.001, value - 0.01), value, min(0.5, value + 0.01)]
                else:
                    param_grid_fine[key] = [value]
        
        grid_size_fine = 1
        for v in param_grid_fine.values():
            grid_size_fine *= len(v)
        print(f"Fine grid size: {grid_size_fine} combinations")
        
        grid_search_fine = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid_fine,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=1
        )
        
        grid_search_fine.fit(X_train, y_train.ravel() if model_type == 'SVR' else y_train)
        
        if grid_search_fine.best_score_ > best_score:
            print(f"✓ Fine-tuning found better parameters!")
            print(f"  Improvement: {-(grid_search_fine.best_score_ - best_score):.6f} MSE")
            best_params = grid_search_fine.best_params_
            best_score = grid_search_fine.best_score_
            best_estimator = grid_search_fine.best_estimator_
        else:
            print(f"Fine-tuning did not improve (best from previous stage retained)")
        
        print(f"Stage 3 Best Parameters: {best_params}")
        print(f"Stage 3 Best CV Score (MSE): {-best_score:.6f}")
    
    print(f"\n{'='*60}")
    print(f"Final Best Parameters for {model_type}: {best_params}")
    print(f"Final Best CV Score (MSE): {-best_score:.6f}")
    print(f"{'='*60}\n")
    
    return best_estimator, best_params


def train_models(X_train, y_train, X_test, y_test, feature_names=None, use_grid_search=True, tuning_stages=2, fast_refined=True, horizon_length=None):

    results = {}
    feature_importance = {}

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)
    
    y_train_mean = np.mean(y_train_scaled)
    y_train_std = np.std(y_train_scaled)
    y_train_skew = ((y_train_scaled - y_train_mean) ** 3).mean() / (y_train_std ** 3)
    
    if use_grid_search:
        print(f"\n{'#'*60}")
        print(f"Using Multi-Stage Grid Search for Hyperparameter Optimization")
        print(f"Tuning Stages: {tuning_stages}")
        print(f"{'#'*60}")
        
        xgb_model, xgb_params = optimize_hyperparameters(X_train_scaled, y_train_scaled, 'XGBoost', 
                                                          tuning_stages=tuning_stages, fast_refined=fast_refined,
                                                          horizon_length=horizon_length)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_pred_orig = scaler_y.inverse_transform(xgb_pred.reshape(-1, 1))
        xgb_metrics = calculate_metrics(y_test_orig, xgb_pred_orig)
        results['XGBoost'] = (xgb_pred_orig, xgb_metrics)
        
        rf_model, rf_params = optimize_hyperparameters(X_train_scaled, y_train_scaled, 'RandomForest', 
                                                        tuning_stages=tuning_stages, fast_refined=fast_refined,
                                                        horizon_length=horizon_length)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_pred_orig = scaler_y.inverse_transform(rf_pred.reshape(-1, 1))
        rf_metrics = calculate_metrics(y_test_orig, rf_pred_orig)
        results['RandomForest'] = (rf_pred_orig, rf_metrics)
        
        svr_model, svr_params = optimize_hyperparameters(X_train_scaled, y_train_scaled, 'SVR', 
                                                          tuning_stages=tuning_stages, fast_refined=fast_refined,
                                                          horizon_length=horizon_length)
        svr_pred = svr_model.predict(X_test_scaled)
        svr_pred_orig = scaler_y.inverse_transform(svr_pred.reshape(-1, 1))
        svr_metrics = calculate_metrics(y_test_orig, svr_pred_orig)
        results['SVR'] = (svr_pred_orig, svr_metrics)
        
        print(f"\n{'#'*60}")
        print(f"Final Optimized Parameters Summary:")
        print(f"{'#'*60}")
        print(f"XGBoost: {xgb_params}")
        print(f"RandomForest: {rf_params}")
        print(f"SVR: {svr_params}")
        print(f"{'#'*60}\n")
        
    else:
        print("Using default parameters...")
        
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_scaled, y_train_scaled)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_pred_orig = scaler_y.inverse_transform(xgb_pred.reshape(-1, 1))
        xgb_metrics = calculate_metrics(y_test_orig, xgb_pred_orig)
        results['XGBoost'] = (xgb_pred_orig, xgb_metrics)
        
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
        rf_model.fit(X_train_scaled, y_train_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_pred_orig = scaler_y.inverse_transform(rf_pred.reshape(-1, 1))
        rf_metrics = calculate_metrics(y_test_orig, rf_pred_orig)
        results['RandomForest'] = (rf_pred_orig, rf_metrics)
        
        svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr_model.fit(X_train_scaled, y_train_scaled.ravel())
        svr_pred = svr_model.predict(X_test_scaled)
        svr_pred_orig = scaler_y.inverse_transform(svr_pred.reshape(-1, 1))
        svr_metrics = calculate_metrics(y_test_orig, svr_pred_orig)
        results['SVR'] = (svr_pred_orig, svr_metrics)
    
    if feature_names is not None:
        xgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance['XGBoost'] = xgb_importance
        
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance['RandomForest'] = rf_importance
        
        svr_perm_importance = permutation_importance(svr_model, X_train_scaled, y_train_scaled.ravel(), n_repeats=10, random_state=42)
        svr_importance_raw = np.abs(svr_perm_importance.importances_mean)
        svr_importance_normalized = svr_importance_raw / svr_importance_raw.sum() if svr_importance_raw.sum() > 0 else svr_importance_raw
        svr_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': svr_importance_normalized,
            'std': svr_perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        feature_importance['SVR'] = svr_importance
    
    print(f"\nFinal Model Performance:")
    print(f"RandomForest: {rf_metrics}")
    print(f"XGBoost: {xgb_metrics}")
    print(f"SVR: {svr_metrics}")
    
    trained_models = {
        'XGBoost': {
            'model': xgb_model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'metrics': xgb_metrics
        },
        'RandomForest': {
            'model': rf_model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'metrics': rf_metrics
        },
        'SVR': {
            'model': svr_model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'metrics': svr_metrics
        }
    }
    
    return results, y_test_orig, feature_importance, trained_models


def map_window_name(window_name):
    """Map window names from 'T-X to T-Y' format to human-readable format."""
    window_mapping = {
        'T-13 to T-11': '11-Month Horizon',
        'T-11 to T-9': '9-Month Horizon',
        'T-8 to T-6': '6-Month Horizon',
        'T-5 to T-3': '3-Month Horizon',
        'T-3 to T-1': '1-Month Horizon'
    }
    return window_mapping.get(window_name, window_name)

def map_feature_names(feature_name):
    """Map feature names to shorter, clearer text for visualization."""
    feature_mapping = {
        # Target lags
        'log_nqh2o_lag1': 'Price (t-1)',
        'log_nqh2o_lag2': 'Price (t-2)', 
        'log_nqh2o_lag3': 'Price (t-3)',
        'log_nqh2o_lag4': 'Price (t-4)',
        'log_nqh2o_lag5': 'Price (t-5)',
        'log_nqh2o_lag6': 'Price (t-6)',
        'log_nqh2o_lag7': 'Price (t-7)',
        'log_nqh2o_lag8': 'Price (t-8)',
        'log_nqh2o_lag9': 'Price (t-9)',
        'log_nqh2o_lag10': 'Price (t-10)',
        'log_nqh2o_lag11': 'Price (t-11)',
        
        # Storage features
        'total_S_lag1': 'Storage (t-1)',
        'total_S_lag2': 'Storage (t-2)',
        'total_S_lag3': 'Storage (t-3)',
        'total_S_lag4': 'Storage (t-4)',
        'total_S_lag5': 'Storage (t-5)',
        'total_S_lag6': 'Storage (t-6)',
        'total_S_lag7': 'Storage (t-7)',
        'total_S_lag8': 'Storage (t-8)',
        'total_S_lag9': 'Storage (t-9)',
        'total_S_lag10': 'Storage (t-10)',
        'total_S_lag11': 'Storage (t-11)',
        
        # Runoff features
        'total_R_lag1': 'Release (t-1)',
        'total_R_lag2': 'Release (t-2)',
        'total_R_lag3': 'Release (t-3)',
        'total_R_lag4': 'Release (t-4)',
        'total_R_lag5': 'Release (t-5)',
        'total_R_lag6': 'Release (t-6)',
        'total_R_lag7': 'Release (t-7)',
        'total_R_lag8': 'Release (t-8)',
        'total_R_lag9': 'Release (t-9)',
        'total_R_lag10': 'Release (t-10)',
        'total_R_lag11': 'Release (t-11)',
        
        # Flow features
        'total_Q_lag1': 'Inflow (t-1)',
        'total_Q_lag2': 'Inflow (t-2)',
        'total_Q_lag3': 'Inflow (t-3)',
        'total_Q_lag4': 'Inflow (t-4)',
        'total_Q_lag5': 'Inflow (t-5)',
        'total_Q_lag6': 'Inflow (t-6)',
        'total_Q_lag7': 'Inflow (t-7)',
        'total_Q_lag8': 'Inflow (t-8)',
        'total_Q_lag9': 'Inflow (t-9)',
        'total_Q_lag10': 'Inflow (t-10)',
        'total_Q_lag11': 'Inflow (t-11)',
        
        # FNF features
        'total_fnf_lag1': 'FNF (t-1)',
        'total_fnf_lag2': 'FNF (t-2)',
        'total_fnf_lag3': 'FNF (t-3)',
        'total_fnf_lag4': 'FNF (t-4)',
        'total_fnf_lag5': 'FNF (t-5)',
        'total_fnf_lag6': 'FNF (t-6)',
        'total_fnf_lag7': 'FNF (t-7)',
        'total_fnf_lag8': 'FNF (t-8)',
        'total_fnf_lag9': 'FNF (t-9)',
        'total_fnf_lag10': 'FNF (t-10)',
        'total_fnf_lag11': 'FNF (t-11)',
        
        # SNPK features
        'total_SNPK_lag1': 'Snowpack (t-1)',
        'total_SNPK_lag2': 'Snowpack (t-2)',
        'total_SNPK_lag3': 'Snowpack (t-3)',
        'total_SNPK_lag4': 'Snowpack (t-4)',
        'total_SNPK_lag5': 'Snowpack (t-5)',
        'total_SNPK_lag6': 'Snowpack (t-6)',
        'total_SNPK_lag7': 'Snowpack (t-7)',
        'total_SNPK_lag8': 'Snowpack (t-8)',
        'total_SNPK_lag9': 'Snowpack (t-9)',
        'total_SNPK_lag10': 'Snowpack (t-10)',
        'total_SNPK_lag11': 'Snowpack (t-11)',
        
        # Pump features
        'delta_HRO_pump_lag1': 'Banks Pumping (t-1)',
        'delta_HRO_pump_lag2': 'Banks Pumping (t-2)',
        'delta_HRO_pump_lag3': 'Banks Pumping (t-3)',
        'delta_HRO_pump_lag4': 'Banks Pumping (t-4)',
        'delta_HRO_pump_lag5': 'Banks Pumping (t-5)',
        'delta_HRO_pump_lag6': 'Banks Pumping (t-6)',
        'delta_HRO_pump_lag7': 'Banks Pumping (t-7)',
        'delta_HRO_pump_lag8': 'Banks Pumping (t-8)',
        'delta_HRO_pump_lag9': 'Banks Pumping (t-9)',
        'delta_HRO_pump_lag10': 'Banks Pumping (t-10)',
        'delta_HRO_pump_lag11': 'Banks Pumping (t-11)',
        
        'delta_TRP_pump_lag1': 'Tracy Pumping (t-1)',
        'delta_TRP_pump_lag2': 'Tracy Pumping (t-2)',
        'delta_TRP_pump_lag3': 'Tracy Pumping (t-3)',
        'delta_TRP_pump_lag4': 'Tracy Pumping (t-4)',
        'delta_TRP_pump_lag5': 'Tracy Pumping (t-5)',
        'delta_TRP_pump_lag6': 'Tracy Pumping (t-6)',
        'delta_TRP_pump_lag7': 'Tracy Pumping (t-7)',
        'delta_TRP_pump_lag8': 'Tracy Pumping (t-8)',
        'delta_TRP_pump_lag9': 'Tracy Pumping (t-9)',
        'delta_TRP_pump_lag10': 'Tracy Pumping (t-10)',
        'delta_TRP_pump_lag11': 'Tracy Pumping (t-11)',
        
        # Momentum features
        'total_S_momentum_11': 'Storage Momentum (t-11)',
        'total_R_momentum_11': 'Release Momentum (t-11)',
        'total_Q_momentum_11': 'Inflow Momentum (t-11)',
        'total_fnf_momentum_11': 'FNF Momentum (t-11)',
        'total_SNPK_momentum_11': 'Snowpack Momentum (t-11)',
        'total_wonderful_delivery_momentum_11': 'Wonderful Delivery Momentum (t-11)',
        'ca_drought_severity_mean_momentum_11': 'Drought Index Momentum (t-11)',
        'delta_HRO_pump_momentum_11': 'Banks Pumping \nMomentum (t-11)',
        'delta_TRP_pump_momentum_11': 'Tracy Pumping \n Momentum (t-11)',
        'delta_HRO_pump_3M_roll_mean3': 'Banks Pumping 3M Mean (t-3)',
        # Acceleration features
        'total_S_acceleration_11': 'Storage Acceleration (t-11)',
        'total_R_acceleration_11': 'Release Acceleration (t-11)',
        'total_Q_acceleration_11': 'Inflow Acceleration (t-11)',
        'total_fnf_acceleration_11': 'FNF Acceleration (t-11)',
        'total_SNPK_acceleration_11': 'Snowpack Acceleration (t-11)',
        'total_wonderful_delivery_acceleration_11': 'Wonderful Delivery Acceleration (t-11)',
        'ca_drought_severity_mean_acceleration_11': 'Drought Index Acceleration (t-11)',
        'ca_drought_severity_mean_acceleration_3': 'Drought Index Acceleration (t-3)',
        'delta_HRO_pump_acceleration_11': 'Banks Pumping Acceleration (t-11)',
        'delta_TRP_pump_acceleration_11': 'Tracy Pumping Acceleration (t-11)',
        
        # Rolling features
        'total_S_3M_roll_std3': 'Storage 3M Std (t-3)',
        'total_SNPK_3M_roll_std3': 'Snowpack 3M Std (t-3)',
        'total_S_3M_roll_mean11': 'Storage 3M Mean (t-11)',
        'total_R_3M_roll_mean11': 'Release 3M Mean (t-11)',
        'total_R_3M_roll_mean5': 'Release 3M Mean (t-5)',
        'total_R_3M_roll_mean3': 'Release 3M Mean (t-3)',
        'total_Q_3M_roll_mean11': 'Inflow 3M Mean (t-11)',
        'total_Q_3M_roll_mean8': 'Inflow 3M Mean (t-8)',
        'total_fnf_3M_roll_mean11': 'FNF 3M Mean (t-11)',
        'total_SNPK_3M_roll_mean11': 'Snowpack 3M Mean (t-11)',
        'total_wonderful_delivery_3M_roll_mean11': 'Wonderful Delivery 3M Mean (t-11)',
        'ca_drought_severity_mean_3M_roll_mean11': 'Drought Index 3M Mean (t-11)',
        'ca_drought_severity_mean_3M_roll_std5': 'Drought Index 3M Std (t-5)',
        'ca_drought_severity_mean_3M_roll_std8': 'Drought Index 3M Std (t-8)',
        'ca_drought_severity_mean_acce': 'Drought Index 3M Std (t-5)',
        'delta_HRO_pump_3M_roll_mean11': 'Banks Pumping 3M Mean (t-11)',
        'delta_TRP_pump_3M_roll_mean11': 'Tracy Pumping 3M Mean (t-11)',
        'delta_TRP_pump_3M_roll_mean8': 'Tracy Pumping 3M Mean (t-8)',
        'delta_HRO_pump_3M_roll_mean8': 'Banks Pumping 3M Mean (t-8)',
        'delta_TRP_pump_3M_roll_mean5': 'Tracy Pumping 3M Mean (t-5)',
        'delta_HRO_pump_3M_roll_mean5': 'Banks Pumping 3M Mean (t-5)',
        'total_S_3M_roll_std11': 'Storage 3M Std (t-11)',
        'total_R_3M_roll_std11': 'Release 3M Std (t-11)',
        'total_Q_3M_roll_std11': 'Inflow 3M Std (t-11)',
        'total_fnf_3M_roll_std11': 'FNF 3M Std (t-11)',
        'total_SNPK_3M_roll_std11': 'Snowpack 3M Std (t-11)',
        'total_wonderful_delivery_3M_roll_std11': 'Wonderful Delivery 3M Std (t-11)',
        'delta_HRO_pump_3M_roll_std11': 'Banks Pumping 3M Std (t-11)',
        'delta_TRP_pump_3M_roll_std11': 'Tracy Pumping 3M Std (t-11)'
    }
    
    return feature_mapping.get(feature_name, feature_name.replace('_', ' ').title())

def plot_feature_importance(all_feature_importance):
    """Plot feature importance for all lag windows and models."""
    if not all_feature_importance:
        print("No feature importance data available")
        return
    
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 16,
        'figure.titlesize': 14
    })

    models = ['XGBoost', 'RandomForest', 'SVR']
    model_colors = {'XGBoost': '#91bfdb', 'RandomForest': '#fc8d59', 'SVR': '#ffffbf'}

    def wrap_label(s, width=28):
        return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

    n_windows = len(all_feature_importance)
    fig, axes = plt.subplots(
        n_windows, 3, figsize=(20, 4.8 * n_windows),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.3, 'left': 0.28, 'right': 0.92}
    )
    if n_windows == 1:
        axes = axes.reshape(1, 3)

    label_counter = 0
    for i in range(n_windows):
        for j in range(3):
            label = chr(ord('a') + label_counter)
            axes[i, j].text(0.02, 1.02, f'({label})', 
                           transform=axes[i, j].transAxes,
                           fontsize=15, fontweight='bold',
                           verticalalignment='bottom',
                           horizontalalignment='left')
            label_counter += 1

    all_windows_reversed = list(reversed(list(all_feature_importance.items())))

    model_max = {m: 0.0 for m in models}
    for _, window_data in all_feature_importance.items():
        for m in models:
            if m in window_data and not window_data[m].empty:
                model_max[m] = max(model_max[m], window_data[m]['importance'].head(6).max())
    for m in model_max:
        model_max[m] = (model_max[m] * 1.1) if model_max[m] > 0 else 1.0

    for i, (window_name, window_data) in enumerate(all_windows_reversed):
        for j, m in enumerate(models):
            ax = axes[i, j]

            if m in window_data and not window_data[m].empty:
                df = window_data[m].head(6).copy().iloc[::-1]
                bars = ax.barh(
                    range(len(df)), df['importance'],
                    color=model_colors[m], alpha=0.9, edgecolor='black', linewidth=0.4
                )

                for y, (_, row) in enumerate(df.iterrows()):
                    val = row['importance']
                    txt = f"{val:.3f}"
                    xpad = model_max[m] * 0.01
                    if val > model_max[m] * 0.15:
                        ax.text(val - xpad, y, txt, va='center', ha='right', fontsize=9, color='white')
                    else:
                        ax.text(val + xpad, y, txt, va='center', ha='left', fontsize=9)

                ax.set_yticks(range(len(df)))
                ax.set_yticklabels([wrap_label(map_feature_names(f), 26) for f in df['feature']], fontsize=13)
                ax.set_xlim(0, model_max[m])
                ax.set_xlabel('Importance', fontsize=11)
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)

        axes[i, 1].set_title(f'{map_window_name(window_name)}', fontsize=14, pad=14, fontweight='bold')

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=model_colors[m], alpha=0.9, label=m) for m in models]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.05),
            ncol=3, fontsize=16, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0.28, 0.03, 0.92, 0.965])
    plt.savefig('hybrid_result/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def _extract_base_feature_name(feature_name):
    """
    Extract the base aggregated feature name by removing all feature engineering suffixes.
    """
    base_name = feature_name
    suffixes_to_remove = [
        '_lag', '_momentum_', '_acceleration_', '_3M_roll_', 
        '_is_peak_', '_consecutive_', '_price_acceleration_'
    ]
    
    for suffix in suffixes_to_remove:
        if suffix in base_name:
            base_name = base_name.split(suffix)[0]
    
    return base_name


RESERVOIR_NAME_MAPPING = {
    'newmelones_S': 'New Melones',
    'exchequer_S': 'Exchequer',
    'shasta_S': 'Shasta',
    'isabella_S': 'Isabella',
    'isabella': 'Isabella',
    'donpedro_S': 'Don Pedro',
    'donpedro': 'Don Pedro',
    'millerton_S': 'Millerton',
    'millerton': 'Millerton',
    'pineflat_S': 'Pine Flat',
    'pineflat': 'Pine Flat',
    'success_S': 'Success',
    'success': 'Success',
    
    'newmelones_R': 'New Melones',
    'exchequer_R': 'Exchequer',
    'shasta_R': 'Shasta',
    'isabella_R': 'Isabella',
    'donpedro_R': 'Don Pedro',
    'millerton_R': 'Millerton',
    'pineflat_R': 'Pine Flat',
    'success_R': 'Success',
    
    'donpedronpk_SNPK': 'Don Pedro',
    'donpedronpk': 'Don Pedro',
    'millertonnpk_SNPK': 'Millerton',
    'millertonnpk': 'Millerton',
    'successnpk_SNPK': 'Success',
    'successnpk': 'Success',
    
    'newmelones_Q': 'New Melones',
    'exchequer_Q': 'Exchequer',
    'shasta_Q': 'Shasta',
    'isabella_Q': 'Isabella',
    'donpedro_Q': 'Don Pedro',
    'millerton_Q': 'Millerton',
    'pineflat_Q': 'Pine Flat',
    'success_Q': 'Success',
}


def _clean_column_name(col_name):
    """
    Clean a column name for display by removing suffixes and formatting.
    Uses a mapping dictionary for human-friendly reservoir names.
    """
    col_lower = col_name.lower()
    for key, value in RESERVOIR_NAME_MAPPING.items():
        if key.lower() == col_lower:
            return value
    
    base_name = col_name.replace('_S', '').replace('_R', '').replace('_SNPK', '')
    base_name = base_name.replace('_fnf', '').replace('_Q', '')
    base_lower = base_name.lower()
    
    for key, value in RESERVOIR_NAME_MAPPING.items():
        key_base = key.replace('_S', '').replace('_R', '').replace('_SNPK', '')
        key_base = key_base.replace('_fnf', '').replace('_Q', '').lower()
        if key_base == base_lower:
            return value
    
    clean_name = base_name.replace('_', ' ').title()
    return clean_name


def _calculate_correlations_by_lag(constituent_cols, supervised_data, target_col, lags=[3, 6]):
    """
    Calculate correlations between lagged constituent columns and the target variable.
    Returns separate correlations for each lag.
    """
    correlations_by_lag = {lag: {} for lag in lags}
    
    for base_col in constituent_cols:
        for lag in lags:
            lagged_col = f"{base_col}_lag{lag}"
            
            if lagged_col in supervised_data.columns:
                try:
                    feature_data = supervised_data[lagged_col].dropna()
                    target_data = supervised_data[target_col].dropna()
                    
                    common_idx = feature_data.index.intersection(target_data.index)
                    if len(common_idx) > 10:
                        feature_aligned = feature_data.loc[common_idx]
                        target_aligned = target_data.loc[common_idx]
                        
                        corr = feature_aligned.corr(target_aligned)
                        if not np.isnan(corr):
                            correlations_by_lag[lag][base_col] = abs(corr)
                        else:
                            correlations_by_lag[lag][base_col] = 0
                    else:
                        correlations_by_lag[lag][base_col] = 0
                except Exception:
                    correlations_by_lag[lag][base_col] = 0
            else:
                correlations_by_lag[lag][base_col] = 0
    
    return correlations_by_lag


def _load_and_align_data(supervised_data, target_col='log_nqh2o'):
    """
    Load original hydro data and align it with supervised data for correlation analysis.
    """
    try:
        hydro_df_orig = pd.read_csv('merged_df_short_test.csv', index_col='Date')
        hydro_df_orig.index = pd.to_datetime(hydro_df_orig.index)
        
        suffix_mapping_for_resample = {
            '_S': 'mean',
            '_R': 'mean', 
            '_SNPK': 'last',
            '_fnf': 'mean',
            '_Q': 'mean',
        }
        
        monthly_hydro = {}
        for suffix, agg_method in suffix_mapping_for_resample.items():
            matching_cols = [col for col in hydro_df_orig.columns if col.endswith(suffix)]
            if matching_cols:
                if agg_method == 'mean':
                    monthly_hydro.update({col: hydro_df_orig[col].resample('M').mean() for col in matching_cols})
                elif agg_method == 'last':
                    monthly_hydro.update({col: hydro_df_orig[col].resample('M').last() for col in matching_cols})
        
        hydro_df_monthly = pd.DataFrame(monthly_hydro)
        hydro_df_monthly.index = hydro_df_monthly.index.to_period('M')
        
        if isinstance(supervised_data.index, pd.PeriodIndex):
            supervised_data_idx_period = supervised_data.index
        elif hasattr(supervised_data.index, 'to_period'):
            supervised_data_idx_period = supervised_data.index.to_period('M')
        else:
            try:
                supervised_data_idx_period = pd.PeriodIndex(supervised_data.index, freq='M')
            except:
                supervised_data_idx_period = pd.to_datetime(supervised_data.index).to_period('M')
        
        common_periods = hydro_df_monthly.index.intersection(supervised_data_idx_period)
        
        if len(common_periods) == 0:
            print(f"Error: No common periods found between hydro data and supervised data")
            print(f"Hydro periods: {hydro_df_monthly.index[:5]} to {hydro_df_monthly.index[-5:]}")
            print(f"Supervised periods: {supervised_data_idx_period[:5]} to {supervised_data_idx_period[-5:]}")
            return None
        
        hydro_df_aligned = hydro_df_monthly.loc[common_periods]
        
        if isinstance(supervised_data.index, pd.PeriodIndex):
            supervised_data_aligned = supervised_data.loc[common_periods]
        else:
            common_dates = [p.to_timestamp() for p in common_periods]
            supervised_data_aligned = supervised_data.loc[common_dates]
        
        if target_col not in supervised_data_aligned.columns:
            print(f"Error: {target_col} not found in supervised_data")
            print(f"Available columns: {list(supervised_data_aligned.columns)}")
            return None
        
        print(f"✓ Aligned data: {len(common_periods)} monthly periods")
        print(f"✓ Target column found: {target_col}")
        
        return hydro_df_aligned, supervised_data_aligned, common_periods
        
    except Exception as e:
        print(f"Error loading/aligning data: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_aggregated_column_mapping():
    """
    Load original hydro data and create mapping from aggregated features 
    
    """
    try:
        hydro_df = pd.read_csv('merged_df_short_test.csv', index_col='Date')
        
        suffix_mapping = {
            '_S': 'total_S',
            '_R': 'total_R', 
            '_SNPK': 'total_SNPK',
            '_fnf': 'total_fnf',
            '_Q': 'total_Q',
        }
        
        aggregated_to_columns = {}
        for suffix, agg_name in suffix_mapping.items():
            matching_cols = [col for col in hydro_df.columns if col.endswith(suffix)]
            if matching_cols:
                aggregated_to_columns[agg_name] = matching_cols
        
        wd_cols = [col for col in hydro_df.columns 
                  if 'wonderful' in col.lower() and 'delivery' in col.lower()]
        if wd_cols:
            aggregated_to_columns['total_wonderful_delivery'] = wd_cols
        
        return aggregated_to_columns
        
    except Exception as e:
        print(f"Warning: Could not load original data to map columns: {e}")
        print("Will analyze based on feature importance only.")
        return {}


def _plot_reservoir_correlation(ax, agg_name, correlations, max_x_limit, top_n=3, lag_label=None, show_title=True):
    """
 top reservoirs by correlation.
    """
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    sorted_cols, sorted_vals = zip(*sorted_corrs) if sorted_corrs else ([], [])
    
    top_n = min(top_n, len(sorted_cols))
    top_cols = list(sorted_cols[:top_n])
    top_vals = list(sorted_vals[:top_n])
    
    if len(top_vals) == 0 or max(top_vals) == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, f'No correlation data\navailable for {agg_name}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11, color='black')
        return
    
    from matplotlib.colors import hex2color
    n_bins = len(top_vals)
    if n_bins == 1:
        colors = [hex2color('#4575b4')]
    else:
        color_dark = hex2color('#4575b4')
        color_light = hex2color('#abd9e9')
        colors = []
        for i, val in enumerate(top_vals):
            t = i / (len(top_vals) - 1) if len(top_vals) > 1 else 0
            r = color_dark[0] * (1 - t) + color_light[0] * t
            g = color_dark[1] * (1 - t) + color_light[1] * t
            b = color_dark[2] * (1 - t) + color_light[2] * t
            colors.append((r, g, b))
    
    bars = ax.barh(range(len(top_vals)), top_vals, 
                  color=colors, edgecolor='#2c3e50', linewidth=1.2, height=0.7)
    
    clean_names = [_clean_column_name(col) for col in top_cols]
    
    ax.set_yticks(range(len(clean_names)))
    ax.set_yticklabels(clean_names, fontsize=23, fontweight='medium', family='Times New Roman')
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=23, width=0.8)
    
    ax.set_xlabel('Absolute Correlation with Target', fontsize=18, labelpad=10, family='Times New Roman')
    ax.tick_params(axis='x', labelsize=17, width=0.8)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x*100)}' if abs(x) < 1 else f'{int(x)}'))
    
    feature_labels = {
        'total_S': 'Reservoir Storage',
        'total_R': 'Reservoir Releases',
        'total_SNPK': 'Snowpack Accumulation',
        'total_Q': 'Reservoir Inflow',
        'total_fnf': 'Full Natural Flow (FNF)'
    }
    feature_label = feature_labels.get(agg_name, agg_name)
    
    if show_title:
        if lag_label:
            title = f'{feature_label}\n({lag_label} Horizon)'
        else:
            title = f'{feature_label}'
        
        ax.set_title(title, fontweight='bold', fontsize=20, pad=12, family='Times New Roman')
    
    ax.set_xlim(0, max_x_limit)
    ax.grid(False)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#888888')
    
    for i, val in enumerate(top_vals):
        if val > 0:
            ax.text(val + max_x_limit * 0.02, i, f'{val:.3f}', 
                   va='center', ha='left', fontsize=15, fontweight='medium', color='black', family='Times New Roman')


def plot_reservoir_contributions(all_feature_importance, supervised_data):
    """
    Reservoir storage drivers and mechanism figure.
    Top row (a-b): top reservoirs by absolute correlation across 3-, 6-month horizons.
    Bottom row (c-d): left 2/3 time series (Oroville vs NQH2O) with seasonal shading;
                      right 1/3 scatter of Apr–May Oroville storage vs Jul–Aug NQH2O peak + regression.
    """
    aggregated_to_columns = _load_aggregated_column_mapping()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from pandas.api.types import CategoricalDtype

    plt.rcParams.update({
        'font.size': 17,
        'axes.titlesize': 19,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 17,
        'legend.fontsize': 17,
        'figure.titlesize': 23,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })

    models = ['XGBoost', 'RandomForest', 'SVR']

    target_windows = ['T-3 to T-1', 'T-5 to T-3', 'T-8 to T-6']
    filtered_feature_importance = {w: d for w, d in all_feature_importance.items() if w in target_windows}
    if not filtered_feature_importance:
        print(f"Warning: None of {target_windows} found.")
        print(f"Available: {list(all_feature_importance.keys())}")
        return
    print(f"\nFiltering to {list(filtered_feature_importance.keys())}")

    agg_feature_importance = {}
    for window_name, window_data in filtered_feature_importance.items():
        for model in models:
            if model in window_data and not window_data[model].empty:
                df = window_data[model]
                for _, row in df.iterrows():
                    feature_name = row['feature']
                    importance = row['importance']
                    base_name = _extract_base_feature_name(feature_name)
                    if base_name in aggregated_to_columns:
                        agg_feature_importance.setdefault((model, base_name), []).append(importance)

    if not agg_feature_importance:
        print("No aggregated feature importance found.")
        return

    agg_avg = {}
    for (model, agg_name), imps in agg_feature_importance.items():
        agg_avg.setdefault(agg_name, {})[model] = float(np.mean(imps))

    print("\n" + "=" * 80)
    print("Reservoir Contribution Analysis for Storage (total_S)")
    print("=" * 80)

    target_col = 'log_nqh2o'
    if target_col not in supervised_data.columns:
        print(f"Error: {target_col} not in supervised_data")
        return

    if 'total_S' not in aggregated_to_columns:
        print("Error: total_S not in aggregated_to_columns")
        return

    print("\nLoading and aligning hydro data ...")
    try:
        hydro_df_orig = pd.read_csv('merged_df_short_test.csv', index_col='Date')
        hydro_df_orig.index = pd.to_datetime(hydro_df_orig.index)

        suffix_mapping_for_resample = {'_S': 'mean', '_R': 'mean', '_SNPK': 'last', '_fnf': 'mean', '_Q': 'mean'}
        monthly_hydro = {}
        for suffix, agg in suffix_mapping_for_resample.items():
            cols = [c for c in hydro_df_orig.columns if c.endswith(suffix)]
            if not cols: continue
            if agg == 'mean':
                monthly_hydro.update({c: hydro_df_orig[c].resample('M').mean() for c in cols})
            else:
                monthly_hydro.update({c: hydro_df_orig[c].resample('M').last() for c in cols})

        hydro_df_monthly = pd.DataFrame(monthly_hydro)
        hydro_df_monthly.index = hydro_df_monthly.index.to_period('M')

        if isinstance(supervised_data.index, pd.PeriodIndex):
            supervised_idx_period = supervised_data.index
        elif hasattr(supervised_data.index, 'to_period'):
            supervised_idx_period = supervised_data.index.to_period('M')
        else:
            supervised_idx_period = pd.to_datetime(supervised_data.index).to_period('M')

        common_periods = hydro_df_monthly.index.intersection(supervised_idx_period)
        hydro_df_aligned = hydro_df_monthly.loc[common_periods]

        if isinstance(supervised_data.index, pd.PeriodIndex):
            hydro_for_alignment = hydro_df_aligned.reindex(supervised_data.index)
        else:
            hydro_for_alignment = hydro_df_aligned.to_timestamp().reindex(pd.to_datetime(supervised_data.index))
            hydro_for_alignment.index = supervised_data.index

        supervised_data_with_lags = supervised_data.copy()
        all_constituent_cols = []
        all_constituent_cols.extend(aggregated_to_columns['total_S'])
        created = 0
        for col in all_constituent_cols:
            if col not in hydro_for_alignment.columns: continue
            supervised_data_with_lags[col] = hydro_for_alignment[col]
            for lag in [1, 3, 6]:
                supervised_data_with_lags[f"{col}_lag{lag}"] = supervised_data_with_lags[col].shift(lag)
                created += 1
        print(f"Created {created} lag features.")

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nCalculating correlations for lags 1/3/6 ...")
    all_corr_by_lag, global_max = {}, 0.0
    try:
        cons = aggregated_to_columns['total_S']
        corr_by_lag = _calculate_correlations_by_lag(cons, supervised_data_with_lags, target_col, lags=[1,3,6])
        all_corr_by_lag['total_S'] = corr_by_lag
        for lag in [1,3,6]:
            if corr_by_lag[lag]:
                global_max = max(global_max, max(corr_by_lag[lag].values()))
    except Exception as e:
        print(f"Warning computing correlations: {e}")
        import traceback; traceback.print_exc()
        return
    if global_max == 0:
        print("No nonzero correlations found.")
        return
    max_x_limit = global_max * 1.2

    fig = plt.figure(figsize=(18, 16), dpi=300)
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1, 1.2],
        hspace=0.25, wspace=0.35,
        left=0.08, right=0.96, top=0.92, bottom=0.12
    )

    fig.text(0.5, 0.98, "Reservoir Storage Importance Across Forecast Horizons",
             ha='center', va='center', fontsize=23, fontweight='bold', color='black', family='Times New Roman')

    top_gs = gs[0, :].subgridspec(1, 2, wspace=0.3)
    ax_a = fig.add_subplot(top_gs[0, 0])
    ax_b = fig.add_subplot(top_gs[0, 1])

    def _plot_corr_block(ax, lag, subtitle, panel_tag):
        _plot_reservoir_correlation(ax, 'total_S',
                                    all_corr_by_lag['total_S'].get(lag, {}),
                                    max_x_limit=max_x_limit, top_n=5, show_title=False)
        ax.set_title(f"{subtitle}", fontsize=25, fontweight='bold', pad=10, y=1.02, family='Times New Roman')
        ax.text(0.02, 1.02, panel_tag, transform=ax.transAxes,
                fontsize=25, va='bottom', ha='left', color='black', family='Times New Roman')

    _plot_corr_block(ax_a, 3, "3-month Horizon", "(a)")
    _plot_corr_block(ax_b, 6, "6-month Horizon", "(b)")

    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.text(0.01, 1.03, '(c)', transform=ax_d.transAxes,
              fontsize=25,  va='bottom', ha='left', color='black', family='Times New Roman')

    try:
        start_date = pd.Timestamp('2013-01-01')
        end_date   = pd.Timestamp('2024-12-31')

        def _to_datetime_series(s):
            if isinstance(s.index, pd.PeriodIndex):
                s = s.copy(); s.index = s.index.to_timestamp()
            else:
                s = s.copy(); s.index = pd.to_datetime(s.index)
            return s

        if 'oroville_S' in hydro_for_alignment.columns:
            s_orov = _to_datetime_series(hydro_for_alignment['oroville_S']).loc[start_date:end_date].dropna()
        else:
            s_orov = pd.Series(dtype=float)

        s_price = _to_datetime_series(supervised_data[target_col]).loc[start_date:end_date].dropna()

        ax_price = ax_d.twinx()
        if len(s_orov):
            ax_d.plot(s_orov.index, s_orov.values, color='#fc8d59', ls='--', lw=2.0, label='Oroville')
        if len(s_price):
            ax_price.plot(s_price.index, s_price.values, color='black', lw=2.5, label='log (NQH2O) Price')

        years = range(start_date.year, end_date.year + 1)
        for y in years:
            am0, am1 = pd.Timestamp(f'{y}-04-01'), pd.Timestamp(f'{y}-05-31')
            ja0, ja1 = pd.Timestamp(f'{y}-07-01'), pd.Timestamp(f'{y}-08-31')
            if am1 >= start_date and am0 <= end_date:
                ax_d.axvspan(max(am0,start_date), min(am1,end_date), color='#ffffbf', alpha=0.25, zorder=0)
            if ja1 >= start_date and ja0 <= end_date:
                ax_d.axvspan(max(ja0,start_date), min(ja1,end_date), color='#91bfdb', alpha=0.25, zorder=0)

        ax_d.set_title("Oroville Reservoir and Water Index (NQH2O) Price", fontsize=23, fontweight='bold', pad=10, family='Times New Roman')
        ax_d.set_ylabel("Reservoir Storage\n(×10³ AF)", fontsize=23, fontweight='bold', color='#2c3e50', family='Times New Roman')

        from matplotlib.dates import YearLocator, DateFormatter
        from matplotlib.ticker import FormatStrFormatter
        ax_d.xaxis.set_major_locator(YearLocator())
        ax_d.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax_d.tick_params(axis='x', rotation=35, labelsize=23)
        ax_d.tick_params(axis='y', labelsize=23)
        ax_d.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax_price.tick_params(axis='y', labelsize=23)
        for label in ax_d.get_xticklabels() + ax_d.get_yticklabels() + ax_price.get_yticklabels():
            label.set_family('Times New Roman')
        ax_d.set_xlim(start_date, end_date)

        april_may_patch = Rectangle((0, 0), 1, 1, facecolor='#ffffbf', alpha=0.25, edgecolor='none', label='April-May')
        july_aug_patch = Rectangle((0, 0), 1, 1, facecolor='#91bfdb', alpha=0.25, edgecolor='none', label='July-August')
        
        handles1, labels1 = ax_d.get_legend_handles_labels()
        handles2, labels2 = ax_price.get_legend_handles_labels()
        all_handles = handles1 + handles2 + [april_may_patch, july_aug_patch]
        all_labels = labels1 + labels2 + ['April-May', 'July-August']
        
        leg1 = ax_d.legend(all_handles, all_labels, loc='upper left', frameon=True, fontsize=28, prop={'family': 'Times New Roman'})
        for ax in (ax_d, ax_price):
            for sp in ax.spines.values():
                sp.set_linewidth(0.9); sp.set_color('#888888')
        ax_d.grid(True, ls='--', lw=0.5, alpha=0.15)

    except Exception as e:
        print(f"Warning: time-series panel failed: {e}")
        import traceback; traceback.print_exc()
        ax_d.axis('off')
        ax_d.text(0.5, 0.5, 'Time series unavailable', ha='center', va='center', color='black')

    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.text(0.01, 1.05, '(d)', transform=ax_e.transAxes,
              fontsize=25,  va='bottom', ha='left', color='black', family='Times New Roman')

    try:
        def _extract_window_mean(s, year, months):
            mask = (s.index.year == year) & (s.index.month.isin(months))
            vals = s[mask]
            return np.nan if len(vals)==0 else float(vals.mean())

        def _extract_window_max(s, year, months):
            mask = (s.index.year == year) & (s.index.month.isin(months))
            vals = s[mask]
            return np.nan if len(vals)==0 else float(vals.max())

        if len(s_orov) and len(s_price):
            years = range(max(s_orov.index.min().year, s_price.index.min().year),
                          min(s_orov.index.max().year, s_price.index.max().year) + 1)
            X, Y, year_list = [], [], []
            for y in years:
                x = _extract_window_mean(s_orov, y, [4,5])
                yv = _extract_window_max(s_price, y, [7,8])
                if not np.isnan(x) and not np.isnan(yv):
                    X.append(x); Y.append(yv); year_list.append(y)

            if X:
                year_array = np.array(year_list)
                year_min, year_max = year_array.min(), year_array.max()
                
                scatter = ax_e.scatter(X, Y, s=120, c=year_array, cmap='viridis', 
                                      edgecolor='black', lw=1.5, alpha=0.85, zorder=3, vmin=year_min, vmax=year_max)

                X_np, Y_np = np.array(X), np.array(Y)
                coeffs = np.polyfit(X_np, Y_np, 1)
                xline = np.linspace(min(X_np), max(X_np), 100)
                yline = coeffs[0]*xline + coeffs[1]
                ax_e.plot(xline, yline, color='black', lw=1.5, zorder=2)

                if len(X_np) > 1:
                    r = float(np.corrcoef(X_np, Y_np)[0,1])
                    ax_e.text(0.98, 0.96, f"Pearson r = {r:.2f}",
                              transform=ax_e.transAxes, ha='right', va='top',
                              fontsize=23, fontweight='bold', color='black',
                              bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9), family='Times New Roman')

                ax_e.set_xlabel("Oroville Storage (×10³ AF)", fontsize=23, family='Times New Roman')
                ax_e.set_ylabel("Log (NQH2O) Peak", fontsize=23, family='Times New Roman')
                ax_e.set_title("Storage vs Price", fontsize=23, fontweight='bold', pad=12, family='Times New Roman')
                ax_e.tick_params(axis='both', labelsize=23)
                for label in ax_e.get_xticklabels() + ax_e.get_yticklabels():
                    label.set_family('Times New Roman')
                for sp in ax_e.spines.values():
                    sp.set_linewidth(0.9); sp.set_color('#888888')
                
                cbar = plt.colorbar(scatter, ax=ax_e, orientation='horizontal', pad=0.2, aspect=40, shrink=0.8)
                cbar.set_label('Year', fontsize=20, fontweight='bold', labelpad=8, family='Times New Roman')
                cbar.ax.tick_params(labelsize=18)
                for label in cbar.ax.get_xticklabels():
                    label.set_family('Times New Roman')

                xr = max(X_np) - min(X_np)
                yr = max(Y_np) - min(Y_np)
                ax_e.set_xlim(min(X_np) - 0.05*xr, max(X_np) + 0.05*xr)
                ax_e.set_ylim(min(Y_np) - 0.05*yr, max(Y_np) + 0.05*yr)
            else:
                ax_e.axis('off'); ax_e.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', color='black')
        else:
            ax_e.axis('off'); ax_e.text(0.5, 0.5, 'Data unavailable', ha='center', va='center', color='black')

    except Exception as e:
        print(f"Warning: scatter panel failed: {e}")
        import traceback; traceback.print_exc()
        ax_e.axis('off'); ax_e.text(0.5, 0.5, 'Scatter failed', ha='center', va='center', color='black')

    os.makedirs('hybrid_result', exist_ok=True)
    plt.savefig('hybrid_result/reservoir_contributions.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    
    # Print summary for total_S only
    agg_name = 'total_S'
    if agg_name in aggregated_to_columns and agg_name in agg_avg:
        print("\nAggregated Feature Importance Summary:")
        print("-" * 80)
        constituent_count = len(aggregated_to_columns[agg_name])
        avg_importance = sum(agg_avg[agg_name].values()) / len(agg_avg[agg_name])
        print(f"\n{agg_name}:")
        print(f"  Average Importance: {avg_importance:.4f}")
        print(f"  Constituent Reservoirs ({constituent_count}):")
        for col in aggregated_to_columns[agg_name][:10]:  # Show first 10
            clean_name = _clean_column_name(col)
            print(f"    - {clean_name}")
        if constituent_count > 10:
            print(f"    ... and {constituent_count - 10} more")
    
    # Print detailed correlation analysis (showing lag 1, lag 3, and lag 6 separately)
    print("\n" + "=" * 80)
    print("Top Individual Reservoirs by Correlation with Target - Storage (total_S)")
    print("(Showing lag 1, lag 3, and lag 6 separately)")
    print("=" * 80)
    
    agg_name = 'total_S'
    if agg_name in aggregated_to_columns and agg_name in agg_avg:
        # Get correlations for all lags
        correlations_by_lag = all_corr_by_lag.get(agg_name, {1: {}, 3: {}, 6: {}})
        avg_importance = sum(agg_avg[agg_name].values()) / len(agg_avg[agg_name])
        
        print(f"\n{agg_name} (Model Importance: {avg_importance:.4f}):")
        
        # Show lag 1 results
        corr_lag1 = correlations_by_lag.get(1, {})
        sorted_corrs_lag1 = sorted(corr_lag1.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 10 Reservoirs by Correlation (1-month horizon):")
        for i, (col, corr_val) in enumerate(sorted_corrs_lag1[:10], 1):
            clean_name = _clean_column_name(col)
            print(f"    {i:2d}. {clean_name:30s}: {corr_val:.4f}")
        
        # Show lag 3 results
        corr_lag3 = correlations_by_lag.get(3, {})
        sorted_corrs_lag3 = sorted(corr_lag3.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 10 Reservoirs by Correlation (3-month horizon):")
        for i, (col, corr_val) in enumerate(sorted_corrs_lag3[:10], 1):
            clean_name = _clean_column_name(col)
            print(f"    {i:2d}. {clean_name:30s}: {corr_val:.4f}")
        
        # Show lag 6 results
        corr_lag6 = correlations_by_lag.get(6, {})
        sorted_corrs_lag6 = sorted(corr_lag6.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 10 Reservoirs by Correlation (6-month horizon):")
        for i, (col, corr_val) in enumerate(sorted_corrs_lag6[:10], 1):
            clean_name = _clean_column_name(col)
            print(f"    {i:2d}. {clean_name:30s}: {corr_val:.4f}")
    
    print("\n" + "=" * 80 + "\n")

def plot_oos_combined(results_df, predictions_by_window):
    plt.rcParams.update({
        "font.size": 22,
        "font.family": "Times New Roman",
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 22,
        "figure.titlesize": 26,
        "axes.spines.top": True,
        "axes.spines.right": True
    })

    models = ["XGBoost", "RandomForest", "SVR"]
    model_color = {"XGBoost": "#91bfdb", "RandomForest": "#fc8d59", "SVR": "#f4d03f"}  
    model_marker = {"XGBoost": "o", "RandomForest": "s", "SVR": "^"}
    
    horizon_to_shift = {
        "T-3 to T-1": 1,   # 1-month horizon
        "T-5 to T-3": 3,   # 3-month horizon
        "T-8 to T-6": 6,   # 6-month horizon
        "T-11 to T-9": 9   # 9-month horizon
    }
    baseline_label = "Baseline"
    baseline_color = {baseline_label: "#5ab4ac"}  
    baseline_marker = {baseline_label: "D"}
    baseline_linestyle = {baseline_label: "--"}

    top_horizons = ["T-3 to T-1", "T-5 to T-3", "T-8 to T-6"]  # 1, 3, 6 month
    top_horizon_labels = [map_window_name(w) for w in top_horizons]
    
    window_order = ["T-3 to T-1", "T-5 to T-3", "T-8 to T-6", "T-11 to T-9"]
    horizon_labels = [map_window_name(w) for w in window_order]

    first_window = next(iter(predictions_by_window))
    y_true = predictions_by_window[first_window]["y_true"]
    y_true_series = pd.Series(y_true)
    n_points = len(y_true)
    start_date = pd.Timestamp("2022-10-01")
    date_range = pd.date_range(start=start_date, periods=n_points, freq="M")
    xt_idx = range(0, n_points, max(1, n_points // 8))
    xt_lbls = [date_range[i].strftime("%Y-%m") for i in xt_idx]

    baseline_predictions_by_horizon = {}  # Store by horizon
    baseline_metrics = []
    
    for window in window_order:
        shift_months = horizon_to_shift.get(window)
        if shift_months is None:
            continue
        
        y_shifted = y_true_series.shift(shift_months).bfill().values.reshape(-1, 1)
        lr_model = LinearRegression()
        lr_model.fit(y_shifted, y_true)
        lr_pred = lr_model.predict(y_shifted)
        
        baseline_predictions_by_horizon[window] = {
            'name': baseline_label,
            'pred': lr_pred,
            'shift': shift_months  # Store shift for display purposes if needed
        }
        
        metrics = calculate_metrics(y_true, lr_pred)
        metrics['Model'] = baseline_label
        metrics['Window'] = window
        baseline_metrics.append(metrics)

    all_series = [np.asarray(y_true)]
    for window, preds in predictions_by_window.items():
        for m in models:
            if m in preds:
                all_series.append(np.asarray(preds[m]))
    for window, baseline_data in baseline_predictions_by_horizon.items():
        all_series.append(np.asarray(baseline_data['pred']))
    y_min, y_max = min(s.min() for s in all_series), max(s.max() for s in all_series)
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_limits = (y_min - pad, y_max + pad)

    fig = plt.figure(figsize=(20, 12))

    for i, window in enumerate(top_horizons, start=1):
        ax = plt.subplot(2, 3, i)
        
        label = chr(ord('a') + i - 1)
        ax.text(0.02, 1.02, f'({label})', transform=ax.transAxes,
                fontsize=24, verticalalignment='bottom', horizontalalignment='left', family='Times New Roman')

        ax.plot(range(n_points), y_true, color="black", lw=3, alpha=0.9, label="Actual")

        preds = predictions_by_window.get(window, {})
        for model in models:
            if model in preds:
                ax.plot(
                    range(n_points),
                    preds[model],
                    color=model_color[model],
                    lw=2.0,
                    ls="-",
                    marker=model_marker[model],
                    ms=4,
                    markevery=max(1, n_points // 20),
                    label=model
                )
        
        if window in baseline_predictions_by_horizon:
            baseline_data = baseline_predictions_by_horizon[window]
            baseline_pred = baseline_data['pred']
            ax.plot(
                range(n_points),
                baseline_pred,
                color=baseline_color[baseline_label],
                lw=1.5,
                ls=baseline_linestyle[baseline_label],
                marker=baseline_marker[baseline_label],
                ms=4,
                markevery=max(1, n_points // 20),
                alpha=0.8,
                label=baseline_label
            )

        ax.set_ylim(*y_limits)
        horizon_label = map_window_name(window)
        ax.set_title(f"{horizon_label} OOS", pad=8, fontsize=24, family='Times New Roman')
        ax.set_ylabel("Log NQH2O Price", fontsize=22, family='Times New Roman')
        ax.set_xticks(list(xt_idx))
        ax.set_xticklabels(xt_lbls, rotation=30, ha="right", fontsize=20)

        ax.legend(["Actual"], loc="upper right", frameon=False, fontsize=20, prop={'family': 'Times New Roman'})

    metrics = ["RMSE", "MAE", "R2"]
    
    baseline_results_df = pd.DataFrame(baseline_metrics)
    baseline_results_df["Window_mapped"] = baseline_results_df["Window"].apply(map_window_name)
    
    results_df = results_df.copy()
    results_df["Window_mapped"] = results_df["Window"].apply(map_window_name)
    combined_results = pd.concat([results_df, baseline_results_df], ignore_index=True)
    
    cat = CategoricalDtype(categories=horizon_labels, ordered=True)
    combined_results["Window_mapped"] = combined_results["Window_mapped"].astype(cat)
    combined_results = combined_results.sort_values("Window_mapped")

    for i, metric in enumerate(metrics, start=4):
        ax = plt.subplot(2, 3, i)
        
        label = chr(ord('a') + i - 1)
        ax.text(0.02, 1.02, f'({label})', transform=ax.transAxes,
                fontsize=24, verticalalignment='bottom', horizontalalignment='left', family='Times New Roman')
        
        for model in models:
            sub = combined_results[combined_results["Model"] == model]
            if sub.empty:
                continue
            ax.plot(
                sub["Window_mapped"],
                sub[metric],
                color=model_color[model],
                marker=model_marker[model],
                ls="-",
                lw=2.2,
                ms=6,
                label=model
            )
        
        sub = combined_results[combined_results["Model"] == baseline_label]
        if not sub.empty:
            ax.plot(
                sub["Window_mapped"],
                sub[metric],
                color=baseline_color[baseline_label],
                marker=baseline_marker[baseline_label],
                ls=baseline_linestyle[baseline_label],
                lw=1.5,
                ms=5,
                alpha=0.8,
                label=baseline_label
            )
        
        ax.set_title(metric, pad=8, fontsize=24, family='Times New Roman')
        ax.set_ylabel(metric, fontsize=22, family='Times New Roman')
        ax.set_xticks(range(len(horizon_labels)))
        short_labels = [label.replace("-Month Horizon", "M") for label in horizon_labels]
        ax.set_xticklabels(short_labels, rotation=0, fontsize=20, ha='center')

    model_handles = [
        Line2D([0], [0], color=model_color[m], lw=3, marker=model_marker[m], ms=7, label=m)
        for m in models
    ]
    baseline_handles = [
        Line2D([0], [0], color=baseline_color[baseline_label], lw=2, marker=baseline_marker[baseline_label], ms=6, 
               ls=baseline_linestyle[baseline_label], label=baseline_label, alpha=0.8)
    ]
    all_handles = model_handles + baseline_handles
    fig.legend(
        handles=all_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.03),
        ncol=4, title="Model / Baseline", frameon=True, fontsize=18, title_fontsize=20, prop={'family': 'Times New Roman'}
    )

    plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.93])
    out_png = "hybrid_result/oos_combined_analysis.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()




def save_trained_models(trained_models, all_feature_importance, results_df, predictions_by_window, save_dir='hybrid_result/models'):
    """
    Save trained models, feature importance, results, and predictions to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_data = {
        'trained_models': trained_models,
        'all_feature_importance': all_feature_importance,
        'results_df': results_df,
        'predictions_by_window': predictions_by_window
    }
    
    save_path = os.path.join(save_dir, 'trained_models.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n✓ Saved trained models and results to: {save_path}")
    print(f"  - Trained models for {len(trained_models)} windows")
    print(f"  - Feature importance data")
    print(f"  - Results DataFrame with {len(results_df)} rows")
    print(f"  - Predictions for {len(predictions_by_window)} windows")

def load_trained_models(save_dir='hybrid_result/models'):
    """
    Load trained models, feature importance, results, and predictions from disk.
    """
    save_path = os.path.join(save_dir, 'trained_models.pkl')
    
    if not os.path.exists(save_path):
        print(f"✗ No saved models found at: {save_path}")
        return None
    
    try:
        with open(save_path, 'rb') as f:
            save_data = pickle.load(f)
        
        print(f"\n✓ Loaded trained models and results from: {save_path}")
        print(f"  - Trained models for {len(save_data['trained_models'])} windows")
        print(f"  - Feature importance data")
        print(f"  - Results DataFrame with {len(save_data['results_df'])} rows")
        print(f"  - Predictions for {len(save_data['predictions_by_window'])} windows")
        
        return save_data
    except Exception as e:
        print(f"✗ Error loading saved models: {e}")
        return None

def main():
    """Main execution function."""
    print("Starting Hybrid Prediction Model for Water Futures")
    print("=" * 60)
    
    supervised_data = load_and_process_data()
    
    target_column = 'log_nqh2o'
    hydro_features = ['total_S', 'total_R', 'total_Q', 'total_fnf', 'total_SNPK', 
                      'total_wonderful_delivery',  'delta_HRO_pump', 'delta_TRP_pump']
    lag_windows = [(11, 9), (8, 6), (5, 3), (3, 1)]  # Reduced lag windows for shorter dataset
    
    print("\n" + "=" * 60)
    print("Checking for saved models...")
    print("=" * 60)
    saved_data = load_trained_models()
    
    if saved_data is not None:
        # Use saved models
        trained_models = saved_data['trained_models']
        all_feature_importance = saved_data['all_feature_importance']
        results_df = saved_data['results_df']
        predictions_by_window = saved_data['predictions_by_window']
        print("\n✓ Using saved models - skipping training!")
    else:
        print("\n" + "=" * 60)
        print("No saved models found. Training new models...")
        print("=" * 60)
        
        all_results = []
        predictions_by_window = {}
        all_feature_importance = {}
        trained_models = {}  # Store trained models for future predictions
        
        print("\nTraining Models...")
        for lag_start, lag_end in lag_windows:
            print(f"\nProcessing window T-{lag_start} to T-{lag_end}...")
            
            df_features = create_lag_features(supervised_data, target=target_column, 
                                            hydro_cols=hydro_features, lag_start=lag_start, lag_end=lag_end)
            
            if df_features.empty:
                print(f"Warning: No data after feature engineering for window T-{lag_start} to T-{lag_end}")
                print(f"Original data shape: {supervised_data.shape}")
                print(f"Original data columns: {supervised_data.columns.tolist()}")
                continue
            
            X = df_features.drop(columns=[target_column])
            y = df_features[target_column]
            
            closest_lag = f'{target_column}_lag{lag_end}'
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_filtered, feature_scores = permutation_feature_selection(X, y, rf_model, n_features=10, always_keep= closest_lag )
            
            split_idx = -24
            X_train, X_test = X_filtered.iloc[:split_idx], X_filtered.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            horizon_length = lag_end
            
            model_results, y_test_orig, feature_importance, window_trained_models = train_models(
                X_train, y_train, X_test, y_test, X_filtered.columns, 
                use_grid_search=True, tuning_stages=2, fast_refined=True, horizon_length=horizon_length)
            
            window_name = f'T-{lag_start} to T-{lag_end}'
            predictions_by_window[window_name] = {
                'y_true': y_test_orig.flatten(),
                **{model: preds.flatten() for model, (preds, _) in model_results.items()}
            }
            
            all_feature_importance[window_name] = feature_importance
            
            trained_models[window_name] = window_trained_models
            trained_models[window_name]['selected_features'] = X_filtered.columns.tolist()
            
            for model_name, (_, metrics) in model_results.items():
                metrics['Model'] = model_name
                metrics['Window'] = window_name
                all_results.append(metrics)
    
        results_df = pd.DataFrame(all_results)
        
        if results_df.empty:
            print("\nNo results to display - all feature engineering failed.")
            print("This might be due to insufficient data or too many lag features.")
            return
        
        save_trained_models(trained_models, all_feature_importance, results_df, predictions_by_window)
    
    plot_oos_combined(results_df, predictions_by_window)
    
    if all_feature_importance:
        plot_feature_importance(all_feature_importance)
        plot_reservoir_contributions(all_feature_importance, supervised_data)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS BY MODEL AND HORIZON")
    print("=" * 80)
    
    if not results_df.empty:
        results_sorted = results_df.sort_values(['Window', 'Model'])
        
        windows = results_sorted['Window'].unique()
        models = results_sorted['Model'].unique()
        
        print(f"\n{'Horizon':<25} {'Model':<15} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
        print("-" * 80)
        
        for window in windows:
            window_data = results_sorted[results_sorted['Window'] == window]
            try:
                window_label = map_window_name(window)
            except:
                window_label = window
            
            for model in models:
                model_data = window_data[window_data['Model'] == model]
                if not model_data.empty:
                    row = model_data.iloc[0]
                    print(f"{window_label:<25} {model:<15} {row['RMSE']:>10.4f} {row['MAE']:>10.4f} {row['R2']:>10.4f}")
        
        print("-" * 80)
        
        print("\nAverage Performance Across All Horizons:")
        print(results_df.groupby('Model')[['RMSE', 'MAE', 'R2']].mean().round(4))
        print("\nAverage Performance Across All Models:")
        horizon_avg = results_df.groupby('Window')[['RMSE', 'MAE', 'R2']].mean().round(4)
        try:
            horizon_avg.index = horizon_avg.index.map(map_window_name)
        except:
            pass
        print(horizon_avg)
    
    print("\n" + "=" * 80)
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()