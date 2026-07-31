#!/usr/bin/env python3
"""
This script implements hedging strategies for water futures using:
- Scenario-specific actual prices from each row in scenario_forecasting_matrix.csv (100 scenarios)
- Trained model structures from hybrid_prediction_decay.py with predictive skill decay
- Different prediction horizons (1, 3, 6, 9 months ahead) with varying accuracy
- Volume requirements based on pumping data for each scenario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import os
from scipy.stats import norm
warnings.filterwarnings('ignore')
from calfews_src.util import *
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_prediction_decay import (
    load_and_process_data, 
    create_lag_features, 
    permutation_feature_selection,
    train_models
)

class FixedScenarioSpecificWaterHedgingStrategy:
    """Fixed scenario-specific water futures hedging strategy with proper feature handling."""
    
    def __init__(self, scenario_matrix_path='hybrid_result/scenario_predictions.csv'):
        self.scenario_matrix_path = scenario_matrix_path
        self.scenario_data = None
        self.pumping_data = None
        self.hedging_results = {}
        self.trained_models = None
        self.supervised_data = None
        
        self.hedging_horizons = [1, 3, 6, 9]
        self.hedging_volumes = [0.5, 1.0]
        self.transaction_costs = 0.02
        self.model_save_path = 'hybrid_result/trained_hedging_models.pkl'
        self.base_hedge_month_indices = [9, 10, 11] # 0-indexed months within a year to hedge
        self.hedging_year_index = 1 #(0-indexed) 1: year-2 hedge    
        self.custom_hedge_months = None
        self.analysis_months = None
        self.price_month_count = 0
        
        print('Fixed Scenario-Specific Water Futures Hedging Strategy Analyzer Initialized')
    
    def load_scenario_data(self):
        """Load scenario-specific forecasting data with actual prices and bounds for each scenario."""
        print('Loading scenario-specific forecasting data with actual prices and uncertainty bounds...')
        
        self.scenario_data = pd.read_csv(self.scenario_matrix_path, index_col=0)
        
        if 'Actual' in self.scenario_data.index:
            self.scenario_data = self.scenario_data.drop('Actual')
        
        self.scenario_prices = self.scenario_data
        self.price_month_count = self.scenario_prices.shape[1]
        if self.analysis_months is None:
            self.analysis_months = self.price_month_count
        
        # Load bounds file
        bounds_path = self.scenario_matrix_path.replace('scenario_predictions.csv', 'scenario_predictions_bounds.csv')
        bounds_data = pd.read_csv(bounds_path, index_col=0)
        
        self.scenario_prices_upper = pd.DataFrame(index=self.scenario_prices.index, 
                                                  columns=self.scenario_prices.columns)
        self.scenario_prices_lower = pd.DataFrame(index=self.scenario_prices.index, 
                                                  columns=self.scenario_prices.columns)
        
        for bounds_idx in bounds_data.index:
            if not isinstance(bounds_idx, str):
                continue
                
            parts = bounds_idx.split('_')
            if len(parts) < 2:
                continue
                
            try:
                scenario_num = int(parts[1])
            except ValueError:
                continue
            
            if scenario_num > len(self.scenario_prices):
                continue
                
            price_idx = self.scenario_prices.index[scenario_num - 1]
            
            if '_upper' in bounds_idx:
                self.scenario_prices_upper.loc[price_idx] = bounds_data.loc[bounds_idx].values
            elif '_lower' in bounds_idx:
                self.scenario_prices_lower.loc[price_idx] = bounds_data.loc[bounds_idx].values
        
        if self.scenario_prices_upper.shape[1] != self.price_month_count:
            self.scenario_prices_upper = self.scenario_prices_upper.iloc[:, :self.price_month_count]
        if self.scenario_prices_lower.shape[1] != self.price_month_count:
            self.scenario_prices_lower = self.scenario_prices_lower.iloc[:, :self.price_month_count]
        
        print(f'Loaded {self.scenario_prices.shape[0]} scenarios with {self.price_month_count} months each')
        print(f'Loaded price bounds for scenarios')
        
        return True
    
    def load_pumping_data(self, scenario_path_template='results/startyear_4_1/2024_{}/results.hdf5'):
        """Load pumping data for volume calculations for each scenario."""
        print('Loading pumping data for volume calculations...')
        
        pumping_scenarios = []
        successful_scenarios = 0
        
        for i in range(1, 101):
            try:
                scenario_path = scenario_path_template.format(i)
                
                hydro_data = get_results_sensitivity_number_outside_model(scenario_path, '')
                
                s_cols = [col for col in hydro_data.columns if col.endswith('_S')]
                r_cols = [col for col in hydro_data.columns if col.endswith('_R')]
                q_cols = [col for col in hydro_data.columns if col.endswith('_Q')]
                fnf_cols = [col for col in hydro_data.columns if col.endswith('_fnf')]
                snpk_cols = [col for col in hydro_data.columns if col.endswith('_SNPK')]
                
                wonderful_cols = [
                    'wonderful_BLR_tableA_delivery',
                    'wonderful_LHL_tableA_delivery', 
                    'wonderful_BDM_tableA_delivery'
                ]
                
                hydro_data['total_S'] = hydro_data[s_cols].sum(axis=1) if s_cols else 0
                hydro_data['total_R'] = hydro_data[r_cols].sum(axis=1) if r_cols else 0
                hydro_data['total_Q'] = hydro_data[q_cols].sum(axis=1) if q_cols else 0
                hydro_data['total_fnf'] = hydro_data[fnf_cols].sum(axis=1) if fnf_cols else 0
                hydro_data['total_SNPK'] = hydro_data[snpk_cols].sum(axis=1) if snpk_cols else 0
                
                available_wonderful_cols = [col for col in wonderful_cols if col in hydro_data.columns]
                hydro_data['total_wonderful_delivery'] = hydro_data[available_wonderful_cols].sum(axis=1) if available_wonderful_cols else 0
                
                hro_pump_cols = [col for col in hydro_data.columns if 'HRO' in col and 'pump' in col.lower()]
                trp_pump_cols = [col for col in hydro_data.columns if 'TRP' in col and 'pump' in col.lower()]
                
                hydro_data['delta_HRO_pump'] = hydro_data[hro_pump_cols].sum(axis=1) if hro_pump_cols else 0
                hydro_data['delta_TRP_pump'] = hydro_data[trp_pump_cols].sum(axis=1) if trp_pump_cols else 0
                pumping_cols = ['wonderful_BLR_pumping', 'wonderful_LHL_pumping', 'wonderful_BDM_pumping']
                available_pumping_cols = [col for col in pumping_cols if col in hydro_data.columns]
                
                if available_pumping_cols:
                    monthly_pumping = hydro_data[available_pumping_cols].resample('M').sum()

                    months_to_use = self.analysis_months if self.analysis_months is not None else 12
                    months_to_use = min(months_to_use, len(monthly_pumping))

                    if months_to_use == 0:
                        print(f'  Warning: No monthly pumping data available for scenario {i}')
                        continue

                    monthly_pumping = monthly_pumping.head(months_to_use)
                    
                    total_volume = monthly_pumping.sum().sum()
                    
                    hydro_features_cols = ['total_S', 'total_R', 'total_Q', 'total_fnf', 'total_SNPK', 
                                          'total_wonderful_delivery', 'delta_HRO_pump', 'delta_TRP_pump']
                    
                    agg_mean_cols = ['total_S', 'total_R', 'total_Q', 'total_fnf']
                    agg_last_cols = ['total_SNPK', 'total_wonderful_delivery', 'delta_HRO_pump', 'delta_TRP_pump']
                    
                    agg_mapping = {}
                    agg_mapping.update({c: 'mean' for c in agg_mean_cols if c in hydro_data.columns})
                    agg_mapping.update({c: 'last' for c in agg_last_cols if c in hydro_data.columns})
                    
                    monthly_hydro = hydro_data[list(agg_mapping.keys())].resample('M').agg(agg_mapping)
                    monthly_hydro = monthly_hydro.head(months_to_use)
                    scenario_hydro_data = monthly_hydro
                    
                    pumping_scenarios.append({
                        'scenario': i,
                        'monthly_pumping': monthly_pumping,
                        'total_volume': total_volume,
                        'monthly_totals': monthly_pumping.sum(axis=1).values,
                        'scenario_type': 'normal',
                        'hydro_data': scenario_hydro_data,
                        'original_hydro_data': hydro_data
                    })
                    
                    successful_scenarios += 1
                
                if i % 20 == 0:
                    print(f'Processed {i} scenarios...')
                    
            except Exception as e:
                print(f'Error loading scenario {i}: {e}')
                continue
        
        self.pumping_data = pumping_scenarios
        print(f'Loaded pumping data for {successful_scenarios} scenarios')
        
        if successful_scenarios > 0:
            for scenario in pumping_scenarios:
                scenario_id = scenario['scenario']
                hydro_data = scenario['hydro_data']
                scenario['scenario_type'] = self._classify_scenario_type(scenario_id, hydro_data)
        
        return successful_scenarios > 0
    
    def _resolve_hedge_months(self):
        """Determine which scenario months to hedge based on configuration."""
        if self.custom_hedge_months is not None:
            hedge_months = list(self.custom_hedge_months)
        else:
            base_months = getattr(self, 'base_hedge_month_indices', [9, 10, 11])
            year_offset = getattr(self, 'hedging_year_index', 0)
            hedge_months = [m + year_offset * 12 for m in base_months]

        hedge_months = sorted({int(m) for m in hedge_months if m is not None})

        if not hedge_months:
            print('  Warning: No hedge months configured; set custom_hedge_months or base_hedge_month_indices')
            return []

        if self.price_month_count and self.price_month_count > 0:
            filtered_months = [m for m in hedge_months if 0 <= m < self.price_month_count]
            if len(filtered_months) < len(hedge_months):
                print(f'  Warning: Trimmed hedge months to available price window (requested {hedge_months}, using {filtered_months})')
            hedge_months = filtered_months

        return hedge_months

    def _resolve_prediction_months(self, hedge_months, horizon):
        """Calculate prediction (decision) months for each hedge month given the horizon."""
        if not hedge_months:
            return []

        prediction_months = []
        for hedge_month in hedge_months:
            decision_time = hedge_month - horizon
            if decision_time < 0:
                print(f'  Warning: Horizon {horizon} exceeds available lookback for hedge month {hedge_month + 1}')
                return []

            if self.price_month_count and decision_time >= self.price_month_count:
                print(f'  Warning: Decision month {decision_time + 1} exceeds available price data window ({self.price_month_count} months)')
                return []

            prediction_months.append(decision_time)

        return prediction_months

    def _classify_scenario_type(self, scenario_id, hydro_data):
        """Classify scenario as dry, normal, or wet based on hydrological conditions."""
        storage_cols = [col for col in hydro_data.columns if 'storage' in col.lower() or col.endswith('_S') or col == 'total_S']
        if not storage_cols:
            return 'normal'
        
        avg_storage = hydro_data[storage_cols].mean().mean()
        
        all_storage_values = []
        for scenario in self.pumping_data:
            scenario_hydro = scenario['hydro_data']
            scenario_storage = scenario_hydro[storage_cols].mean().mean()
            all_storage_values.append(scenario_storage)
        
        if not all_storage_values:
            return 'normal'
        
        all_storage_values = sorted(all_storage_values)
        n = len(all_storage_values)
        dry_threshold = all_storage_values[int(n * 0.33)]
        wet_threshold = all_storage_values[int(n * 0.67)]
        
        if avg_storage < dry_threshold:
            return 'dry'
        elif avg_storage > wet_threshold:
            return 'wet'
        else:
            return 'normal'
    
    def load_trained_models(self):
        """Load trained models from disk or train and save them if they don't exist."""
        print('Loading trained models...')
        
        if os.path.exists(self.model_save_path):
            try:
                with open(self.model_save_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.trained_models = saved_data['trained_models']
                    self.supervised_data = saved_data['supervised_data']
                
                has_old_features = False
                for window_name, model_data in self.trained_models.items():
                    if 'selected_features' in model_data:
                        selected_features = model_data['selected_features']
                        if any('ca_drought_severity_mean' in feat for feat in selected_features):
                            has_old_features = True
                            print(f'\nWARNING: Loaded models contain ca_drought_severity_mean features (old models).')
                            print(f'These features are not available in scenarios and will use historical/default values.')
                            print(f'For best results, delete {self.model_save_path} to retrain models without this feature.\n')
                            break
                
                print(f'Loaded {len(self.trained_models)} trained model sets from disk')
                print('MAE values by horizon:')
                for window_name, model_data in self.trained_models.items():
                    horizon = model_data.get('prediction_horizon', 'unknown')
                    
                    mae = None
                    if 'RandomForest' in model_data:
                        rf_data = model_data['RandomForest']
                        if isinstance(rf_data, dict) and 'metrics' in rf_data:
                            metrics = rf_data['metrics']
                            if isinstance(metrics, dict) and 'MAE' in metrics:
                                mae = metrics['MAE']
                    
                    if mae is not None:
                        self.trained_models[window_name]['mae'] = mae
                        print(f'  {window_name} (horizon: {horizon} months): MAE = {mae:.4f}')
                    else:
                        print(f'  {window_name} (horizon: {horizon} months): MAE NOT FOUND in metrics - will use default 0.1')
                        self.trained_models[window_name]['mae'] = 0.1
                
                return True
            except Exception as e:
                print(f'Error loading saved models: {e}')
                print('Will train new models...')
        
        print('Training models...')
        
        try:
            self.supervised_data = load_and_process_data()
            
            target_column = 'log_nqh2o'
            hydro_features = ['total_S', 'total_R', 'total_Q', 'total_fnf', 'total_SNPK', 
                            'total_wonderful_delivery', 'delta_HRO_pump', 'delta_TRP_pump']
            lag_windows = [(13,11), (11, 9), (8, 6), (5, 3), (3, 1)]
            
            self.trained_models = {}
            all_feature_importance = {}
            
            for lag_start, lag_end in lag_windows:
                prediction_horizon = lag_end
                
                df_features = create_lag_features(self.supervised_data, target=target_column, 
                                                hydro_cols=hydro_features, lag_start=lag_start, lag_end=lag_end)
                
                if df_features.empty:
                    continue
                
                X = df_features.drop(columns=[target_column])
                y = df_features[target_column]
                
                closest_lag = f'{target_column}_lag{lag_end}'
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                X_filtered, feature_scores = permutation_feature_selection(X, y, rf_model, n_features=10, always_keep=closest_lag)
                
                split_idx = -24
                X_train, X_test = X_filtered.iloc[:split_idx], X_filtered.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                model_results, y_test_orig, feature_importance, window_trained_models = train_models(
                    X_train, y_train, X_test, y_test, X_filtered.columns, use_grid_search=True)
                
                window_name = f'T-{lag_start} to T-{lag_end}'
                self.trained_models[window_name] = window_trained_models
                self.trained_models[window_name]['selected_features'] = X_filtered.columns.tolist()
                self.trained_models[window_name]['prediction_horizon'] = prediction_horizon
                
                if 'RandomForest' in model_results and 'mae' in model_results['RandomForest']:
                    self.trained_models[window_name]['mae'] = model_results['RandomForest']['mae']
                    print(f'  Stored MAE from model_results: {model_results["RandomForest"]["mae"]:.4f} for {window_name} (horizon: {prediction_horizon} months)')
                elif 'RandomForest' in model_results:
                    from sklearn.metrics import mean_absolute_error
                    rf_model = window_trained_models['RandomForest']['model']
                    rf_scaler_X = window_trained_models['RandomForest']['scaler_X']
                    rf_scaler_y = window_trained_models['RandomForest']['scaler_y']
                    X_test_scaled = rf_scaler_X.transform(X_test)
                    y_pred_scaled = rf_model.predict(X_test_scaled)
                    y_pred = rf_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    mae = mean_absolute_error(y_test_orig, y_pred)
                    self.trained_models[window_name]['mae'] = mae
                    print(f'  Calculated MAE: {mae:.4f} for {window_name} (horizon: {prediction_horizon} months)')
                else:
                    mae_values = []
                    for model_name in ['XGBoost', 'RandomForest', 'SVR']:
                        if model_name in model_results and 'mae' in model_results[model_name]:
                            mae_values.append(model_results[model_name]['mae'])
                    default_mae = np.mean(mae_values) if mae_values else 0.1
                    self.trained_models[window_name]['mae'] = default_mae
                    print(f'  Using default/average MAE: {default_mae:.4f} for {window_name} (horizon: {prediction_horizon} months)')
                
                all_feature_importance[window_name] = feature_importance
            
            try:
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                save_data = {
                    'trained_models': self.trained_models,
                    'supervised_data': self.supervised_data,
                    'feature_importance': all_feature_importance
                }
                with open(self.model_save_path, 'wb') as f:
                    pickle.dump(save_data, f)
            except Exception as e:
                print(f'Warning: Could not save models to disk: {e}')
            
            print(f'Trained and loaded {len(self.trained_models)} model sets')
            return True
            
        except Exception as e:
            print(f'Error loading trained models: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_lag_from_feature_name(self, feature_name):
        """Extract base feature name and lag number from feature name.
        
        Returns:
            tuple: (base_feature, lag_num, feature_type)
                base_feature: Base feature name (e.g., 'log_nqh2o', 'total_S')
                lag_num: Lag number (None if no lag)
                feature_type: 'price', 'drought', 'hydro', or None
        """
        if 'log_nqh2o_lag' in feature_name:
            lag_num = int(feature_name.split('lag')[1])
            return 'log_nqh2o', lag_num, 'price'
        
        elif 'ca_drought_severity_mean' in feature_name:
            lag_num = None
            if '_lag' in feature_name:
                lag_num = int(feature_name.split('_lag')[1])
            elif '_momentum_' in feature_name:
                lag_num = int(feature_name.split('_momentum_')[1])
            elif '_3M_roll_' in feature_name:
                roll_part = feature_name.split('_3M_roll_')[1]
                if 'mean' in roll_part or 'std' in roll_part:
                    lag_num = int(roll_part.replace('mean', '').replace('std', ''))
            return 'ca_drought_severity_mean', lag_num, 'drought'
        
        elif any(hydro_feat in feature_name for hydro_feat in 
                 ['total_S', 'total_R', 'total_Q', 'total_fnf', 'total_SNPK', 
                  'total_wonderful_delivery', 'delta_HRO_pump', 'delta_TRP_pump']):
            lag_num = None
            base_feature = feature_name
            
            if '_lag' in feature_name:
                parts = feature_name.split('_lag')
                base_feature = parts[0]
                lag_num = int(parts[1])
            elif '_momentum_' in feature_name:
                parts = feature_name.split('_momentum_')
                base_feature = parts[0]
                lag_num = int(parts[1])
            elif '_acceleration_' in feature_name:
                parts = feature_name.split('_acceleration_')
                base_feature = parts[0]
                lag_num = int(parts[1])
            elif '_3M_roll_' in feature_name:
                parts = feature_name.split('_3M_roll_')
                base_feature = parts[0]
                roll_part = parts[1]
                if 'mean' in roll_part or 'std' in roll_part:
                    lag_num = int(roll_part.replace('mean', '').replace('std', ''))
            elif '_volume_weighted_' in feature_name:
                parts = feature_name.split('_volume_weighted_')
                base_feature = parts[0]
                lag_num = int(parts[1])
            
            return base_feature, lag_num, 'hydro'
        
        return None, None, None
    
    def _get_feature_value_with_fallback(self, base_feature, lag_num, feature_type, 
                                         scenario_prices, scenario_hydro_data, 
                                         decision_time, hedge_months, prediction_months,
                                         historical_length, scenario_length):
        """Get feature value with fallback chain: scenario -> historical -> default.
        
        Returns:
            float: Feature value
        """
        # Handle price features
        if feature_type == 'price':
            log_prices = np.log(scenario_prices) if scenario_prices is not None else None
            
            if decision_time is not None and scenario_prices is not None and len(scenario_prices) > 0:
                hedge_month_idx = prediction_months.index(decision_time) if decision_time in prediction_months else 0
                hedge_month = hedge_months[hedge_month_idx]
                scenario_lag_idx = hedge_month - lag_num
                
                if 0 <= scenario_lag_idx < len(scenario_prices):
                    return log_prices[scenario_lag_idx]
                elif scenario_lag_idx < 0:
                    if decision_time < len(scenario_prices):
                        return log_prices[decision_time]
                    elif lag_num <= historical_length:
                        return self.supervised_data['log_nqh2o'].iloc[-lag_num]
                    else:
                        return self.supervised_data['log_nqh2o'].iloc[-1]
                else:
                    if decision_time < len(scenario_prices):
                        return log_prices[decision_time]
                    else:
                        return log_prices[-1]
            elif lag_num <= historical_length:
                return self.supervised_data['log_nqh2o'].iloc[-lag_num]
            elif scenario_prices is not None and len(scenario_prices) > 0:
                return log_prices[-1]
            else:
                raise ValueError(f"Lag {lag_num} exceeds available data (historical: {historical_length}, scenario: {scenario_length})")
        
        # Handle drought features
        elif feature_type == 'drought':
            if 'ca_drought_severity_mean' not in self.supervised_data.columns:
                return 2.0  # Default value
            
            if lag_num is not None and lag_num <= historical_length:
                return self.supervised_data['ca_drought_severity_mean'].iloc[-lag_num]
            else:
                return self.supervised_data['ca_drought_severity_mean'].iloc[-1]
        
        # Handle hydro features
        elif feature_type == 'hydro':
            if base_feature not in self.supervised_data.columns:
                raise ValueError(f"Base feature '{base_feature}' not found in supervised_data")
            
            if lag_num is not None and scenario_hydro_data is not None and base_feature in scenario_hydro_data.columns:
                if decision_time is not None:
                    hedge_month_idx = prediction_months.index(decision_time) if decision_time in prediction_months else 0
                    hedge_month = hedge_months[hedge_month_idx]
                    scenario_lag_idx = hedge_month - lag_num
                    
                    if 0 <= scenario_lag_idx < len(scenario_hydro_data):
                        return scenario_hydro_data[base_feature].iloc[scenario_lag_idx]
                    elif scenario_lag_idx < 0:
                        if decision_time < len(scenario_hydro_data):
                            return scenario_hydro_data[base_feature].iloc[decision_time]
                        elif lag_num <= historical_length:
                            return self.supervised_data[base_feature].iloc[-lag_num]
                        else:
                            return self.supervised_data[base_feature].iloc[-1]
                    else:
                        if decision_time < len(scenario_hydro_data):
                            return scenario_hydro_data[base_feature].iloc[decision_time]
                        else:
                            return scenario_hydro_data[base_feature].iloc[-1]
                else:
                    if lag_num <= historical_length:
                        return self.supervised_data[base_feature].iloc[-lag_num]
                    elif scenario_length > 0:
                        return scenario_hydro_data[base_feature].iloc[scenario_length - 1]
                    else:
                        return self.supervised_data[base_feature].iloc[-lag_num] if lag_num <= historical_length else self.supervised_data[base_feature].iloc[-1]
            elif lag_num is not None and lag_num <= historical_length:
                return self.supervised_data[base_feature].iloc[-lag_num]
            elif lag_num is not None:
                raise ValueError(f"Lag {lag_num} for feature '{base_feature}' exceeds available data")
            else:
                if scenario_hydro_data is not None and base_feature in scenario_hydro_data.columns and scenario_length > 0:
                    return scenario_hydro_data[base_feature].iloc[scenario_length - 1]
                else:
                    return self.supervised_data[base_feature].iloc[-1]
        
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    def create_scenario_specific_features(self, scenario_prices, model_key, scenario_hydro_data=None, decision_time=None, hedge_months=None, prediction_months=None):
        """Create scenario-specific lagged features from scenario prices and hydro data."""
        try:
            selected_features = self.trained_models[model_key]['selected_features']
            
            if self.supervised_data is None:
                raise ValueError("supervised_data must be available to create lagged features")
            
            historical_length = len(self.supervised_data)
            scenario_length = len(scenario_prices) if scenario_prices is not None else 0
            
            scenario_features = {}
            
            for feature_name in selected_features:
                base_feature, lag_num, feature_type = self._parse_lag_from_feature_name(feature_name)
                
                if feature_type is None:
                    raise ValueError(f"Unknown feature type: '{feature_name}'. Cannot create feature without historical data.")
                
                # Special handling for drought features with warnings
                if feature_type == 'drought' and 'ca_drought_severity_mean' not in self.supervised_data.columns:
                    scenario_features[feature_name] = 2.0
                    print(f'Warning: Using default value for {feature_name} (ca_drought_severity_mean not available). Consider retraining models without this feature.')
                    continue
                
                feature_value = self._get_feature_value_with_fallback(
                    base_feature, lag_num, feature_type,
                    scenario_prices, scenario_hydro_data,
                    decision_time, hedge_months, prediction_months,
                    historical_length, scenario_length
                )
                
                scenario_features[feature_name] = feature_value
            
            feature_vector = np.array([scenario_features.get(feat, 0.0) for feat in selected_features])
            return feature_vector
            
        except Exception as e:
            print(f'Error creating scenario-specific features: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_prediction_based_hedging_strategies(self):
        """Calculate hedging strategies using trained models for prediction-based hedging."""
        print('Calculating prediction-based hedging strategies...')
        
        if self.scenario_prices is None or self.pumping_data is None or self.trained_models is None:
            print('Error: Missing scenario data, pumping data, or trained models')
            return False
        
        strategies = {}
        
        for horizon in self.hedging_horizons:
            print(f'Processing {horizon}-month hedging horizon...')
            
            model_key = None
            for window_name, model_data in self.trained_models.items():
                if model_data['prediction_horizon'] == horizon:
                    model_key = window_name
                    break
            
            if model_key is None:
                print(f'Warning: No trained model found for {horizon}-month horizon')
                continue
            
            horizon_strategies = {}

            hedge_months = self._resolve_hedge_months()
            if not hedge_months:
                continue

            prediction_months = self._resolve_prediction_months(hedge_months, horizon)
            if not prediction_months or len(prediction_months) != len(hedge_months):
                continue

            for volume_fraction in self.hedging_volumes:
                
                strategy_results = []
                
                for i, scenario in enumerate(self.pumping_data):
                    scenario_id = scenario['scenario']
                    
                    if scenario_id > len(self.scenario_prices):
                        continue
                        
                    scenario_actual_prices = self.scenario_prices.iloc[scenario_id-1].values
                    scenario_upper_prices = self.scenario_prices_upper.iloc[scenario_id-1].values
                    scenario_lower_prices = self.scenario_prices_lower.iloc[scenario_id-1].values
                    
                    scenario_hydro_data = scenario.get('hydro_data', None)
                    
                    month_specific_predictions = {}
                    for hedge_month_idx, prediction_month in enumerate(prediction_months):
                        hedge_month = hedge_months[hedge_month_idx]
                        decision_time = prediction_month
                        
                        prediction_result = self._generate_scenario_predictions_fixed(
                            scenario_id, scenario_actual_prices, model_key, horizon,
                            scenario_hydro_data=scenario_hydro_data,
                            decision_time=decision_time,
                            hedge_months=hedge_months,
                            prediction_months=prediction_months
                        )
                        
                        if prediction_result is not None:
                            month_specific_predictions[hedge_month] = prediction_result
                    
                    if len(month_specific_predictions) == len(hedge_months):
                        hedging_result = self._calculate_prediction_based_scenario_hedge(
                            scenario_actual_prices, month_specific_predictions, scenario, 
                            horizon, volume_fraction, hedge_months, prediction_months,
                            settlement_upper_prices=scenario_upper_prices,
                            settlement_lower_prices=scenario_lower_prices
                        )
                        
                        if hedging_result:
                            hedging_result['scenario_id'] = scenario_id
                            hedging_result['scenario_type'] = scenario['scenario_type']
                            hedging_result['model_key'] = model_key
                            strategy_results.append(hedging_result)
                
                if strategy_results:
                    horizon_strategies[f'{int(volume_fraction*100)}%_target'] = strategy_results
            
            if horizon_strategies:
                strategies[f'{horizon}_month_horizon'] = horizon_strategies
        
        self.hedging_results = strategies
        print(f'Calculated prediction-based hedging strategies for {len(strategies)} horizons')
        
        return True
    
    def _generate_scenario_predictions_fixed(self, scenario_id, scenario_prices, model_key, horizon, 
                                          scenario_hydro_data=None, decision_time=None, hedge_months=None, prediction_months=None):
        """Generate predictions for a specific scenario using trained models with proper feature handling.
        """
        try:
            window_models = self.trained_models[model_key]
            selected_features = window_models['selected_features']
            
            scenario_feature_vector = self.create_scenario_specific_features(
                scenario_prices, model_key, 
                scenario_hydro_data=scenario_hydro_data, 
                decision_time=decision_time,
                hedge_months=hedge_months,
                prediction_months=prediction_months
            )
            
            if scenario_feature_vector is None:
                print(f'Error: Could not create features for scenario {scenario_id}')
                return None
            
            if len(scenario_feature_vector) != len(selected_features):
                print(f'Warning: Feature vector length {len(scenario_feature_vector)} != expected {len(selected_features)}')
                if len(scenario_feature_vector) < len(selected_features):
                    padded_vector = np.zeros(len(selected_features))
                    padded_vector[:len(scenario_feature_vector)] = scenario_feature_vector
                    scenario_feature_vector = padded_vector
                else:
                    scenario_feature_vector = scenario_feature_vector[:len(selected_features)]
            
            scenario_features = scenario_feature_vector.reshape(1, -1)
            
            predictions = {}
            for model_name in ['XGBoost', 'RandomForest', 'SVR']:
                if model_name in window_models:
                    model_data = window_models[model_name]
                    model = model_data['model']
                    scaler_X = model_data['scaler_X']
                    scaler_y = model_data['scaler_y']
                    
                    try:
                        features_scaled = scaler_X.transform(scenario_features)
                        pred_scaled = model.predict(features_scaled)
                        pred_orig = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                        predictions[model_name] = pred_orig
                    except Exception as e:
                        print(f'Error with {model_name} prediction: {e}')
                        continue
            
            if 'RandomForest' in predictions:
                rf_prediction = predictions['RandomForest']
                mae = self.trained_models[model_key].get('mae', 0.1)
                if mae == 0.1:
                    print(f'  Warning: Using default MAE (0.1) for model_key={model_key}, horizon={horizon}')
                return {
                    'ensemble': rf_prediction,
                    'individual': predictions,
                    'horizon': horizon,
                    'mae': mae
                }
            else:
                print(f'Error: RandomForest prediction not available for scenario {scenario_id}')
                return None
            
        except Exception as e:
            print(f'Error generating predictions for scenario {scenario_id}: {e}')
            return None
        
        return None
    
    def _make_hedge_decision_for_month(self, predicted_price, settlement_price, mae_log, horizon):
        """
        Make hedging decision for a single hedge month using its specific prediction and price.
        Args:
            predicted_price: Predicted price for this specific hedge month
            settlement_price: Settlement price for this specific hedge month
            mae_log: MAE in log space for this prediction (uncertainty measure)
            horizon: Hedging horizon (1, 3, or 6 months) - kept for reference but not used in thresholds
        
        Returns:
            tuple: (position_type, should_hedge, prob_profitable, hedge_ratio)
                position_type: 'LONG', 'SHORT', or 'NONE'
                should_hedge: Boolean indicating whether to hedge this month
                prob_profitable: Probability of profit if hedging
                hedge_ratio: Fraction of volume to hedge (shrinks with uncertainty, range 0-1)
        """
        pred_log_mean = np.log(predicted_price)
        settlement_log = np.log(settlement_price)
        
        pred_log_lower = pred_log_mean - 2 * mae_log
        pred_log_upper = pred_log_mean + 2 * mae_log
        predicted_price_lower = np.exp(pred_log_lower)
        predicted_price_upper = np.exp(pred_log_upper)
        
        prob_profitable_long = 1 - norm.cdf(settlement_log, loc=pred_log_mean, scale=mae_log)
        prob_profitable_short = norm.cdf(settlement_log, loc=pred_log_mean, scale=mae_log)
        
        min_price_difference = 25
        min_prob_difference = 0.15
        
        settlement_within_range = (predicted_price_lower <= settlement_price <= predicted_price_upper)
        
        if settlement_within_range:
            prob_profitable_threshold = 0.80
        else:
            prob_profitable_threshold = 0.70
        
        price_difference = abs(predicted_price - settlement_price)
        prob_difference = abs(prob_profitable_long - prob_profitable_short)
        
        uncertainty_decay_rate = 3.0
        hedge_ratio = np.exp(-uncertainty_decay_rate * mae_log)
        hedge_ratio = max(0.3, min(1.0, hedge_ratio))
        
        if price_difference < min_price_difference:
            position_type = 'NONE'
            should_hedge = False
            prob_profitable = 0.0
            hedge_ratio = 0.0
        elif prob_difference < min_prob_difference:
            position_type = 'NONE'
            should_hedge = False
            prob_profitable = max(prob_profitable_long, prob_profitable_short)
            hedge_ratio = 0.0
        elif prob_profitable_long >= prob_profitable_short:
            position_type = 'LONG'
            prob_profitable = prob_profitable_long
            should_hedge = (prob_profitable >= prob_profitable_threshold)
        else:
            position_type = 'SHORT'
            prob_profitable = prob_profitable_short
            should_hedge = (prob_profitable >= prob_profitable_threshold)
        
        if prob_profitable < prob_profitable_threshold:
            should_hedge = False
        
        if not should_hedge:
            hedge_ratio = 0.0
        
        return position_type, should_hedge, prob_profitable, hedge_ratio

    def _calculate_prediction_based_scenario_hedge(self, actual_prices, month_specific_predictions, pumping_scenario, 
                                                 horizon, volume_fraction, hedge_months, prediction_months,
                                                 settlement_upper_prices=None, settlement_lower_prices=None):
        """
        Calculate hedging strategy with month-specific predictions and settlement price uncertainty.
        
        Args:
            actual_prices: Array of base actual prices for the scenario (length matches scenario data)
            settlement_upper_prices: Array of upper bound settlement prices (optional)
            settlement_lower_prices: Array of lower bound settlement prices (optional)
            month_specific_predictions: Dictionary mapping hedge month index to prediction dict
            pumping_scenario: Scenario data with monthly volumes
            horizon: Hedging horizon (1, 3, or 6 months)
            volume_fraction: Fraction of total volume to hedge
            hedge_months: List of month indices to hedge (e.g., [8, 9, 10])
            prediction_months: List of month indices when predictions were made (e.g., [5, 6, 7])
        
        All horizons hedge the configured months (via _resolve_hedge_months).
        Each hedge month uses its own prediction made at the appropriate decision time.
        DECISION IS MADE PER HEDGE MONTH using that month's specific prediction and price.
        Hedged volume is spread evenly across hedge months that meet the hedging criteria.
        Settlement price uncertainty is incorporated into P&L calculations.
        """
        try:
            monthly_volumes = pumping_scenario['monthly_totals']
            total_volume = pumping_scenario['total_volume']
            
            if not hedge_months:
                return None
            
            months_to_hedge = len(hedge_months)
            if months_to_hedge == 0:
                print('  Warning: No hedge months provided to _calculate_prediction_based_scenario_hedge')
                return None

            total_volume_to_hedge = total_volume * volume_fraction
            hedge_volume_per_position = total_volume_to_hedge / months_to_hedge
            
            predicted_prices = {}
            prediction_maes = {}  # Store MAE for each prediction (in log space)
            for hedge_month in hedge_months:
                if hedge_month in month_specific_predictions:
                    pred_dict = month_specific_predictions[hedge_month]
                    pred_log = pred_dict['ensemble']
                    predicted_prices[hedge_month] = np.exp(pred_log)  # Convert log price back to original scale
                    prediction_maes[hedge_month] = pred_dict.get('mae', 0.1)  # Default 0.1 if not available
            
            if len(predicted_prices) != len(hedge_months):
                print(f'Warning: Missing predictions for some hedge months. Expected {len(hedge_months)}, got {len(predicted_prices)}')
                return None
            

            hedge_decisions = {}  
            hedged_months = []
            
            for month_idx in hedge_months:
                if month_idx >= len(actual_prices) or month_idx not in predicted_prices:
                    continue
                
                predicted_price_month = predicted_prices[month_idx]
                settlement_price_base = actual_prices[month_idx]
                
                settlement_price_upper = settlement_upper_prices[month_idx] if month_idx < len(settlement_upper_prices) else settlement_price_base
                settlement_price_lower = settlement_lower_prices[month_idx] if month_idx < len(settlement_lower_prices) else settlement_price_base
                
                mae_log_month = prediction_maes.get(month_idx, 0.1)
                
                position_type, should_hedge, prob_profitable, hedge_ratio = self._make_hedge_decision_for_month(
                    predicted_price_month, settlement_price_base, mae_log_month, horizon
                )
                
                hedge_decisions[month_idx] = (position_type, should_hedge, prob_profitable, hedge_ratio, 
                                             settlement_price_base, settlement_price_upper, settlement_price_lower)
                if should_hedge:
                    hedged_months.append(month_idx)
            
            if not hedged_months:
                months_available = min(len(actual_prices), len(monthly_volumes))
                total_actual_cost = sum([monthly_volumes[m] * actual_prices[m] for m in range(months_available)])
                
                avg_predicted_price = np.mean([predicted_prices[m] for m in hedge_months if m in predicted_prices])
                avg_settlement_price = np.mean([actual_prices[m] for m in hedge_months if m < len(actual_prices)])
                
                prob_profitable_values = []
                for month_idx in hedge_months:
                    if month_idx in hedge_decisions:
                        _, _, prob_prof, _, _, _, _ = hedge_decisions[month_idx]
                        prob_profitable_values.append(prob_prof)
                avg_prob_profitable = np.mean(prob_profitable_values) if prob_profitable_values else 0.0
                
                position_types = [hedge_decisions[m][0] for m in hedge_months if m in hedge_decisions]
                if position_types:
                    from collections import Counter
                    position_type_counts = Counter(position_types)
                    dominant_position_type = position_type_counts.most_common(1)[0][0]
                else:
                    dominant_position_type = 'NONE'
                
                return {
                    'total_volume': total_volume,
                    'volume_fraction': volume_fraction,
                    'hedged_volume': 0,
                    'hedge_volume_per_position': 0,
                    'hedge_months': [],
                    'predicted_price': avg_predicted_price,
                    'predicted_price_best': avg_predicted_price,
                    'predicted_price_worst': avg_predicted_price,
                    'avg_settlement_price': avg_settlement_price,
                    'should_hedge': False,
                    'position_type': dominant_position_type,
                    'prob_profitable': avg_prob_profitable,
                    'hedge_positions': [],
                    'cost_at_predicted_price': 0,
                    'total_settlement_cost': 0,
                    'hedge_cost': 0,
                    'locked_in_cost': 0,
                    'total_actual_cost': total_actual_cost,
                    'total_actual_cost_hedged': 0,
                    'hedging_benefit': 0,
                    'adjusted_hedging_benefit': 0,
                    'adjusted_benefit': 0,
                    'price_volatility': np.std(actual_prices),
                    'price_range': np.max(actual_prices) - np.min(actual_prices),
                    'avg_price': np.mean(actual_prices),
                    'prediction_error': abs(avg_predicted_price - np.mean(actual_prices)) / np.mean(actual_prices) if np.mean(actual_prices) > 0 else 0,
                }
            
            hedge_positions = []
            total_cost_at_predicted_price = 0
            total_settlement_cost = 0
            total_cost_at_predicted_price_uncertainty = 0
            total_pnl_uncertainty = 0
            total_pnl_best = 0
            total_pnl_worst = 0
            total_pnl_5th = 0
            total_pnl_25th = 0
            total_pnl_50th = 0
            total_pnl_75th = 0
            total_pnl_95th = 0
            total_predicted_price_best_weighted = 0
            total_predicted_price_worst_weighted = 0
            total_volume_for_predicted_prices = 0
            
            for month_idx in hedged_months:
                if month_idx >= len(actual_prices) or month_idx >= len(monthly_volumes):
                    continue
                
                position_type, should_hedge, prob_profitable, hedge_ratio, settlement_base, settlement_upper, settlement_lower = hedge_decisions[month_idx]
                
                base_volume_per_month = total_volume_to_hedge / len(hedged_months) if len(hedged_months) > 0 else 0
                hedge_volume_per_position = base_volume_per_month * hedge_ratio
                
                predicted_price_month = predicted_prices[month_idx]
                actual_volume_month = monthly_volumes[month_idx]
                
                mae_log = prediction_maes.get(month_idx, 0.1)
                pred_log = np.log(predicted_price_month)
                
                price_uncertainty_upper = predicted_price_month * (np.exp(mae_log) - 1)
                price_uncertainty_lower = predicted_price_month * (1 - np.exp(-mae_log))
                price_uncertainty = (price_uncertainty_upper + price_uncertainty_lower) / 2
                
                pred_log_5th = pred_log - 1.645 * mae_log
                pred_log_25th = pred_log - 0.675 * mae_log
                pred_log_50th = pred_log
                pred_log_75th = pred_log + 0.675 * mae_log
                pred_log_95th = pred_log + 1.645 * mae_log
                
                predicted_price_5th = np.exp(pred_log_5th)
                predicted_price_25th = np.exp(pred_log_25th)
                predicted_price_50th = predicted_price_month
                predicted_price_75th = np.exp(pred_log_75th)
                predicted_price_95th = np.exp(pred_log_95th)
                
                predicted_price_best = predicted_price_month - price_uncertainty_lower
                predicted_price_worst = predicted_price_month + price_uncertainty_upper
                
                total_predicted_price_best_weighted += predicted_price_best * hedge_volume_per_position
                total_predicted_price_worst_weighted += predicted_price_worst * hedge_volume_per_position
                total_volume_for_predicted_prices += hedge_volume_per_position
                
                settlement_uncertainty = (settlement_upper - settlement_lower) * hedge_volume_per_position
                
                if position_type == 'LONG':
                    cost_at_predicted_price_month = hedge_volume_per_position * predicted_price_month
                    cost_at_predicted_price_uncertainty_month = hedge_volume_per_position * price_uncertainty
                    
                    transaction_cost_month = hedge_volume_per_position * settlement_base * self.transaction_costs
                    transaction_cost_uncertainty_month = 0
                    
                    settlement_cost_month = hedge_volume_per_position * settlement_base
                    
                    pnl_month_5th_base = (predicted_price_5th - settlement_base) * hedge_volume_per_position
                    pnl_month_25th_base = (predicted_price_25th - settlement_base) * hedge_volume_per_position
                    pnl_month_50th_base = (predicted_price_50th - settlement_base) * hedge_volume_per_position
                    pnl_month_75th_base = (predicted_price_75th - settlement_base) * hedge_volume_per_position
                    pnl_month_95th_base = (predicted_price_95th - settlement_base) * hedge_volume_per_position
                    
                    pnl_month_best = (predicted_price_5th - settlement_lower) * hedge_volume_per_position
                    
                    pnl_month_worst = (predicted_price_95th - settlement_upper) * hedge_volume_per_position
                    
                    pnl_month_5th = pnl_month_5th_base - transaction_cost_month
                    pnl_month_25th = pnl_month_25th_base - transaction_cost_month
                    pnl_month_50th = pnl_month_50th_base - transaction_cost_month
                    pnl_month_75th = pnl_month_75th_base - transaction_cost_month
                    pnl_month_95th = pnl_month_95th_base - transaction_cost_month
                    
                    pnl_month_best = pnl_month_best - transaction_cost_month
                    pnl_month_worst = pnl_month_worst - transaction_cost_month
                    
                    weights = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
                    pnl_values = np.array([pnl_month_5th, pnl_month_25th, pnl_month_50th, pnl_month_75th, pnl_month_95th])
                    pnl_month = np.sum(weights * pnl_values)
                    
                    pnl_uncertainty_month = price_uncertainty * hedge_volume_per_position + settlement_uncertainty * 0.5
                    
                elif position_type == 'SHORT':
                    cost_at_predicted_price_month = hedge_volume_per_position * predicted_price_month
                    cost_at_predicted_price_uncertainty_month = hedge_volume_per_position * price_uncertainty
                    
                    transaction_cost_month = hedge_volume_per_position * settlement_base * self.transaction_costs
                    transaction_cost_uncertainty_month = 0
                    
                    settlement_cost_month = hedge_volume_per_position * settlement_base
                    
                    pnl_month_5th_base = (settlement_base - predicted_price_5th) * hedge_volume_per_position
                    pnl_month_25th_base = (settlement_base - predicted_price_25th) * hedge_volume_per_position
                    pnl_month_50th_base = (settlement_base - predicted_price_50th) * hedge_volume_per_position
                    pnl_month_75th_base = (settlement_base - predicted_price_75th) * hedge_volume_per_position
                    pnl_month_95th_base = (settlement_base - predicted_price_95th) * hedge_volume_per_position
                    
                    pnl_month_best = (settlement_upper - predicted_price_5th) * hedge_volume_per_position
                    
                    pnl_month_worst = (settlement_lower - predicted_price_95th) * hedge_volume_per_position
                    
                    pnl_month_5th = pnl_month_5th_base - transaction_cost_month
                    pnl_month_25th = pnl_month_25th_base - transaction_cost_month
                    pnl_month_50th = pnl_month_50th_base - transaction_cost_month
                    pnl_month_75th = pnl_month_75th_base - transaction_cost_month
                    pnl_month_95th = pnl_month_95th_base - transaction_cost_month
                    
                    pnl_month_best = pnl_month_best - transaction_cost_month
                    pnl_month_worst = pnl_month_worst - transaction_cost_month
                    
                    weights = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
                    pnl_values = np.array([pnl_month_5th, pnl_month_25th, pnl_month_50th, pnl_month_75th, pnl_month_95th])
                    pnl_month = np.sum(weights * pnl_values)
                    
                    pnl_uncertainty_month = price_uncertainty * hedge_volume_per_position + settlement_uncertainty * 0.5
                else:
                    cost_at_predicted_price_month = 0
                    cost_at_predicted_price_uncertainty_month = 0
                    transaction_cost_month = 0
                    transaction_cost_uncertainty_month = 0
                    settlement_cost_month = 0
                    pnl_month = 0
                    pnl_month_5th = 0
                    pnl_month_25th = 0
                    pnl_month_50th = 0
                    pnl_month_75th = 0
                    pnl_month_95th = 0
                    pnl_month_best = 0
                    pnl_month_worst = 0
                    pnl_uncertainty_month = 0
                
                total_cost_at_predicted_price += cost_at_predicted_price_month + transaction_cost_month
                total_settlement_cost += settlement_cost_month
                total_cost_at_predicted_price_uncertainty += cost_at_predicted_price_uncertainty_month
                total_pnl_uncertainty += pnl_uncertainty_month
                total_pnl_5th += pnl_month_5th
                total_pnl_25th += pnl_month_25th
                total_pnl_50th += pnl_month_50th
                total_pnl_75th += pnl_month_75th
                total_pnl_95th += pnl_month_95th
                total_pnl_best += pnl_month_best
                total_pnl_worst += pnl_month_worst
                
                hedge_positions.append({
                    'month': month_idx + 1,
                    'hedged_volume': hedge_volume_per_position,
                    'hedge_ratio': hedge_ratio,
                    'predicted_price': predicted_price_month,
                    'predicted_price_best': predicted_price_best,
                    'predicted_price_worst': predicted_price_worst,
                    'settlement_price_base': settlement_base,
                    'settlement_price_upper': settlement_upper,
                    'settlement_price_lower': settlement_lower,
                    'actual_price': settlement_base,  # Keep for backward compatibility
                    'position_type': position_type,
                    'cost_at_predicted_price': cost_at_predicted_price_month + transaction_cost_month,
                    'cost_at_predicted_price_uncertainty': cost_at_predicted_price_uncertainty_month + transaction_cost_uncertainty_month,
                    'transaction_cost': transaction_cost_month,
                    'settlement_cost': settlement_cost_month,
                    'pnl': pnl_month,
                    'pnl_5th': pnl_month_5th,
                    'pnl_25th': pnl_month_25th,
                    'pnl_50th': pnl_month_50th,
                    'pnl_75th': pnl_month_75th,
                    'pnl_95th': pnl_month_95th,
                    'pnl_best': pnl_month_best,
                    'pnl_worst': pnl_month_worst,
                    'pnl_uncertainty': pnl_uncertainty_month,
                    'price_uncertainty': price_uncertainty,
                    'settlement_uncertainty': settlement_uncertainty / hedge_volume_per_position if hedge_volume_per_position > 0 else 0
                })
            
            avg_predicted_price_best = total_predicted_price_best_weighted / total_volume_for_predicted_prices if total_volume_for_predicted_prices > 0 else 0
            avg_predicted_price_worst = total_predicted_price_worst_weighted / total_volume_for_predicted_prices if total_volume_for_predicted_prices > 0 else 0
            
            total_hedged_volume = 0
            total_predicted_weighted = 0
            total_settlement_weighted = 0
            total_prob_weighted = 0
            
            for month_idx in hedged_months:
                if month_idx not in hedge_decisions or month_idx >= len(actual_prices) or month_idx not in predicted_prices:
                    continue
                position_type, should_hedge, prob_profitable, hedge_ratio, settlement_base, settlement_upper, settlement_lower = hedge_decisions[month_idx]
                base_volume_per_month = total_volume_to_hedge / len(hedged_months) if len(hedged_months) > 0 else 0
                month_hedged_volume = base_volume_per_month * hedge_ratio
                
                total_hedged_volume += month_hedged_volume
                total_predicted_weighted += predicted_prices[month_idx] * month_hedged_volume
                total_settlement_weighted += settlement_base * month_hedged_volume
                total_prob_weighted += prob_profitable * month_hedged_volume
            
            avg_predicted_price = total_predicted_weighted / total_hedged_volume if total_hedged_volume > 0 else 0
            avg_settlement_price = total_settlement_weighted / total_hedged_volume if total_hedged_volume > 0 else 0
            avg_prob_profitable = total_prob_weighted / total_hedged_volume if total_hedged_volume > 0 else 0
            
            # Determine dominant position type (most common among hedged months)
            position_types = [hedge_decisions[m][0] for m in hedged_months if m in hedge_decisions]
            if position_types:
                from collections import Counter
                position_type_counts = Counter(position_types)
                dominant_position_type = position_type_counts.most_common(1)[0][0]
            else:
                dominant_position_type = 'NONE'
            
            total_actual_cost = 0
            months_available = min(len(actual_prices), len(monthly_volumes))
            for month in range(months_available):
                total_actual_cost += monthly_volumes[month] * actual_prices[month]
            
            weights = np.array([0.05, 0.25, 0.40, 0.25, 0.05])
            total_pnl_values = np.array([total_pnl_5th, total_pnl_25th, total_pnl_50th, total_pnl_75th, total_pnl_95th])
            total_hedging_benefit = np.sum(weights * total_pnl_values)  # Probability-weighted expected P&L
            
            # Use best/worst P&L that incorporate both prediction uncertainty AND settlement price uncertainty
            total_hedging_benefit_best = total_pnl_best  # Best case: best prediction vs best settlement
            total_hedging_benefit_worst = total_pnl_worst  # Worst case: worst prediction vs worst settlement
            
            cost_at_predicted_price_lower = total_cost_at_predicted_price - total_cost_at_predicted_price_uncertainty
            cost_at_predicted_price_upper = total_cost_at_predicted_price + total_cost_at_predicted_price_uncertainty
            
            hedging_benefit_lower = total_hedging_benefit - total_pnl_uncertainty
            hedging_benefit_upper = total_hedging_benefit + total_pnl_uncertainty
            
            total_cost_with_hedging_expected = total_actual_cost - total_hedging_benefit
            total_cost_with_hedging_best = total_actual_cost - total_hedging_benefit_best
            total_cost_with_hedging_worst = total_actual_cost - total_hedging_benefit_worst
            
            adjusted_hedging_benefit = total_hedging_benefit
            
            price_volatility = np.std(actual_prices)
            price_range = np.max(actual_prices) - np.min(actual_prices)
            avg_price = np.mean(actual_prices)
            prediction_error = abs(avg_predicted_price - avg_price) / avg_price if avg_price > 0 else 0
            
            return {
                'total_volume': total_volume,
                'volume_fraction': volume_fraction,
                'hedged_volume': total_hedged_volume,
                'hedge_volume_per_position': total_hedged_volume / len(hedged_months) if len(hedged_months) > 0 else 0,
                'hedge_months': [m + 1 for m in hedged_months],
                'predicted_price': avg_predicted_price,
                'predicted_price_best': avg_predicted_price_best,
                'predicted_price_worst': avg_predicted_price_worst,
                'avg_settlement_price': avg_settlement_price,
                'should_hedge': True,
                'position_type': dominant_position_type,
                'prob_profitable': avg_prob_profitable,
                'hedge_positions': hedge_positions,
                'cost_at_predicted_price': total_cost_at_predicted_price,
                'cost_at_predicted_price_uncertainty': total_cost_at_predicted_price_uncertainty,
                'cost_at_predicted_price_lower': cost_at_predicted_price_lower,
                'cost_at_predicted_price_upper': cost_at_predicted_price_upper,
                'total_settlement_cost': total_settlement_cost,
                'hedge_cost': total_cost_at_predicted_price,
                'hedge_cost_uncertainty': total_cost_at_predicted_price_uncertainty,
                'hedge_cost_lower': cost_at_predicted_price_lower,
                'hedge_cost_upper': cost_at_predicted_price_upper,
                'locked_in_cost': total_cost_at_predicted_price,
                'locked_in_cost_uncertainty': total_cost_at_predicted_price_uncertainty,
                'locked_in_cost_lower': cost_at_predicted_price_lower,
                'locked_in_cost_upper': cost_at_predicted_price_upper,
                'total_actual_cost': total_actual_cost,
                'total_actual_cost_hedged': total_settlement_cost,
                'hedging_benefit': total_hedging_benefit,
                'hedging_benefit_best': total_hedging_benefit_best,
                'hedging_benefit_worst': total_hedging_benefit_worst,
                'hedging_benefit_5th': total_pnl_5th,
                'hedging_benefit_25th': total_pnl_25th,
                'hedging_benefit_50th': total_pnl_50th,
                'hedging_benefit_75th': total_pnl_75th,
                'hedging_benefit_95th': total_pnl_95th,
                'hedging_benefit_uncertainty': total_pnl_uncertainty,
                'hedging_benefit_lower': hedging_benefit_lower,
                'hedging_benefit_upper': hedging_benefit_upper,
                'total_cost_with_hedging_expected': total_cost_with_hedging_expected,
                'total_cost_with_hedging_worst': total_cost_with_hedging_worst,
                'total_cost_with_hedging_best': total_cost_with_hedging_best,
                'adjusted_hedging_benefit': adjusted_hedging_benefit,
                'adjusted_benefit': adjusted_hedging_benefit,
                'price_volatility': price_volatility,
                'price_range': price_range,
                'avg_price': avg_price,
                'prediction_error': prediction_error,
            }
            
        except Exception as e:
            print(f'Error calculating prediction-based hedge for scenario: {e}')
            import traceback
            traceback.print_exc()
        return None
    
    def analyze_prediction_based_hedging_performance(self):
        """Analyze prediction-based hedging performance with skill decay considerations."""
        print('Analyzing prediction-based hedging performance...')
        
        if not self.hedging_results:
            print('No hedging results to analyze')
            return None
        
        analysis_results = {}
        
        for horizon_name, horizon_strategies in self.hedging_results.items():
            print(f'Analyzing {horizon_name}...')
            
            horizon_analysis = {}
            
            for volume_name, strategy_results in horizon_strategies.items():
                if not strategy_results:
                    continue
                
                df = pd.DataFrame(strategy_results)
                
                performance_metrics = {
                    'scenarios_count': len(df),
                    'avg_hedging_benefit': df['adjusted_hedging_benefit'].mean(),
                    'avg_adjusted_benefit': df['adjusted_benefit'].mean(),
                    'std_hedging_benefit': df['adjusted_hedging_benefit'].std(),
                    'min_hedging_benefit': df['adjusted_hedging_benefit'].min(),
                    'max_hedging_benefit': df['adjusted_hedging_benefit'].max(),
                    'avg_hedge_cost': df['hedge_cost'].mean(),
                    'avg_total_actual_cost': df['total_actual_cost'].mean(),
                    'avg_total_actual_cost_hedged': df['total_actual_cost_hedged'].mean(),
                    'avg_hedged_volume': df['hedged_volume'].mean(),
                    'avg_price_volatility': df['price_volatility'].mean(),
                    'avg_price_range': df['price_range'].mean(),
                    'avg_prediction_error': df['prediction_error'].mean(),
                    'success_rate': (df['adjusted_hedging_benefit'] > 0).mean(),
                    'adjusted_success_rate': (df['adjusted_benefit'] > 0).mean()
                }
                
                risk_metrics = {
                    'var_95': np.percentile(df['adjusted_hedging_benefit'], 5),
                    'var_99': np.percentile(df['adjusted_hedging_benefit'], 1),
                    'adjusted_var_95': np.percentile(df['adjusted_benefit'], 5),
                    'expected_shortfall': df[df['adjusted_hedging_benefit'] <= np.percentile(df['adjusted_hedging_benefit'], 5)]['adjusted_hedging_benefit'].mean(),
                    'sharpe_ratio': df['adjusted_hedging_benefit'].mean() / df['adjusted_hedging_benefit'].std() if df['adjusted_hedging_benefit'].std() > 0 else 0,
                    'adjusted_sharpe': df['adjusted_benefit'].mean() / df['adjusted_benefit'].std() if df['adjusted_benefit'].std() > 0 else 0,
                    'prediction_accuracy': 1 - df['prediction_error'].mean()
                }
                
                scenario_type_analysis = {}
                for scenario_type in df['scenario_type'].unique():
                    type_data = df[df['scenario_type'] == scenario_type]
                    scenario_type_analysis[scenario_type] = {
                        'count': len(type_data),
                        'avg_benefit': type_data['adjusted_hedging_benefit'].mean(),
                        'success_rate': (type_data['adjusted_hedging_benefit'] > 0).mean(),
                        'avg_volatility': type_data['price_volatility'].mean(),
                        'avg_prediction_error': type_data['prediction_error'].mean()
                    }
                
                horizon_analysis[volume_name] = {
                    'performance': performance_metrics,
                    'risk': risk_metrics,
                    'scenario_types': scenario_type_analysis,
                    'data': df
                }
            
            analysis_results[horizon_name] = horizon_analysis
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def create_prediction_based_visualization(self):
        """Create visualization of total cost comparison with percentiles."""
        print('Creating prediction-based hedging visualization...')
        
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            print('No analysis results to visualize')
            return
        
        strategy_data = []
        original_costs_all = []  
        
        for horizon_name, horizon_data in self.analysis_results.items():
            if '9_month' in horizon_name or '9mo' in horizon_name.lower():
                continue
                
            for volume_name, volume_data in horizon_data.items():
                df = volume_data['data']  # DataFrame with all scenario results

                original_costs = df['total_actual_cost'].values
                original_costs_all.extend(original_costs.tolist())
                

                hedged_costs_expected = df['total_cost_with_hedging_expected'].values if 'total_cost_with_hedging_expected' in df.columns else df['total_actual_cost'] - df['hedging_benefit']
                hedged_costs_worst = df['total_cost_with_hedging_worst'].values if 'total_cost_with_hedging_worst' in df.columns else hedged_costs_expected
                hedged_costs_best = df['total_cost_with_hedging_best'].values if 'total_cost_with_hedging_best' in df.columns else hedged_costs_expected
                
                strategy_name = f'{horizon_name.split("_")[0]}mo—{volume_name.replace("_target", "")} target'
                
                hedged_costs_expected = hedged_costs_expected[np.isfinite(hedged_costs_expected)]
                hedged_costs_worst = hedged_costs_worst[np.isfinite(hedged_costs_worst)]
                hedged_costs_best = hedged_costs_best[np.isfinite(hedged_costs_best)]
                
                if len(hedged_costs_expected) == 0:
                    print(f'  Warning: No valid hedged costs for strategy {strategy_name}, skipping')
                    continue
                

                hedged_mean = np.mean(hedged_costs_expected)
                
                all_costs_combined = np.concatenate([hedged_costs_best, hedged_costs_expected, hedged_costs_worst])
                all_costs_combined = all_costs_combined[np.isfinite(all_costs_combined)]
                
                if len(all_costs_combined) > 0:
                    hedged_overall_lower = np.min(all_costs_combined)  # Minimum from combined distribution
                    hedged_overall_upper = np.max(all_costs_combined)  # Maximum from combined distribution
                else:
                    hedged_overall_lower = np.min(hedged_costs_best) if len(hedged_costs_best) > 0 else np.nan
                    hedged_overall_upper = np.max(hedged_costs_worst) if len(hedged_costs_worst) > 0 else np.nan
                
                hedged_worst_case = np.max(hedged_costs_worst)  # Worst-case including uncertainty
                hedged_best_case = np.min(hedged_costs_best)    # Best-case including uncertainty
                
                uncertainty_upper = hedged_overall_upper - hedged_mean  # How much worse than expected (worst scenario worst-case)
                uncertainty_lower = hedged_mean - hedged_overall_lower    # How much better than expected (best scenario best-case)
                
                mean_uncertainty_upper = np.mean(hedged_costs_worst - hedged_costs_expected)
                mean_uncertainty_lower = np.mean(hedged_costs_expected - hedged_costs_best)
                
                strategy_data.append({
                    'strategy': strategy_name,
                    'hedged_mean': hedged_mean,
                    'hedged_overall_lower': hedged_overall_lower,
                    'hedged_overall_upper': hedged_overall_upper,
                    'hedged_best_case': hedged_best_case,  # Best-case including prediction uncertainty
                    'hedged_worst_case': hedged_worst_case,  # Worst-case including prediction uncertainty
                    'uncertainty_upper': uncertainty_upper,  # Full range upper bound
                    'uncertainty_lower': uncertainty_lower,    # Full range lower bound
                    'mean_uncertainty_upper': mean_uncertainty_upper,  # Average uncertainty upper
                    'mean_uncertainty_lower': mean_uncertainty_lower,  # Average uncertainty lower
                    'hedged_min_expected': np.min(hedged_costs_expected),
                    'hedged_max_expected': np.max(hedged_costs_expected),
                })
        
        if not strategy_data:
            print('No strategy data to visualize')
            return
        

        original_costs_all = np.array(original_costs_all)
        original_costs_all = original_costs_all[np.isfinite(original_costs_all)]
        
        if len(original_costs_all) == 0:
            print('Warning: No valid original costs found for visualization')
            return
        
        original_cost_min = np.min(original_costs_all)
        original_cost_mean = np.mean(original_costs_all)  # Mean for line and label
        original_cost_max = np.max(original_costs_all)
        
        original_cost_p5 = np.percentile(original_costs_all, 5)
        original_cost_p20 = np.percentile(original_costs_all, 20)
        original_cost_p50 = np.percentile(original_costs_all, 50)  # 50th percentile (median)
        original_cost_p80 = np.percentile(original_costs_all, 80)
        original_cost_p95 = np.percentile(original_costs_all, 95)
        original_cost_median = original_cost_p50
        
        if not all(np.isfinite([original_cost_min, original_cost_median, original_cost_mean, original_cost_max,
                                original_cost_p5, original_cost_p20, original_cost_p50, original_cost_p80, original_cost_p95])):
            print('Warning: Some original cost statistics are not finite, skipping visualization')
            return
        
        hedged_means = [s['hedged_mean'] for s in strategy_data]  # Mean values for line and label
        hedged_overall_lower = [s['hedged_overall_lower'] for s in strategy_data]
        hedged_overall_upper = [s['hedged_overall_upper'] for s in strategy_data]
        hedged_best_case = [s.get('hedged_best_case', s['hedged_overall_lower']) for s in strategy_data]  # Best-case with uncertainty
        hedged_worst_case = [s.get('hedged_worst_case', s['hedged_overall_upper']) for s in strategy_data]  # Worst-case with uncertainty
        
        hedged_means = [m if np.isfinite(m) else np.nan for m in hedged_means]
        hedged_overall_lower = [l if np.isfinite(l) else np.nan for l in hedged_overall_lower]
        hedged_overall_upper = [u if np.isfinite(u) else np.nan for u in hedged_overall_upper]
        hedged_best_case = [b if np.isfinite(b) else np.nan for b in hedged_best_case]
        hedged_worst_case = [w if np.isfinite(w) else np.nan for w in hedged_worst_case]
        
 
        hedged_percentiles = []  # List of dicts with p5, p20, p50, p80, p95, min, max for each strategy
        hedged_means_combined = []  # Mean from combined distribution (for display, to match percentiles)
        for s in strategy_data:
            for horizon_name, horizon_data in self.analysis_results.items():
                if '9_month' in horizon_name or '9mo' in horizon_name.lower():
                    continue
                for volume_name, volume_data in horizon_data.items():
                    matching_strategy_name = f'{horizon_name.split("_")[0]}mo—{volume_name.replace("_target", "")} target'
                    
                    if matching_strategy_name == s['strategy']:
                        df = volume_data['data']
                        hedged_costs_expected = df['total_cost_with_hedging_expected'].values if 'total_cost_with_hedging_expected' in df.columns else df['total_actual_cost'] - df['hedging_benefit']
                        hedged_costs_worst = df['total_cost_with_hedging_worst'].values if 'total_cost_with_hedging_worst' in df.columns else hedged_costs_expected
                        hedged_costs_best = df['total_cost_with_hedging_best'].values if 'total_cost_with_hedging_best' in df.columns else hedged_costs_expected
                        
                        hedged_costs_expected = hedged_costs_expected[np.isfinite(hedged_costs_expected)]
                        hedged_costs_worst = hedged_costs_worst[np.isfinite(hedged_costs_worst)]
                        hedged_costs_best = hedged_costs_best[np.isfinite(hedged_costs_best)]
                        
                        if len(hedged_costs_expected) > 0:
                            all_costs = np.concatenate([hedged_costs_best, hedged_costs_expected, hedged_costs_worst])
                            all_costs = all_costs[np.isfinite(all_costs)]
                            
                            if len(all_costs) > 0:
                                mean_combined = np.mean(all_costs)
                                hedged_means_combined.append(mean_combined)
                                
                                hedged_percentiles.append({
                                    'min': np.min(all_costs),
                                    'p5': np.percentile(all_costs, 5),
                                    'p20': np.percentile(all_costs, 20),
                                    'p50': np.percentile(all_costs, 50),
                                    'p80': np.percentile(all_costs, 80),
                                    'p95': np.percentile(all_costs, 95),
                                    'max': np.max(all_costs)
                                })
                            else:
                                hedged_means_combined.append(np.nan)
                                hedged_percentiles.append({
                                    'min': np.nan, 'p5': np.nan, 'p20': np.nan, 'p50': np.nan,
                                    'p80': np.nan, 'p95': np.nan, 'max': np.nan
                                })
                        else:
                            hedged_means_combined.append(np.nan)
                            hedged_percentiles.append({
                                'min': np.nan, 'p5': np.nan, 'p20': np.nan, 'p50': np.nan,
                                'p80': np.nan, 'p95': np.nan, 'max': np.nan
                            })
                        break
                if len(hedged_percentiles) == len(hedged_means):
                    break
            if len(hedged_percentiles) == len(hedged_means):
                break
        
        hedged_medians = [p['p50'] if np.isfinite(p['p50']) else np.nan for p in hedged_percentiles]
        
        valid_indices = []
        for i in range(len(strategy_data)):
            p = hedged_percentiles[i] if i < len(hedged_percentiles) else {}
            mean_combined = hedged_means_combined[i] if i < len(hedged_means_combined) else np.nan
            is_valid = (np.isfinite(hedged_means[i]) and np.isfinite(mean_combined) and 
                       np.isfinite(hedged_overall_lower[i]) and np.isfinite(hedged_overall_upper[i]) and 
                       np.isfinite(hedged_medians[i]) and np.isfinite(hedged_best_case[i]) and 
                       np.isfinite(hedged_worst_case[i]) and
                       all(np.isfinite([p.get('min', np.nan), p.get('p5', np.nan), p.get('p20', np.nan),
                                        p.get('p50', np.nan), p.get('p80', np.nan), p.get('p95', np.nan), p.get('max', np.nan)])))
            if is_valid:
                valid_indices.append(i)
            else:
                print(f'  Filtering out strategy {strategy_data[i]["strategy"]}: '
                      f'mean={hedged_means[i]:.2f}, mean_combined={mean_combined:.2f}, lower={hedged_overall_lower[i]:.2f}, '
                      f'upper={hedged_overall_upper[i]:.2f}, median={hedged_medians[i]:.2f}')
        

        if valid_indices:
            print(f'  Found {len(valid_indices)} valid hedged strategies out of {len(strategy_data)} total')
            strategy_data = [strategy_data[i] for i in valid_indices]
            hedged_means = [hedged_means[i] for i in valid_indices]  # Keep original for reference
            hedged_means_combined = [hedged_means_combined[i] for i in valid_indices]  # Use combined for display
            hedged_overall_lower = [hedged_overall_lower[i] for i in valid_indices]
            hedged_overall_upper = [hedged_overall_upper[i] for i in valid_indices]
            hedged_medians = [hedged_medians[i] for i in valid_indices]
            hedged_best_case = [hedged_best_case[i] for i in valid_indices]
            hedged_worst_case = [hedged_worst_case[i] for i in valid_indices]
            hedged_percentiles = [hedged_percentiles[i] for i in valid_indices]
        else:
            print(f'Warning: No valid hedged cost data found (checked {len(strategy_data)} strategies), showing only original cost')
            strategy_data = []
            hedged_means = []
            hedged_means_combined = []
            hedged_overall_lower = []
            hedged_overall_upper = []
            hedged_medians = []
            hedged_best_case = []
            hedged_worst_case = []
            hedged_percentiles = []
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        n_strategies = len(strategy_data)
        x_original = 0
        x_hedged = np.arange(1, n_strategies + 1)
        x_positions = np.concatenate([[x_original], x_hedged])
        width = 0.6  # Width of bars
        
        strategy_names = ['No Hedging'] + [s['strategy'] for s in strategy_data]
        
        # Create stacked bars for original cost: min to median, median to p95, p95 to max (top 5%)
        original_bottom = max(0, original_cost_min) if np.isfinite(original_cost_min) else 0
        original_seg1 = max(0, original_cost_median - original_cost_min) if np.isfinite(original_cost_median) and np.isfinite(original_cost_min) else 0
        original_seg2 = max(0, original_cost_p95 - original_cost_median) if np.isfinite(original_cost_p95) and np.isfinite(original_cost_median) else 0
        original_seg3 = max(0, original_cost_max - original_cost_p95) if np.isfinite(original_cost_max) and np.isfinite(original_cost_p95) else 0
        
        original_bottom = original_bottom if np.isfinite(original_bottom) else 0
        original_seg1 = original_seg1 if np.isfinite(original_seg1) else 0
        original_seg2 = original_seg2 if np.isfinite(original_seg2) else 0
        original_seg3 = original_seg3 if np.isfinite(original_seg3) else 0
        
        ax.bar(x_original, original_seg1, width, bottom=original_bottom,
               label='Min to Median', color='#91bfdb', alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.bar(x_original, original_seg2, width, bottom=original_bottom + original_seg1,
               label='Median to 95th Percentile', color='#fc8d59', alpha=0.6, edgecolor='white', linewidth=0.5)
        ax.bar(x_original, original_seg3, width, bottom=original_bottom + original_seg1 + original_seg2,
               label='Top 5% High Cost', color='#ca0020', alpha=0.8, edgecolor='white', linewidth=0.5)
        
        if len(strategy_data) > 0:
            hedged_bottoms = [max(0, lower) if np.isfinite(lower) else 0 for lower in hedged_overall_lower]
            hedged_seg1 = []  # min to median
            hedged_seg2 = []  # median to p95
            hedged_seg3 = []  # p95 to max (top 5%)
            
            for i, p in enumerate(hedged_percentiles):
                p_min = p['min'] if np.isfinite(p['min']) else hedged_overall_lower[i]
                p_median = p['p50'] if np.isfinite(p['p50']) else p_min
                p_p95 = p['p95'] if np.isfinite(p['p95']) else p_median
                p_max = p['max'] if np.isfinite(p['max']) else p_p95
                
                seg1 = max(0, p_median - p_min) if np.isfinite(p_median) and np.isfinite(p_min) else 0
                seg2 = max(0, p_p95 - p_median) if np.isfinite(p_p95) and np.isfinite(p_median) else 0
                seg3 = max(0, p_max - p_p95) if np.isfinite(p_max) and np.isfinite(p_p95) else 0
                
                hedged_seg1.append(seg1 if np.isfinite(seg1) else 0)
                hedged_seg2.append(seg2 if np.isfinite(seg2) else 0)
                hedged_seg3.append(seg3 if np.isfinite(seg3) else 0)
            
            bottom1 = hedged_bottoms
            bottom2 = [b + s1 for b, s1 in zip(hedged_bottoms, hedged_seg1)]
            
            ax.bar(x_hedged, hedged_seg1, width, bottom=bottom1,
                   color='#91bfdb', alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.bar(x_hedged, hedged_seg2, width, bottom=bottom2,
                   color='#fc8d59', alpha=0.6, edgecolor='white', linewidth=0.5)
            ax.bar(x_hedged, hedged_seg3, width, bottom=[b + s1 + s2 for b, s1, s2 in zip(hedged_bottoms, hedged_seg1, hedged_seg2)],
                   color='#ca0020', alpha=0.8, edgecolor='white', linewidth=0.5)
        
        print('\n' + '='*80)
        print('REFERENCE STATISTICS FOR EACH HEDGING STRATEGY')
        print('='*80)
        print(f'\nNo Hedging:')
        print(f'  Min:     ${original_cost_min/1000:.2f}M' if np.isfinite(original_cost_min) else '  Min:     N/A')
        print(f'  Median: ${original_cost_median/1000:.2f}M' if np.isfinite(original_cost_median) else '  Median: N/A')
        print(f'  95th %: ${original_cost_p95/1000:.2f}M' if np.isfinite(original_cost_p95) else '  95th %: N/A')
        print(f'  Max:     ${original_cost_max/1000:.2f}M' if np.isfinite(original_cost_max) else '  Max:     N/A')
        
        y_range = original_cost_max - original_cost_min if np.isfinite(original_cost_max) and np.isfinite(original_cost_min) else original_cost_max * 0.05
        label_offset = y_range * 0.02  # 2% of range
        
        if np.isfinite(original_cost_median):
            ax.text(x_original, original_cost_median + label_offset, f'${original_cost_median/1000:.1f}M',
                   ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
        if np.isfinite(original_cost_p95):
            ax.text(x_original, original_cost_p95 + label_offset, f'${original_cost_p95/1000:.1f}M',
                   ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
        
        if len(strategy_data) > 0 and len(hedged_percentiles) > 0:
            for i, (x_pos, p, strategy) in enumerate(zip(x_hedged, hedged_percentiles, strategy_data)):
                p_min = p['min'] if np.isfinite(p['min']) else None
                p_median = p['p50'] if np.isfinite(p['p50']) else None
                p_p95 = p['p95'] if np.isfinite(p['p95']) else None
                p_max = p['max'] if np.isfinite(p['max']) else None
                
                strategy_name = strategy['strategy']
                print(f'\n{strategy_name}:')
                print(f'  Min:     ${p_min/1000:.2f}M' if p_min is not None and np.isfinite(p_min) else '  Min:     N/A')
                print(f'  Median: ${p_median/1000:.2f}M' if p_median is not None and np.isfinite(p_median) else '  Median: N/A')
                print(f'  95th %: ${p_p95/1000:.2f}M' if p_p95 is not None and np.isfinite(p_p95) else '  95th %: N/A')
                print(f'  Max:     ${p_max/1000:.2f}M' if p_max is not None and np.isfinite(p_max) else '  Max:     N/A')
                
                strategy_y_range = p_max - p_min if (p_max is not None and p_min is not None and np.isfinite(p_max) and np.isfinite(p_min)) else (p_max * 0.05 if p_max is not None and np.isfinite(p_max) else 1000)
                strategy_label_offset = strategy_y_range * 0.02  # 2% of range
                
                if p_median is not None and np.isfinite(p_median):
                    ax.text(x_pos, p_median + strategy_label_offset, f'${p_median/1000:.1f}M',
                           ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
                if p_p95 is not None and np.isfinite(p_p95):
                    ax.text(x_pos, p_p95 + strategy_label_offset, f'${p_p95/1000:.1f}M',
                           ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
        
        print('\n' + '='*80 + '\n')

        ax.set_ylabel('Total Cost (Million $)', fontsize=20)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=20)
        
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}M'))
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#91bfdb', alpha=0.8, label='Min to Median'),
            Patch(facecolor='#fc8d59', alpha=0.6, label='Median to 95th Percentile'),
            Patch(facecolor='#ca0020', alpha=0.8, label='Top 5% Cost')
        ]
        ax.legend(handles=legend_elements, loc='center left', fontsize=20, framealpha=0.9, 
                 bbox_to_anchor=(1.0, 0.5), handlelength=1.5, handletextpad=0.5, columnspacing=0.8)
        
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout(rect=[0, 0, 0.95, 1])  # Reserve 5% for legend on the right
        plt.savefig('hybrid_result/fixed_prediction_based_hedging_all_horizons.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_prediction_based_results(self):
        """Save prediction-based hedging results to CSV files with detailed P&L for each scenario."""
        print('Saving prediction-based hedging results...')
        
        if not self.hedging_results:
            print('No results to save')
            return
        
        for horizon_name, horizon_strategies in self.hedging_results.items():
            for volume_name, strategy_results in horizon_strategies.items():
                if not strategy_results:
                    continue
                
                detailed_results = []
                
                for result in strategy_results:
                    scenario_id = result['scenario_id']
                    total_volume = result['total_volume']
                    hedged_volume = result['hedged_volume']
                    cost_at_predicted_price = result.get('cost_at_predicted_price', result.get('hedge_cost', result.get('locked_in_cost', 0)))
                    total_actual_cost = result['total_actual_cost']
                    total_settlement_cost = result.get('total_settlement_cost', result.get('total_actual_cost_hedged', 0))
                    hedging_benefit = result['hedging_benefit']
                    adjusted_benefit = result['adjusted_hedging_benefit']
                    hedge_months = result['hedge_months']
                    volume_fraction = result['volume_fraction']
                    
                    avg_april_may_oroville_s = None
                    if self.pumping_data:
                        scenario_data = next((s for s in self.pumping_data if s['scenario'] == scenario_id), None)
                        if scenario_data:
                            hydro_data_to_use = scenario_data.get('original_hydro_data')
                            if hydro_data_to_use is None:
                                hydro_data_to_use = scenario_data.get('hydro_data')
                            
                            if hydro_data_to_use is not None:
                                oroville_s_cols = [col for col in hydro_data_to_use.columns 
                                                 if 'oroville' in col.lower() and col.endswith('_S')]
                                
                                if oroville_s_cols:
                                    oroville_s_col = oroville_s_cols[0] if len(oroville_s_cols) == 1 else oroville_s_cols
                                    
                                    if hasattr(hydro_data_to_use.index, 'month'):
                                        april_may_data = hydro_data_to_use[
                                            (hydro_data_to_use.index.month == 4) | (hydro_data_to_use.index.month == 5)
                                        ]
                                    elif hasattr(hydro_data_to_use.index, 'to_period'):
                                        april_may_mask = hydro_data_to_use.index.to_timestamp().month.isin([4, 5])
                                        april_may_data = hydro_data_to_use[april_may_mask]
                                    else:
                                        if len(hydro_data_to_use) > 7:
                                            april_may_data = hydro_data_to_use.iloc[6:8]  # Months 6 and 7
                                        else:
                                            april_may_data = pd.DataFrame()
                                    
                                    if len(april_may_data) > 0:
                                        if isinstance(oroville_s_col, list):
                                            # Sum multiple columns
                                            avg_april_may_oroville_s = april_may_data[oroville_s_col].sum(axis=1).mean()
                                        else:
                                            avg_april_may_oroville_s = april_may_data[oroville_s_col].mean()
                    
                    avg_settlement_price = result.get('avg_settlement_price', result.get('predicted_price', 0))
                    should_hedge = result.get('should_hedge', True)
                    position_type = result.get('position_type', 'NONE')
                    
                    total_cost_with_hedging_expected = result.get('total_cost_with_hedging_expected', np.nan)
                    total_cost_with_hedging_best = result.get('total_cost_with_hedging_best', np.nan)
                    total_cost_with_hedging_worst = result.get('total_cost_with_hedging_worst', np.nan)
                    
                    hedging_benefit = result.get('hedging_benefit', result.get('profit_loss', 0))  # Expected P&L
                    hedging_benefit_best = result.get('hedging_benefit_best', np.nan)  # Best-case P&L
                    hedging_benefit_worst = result.get('hedging_benefit_worst', np.nan)  # Worst-case P&L
                    
                    hedge_positions_list = result.get('hedge_positions', [])
                    num_hedged_positions = len(hedge_positions_list)
                    
                    detailed_results.append({
                        'scenario_id': scenario_id,
                        'scenario_type': result['scenario_type'],
                        'avg_april_may_oroville_s': avg_april_may_oroville_s if avg_april_may_oroville_s is not None else np.nan,
                        'volume_fraction': volume_fraction,
                        'total_volume': total_volume,
                        'total_hedged_volume': hedged_volume,
                        'hedge_volume_per_position': result['hedge_volume_per_position'],
                        'num_hedged_positions': num_hedged_positions,  # Number of positions actually hedged
                        'hedge_months': str(hedge_months),  # Months that were hedged (1-indexed)
                        'predicted_price': result['predicted_price'],
                        'predicted_price_best': result.get('predicted_price_best', np.nan),  # Best-case predicted price (with MAE uncertainty)
                        'predicted_price_worst': result.get('predicted_price_worst', np.nan),  # Worst-case predicted price (with MAE uncertainty)
                        'avg_settlement_price': avg_settlement_price,
                        'should_hedge': should_hedge,
                        'position_type': position_type,
                        'prob_profitable': result.get('prob_profitable', np.nan),  # Probability of profit for hedging (>= 0.8 to hedge)
                        'cost_at_predicted_price': cost_at_predicted_price,  # Cost at predicted market price (without hedging)
                        'total_actual_cost': total_actual_cost,  # Total actual cost without hedging
                        'total_settlement_cost': total_settlement_cost,  # Total actual cost at settlement (with hedging)
                        'total_cost_with_hedging_expected': total_cost_with_hedging_expected,  # Expected cost with hedging (accounting for prediction uncertainty)
                        'total_cost_with_hedging_best': total_cost_with_hedging_best,  # Best-case cost with hedging (including MAE uncertainty)
                        'total_cost_with_hedging_worst': total_cost_with_hedging_worst,  # Worst-case cost with hedging (including MAE uncertainty)
                        'profit_loss': hedging_benefit,  # Expected P&L (profit/loss)
                        'profit_loss_best': hedging_benefit_best,  # Best-case P&L (with predicted price - MAE)
                        'profit_loss_worst': hedging_benefit_worst,  # Worst-case P&L (with predicted price + MAE)
                        'prediction_error': result['prediction_error'],
       
                    })
                    
                    avg_settlement_price = result.get('avg_settlement_price', result.get('predicted_price', 0))
                    
     
                df = pd.DataFrame(detailed_results)
                filename = f'hybrid_result/fixed_prediction_based_hedging_{horizon_name}_{volume_name}.csv'
                df.to_csv(filename, index=False)


def main():
    """Main execution function for fixed prediction-based hedging strategy analysis."""
    print('FIXED PREDICTION-BASED WATER FUTURES HEDGING STRATEGY ANALYSIS')
    print('=' * 75)
    
    hedger = FixedScenarioSpecificWaterHedgingStrategy()
    
    if not hedger.load_scenario_data():
        print('Failed to load scenario data')
        return
    
    if not hedger.load_pumping_data():
        print('Failed to load pumping data')
        return
    
    if not hedger.load_trained_models():
        print('Failed to load trained models')
        return
    
    if not hedger.calculate_prediction_based_hedging_strategies():
        print('Failed to calculate prediction-based hedging strategies')
        return
    
    hedger.analyze_prediction_based_hedging_performance()
    
    hedger.create_prediction_based_visualization()
    
    hedger.save_prediction_based_results()
    
    print('\nFixed prediction-based hedging strategy analysis completed!')
    print('Results saved in hybrid_result/ directory')


if __name__ == '__main__':
    main()
