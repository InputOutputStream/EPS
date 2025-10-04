# scripts/processor.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import joblib
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

class AdaptiveExoplanetProcessor:
    def __init__(self, n_components=3, variance_threshold=0.01):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.label_encoders = {}
        self.fitted = False
        
        # NASA-prioritized critical features (decreasing importance)
        # Source: https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
        # https://science.nasa.gov/exoplanets/exoplanet-catalog/
        self.critical_features = [
            'disposition',           # Classification status (essential)
            'pl_orbper',            # Orbital period (primary detection parameter)
            'pl_rade',              # Planet radius (confirms planetary nature)
            'pl_trandur',           # Transit duration (validates signal)
            'st_teff',              # Stellar temp (host star characterization)
            'st_rad',               # Stellar radius (scales planet properties)
            'discoverymethod'       # Detection method (validation pathway)
        ]
        
        # NASA feature importance hierarchy
        # https://exoplanetarchive.ipac.caltech.edu/docs/transit_algorithms.html
        self.nasa_priority_features = {
            'pl_orbper': 1.0,       # Period - most fundamental
            'pl_trandur': 0.95,     # Duration - transit signature
            'pl_trandep': 0.90,     # Depth - signal strength
            'pl_rade': 0.95,        # Planet radius - physical property
            'pl_masse': 0.85,       # Mass - density constraint
            'st_teff': 0.90,        # Stellar temp - host characterization
            'st_rad': 0.85,         # Stellar radius - system scale
            'st_mass': 0.80,        # Stellar mass - dynamics
            'pl_orbincl': 0.75,     # Inclination - geometry
            'pl_imppar': 0.70       # Impact parameter - transit geometry
        }
        
        # Categorical columns to encode
        self.categorical_columns = [
            'discoverymethod', 'disc_locale', 'disc_facility',
            'disc_telescope', 'disc_instrument', 'st_spectype'
        ]
        
        # Columns to drop
        self.drop_columns = [
            'rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name',
            'epic_hostname', 'epic_candname', 'hd_name', 'hip_name',
            'pl_refname', 'disc_refname', 'pl_pubdate', 'releasedate',
            'rastr', 'decstr', 'st_refname', 'sy_refname',
            'rowupdate', 'default_flag'
        ]
        
        self.selected_features = None
        self.feature_importance = None
        self.calculated_reputations = {}
    
    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        try:
            file_ext = Path(filepath).suffix.lower()
            df = pd.read_csv(filepath) if file_ext == '.csv' else pd.read_json(filepath)
            print(f"Loaded {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print("\n=== PREPROCESSING PIPELINE ===")
        
        # 1. Calculate data-driven reputations FIRST (before filtering)
        self._calculate_reputations(df)
        
        # 2. Filter records with critical missing values
        df = self._filter_critical_missing(df)
        
        # 3. Add reputation-based features
        df = self._add_reputation_features(df)
        
        # 4. Encode categorical features
        df = self._encode_categorical(df)
        
        # 5. Drop unnecessary columns
        df = self._drop_columns(df)
        
        # 6. Extract labels
        y = self._extract_labels(df)
        df = df.drop(columns=['disposition'], errors='ignore')
        
        # 7. Handle missing values with NASA priority awareness
        df = self._handle_missing_values_priority(df)
        
        # 8. Feature selection with NASA hierarchy
        X, selected_features = self._select_features_nasa(df, y)
        
        # 9. Handle outliers
        X = self._handle_outliers(X)
        
        # 10. Log-transform skewed features
        X = self._log_transform_skewed(X, selected_features)
        
        print(f"\n✓ Final: {X.shape[0]} records × {X.shape[1]} features")
        
        return X, y, selected_features
    
    def _calculate_reputations(self, df: pd.DataFrame):
        """Calculate success rates from actual data"""
        print("\nCalculating data-driven reputations...")
        
        if 'disposition' not in df.columns:
            return
        
        # Map dispositions to success (CONFIRMED = success)
        success_map = {'CONFIRMED': 1, 'Confirmed': 1}
        df['_success'] = df['disposition'].map(success_map).fillna(0)
        
        # Calculate telescope reputation
        if 'disc_telescope' in df.columns:
            telescope_stats = df.groupby('disc_telescope')['_success'].agg(['sum', 'count'])
            telescope_stats['success_rate'] = telescope_stats['sum'] / telescope_stats['count']
            # Weight by volume (more detections = higher confidence)
            telescope_stats['weighted_score'] = (
                telescope_stats['success_rate'] * 0.7 + 
                (telescope_stats['count'] / telescope_stats['count'].max()) * 0.3
            )
            self.calculated_reputations['telescope'] = telescope_stats['weighted_score'].to_dict()
            print(f"  Telescopes: {len(self.calculated_reputations['telescope'])} analyzed")
        
        # Calculate instrument reputation
        if 'disc_instrument' in df.columns:
            instrument_stats = df.groupby('disc_instrument')['_success'].agg(['sum', 'count'])
            instrument_stats['success_rate'] = instrument_stats['sum'] / instrument_stats['count']
            instrument_stats['weighted_score'] = (
                instrument_stats['success_rate'] * 0.7 + 
                (instrument_stats['count'] / instrument_stats['count'].max()) * 0.3
            )
            self.calculated_reputations['instrument'] = instrument_stats['weighted_score'].to_dict()
            print(f"  Instruments: {len(self.calculated_reputations['instrument'])} analyzed")
        
        # Calculate method confidence
        if 'discoverymethod' in df.columns:
            method_stats = df.groupby('discoverymethod')['_success'].agg(['sum', 'count'])
            method_stats['success_rate'] = method_stats['sum'] / method_stats['count']
            method_stats['weighted_score'] = (
                method_stats['success_rate'] * 0.8 + 
                (method_stats['count'] / method_stats['count'].max()) * 0.2
            )
            self.calculated_reputations['method'] = method_stats['weighted_score'].to_dict()
            print(f"  Methods: {len(self.calculated_reputations['method'])} analyzed")
        
        # Calculate author/reference reputation
        if 'disp_refname' in df.columns:
            # Extract first author from reference
            df['_first_author'] = df['disp_refname'].str.extract(r'([A-Z][a-z]+)', expand=False)
            author_stats = df.groupby('_first_author')['_success'].agg(['sum', 'count'])
            author_stats = author_stats[author_stats['count'] >= 3]  # Min 3 papers
            author_stats['success_rate'] = author_stats['sum'] / author_stats['count']
            author_stats['weighted_score'] = (
                author_stats['success_rate'] * 0.6 + 
                (author_stats['count'] / author_stats['count'].max()) * 0.4
            )
            self.calculated_reputations['author'] = author_stats['weighted_score'].to_dict()
            print(f"  Authors: {len(self.calculated_reputations['author'])} analyzed (min 3 papers)")
        
        df.drop(columns=['_success', '_first_author'], errors='ignore', inplace=True)
    
    def _filter_critical_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        critical_present = [col for col in self.critical_features if col in df.columns]
        
        # Progressive filtering: drop if top 3 critical are missing
        top_critical = critical_present[:3]
        df = df.dropna(subset=top_critical, how='any')
        print(f"Critical filter: {initial} → {len(df)} records")
        return df
    
    def _add_reputation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nApplying calculated reputations...")
        
        if 'telescope' in self.calculated_reputations and 'disc_telescope' in df.columns:
            df['telescope_reputation'] = df['disc_telescope'].map(
                self.calculated_reputations['telescope']
            ).fillna(0.5)
        
        if 'instrument' in self.calculated_reputations and 'disc_instrument' in df.columns:
            df['instrument_reputation'] = df['disc_instrument'].map(
                self.calculated_reputations['instrument']
            ).fillna(0.5)
        
        if 'method' in self.calculated_reputations and 'discoverymethod' in df.columns:
            df['method_reputation'] = df['discoverymethod'].map(
                self.calculated_reputations['method']
            ).fillna(0.5)
        
        if 'author' in self.calculated_reputations and 'disp_refname' in df.columns:
            df['_first_author'] = df['disp_refname'].str.extract(r'([A-Z][a-z]+)', expand=False)
            df['author_reputation'] = df['_first_author'].map(
                self.calculated_reputations['author']
            ).fillna(0.5)
            df.drop(columns=['_first_author'], inplace=True)
        
        # Composite confidence
        rep_cols = [c for c in df.columns if c.endswith('_reputation')]
        if rep_cols:
            df['confidence_score'] = df[rep_cols].mean(axis=1)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nEncoding categorical features...")
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = df[col].fillna('unknown')
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = df[col].fillna('unknown')
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
            
            df = df.drop(columns=[col])
        
        return df
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = [col for col in self.drop_columns if col in df.columns]
        df = df.drop(columns=to_drop)
        print(f"Dropped {len(to_drop)} identifier columns")
        return df
    
    def _extract_labels(self, df: pd.DataFrame) -> np.ndarray:
        if 'disposition' not in df.columns:
            return None
        
        label_map = {
            'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0,
            'Confirmed': 2, 'Candidate': 1, 'False Positive': 0
        }
        
        y = df['disposition'].map(label_map).fillna(0).astype(int)
        counts = np.bincount(y)
        print(f"\nLabels: FP={counts[0]}, Candidate={counts[1]}, Confirmed={counts[2]}")
        return y.values
    
    def _handle_missing_values_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing with NASA priority awareness"""
        print("\nHandling missing values (NASA priority)...")
        
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            
            # Check if NASA priority feature
            is_priority = any(p in col for p in self.nasa_priority_features.keys())
            
            if missing_pct > 70:
                if is_priority:
                    print(f"  ⚠ NASA priority {col} has {missing_pct:.1f}% missing - imputing median")
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df = df.drop(columns=[col])
                continue
            
            if df[col].dtype == 'object':
                df = self._frequency_encode(df, col)
            else:
                if missing_pct > 30:
                    df[col] = df[col].fillna(df[col].median())
                elif missing_pct > 0:
                    df[col] = df[col].ffill().bfill().fillna(df[col].median())
        
        return df
    
    def _frequency_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        freq = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)
        
        count = df[col].value_counts().to_dict()
        df[f'{col}_count'] = df[col].map(count).fillna(0)
        max_count = df[f'{col}_count'].max()
        if max_count > 0:
            df[f'{col}_count'] = df[f'{col}_count'] / max_count
        
        df = df.drop(columns=[col])
        return df
    
    def _select_features_nasa(self, df: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Feature selection with NASA priority weighting"""
        print("\nFeature selection (NASA-weighted)...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Variance filter
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = selector.fit_transform(numeric_df)
        selected_cols = numeric_df.columns[selector.get_support()].tolist()
        print(f"  Variance filter: {len(numeric_df.columns)} → {len(selected_cols)}")
        
        if y is not None and len(selected_cols) > 0:
            # Calculate MI scores
            mi_scores = mutual_info_classif(X_var, y, random_state=42)
            
            # Apply NASA priority weighting
            weighted_scores = []
            for i, col in enumerate(selected_cols):
                nasa_weight = max([v for k, v in self.nasa_priority_features.items() if k in col] or [1.0])
                weighted_scores.append(mi_scores[i] * nasa_weight)
            
            self.feature_importance = dict(zip(selected_cols, weighted_scores))
            
            # Select top features
            top_n = min(50, len(selected_cols))
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            selected_cols = [f[0] for f in top_features]
            
            print(f"  NASA-weighted MI: Top {len(selected_cols)} features")
            print(f"  Top 5: {[f[0] for f in top_features[:5]]}")
        
        self.selected_features = selected_cols
        return numeric_df[selected_cols].values, selected_cols
    
    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        X_clean = X.copy()
        for i in range(X.shape[1]):
            q1, q3 = np.nanpercentile(X[:, i], [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 3*iqr, q3 + 3*iqr
            outliers = (X[:, i] < lower) | (X[:, i] > upper)
            if outliers.sum() > 0:
                X_clean[outliers, i] = np.nanmedian(X[:, i])
        return X_clean
    
    def _log_transform_skewed(self, X: np.ndarray, features: List[str]) -> np.ndarray:
        skew_cols = ['pl_orbper', 'pl_trandep', 'pl_masse', 'pl_massj']
        for i, feat in enumerate(features):
            if any(skew in feat for skew in skew_cols) and np.all(X[:, i] > 0):
                X[:, i] = np.log10(X[:, i] + 1e-10)
        return X
    
    def fit_transform(self, X: np.ndarray, save_path: str = 'models/') -> np.ndarray:
        Path(save_path).mkdir(exist_ok=True)
        
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.fitted = True
        
        joblib.dump(self.scaler, f'{save_path}/scaler.joblib')
        joblib.dump(self.pca, f'{save_path}/pca.joblib')
        joblib.dump(self.label_encoders, f'{save_path}/encoders.joblib')
        
        metadata = {
            'n_components': self.n_components,
            'explained_variance': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            'selected_features': self.selected_features,
            'feature_importance': {k: float(v) for k, v in (self.feature_importance or {}).items()},
            'calculated_reputations': {k: {str(k2): float(v2) for k2, v2 in v.items()} 
                                      for k, v in self.calculated_reputations.items()}
        }
        
        with open(f'{save_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ PCA variance: {self.pca.explained_variance_ratio_}")
        return X_pca
    
    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self.pca.transform(self.scaler.transform(X))
    
    def export_for_js(self, output_path: str = 'data/'):
        Path(output_path).mkdir(exist_ok=True)
        js_export = {
            'components': self.pca.components_.tolist(),
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'features': self.selected_features,
            'variance': self.pca.explained_variance_ratio_.tolist()
        }
        with open(f'{output_path}/pca_params.json', 'w') as f:
            json.dump(js_export, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', default='data/')
    parser.add_argument('-c', '--components', type=int, default=3)
    parser.add_argument('-t', '--test-size', type=float, default=0.2)
    parser.add_argument('--variance-threshold', type=float, default=0.01)
    
    args = parser.parse_args()
    
    processor = AdaptiveExoplanetProcessor(
        n_components=args.components,
        variance_threshold=args.variance_threshold
    )
    
    df = processor.load_data(args.input)
    if df is None:
        return
    
    X, y, features = processor.preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print(f"\n{'='*50}")
    print("APPLYING PCA")
    X_train_pca = processor.fit_transform(X_train, 'models/')
    X_test_pca = processor.transform_new_data(X_test)
    
    Path(args.output).mkdir(exist_ok=True)
    np.save(f'{args.output}/X_train_pca.npy', X_train_pca)
    np.save(f'{args.output}/X_test_pca.npy', X_test_pca)
    np.save(f'{args.output}/y_train.npy', y_train)
    np.save(f'{args.output}/y_test.npy', y_test)
    
    processor.export_for_js(args.output)
    print(f"\n✓ Saved: {X_train_pca.shape} train, {X_test_pca.shape} test")

if __name__ == "__main__":
    main()