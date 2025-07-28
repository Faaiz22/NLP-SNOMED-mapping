import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class DataProcessor:
    """
    Data processing utilities for SNOMED mapping dataset
    """
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the SNOMED mapping dataset.
        This method now also engineers features and ensures all necessary columns are present.
        
        Args:
            df: Raw dataframe from CSV
            
        Returns:
            Cleaned and feature-engineered dataframe
        """
        self.original_data = df.copy()
        
        # Standardize column names
        df_clean = df.copy()
        df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
        
        # Handle missing values - fill with 0 for numerical columns, empty string for text
        for col in ['dx', 'abbreviation', 'snomed_ct_code']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('')
        
        # Fill numerical dataset columns and 'total' with 0 if NaN
        dataset_cols = ['cpsc', 'cpsc_extra', 'stpetersburg', 'ptb', 'ptb_xl', 'georgia', 'total']
        for col in dataset_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            else:
                # If a column is missing, add it with zeros to prevent KeyError later
                df_clean[col] = 0

        # Clean diagnosis text
        df_clean['dx'] = df_clean['dx'].str.strip()
        df_clean['dx'] = df_clean['dx'].str.lower()
        df_clean['dx'] = df_clean['dx'].str.replace(r'[^\\w\\s]', '', regex=True) # Keep alphanumeric and spaces
        
        # Clean abbreviations
        df_clean['abbreviation'] = df_clean['abbreviation'].str.strip()
        
        # Ensure SNOMED codes are strings
        df_clean['snomed_ct_code'] = df_clean['snomed_ct_code'].astype(str)
        
        # Now, engineer additional features directly into this DataFrame
        df_clean = self._engineer_features(df_clean)

        self.processed_data = df_clean
        return df_clean

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers additional features for the DataFrame.
        These features are expected by the ML models.
        """
        # Text features
        df['dx_length'] = df['dx'].apply(len)
        df['dx_word_count'] = df['dx'].apply(lambda x: len(x.split()))
        df['has_numbers'] = df['dx'].apply(lambda x: bool(re.search(r'\d', x))).astype(int)

        # SNOMED code features
        df['snomed_length'] = df['snomed_ct_code'].apply(len)
        df['snomed_numeric'] = df['snomed_ct_code'].apply(lambda x: x.isdigit()).astype(int)

        # Dataset distribution features
        dataset_cols = ['cpsc', 'cpsc_extra', 'stpetersburg', 'ptb', 'ptb_xl', 'georgia']
        
        # Ensure these columns exist before attempting operations
        for col in dataset_cols:
            if col not in df.columns:
                df[col] = 0 # Add missing dataset columns with default 0

        df['dataset_count'] = df[dataset_cols].apply(lambda row: (row > 0).sum(), axis=1)
        df['max_dataset_value'] = df[dataset_cols].max(axis=1)
        df['dataset_variance'] = df[dataset_cols].var(axis=1).fillna(0) # Fill NaN for single-value variance

        # Log transform of total cases for better distribution in models
        df['log_total'] = np.log1p(df['total']) # log1p handles total=0 gracefully

        # Simple rarity/commonality indicators based on total cases
        total_median = df['total'].median()
        total_q1 = df['total'].quantile(0.25)
        total_q3 = df['total'].quantile(0.75)

        df['is_rare'] = (df['total'] < total_q1).astype(int)
        df['is_common'] = (df['total'] > total_q3).astype(int)

        return df

    def get_medical_term_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify and categorize common medical terms or phrases based on patterns.
        This is an example and can be expanded.
        """
        medical_terms = defaultdict(list)
        
        # Example patterns for categorization
        patterns = {
            'cardiac': [r'cardiac', r'heart', r'atrial', r'ventricular', r'myocardial', r'ecg', r'arrhythmia'],
            'pulmonary': [r'lung', r'pulmonary', r'respiratory', r'pneumonia', r'asthma'],
            'neurological': [r'brain', r'neuro', r'stroke', r'epilepsy', r'headache'],
            'vascular': [r'vascular', r'artery', r'vein', r'thrombosis', r'embolism']
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = df[df['dx'].str.contains(pattern, regex=True, case=False)]['dx'].unique()
                medical_terms[category].extend(matches)
        
        # Remove duplicates
        for category in medical_terms:
            medical_terms[category] = list(set(medical_terms[category]))
        
        return medical_terms
    
    # The create_feature_matrix is now primarily for generating the numpy array for ML models,
    # assuming the DataFrame already has the necessary columns from _engineer_features.
    def create_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for machine learning.
        Assumes df already contains engineered features from _engineer_features.
        
        Args:
            df: Processed dataframe with features
            
        Returns:
            Feature matrix and feature names
        """
        # Numerical features
        numerical_features = [
            'dx_length', 'dx_word_count', 'has_numbers',
            'snomed_length', 'snomed_numeric',
            'dataset_count', 'max_dataset_value', 'dataset_variance',
            'log_total', 'is_rare', 'is_common'
        ]
        
        # Dataset distribution features (ensure these are present and correctly named)
        dataset_features = ['cpsc', 'cpsc_extra', 'stpetersburg', 'ptb', 'ptb_xl', 'georgia']
        
        all_features = numerical_features + dataset_features
        
        # Select available features - this is a safeguard, all should be present now
        available_features = [f for f in all_features if f in df.columns]
        
        feature_matrix = df[available_features].values
        
        return feature_matrix, available_features


