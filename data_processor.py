import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

class DataProcessor:
    """
    Data processing utilities for SNOMED mapping dataset
    """
    
    def __init__(self):
        self.original_data = None
        self.processed_data = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the SNOMED mapping dataset
        
        Args:
            df: Raw dataframe from CSV
            
        Returns:
            Cleaned dataframe
        """
        self.original_data = df.copy()
        
        # Standardize column names
        df_clean = df.copy()
        df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
        
        # Handle missing values
        df_clean = df_clean.fillna(0)
        
        # Clean diagnosis text
        df_clean['dx'] = df_clean['dx'].str.strip()
        df_clean['dx'] = df_clean['dx'].str.lower()
        df_clean['dx'] = df_clean['dx'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Clean abbreviations
        df_clean['abbreviation'] = df_clean['abbreviation'].str.strip()
        
        # Ensure SNOMED codes are strings
        df_clean['snomed_ct_code'] = df_clean['snomed_ct_code'].astype(str)
        
        # Convert numeric columns
        numeric_cols = ['cpsc', 'cpsc_extra', 'st_petersburg', 'ptb', 'ptb_xl', 'georgia', 'total']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
        # Remove rows with empty diagnosis or SNOMED code
        df_clean = df_clean[(df_clean['dx'] != '') & (df_clean['snomed_ct_code'] != '')]
        
        self.processed_data = df_clean
        return df_clean
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features for machine learning
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dataframe with additional features
        """
        df_features = df.copy()
        
        # Text features
        df_features['dx_length'] = df_features['dx'].str.len()
        df_features['dx_word_count'] = df_features['dx'].str.split().str.len()
        df_features['has_numbers'] = df_features['dx'].str.contains(r'\d', regex=True).astype(int)
        
        # SNOMED code features
        df_features['snomed_length'] = df_features['snomed_ct_code'].str.len()
        df_features['snomed_numeric'] = pd.to_numeric(df_features['snomed_ct_code'], errors='coerce').notna().astype(int)
        
        # Dataset distribution features
        dataset_cols = ['cpsc', 'cpsc_extra', 'st_petersburg', 'ptb', 'ptb_xl', 'georgia']
        df_features['dataset_count'] = (df_features[dataset_cols] > 0).sum(axis=1)
        df_features['max_dataset_value'] = df_features[dataset_cols].max(axis=1)
        df_features['dataset_variance'] = df_features[dataset_cols].var(axis=1)
        
        # Frequency features
        df_features['log_total'] = np.log1p(df_features['total'])
        df_features['is_rare'] = (df_features['total'] <= 10).astype(int)
        df_features['is_common'] = (df_features['total'] >= 100).astype(int)
        
        return df_features
    
    def create_text_corpus(self, df: pd.DataFrame) -> List[str]:
        """
        Create text corpus for semantic analysis
        
        Args:
            df: Processed dataframe
            
        Returns:
            List of diagnosis texts
        """
        corpus = []
        for _, row in df.iterrows():
            # Combine diagnosis and abbreviation for richer context
            text = f"{row['dx']} {row['abbreviation']}"
            corpus.append(text)
        return corpus
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive dataset statistics
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Basic statistics
        stats['total_diagnoses'] = len(df)
        stats['unique_snomed_codes'] = df['snomed_ct_code'].nunique()
        stats['total_cases'] = df['total'].sum()
        stats['mean_cases'] = df['total'].mean()
        stats['median_cases'] = df['total'].median()
        
        # Dataset distribution
        dataset_cols = ['cpsc', 'cpsc_extra', 'st_petersburg', 'ptb', 'ptb_xl', 'georgia']
        stats['dataset_totals'] = {}
        for col in dataset_cols:
            stats['dataset_totals'][col] = df[col].sum()
        
        # Frequency distribution
        stats['frequency_distribution'] = {
            'rare_cases_1_10': len(df[(df['total'] >= 1) & (df['total'] <= 10)]),
            'medium_cases_11_100': len(df[(df['total'] >= 11) & (df['total'] <= 100)]),
            'common_cases_100plus': len(df[df['total'] > 100]),
            'zero_cases': len(df[df['total'] == 0])
        }
        
        # Text statistics
        stats['text_stats'] = {
            'avg_diagnosis_length': df['dx'].str.len().mean(),
            'avg_word_count': df['dx'].str.split().str.len().mean(),
            'diagnoses_with_numbers': df['dx'].str.contains(r'\d', regex=True).sum()
        }
        
        return stats
    
    def normalize_diagnosis_text(self, text: str) -> str:
        """
        Normalize diagnosis text for better matching
        
        Args:
            text: Raw diagnosis text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^\w\s\-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Common medical abbreviations standardization
        abbreviations = {
            'myocardial infarction': 'mi',
            'atrial fibrillation': 'af',
            'ventricular tachycardia': 'vt',
            'atrioventricular': 'av',
            'electrocardiogram': 'ecg',
            'electrocardiographic': 'ecg'
        }
        
        for full_form, abbrev in abbreviations.items():
            text = text.replace(full_form, abbrev)
        
        return text.strip()
    
    def identify_medical_terms(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify common medical terms and patterns
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dictionary categorizing medical terms
        """
        medical_terms = {
            'cardiac_conditions': [],
            'rhythm_disorders': [],
            'conduction_blocks': [],
            'ischemic_conditions': [],
            'anatomical_terms': []
        }
        
        # Define patterns for different categories
        patterns = {
            'cardiac_conditions': [
                r'.*heart.*', r'.*cardiac.*', r'.*myocardial.*', r'.*coronary.*'
            ],
            'rhythm_disorders': [
                r'.*rhythm.*', r'.*tachycardia.*', r'.*fibrillation.*', r'.*flutter.*'
            ],
            'conduction_blocks': [
                r'.*block.*', r'.*bundle.*', r'.*fascicular.*'
            ],
            'ischemic_conditions': [
                r'.*ischemia.*', r'.*infarction.*', r'.*ischemic.*'
            ],
            'anatomical_terms': [
                r'.*atrial.*', r'.*ventricular.*', r'.*left.*', r'.*right.*', r'.*anterior.*', r'.*inferior.*'
            ]
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = df[df['dx'].str.contains(pattern, regex=True, case=False)]['dx'].unique()
                medical_terms[category].extend(matches)
        
        # Remove duplicates
        for category in medical_terms:
            medical_terms[category] = list(set(medical_terms[category]))
        
        return medical_terms
    
    def create_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for machine learning
        
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
        
        # Dataset distribution features
        dataset_features = ['cpsc', 'cpsc_extra', 'st_petersburg', 'ptb', 'ptb_xl', 'georgia']
        
        all_features = numerical_features + dataset_features
        
        # Select available features
        available_features = [f for f in all_features if f in df.columns]
        
        feature_matrix = df[available_features].values
        
        return feature_matrix, available_features
