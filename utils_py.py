import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Union
import streamlit as st
from fuzzywuzzy import fuzz, process
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SNOMEDUtils:
    """Utility functions for SNOMED data processing and visualization"""
    
    @staticmethod
    def clean_diagnosis_text(text: str) -> str:
        """Clean and standardize diagnosis text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\(\)]', '', text)
        
        return text.strip()
    
    @staticmethod
    def validate_snomed_code(code: Union[str, int]) -> bool:
        """Validate SNOMED CT code format"""
        if pd.isna(code):
            return False
        
        code_str = str(code).strip()
        
        # SNOMED codes should be numeric and typically 6-18 digits
        if not code_str.isdigit():
            return False
        
        if len(code_str) < 6 or len(code_str) > 18:
            return False
        
        return True
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str, method: str = 'ratio') -> float:
        """Calculate text similarity using various methods"""
        if pd.isna(text1) or pd.isna(text2):
            return 0.0
        
        text1 = SNOMEDUtils.clean_diagnosis_text(text1)
        text2 = SNOMEDUtils.clean_diagnosis_text(text2)
        
        if method == 'ratio':
            return fuzz.ratio(text1, text2) / 100.0
        elif method == 'partial_ratio':
            return fuzz.partial_ratio(text1, text2) / 100.0
        elif method == 'token_sort':
            return fuzz.token_sort_ratio(text1, text2) / 100.0
        elif method == 'token_set':
            return fuzz.token_set_ratio(text1, text2) / 100.0
        else:
            return fuzz.ratio(text1, text2) / 100.0
    
    @staticmethod
    def extract_medical_keywords(text: str) -> List[str]:
        """Extract medical keywords from diagnosis text"""
        if pd.isna(text):
            return []
        
        # Common medical keywords and their variations
        medical_patterns = [
            r'\b(atrial|ventricular|cardiac|heart|myocardial)\b',
            r'\b(ischemia|infarction|stenosis|hypertrophy)\b',
            r'\b(tachycardia|bradycardia|arrhythmia|fibrillation)\b',
            r'\b(block|syndrome|disease|disorder|abnormal)\b',
            r'\b(left|right|anterior|posterior|inferior|lateral)\b'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    @staticmethod
    def create_diagnosis_hierarchy(df: pd.DataFrame) -> Dict:
        """Create a hierarchical structure of diagnoses based on keywords"""
        hierarchy = {
            'cardiac': [],
            'myocardial': [],
            'atrial': [],
            'ventricular': [],
            'block': [],
            'hypertrophy': [],
            'other': []
        }
        
        for _, row in df.iterrows():
            dx = row['Dx'].lower()
            classified = False
            
            for category in hierarchy.keys():
                if category != 'other' and category in dx:
                    hierarchy[category].append({
                        'diagnosis': row['Dx'],
                        'snomed_code': row['SNOMED CT Code'],
                        'abbreviation': row['Abbreviation'],
                        'total': row['Total']
                    })
                    classified = True
                    break
            
            if not classified:
                hierarchy['other'].append({
                    'diagnosis': row['Dx'],
                    'snomed_code': row['SNOMED CT Code'],
                    'abbreviation': row['Abbreviation'],
                    'total': row['Total']
                })
        
        return hierarchy
    
    @staticmethod
    def calculate_dataset_statistics(df: pd.DataFrame) -> Dict:
        """Calculate comprehensive dataset statistics"""
        dataset_cols = [col for col in df.columns 
                       if col not in ['Dx', 'SNOMED CT Code', 'Abbreviation', 'Total']]
        
        stats = {
            'total_diagnoses': len(df),
            'total_cases': df['Total'].sum(),
            'unique_snomed_codes': df['SNOMED CT Code'].nunique(),
            'avg_cases_per_diagnosis': df['Total'].mean(),
            'median_cases_per_diagnosis': df['Total'].median(),
            'std_cases_per_diagnosis': df['Total'].std(),
            'dataset_distribution': {}
        }
        
        # Dataset-specific statistics
        for col in dataset_cols:
            stats['dataset_distribution'][col] = {
                'total_cases': df[col].sum(),
                'diagnoses_with_cases': (df[col] > 0).sum(),
                'percentage_of_total': (df[col].sum() / df['Total'].sum()) * 100,
                'avg_cases_per_diagnosis': df[col].mean(),
                'max_cases': df[col].max(),
                'top_diagnosis': df.loc[df[col].idxmax(), 'Dx'] if df[col].max() > 0 else None
            }
        
        return stats
    
    @staticmethod
    def find_similar_diagnoses(target_diagnosis: str, df: pd.DataFrame, 
                             top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Find similar diagnoses using fuzzy matching"""
        similarities = []
        
        for _, row in df.iterrows():
            if row['Dx'] != target_diagnosis:
                similarity = SNOMEDUtils.calculate_text_similarity(
                    target_diagnosis, row['Dx'], method='token_sort'
                )
                
                if similarity >= threshold:
                    similarities.append({
                        'diagnosis': row['Dx'],
                        'similarity': similarity,
                        'snomed_code': row['SNOMED CT Code'],
                        'abbreviation': row['Abbreviation'],
                        'total_cases': row['Total']
                    })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def create_sunburst_chart(df: pd.DataFrame, title: str = "SNOMED Diagnosis Distribution") -> go.Figure:
        """Create a sunburst chart for hierarchical diagnosis visualization"""
        hierarchy = SNOMEDUtils.create_diagnosis_hierarchy(df)
        
        # Prepare data for sunburst
        ids = []
        labels = []
        parents = []
        values = []
        