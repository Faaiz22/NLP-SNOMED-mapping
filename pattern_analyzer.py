"""
Pattern Analyzer Module for SNOMED CT Mapping Analysis

This module provides comprehensive pattern analysis capabilities for SNOMED CT 
mapping datasets, focusing on identifying recurring patterns, associations, 
and potential inconsistencies within diagnostic text and SNOMED code relationships.

Author: Healthcare Data Analytics Team
Purpose: Bio AI and in-silico medicine pattern detection
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
from itertools import combinations


class PatternAnalyzer:
    """
    A comprehensive pattern analyzer for SNOMED CT mapping datasets.
    
    This class identifies and analyzes recurring patterns, associations, and 
    potential inconsistencies within SNOMED mapping data, specifically focusing 
    on diagnostic text relationships with SNOMED codes and metadata.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the PatternAnalyzer with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The SNOMED mapping dataset containing columns like
                              'Dx', 'SNOMED CT Code', 'Abbreviation', and dataset counts
        """
        self.df = df.copy()  # Prevent modification of original DataFrame
        self.dataset_columns = [col for col in self.df.columns 
                               if col not in ['Dx', 'SNOMED CT Code', 'Abbreviation']]
        
        # Cache for computed results to improve performance
        self._phrase_cache = {}
        self._association_cache = {}
    
    def _clean_text(self, text: str) -> str:
        """
        Clean diagnostic text for analysis.
        
        Args:
            text (str): Raw diagnostic text
            
        Returns:
            str: Cleaned text suitable for pattern analysis
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract n-grams from cleaned text.
        
        Args:
            text (str): Cleaned input text
            n (int): N-gram size
            
        Returns:
            List[str]: List of n-grams
        """
        words = text.split()
        if len(words) < n:
            return []
        
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def analyze_common_phrases(self, dx_column: str = 'Dx', top_n: int = 10, 
                             ngram_range: Tuple[int, int] = (2, 3)) -> Dict[str, List[Tuple[str, int]]]:
        """
        Identify the most common n-grams (phrases) within the diagnosis text column.
        
        Args:
            dx_column (str): Name of the diagnosis column (default: 'Dx')
            top_n (int): Number of top phrases to return for each n-gram size
            ngram_range (Tuple[int, int]): Range of n-gram sizes to analyze
            
        Returns:
            Dict[str, List[Tuple[str, int]]]: Dictionary with n-gram sizes as keys
                                            and lists of (phrase, count) tuples as values
        """
        cache_key = f"{dx_column}_{top_n}_{ngram_range}"
        if cache_key in self._phrase_cache:
            return self._phrase_cache[cache_key]
        
        if dx_column not in self.df.columns:
            return {}
        
        results = {}
        
        # Clean all diagnosis texts
        cleaned_texts = self.df[dx_column].apply(self._clean_text)
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngram_counter = Counter()
            
            for text in cleaned_texts:
                if text:  # Skip empty texts
                    ngrams = self._extract_ngrams(text, n)
                    ngram_counter.update(ngrams)
            
            # Get top n-grams
            top_ngrams = ngram_counter.most_common(top_n)
            results[f"{n}-grams"] = top_ngrams
        
        self._phrase_cache[cache_key] = results
        return results
    
    def identify_snomed_dx_associations(self, min_support: float = 0.01) -> List[Dict]:
        """
        Analyze associations between SNOMED CT Codes and diagnosis texts.
        
        This method identifies common diagnosis phrases associated with particular 
        SNOMED codes and finds frequent co-occurrence patterns.
        
        Args:
            min_support (float): Minimum support threshold for associations
            
        Returns:
            List[Dict]: List of association dictionaries containing SNOMED codes,
                       diagnosis phrases, counts, and support values
        """
        cache_key = f"associations_{min_support}"
        if cache_key in self._association_cache:
            return self._association_cache[cache_key]
        
        required_columns = ['Dx', 'SNOMED CT Code']
        if not all(col in self.df.columns for col in required_columns):
            return []
        
        associations = []
        total_records = len(self.df)
        min_count = max(1, int(min_support * total_records))
        
        # Group by SNOMED code and analyze associated diagnosis terms
        snomed_groups = self.df.groupby('SNOMED CT Code')
        
        for snomed_code, group in snomed_groups:
            if pd.isna(snomed_code):
                continue
                
            # Extract common words/phrases from diagnoses for this SNOMED code
            diagnoses = group['Dx'].apply(self._clean_text)
            all_words = []
            
            for dx in diagnoses:
                if dx:
                    words = dx.split()
                    all_words.extend(words)
                    # Also add 2-grams for phrase analysis
                    all_words.extend(self._extract_ngrams(dx, 2))
            
            word_counts = Counter(all_words)
            
            # Find significant associations
            for phrase, count in word_counts.items():
                if count >= min_count and len(phrase.strip()) > 2:
                    support = count / total_records
                    confidence = count / len(group)
                    
                    associations.append({
                        'snomed_code': snomed_code,
                        'diagnosis_phrase': phrase,
                        'count': count,
                        'support': round(support, 4),
                        'confidence': round(confidence, 4),
                        'snomed_frequency': len(group)
                    })
        
        # Sort by support and confidence
        associations.sort(key=lambda x: (x['support'], x['confidence']), reverse=True)
        
        self._association_cache[cache_key] = associations
        return associations
    
    def detect_abbreviation_inconsistencies(self) -> List[Dict]:
        """
        Identify inconsistencies in abbreviation usage.
        
        Detects cases where:
        1. Same abbreviation maps to different diagnoses
        2. Same diagnosis has different abbreviations
        
        Returns:
            List[Dict]: List of inconsistency dictionaries describing the type
                       and details of each inconsistency found
        """
        required_columns = ['Dx', 'Abbreviation']
        if not all(col in self.df.columns for col in required_columns):
            return []
        
        inconsistencies = []
        
        # Remove rows with missing values
        clean_df = self.df.dropna(subset=required_columns)
        
        # Check for ambiguous abbreviations (same abbrev, different diagnoses)
        abbrev_to_dx = defaultdict(set)
        dx_to_abbrev = defaultdict(set)
        
        for _, row in clean_df.iterrows():
            abbrev = str(row['Abbreviation']).strip()
            dx = str(row['Dx']).strip().lower()
            
            if abbrev and dx:
                abbrev_to_dx[abbrev].add(dx)
                dx_to_abbrev[dx].add(abbrev)
        
        # Find abbreviations with multiple diagnoses
        for abbrev, diagnoses in abbrev_to_dx.items():
            if len(diagnoses) > 1:
                inconsistencies.append({
                    'type': 'ambiguous_abbreviation',
                    'abbreviation': abbrev,
                    'diagnoses': list(diagnoses),
                    'count': len(diagnoses),
                    'severity': 'high' if len(diagnoses) > 2 else 'medium'
                })
        
        # Find diagnoses with multiple abbreviations
        for dx, abbreviations in dx_to_abbrev.items():
            if len(abbreviations) > 1:
                inconsistencies.append({
                    'type': 'inconsistent_abbreviation',
                    'diagnosis': dx,
                    'abbreviations': list(abbreviations),
                    'count': len(abbreviations),
                    'severity': 'medium' if len(abbreviations) > 2 else 'low'
                })
        
        # Sort by severity and count
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        inconsistencies.sort(key=lambda x: (severity_order.get(x['severity'], 0), x['count']), 
                           reverse=True)
        
        return inconsistencies
    
    def analyze_dataset_distribution_patterns(self, top_n: int = 5) -> Dict[str, List[Dict]]:
        """
        Analyze diagnosis prevalence patterns across different datasets.
        
        Args:
            top_n (int): Number of top diagnoses to return for each dataset
            
        Returns:
            Dict[str, List[Dict]]: Dictionary with dataset names as keys and 
                                 lists of top diagnoses with counts as values
        """
        if 'Dx' not in self.df.columns:
            return {}
        
        results = {}
        
        # Analyze each dataset column
        for dataset_col in self.dataset_columns:
            if dataset_col == 'Total':  # Skip total column
                continue
                
            if dataset_col not in self.df.columns:
                continue
            
            # Filter rows where this dataset has non-zero counts
            dataset_df = self.df[self.df[dataset_col] > 0].copy()
            
            if dataset_df.empty:
                results[dataset_col] = []
                continue
            
            # Calculate diagnosis frequencies for this dataset
            dx_counts = []
            for _, row in dataset_df.iterrows():
                dx = row['Dx']
                count = row[dataset_col]
                if pd.notna(dx) and count > 0:
                    dx_counts.append({
                        'diagnosis': dx,
                        'count': int(count),
                        'snomed_code': row.get('SNOMED CT Code', ''),
                        'abbreviation': row.get('Abbreviation', '')
                    })
            
            # Sort by count and take top N
            dx_counts.sort(key=lambda x: x['count'], reverse=True)
            results[dataset_col] = dx_counts[:top_n]
        
        return results
    
    def _calculate_dataset_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistics about the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing various dataset statistics
        """
        stats = {
            'total_records': len(self.df),
            'unique_diagnoses': self.df['Dx'].nunique() if 'Dx' in self.df.columns else 0,
            'unique_snomed_codes': self.df['SNOMED CT Code'].nunique() if 'SNOMED CT Code' in self.df.columns else 0,
            'unique_abbreviations': self.df['Abbreviation'].nunique() if 'Abbreviation' in self.df.columns else 0,
            'datasets_analyzed': len(self.dataset_columns)
        }
        
        # Calculate total cases across all datasets
        if self.dataset_columns:
            numeric_cols = []
            for col in self.dataset_columns:
                if col in self.df.columns:
                    # Convert to numeric, replacing any non-numeric with 0
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                    numeric_cols.append(col)
            
            if numeric_cols:
                stats['total_cases'] = int(self.df[numeric_cols].sum().sum())
                stats['average_cases_per_diagnosis'] = round(stats['total_cases'] / max(1, stats['total_records']), 2)
        
        return stats
    
    def get_audit_summary(self) -> Dict:
        """
        Provide a high-level summary of patterns identified across all analyses.
        
        Returns:
            Dict: Comprehensive summary dictionary containing counts and insights
                 from all pattern analysis methods
        """
        summary = {
            'dataset_statistics': self._calculate_dataset_statistics(),
            'analysis_summary': {}
        }
        
        # Common phrases analysis
        try:
            phrases = self.analyze_common_phrases(top_n=5)
            phrase_counts = {ngram_type: len(phrase_list) for ngram_type, phrase_list in phrases.items()}
            summary['analysis_summary']['common_phrases'] = {
                'ngram_types_analyzed': len(phrases),
                'phrase_counts_by_type': phrase_counts,
                'total_unique_phrases': sum(len(phrase_list) for phrase_list in phrases.values())
            }
        except Exception as e:
            summary['analysis_summary']['common_phrases'] = {'error': str(e)}
        
        # SNOMED-DX associations
        try:
            associations = self.identify_snomed_dx_associations()
            summary['analysis_summary']['snomed_associations'] = {
                'total_associations_found': len(associations),
                'high_confidence_associations': len([a for a in associations if a.get('confidence', 0) > 0.7]),
                'high_support_associations': len([a for a in associations if a.get('support', 0) > 0.05])
            }
        except Exception as e:
            summary['analysis_summary']['snomed_associations'] = {'error': str(e)}
        
        # Abbreviation inconsistencies
        try:
            inconsistencies = self.detect_abbreviation_inconsistencies()
            summary['analysis_summary']['abbreviation_inconsistencies'] = {
                'total_inconsistencies': len(inconsistencies),
                'high_severity': len([i for i in inconsistencies if i.get('severity') == 'high']),
                'medium_severity': len([i for i in inconsistencies if i.get('severity') == 'medium']),
                'low_severity': len([i for i in inconsistencies if i.get('severity') == 'low'])
            }
        except Exception as e:
            summary['analysis_summary']['abbreviation_inconsistencies'] = {'error': str(e)}
        
        # Dataset distribution patterns
        try:
            distributions = self.analyze_dataset_distribution_patterns()
            summary['analysis_summary']['dataset_distributions'] = {
                'datasets_with_data': len([d for d in distributions if distributions[d]]),
                'total_datasets_analyzed': len(distributions),
                'average_diagnoses_per_dataset': round(
                    sum(len(distributions[d]) for d in distributions) / max(1, len(distributions)), 2
                )
            }
        except Exception as e:
            summary['analysis_summary']['dataset_distributions'] = {'error': str(e)}
        
        # Add timestamp and analysis metadata
        summary['analysis_metadata'] = {
            'analyzer_version': '1.0.0',
            'methods_available': [
                'analyze_common_phrases',
                'identify_snomed_dx_associations', 
                'detect_abbreviation_inconsistencies',
                'analyze_dataset_distribution_patterns'
            ]
        }
        
        return summary