import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore # For outlier detection
import warnings
warnings.filterwarnings('ignore')

class PatternAnalyzer:
    """
    Analyzes patterns, associations, and anomalies within SNOMED mapping dataset.
    This class now includes methods for common phrases, associations, abbreviation
    consistency, dataset distribution, statistical outliers, and clustering.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Ensure 'dx' column is string type and handle potential NaNs for TF-IDF
        self.df['dx'] = self.df['dx'].astype(str).fillna('')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1500, stop_words='english')
        # Fit TF-IDF on initialization for efficiency in multiple methods
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['dx'])
        
    def analyze_common_phrases(self, dx_column: str = 'dx', top_n: int = 10, 
                               ngram_range: Tuple[int, int] = (2, 3)) -> Dict[str, List[Tuple[str, int]]]:
        """
        Identifies the most common n-grams (phrases) within the diagnosis text column.
        Leverages TF-IDF vectorizer's capabilities for n-gram extraction.
        """
        results = {}
        for n in range(ngram_range[0], ngram_range[1] + 1):
            # Create a temporary TF-IDF vectorizer for specific n-gram range
            temp_vectorizer = TfidfVectorizer(ngram_range=(n, n), stop_words='english', max_features=500)
            temp_tfidf_matrix = temp_vectorizer.fit_transform(self.df[dx_column])
            
            # Sum the TF-IDF scores for each n-gram across all documents
            sums = temp_tfidf_matrix.sum(axis=0)
            features = temp_vectorizer.get_feature_names_out()
            
            # Create a list of (n-gram, sum_score) tuples
            ngram_scores = [(feature, sums[0, idx]) for idx, feature in enumerate(features)]
            ngram_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Convert scores to counts for a more intuitive "common phrases" output
            # This is an approximation; true counts would require a CountVectorizer
            # For simplicity, we'll just use the sorted scores as a proxy for frequency
            
            # For a more direct 'count' of phrases, we can use Counter
            all_ngrams = []
            for text in self.df[dx_column]:
                words = text.split()
                if len(words) >= n:
                    for i in range(len(words) - n + 1):
                        all_ngrams.append(" ".join(words[i:i+n]))
            
            ngram_counts = Counter(all_ngrams)
            sorted_ngram_counts = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)

            results[f"{n}-grams"] = sorted_ngram_counts[:top_n]
        return results

    def identify_snomed_dx_associations(self, min_support: float = 0.01) -> List[Dict]:
        """
        Analyzes the association between specific SNOMED CT Codes and the diagnosis texts.
        Finds common diagnosis phrases (using 2-grams) associated with SNOMED codes.
        """
        associations = []
        
        # Generate 2-grams for each diagnosis
        temp_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', max_features=500)
        dx_ngrams = temp_vectorizer.fit_transform(self.df['dx'])
        ngram_features = temp_vectorizer.get_feature_names_out()

        # Create a mapping of SNOMED code to relevant diagnosis texts/ngrams
        snomed_groups = self.df.groupby('snomed_ct_code')['dx'].apply(list).to_dict()

        total_diagnoses = len(self.df)

        for snomed_code, diagnoses_list in snomed_groups.items():
            if snomed_code == '0' or pd.isna(snomed_code): # Skip invalid/missing codes
                continue

            # Get n-grams specific to this SNOMED code's diagnoses
            snomed_dx_indices = self.df[self.df['snomed_ct_code'] == snomed_code].index
            relevant_ngrams_matrix = dx_ngrams[snomed_dx_indices]
            
            if relevant_ngrams_matrix.shape[0] == 0:
                continue

            # Sum up the presence of each ngram for this SNOMED code
            ngram_presence_counts = relevant_ngrams_matrix.sum(axis=0).tolist()[0]
            
            for i, count in enumerate(ngram_presence_counts):
                if count > 0:
                    ngram = ngram_features[i]
                    support = count / total_diagnoses # Simple support calculation
                    
                    if support >= min_support:
                        associations.append({
                            'snomed_code': snomed_code,
                            'diagnosis_phrase': ngram,
                            'count': int(count),
                            'support': support
                        })
        
        # Sort by support
        associations.sort(key=lambda x: x['support'], reverse=True)
        return associations

    def detect_abbreviation_inconsistencies(self) -> List[Dict]:
        """
        Identifies cases where the same abbreviation is used for different full diagnosis texts,
        or different abbreviations are used for the same diagnosis text.
        """
        inconsistencies = []

        # Check for ambiguous abbreviations (one abbreviation, multiple diagnoses)
        abbrev_to_dx = defaultdict(set)
        for _, row in self.df.iterrows():
            abbrev_to_dx[row['abbreviation']].add(row['dx'])
        
        for abbrev, diagnoses_set in abbrev_to_dx.items():
            if len(diagnoses_set) > 1 and abbrev not in ['', '0']: # Ignore empty or '0' abbreviations
                inconsistencies.append({
                    'type': 'ambiguous_abbreviation',
                    'abbreviation': abbrev,
                    'unique_diagnoses_count': len(diagnoses_set),
                    'diagnoses': list(diagnoses_set)
                })

        # Check for multiple abbreviations for a single diagnosis
        dx_to_abbrev = defaultdict(set)
        for _, row in self.df.iterrows():
            dx_to_abbrev[row['dx']].add(row['abbreviation'])

        for dx, abbrevs_set in dx_to_abbrev.items():
            if len(abbrevs_set) > 1 and dx not in ['', '0']: # Ignore empty or '0' diagnoses
                inconsistencies.append({
                    'type': 'multiple_abbreviations_for_diagnosis',
                    'diagnosis': dx,
                    'unique_abbreviations_count': len(abbrevs_set),
                    'abbreviations': list(abbrevs_set)
                })
        
        return inconsistencies

    def analyze_dataset_distribution_patterns(self, top_n: int = 5) -> Dict[str, List[Dict]]:
        """
        Analyzes which diagnoses are most prevalent in specific datasets.
        Returns a dictionary where keys are dataset names and values are lists of top diagnoses (diagnosis, count) for that dataset.
        """
        dataset_cols = ['cpsc', 'cpsc_extra', 'stpetersburg', 'ptb', 'ptb_xl', 'georgia']
        patterns = {}

        for col in dataset_cols:
            if col in self.df.columns:
                # Get top diagnoses for each dataset based on their specific count
                top_diagnoses_in_dataset = self.df.nlargest(top_n, col)[['dx', col]]
                patterns[col.replace('_', ' ').title()] = top_diagnoses_in_dataset.rename(columns={col: 'count'}).to_dict(orient='records')
            else:
                patterns[col.replace('_', ' ').title()] = [] # Handle missing column gracefully
        return patterns

    def find_outliers(self, total_column: str = 'total', threshold_z: float = 2.0) -> pd.DataFrame:
        """
        Detects statistical outliers in the 'total' case counts using Z-score.
        Returns a DataFrame of potential outliers.
        """
        if total_column not in self.df.columns or self.df[total_column].empty:
            return pd.DataFrame() # Return empty if column is missing or empty

        # Calculate Z-scores for the total column
        # Handle cases where std dev is 0 (all values are same)
        if self.df[total_column].std() == 0:
            return pd.DataFrame() # No outliers if all values are the same

        self.df['z_score'] = np.abs(zscore(self.df[total_column]))
        
        # Identify outliers based on threshold
        outliers_df = self.df[self.df['z_score'] > threshold_z].sort_values(by='z_score', ascending=False)
        return outliers_df[['dx', total_column, 'z_score']]

    def perform_clustering(self, n_clusters: int = 5) -> pd.DataFrame:
        """
        Performs KMeans clustering on diagnosis texts using TF-IDF features.
        Returns the DataFrame with an added 'cluster' column.
        """
        if self.tfidf_matrix.shape[0] < n_clusters:
            # Cannot cluster if fewer samples than clusters
            warnings.warn(f"Cannot perform clustering with {self.tfidf_matrix.shape[0]} samples and {n_clusters} clusters. Returning empty clusters.")
            self.df['cluster'] = -1 # Assign a default non-cluster value
            return self.df

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.tfidf_matrix)
        self.df['cluster'] = clusters
        return self.df

    def get_pca_visualization(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates data for PCA visualization of diagnosis clusters.
        Assumes clustered_df already has a 'cluster' column and TF-IDF matrix is available.
        """
        if 'cluster' not in clustered_df.columns:
            warnings.warn("Clustered DataFrame must contain a 'cluster' column for PCA visualization.")
            return pd.DataFrame()

        # Ensure TF-IDF matrix is available and matches DataFrame size
        if self.tfidf_matrix.shape[0] != clustered_df.shape[0]:
            warnings.warn("TF-IDF matrix size mismatch with clustered DataFrame. Re-fitting TF-IDF.")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(clustered_df['dx'])

        if self.tfidf_matrix.shape[1] < 2:
            warnings.warn("Not enough features for PCA (need at least 2). Returning empty PCA data.")
            return pd.DataFrame()

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.tfidf_matrix.toarray()) # Convert sparse to dense for PCA

        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['dx'] = clustered_df['dx'].reset_index(drop=True)
        pca_df['total'] = clustered_df['total'].reset_index(drop=True)
        pca_df['cluster'] = clustered_df['cluster'].reset_index(drop=True).astype(str) # Convert to string for coloring in Plotly

        return pca_df

    def get_cluster_stats(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Provides descriptive statistics for each cluster.
        """
        if 'cluster' not in clustered_df.columns:
            warnings.warn("Clustered DataFrame must contain a 'cluster' column for cluster stats.")
            return pd.DataFrame()

        cluster_stats = clustered_df.groupby('cluster').agg(
            count=('dx', 'size'),
            avg_total_cases=('total', 'mean'),
            unique_snomed_codes=('snomed_ct_code', 'nunique')
        ).reset_index()

        # Get top 3 common diagnoses for each cluster
        top_diagnoses_per_cluster = []
        for cluster_id in clustered_df['cluster'].unique():
            cluster_dx = clustered_df[clustered_df['cluster'] == cluster_id]['dx']
            common_dx = Counter(cluster_dx).most_common(3)
            top_diagnoses_per_cluster.append({
                'cluster': cluster_id,
                'top_diagnoses': [f"{dx} ({count})" for dx, count in common_dx]
            })
        
        top_dx_df = pd.DataFrame(top_diagnoses_per_cluster)
        cluster_stats = cluster_stats.merge(top_dx_df, on='cluster', how='left')

        return cluster_stats

    def get_audit_summary(self) -> Dict:
        """
        Provides a high-level summary of the patterns identified.
        This method will return summaries from other analysis methods.
        """
        summary = {}

        # Common Phrases
        summary['common_phrases'] = self.analyze_common_phrases(top_n=5)

        # SNOMED-Diagnosis Associations
        summary['snomed_dx_associations'] = self.identify_snomed_dx_associations(min_support=0.005)[:5] # Top 5 associations

        # Abbreviation Inconsistencies
        summary['abbreviation_inconsistencies'] = self.detect_abbreviation_inconsistencies()

        # Dataset Distribution Patterns
        summary['dataset_distribution_patterns'] = self.analyze_dataset_distribution_patterns(top_n=3)

        # Outlier count
        outliers_df = self.find_outliers()
        summary['outlier_count'] = len(outliers_df)

        # Basic cluster info (if clustering was run)
        if 'cluster' in self.df.columns:
            summary['num_clusters'] = self.df['cluster'].nunique()
            summary['cluster_sizes'] = self.df['cluster'].value_counts().to_dict()
        else:
            summary['num_clusters'] = 0
            summary['cluster_sizes'] = {}

        return summary
