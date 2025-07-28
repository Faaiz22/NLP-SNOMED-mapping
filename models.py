import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom modules - ONLY import modules defined in *other* files
from data_processor import DataProcessor # This is correct, as DataProcessor is in data_processor.py
# REMOVED: from models import MappingClassifier, ConfidenceEstimator, SemanticRetrieval
# This line caused the circular import because models.py defines these classes, it doesn't import them from itself.

class MappingClassifier:
    """
    Multi-class classifier for SNOMED mapping tasks
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df # This df should now contain all engineered features
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # Initialize TF-IDF vectorizer here, fit it in prepare_training_data
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_names = []
    
    def prepare_training_data(self):
        """
        Prepare training data with simulated labels for demonstration.
        Assumes self.df already contains all engineered features from DataProcessor.
        """
        # Text features using TF-IDF
        text_corpus = self.df['dx'].fillna('').tolist()
        # Fit and transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.fit_transform(text_corpus).to_array()
        
        # Numerical features - these columns are now guaranteed to exist in self.df
        numerical_feature_cols = [
            'dx_length', 'dx_word_count', 'has_numbers',
            'snomed_length', 'snomed_numeric',
            'dataset_count', 'max_dataset_value', 'dataset_variance',
            'log_total', 'is_rare', 'is_common'
        ]
        
        # Ensure all expected numerical feature columns are in the DataFrame
        # This check is a safeguard, as DataProcessor should now ensure their presence
        missing_numerical_cols = [col for col in numerical_feature_cols if col not in self.df.columns]
        if missing_numerical_cols:
            raise KeyError(f"Missing engineered numerical features in DataFrame: {missing_numerical_cols}. "
                           "Ensure DataProcessor correctly adds these columns.")

        numerical_features_df = self.df[numerical_feature_cols].copy()

        # Handle potential NaN values after feature extraction, fill with 0 or mean/median
        numerical_features_df = numerical_features_df.fillna(0) # Filling NaN for simplicity

        # Combine TF-IDF features with numerical features
        X_numerical = numerical_features_df.values
        
        # Ensure feature dimensions match
        if tfidf_features.shape[0] != X_numerical.shape[0]:
            raise ValueError("Mismatch in number of samples between text and numerical features.")

        # Concatenate features
        X = np.hstack((tfidf_features, X_numerical))
        
        # Generate simulated labels for demonstration
        y_simulated = []
        for index, row in self.df.iterrows():
            # These column names are now guaranteed to be correct and present by DataProcessor
            total_dataset_cases = row['cpsc'] + row['cpsc_extra'] + row['stpetersburg'] + \
                                  row['ptb'] + row['ptb_xl'] + row['georgia']
            
            if pd.notna(row['snomed_ct_code']) and row['snomed_ct_code'] != '0' and total_dataset_cases > 50:
                y_simulated.append('mapped')
            elif pd.notna(row['snomed_ct_code']) and row['snomed_ct_code'] != '0' and total_dataset_cases <= 50:
                y_simulated.append('needs_review')
            else:
                y_simulated.append('unmapped')
        
        y_simulated = self.label_encoder.fit_transform(y_simulated)
        
        # Store feature names for importance analysis
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist() + numerical_features_df.columns.tolist()
        
        return X, y_simulated
    
    def train_model(self, X, y):
        """
        Train a classification model (RandomForestClassifier)
        """
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        
        return accuracy, report
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            return pd.DataFrame()
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            return importance_df.sort_values(by='importance', ascending=False)
        return pd.DataFrame()

    def train_code_classifier(self, target_snomed_code: str):
        """
        Train a binary classifier to predict if a diagnosis maps to a specific SNOMED code
        """
        df_filtered = self.df.copy()
        df_filtered['is_target'] = (df_filtered['snomed_ct_code'] == target_snomed_code).astype(int)
        
        # Prepare features for this specific classification task
        text_corpus = df_filtered['dx'].fillna('').tolist()
        tfidf_features = self.tfidf_vectorizer.fit_transform(text_corpus).to_array()
        
        numerical_feature_cols = [
            'dx_length', 'dx_word_count', 'has_numbers',
            'snomed_length', 'snomed_numeric',
            'dataset_count', 'max_dataset_value', 'dataset_variance',
            'log_total', 'is_rare', 'is_common'
        ]
        numerical_features_df = df_filtered[numerical_feature_cols].fillna(0)

        X_numerical = numerical_features_df.values
        X = np.hstack((tfidf_features, X_numerical))
        y = df_filtered['is_target'].values
        
        if len(np.unique(y)) < 2:
            print(f"Not enough variation in target code '{target_snomed_code}' for classification.")
            return 0.0 # Cannot train if only one class is present

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

class ConfidenceEstimator:
    """
    Estimates confidence scores for SNOMED mappings
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['dx'].fillna(''))
        
    def estimate_confidence(self, query_diagnosis: str) -> pd.DataFrame:
        """
        Estimate confidence for a query diagnosis by finding similar existing mappings.
        Confidence is based on semantic similarity and frequency.
        """
        query_vec = self.tfidf_vectorizer.transform([query_diagnosis])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Create a DataFrame of similarities
        sim_df = pd.DataFrame({
            'dx': self.df['dx'],
            'snomed_ct_code': self.df['snomed_ct_code'],
            'abbreviation': self.df['abbreviation'],
            'total': self.df['total'],
            'similarity_score': cosine_sim
        })
        
        # Filter out self-similarity if the query diagnosis is in the dataset
        sim_df = sim_df[sim_df['dx'] != query_diagnosis]
        
        # Calculate a combined confidence score
        # Combine similarity with a weighted frequency component
        # Normalize total cases to avoid dominance by very high counts
        max_total = self.df['total'].max()
        sim_df['normalized_total'] = sim_df['total'] / max_total if max_total > 0 else 0
        
        # Simple weighted sum for confidence: adjust weights as needed
        sim_df['confidence_score'] = (sim_df['similarity_score'] * 0.7) + (sim_df['normalized_total'] * 0.3)
        
        return sim_df.sort_values(by='confidence_score', ascending=False)

class SemanticRetrieval:
    """
    Semantic retrieval system for SNOMED diagnoses
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['dx'].fillna(''))
        
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search for diagnoses using TF-IDF and cosine similarity.
        """
        query_vec = self.tfidf_vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            row = self.df.iloc[i].to_dict()
            result = {
                'dx': row['dx'],
                'snomed_ct_code': row['snomed_ct_code'],
                'abbreviation': row['abbreviation'],
                'total': row['total'],
                'similarity_score': cosine_similarities[i],
                # Include dataset counts for display
                'cpsc': row.get('cpsc', 0),
                'cpsc_extra': row.get('cpsc_extra', 0),
                'stpetersburg': row.get('stpetersburg', 0), # Corrected column name
                'ptb': row.get('ptb', 0),
                'ptb_xl': row.get('ptb_xl', 0),
                'georgia': row.get('georgia', 0)
            }
            results.append(result)
        
        return results

    def suggest_codes(self, query_diagnosis: str, top_k: int = 5) -> List[Dict]:
        """
        Suggest SNOMED codes based on semantic similarity to existing diagnoses.
        Combines semantic similarity with a frequency boost.
        """
        query_vec = self.tfidf_vectorizer.transform([query_diagnosis])
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        suggestions = []
        for i, score in enumerate(cosine_similarities):
            row = self.df.iloc[i].to_dict()
            
            # Calculate word overlap
            word_overlap = self._calculate_word_overlap(query_diagnosis, row['dx'])
            
            # Apply a frequency boost for more common diagnoses
            frequency_boost = np.log1p(row['total']) # log1p to handle zero totals gracefully
            
            # Combine scores (adjust weights as needed)
            combined_similarity = (
                score * 0.7 + # Semantic similarity
                word_overlap * 0.2 + # Word overlap for direct term matches
                frequency_boost * 0.1 # Frequency boost for common terms
            )
            
            suggestions.append({
                **row, # Include all original row data
                'similarity': combined_similarity,
                'word_overlap': word_overlap
            })
        
        # Sort by combined similarity
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        return suggestions[:top_k]
    
    def _calculate_word_overlap(self, text1, text2):
        """
        Calculate word overlap between two texts
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_diagnoses(self, snomed_code, top_k=10):
        """
        Find diagnoses similar to those mapped to a specific SNOMED code
        """
        # Get diagnoses with the target SNOMED code
        target_diagnoses = self.df[self.df['snomed_ct_code'] == snomed_code]
        
        if target_diagnoses.empty:
            return []
        
        # Use the first diagnosis as query
        query_diagnosis = target_diagnoses.iloc[0]['dx']
        
        # Find similar diagnoses
        results = self.semantic_search(query_diagnosis, top_k)
        
        # Filter out the original diagnosis
        filtered_results = [r for r in results if r['snomed_ct_code'] != snomed_code]
        return filtered_results

