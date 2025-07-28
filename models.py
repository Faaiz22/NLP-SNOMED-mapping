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
import warnings
warnings.filterwarnings('ignore')

class MappingClassifier:
    """
    Multi-class classifier for SNOMED mapping tasks
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_names = []
    
    def prepare_training_data(self):
        """
        Prepare training data with simulated labels for demonstration
        Since the dataset doesn't include mapping status, we simulate it based on patterns
        """
        # Extract features
        features = []
        
        # Text features using TF-IDF
        text_corpus = self.df['dx'].fillna('').tolist()
        tfidf_features = self.tfidf_vectorizer.fit_transform(text_corpus).toarray()
        
        # Numerical features
        numerical_features = []
        for _, row in self.df.iterrows():
            num_features = [
                len(str(row['dx'])),  # diagnosis length
                len(str(row['snomed_ct_code'])),  # SNOMED code length
                row['total'],  # total cases
                row['cpsc'] + row['cpsc_extra'] + row['st_petersburg'] + 
                row['ptb'] + row['ptb_xl'] + row['georgia'],  # sum across datasets
                (np.array([row['cpsc'], row['cpsc_extra'], row['st_petersburg'], 
                          row['ptb'], row['ptb_xl'], row['georgia']]) > 0).sum()  # number of datasets
            ]
            numerical_features.append(num_features)
        
        numerical_features = np.array(numerical_features)
        
        # Combine features
        X = np.hstack([tfidf_features, numerical_features])
        
        # Simulate mapping status labels based on heuristics
        y_simulated = []
        for _, row in self.df.iterrows():
            # Simulate labels based on total cases and dataset distribution
            if row['total'] > 100:  # High confidence - accepted
                label = 'accepted'
            elif row['total'] < 5:  # Low confidence - needs review
                label = 'needs_review'
            elif row['total'] == 0:  # Zero cases - rejected
                label = 'rejected'
            else:  # Medium cases - could be any
                # Add some randomness based on dataset distribution
                dataset_count = (np.array([row['cpsc'], row['cpsc_extra'], row['st_petersburg'], 
                                         row['ptb'], row['ptb_xl'], row['georgia']]) > 0).sum()
                if dataset_count >= 3:
                    label = 'accepted'
                elif dataset_count == 1:
                    label = 'needs_review'
                else:
                    label = 'accepted'
            y_simulated.append(label)
        
        return X, np.array(y_simulated)
    
    def train_model(self, X, y, model_type='rf'):
        """
        Train the classification model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.model = SVC(random_state=42, probability=True)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def predict_mapping_status(self, diagnosis_text, snomed_code):
        """
        Predict mapping status for a new diagnosis-SNOMED pair
        """
        if self.model is None:
            return None
        
        # Prepare features (simplified for demo)
        text_features = self.tfidf_vectorizer.transform([diagnosis_text]).toarray()
        num_features = np.array([[len(diagnosis_text), len(snomed_code), 0, 0, 1]])
        
        X = np.hstack([text_features, num_features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0].max()
        
        return prediction, probability
    
    def get_feature_importance(self):
        """
        Get feature importance for RandomForest models
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        # Get TF-IDF feature names
        tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
        numerical_feature_names = ['dx_length', 'snomed_length', 'total_cases', 'sum_datasets', 'num_datasets']
        
        all_feature_names = list(tfidf_features) + numerical_feature_names
        
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        return importance_df
    
    def train_code_classifier(self, target_code):
        """
        Train a binary classifier to predict if a diagnosis maps to a specific SNOMED code
        """
        # Prepare binary labels
        y_binary = (self.df['snomed_ct_code'] == target_code).astype(int)
        
        # Get features
        X, _ = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train binary classifier
        binary_model = RandomForestClassifier(n_estimators=100, random_state=42)
        binary_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = binary_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy


class ConfidenceEstimator:
    """
    Regression model to estimate mapping confidence scores
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model = None
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def prepare_confidence_features(self):
        """
        Prepare features for confidence estimation
        """
        # Text similarity features
        text_corpus = self.df['dx'].fillna('').tolist()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_corpus).toarray()
        
        # Calculate self-similarity as a baseline confidence measure
        confidence_scores = []
        
        for i, row in self.df.iterrows():
            # Confidence based on frequency, dataset distribution, and text characteristics
            freq_score = min(row['total'] / self.df['total'].max(), 1.0)  # Normalized frequency
            
            # Dataset consistency score
            dataset_values = [row['cpsc'], row['cpsc_extra'], row['st_petersburg'], 
                            row['ptb'], row['ptb_xl'], row['georgia']]
            non_zero_datasets = sum(1 for x in dataset_values if x > 0)
            consistency_score = non_zero_datasets / 6.0  # Normalized by total datasets
            
            # Text quality score (longer, more specific diagnoses get higher scores)
            text_quality = min(len(row['dx']) / 50.0, 1.0)  # Normalized text length
            
            # Combined confidence score
            confidence = (freq_score * 0.5 + consistency_score * 0.3 + text_quality * 0.2)
            confidence_scores.append(confidence)
        
        return tfidf_matrix, np.array(confidence_scores)
    
    def train_confidence_model(self):
        """
        Train regression model for confidence estimation
        """
        X, y = self.prepare_confidence_features()
        
        # Add numerical features
        numerical_features = []
        for _, row in self.df.iterrows():
            num_features = [
                row['total'],
                len(str(row['dx'])),
                len(str(row['snomed_ct_code'])),
                (np.array([row['cpsc'], row['cpsc_extra'], row['st_petersburg'], 
                          row['ptb'], row['ptb_xl'], row['georgia']]) > 0).sum()
            ]
            numerical_features.append(num_features)
        
        numerical_features = np.array(numerical_features)
        X_combined = np.hstack([X, numerical_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        return mse
    
    def estimate_confidence(self, diagnosis_text, top_k=10):
        """
        Estimate confidence for a given diagnosis against all SNOMED codes
        """
        # Find similar diagnoses
        similarities = []
        
        for _, row in self.df.iterrows():
            # Simple text similarity (can be enhanced with embeddings)
            common_words = set(diagnosis_text.lower().split()) & set(row['dx'].lower().split())
            similarity = len(common_words) / max(len(diagnosis_text.split()), len(row['dx'].split()))
            
            similarities.append({
                'dx': row['dx'],
                'snomed_ct_code': row['snomed_ct_code'],
                'similarity': similarity,
                'total': row['total'],
                'confidence_score': similarity * (1 + np.log1p(row['total']) / 10)  # Boost by frequency
            })
        
        # Sort by confidence and return top-k
        similarities.sort(key=lambda x: x['confidence_score'], reverse=True)
        return pd.DataFrame(similarities[:top_k])


class SemanticRetrieval:
    """
    Semantic retrieval system for SNOMED code suggestions
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.tfidf_matrix = None
        self._build_index()
    
    def _build_index(self):
        """
        Build TF-IDF index for semantic search
        """
        # Prepare text corpus
        corpus = []
        for _, row in self.df.iterrows():
            # Combine diagnosis text with abbreviation for richer context
            text = f"{row['dx']} {row['abbreviation']}"
            corpus.append(text)
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
    
    def semantic_search(self, query, top_k=10):
        """
        Perform semantic search using TF-IDF and cosine similarity
        """
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'dx': row['dx'],
                'snomed_ct_code': row['snomed_ct_code'],
                'abbreviation': row['abbreviation'],
                'total': row['total'],
                'similarity_score': similarities[idx],
                'cpsc': row['cpsc'],
                'cpsc_extra': row['cpsc_extra'],
                'st_petersburg': row['st_petersburg'],
                'ptb': row['ptb'],
                'ptb_xl': row['ptb_xl'],
                'georgia': row['georgia']
            })
        
        return results
    
    def suggest_codes(self, diagnosis_text, top_k=5):
        """
        Suggest SNOMED codes for a new diagnosis
        """
        # Use semantic search
        search_results = self.semantic_search(diagnosis_text, top_k)
        
        # Enhance with additional features
        suggestions = []
        for result in search_results:
            # Calculate additional similarity metrics
            word_overlap = self._calculate_word_overlap(diagnosis_text, result['dx'])
            frequency_boost = np.log1p(result['total']) / 10.0
            
            # Combined similarity score
            combined_similarity = (
                result['similarity_score'] * 0.7 +
                word_overlap * 0.2 +
                frequency_boost * 0.1
            )
            
            suggestions.append({
                **result,
                'similarity': combined_similarity,
                'word_overlap': word_overlap
            })
        
        # Sort by combined similarity
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        return suggestions
    
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
        
        return filtered_results[:top_k]