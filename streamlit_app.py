import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from models import MappingClassifier, ConfidenceEstimator, SemanticRetrieval
from quality_auditor import SNOMEDQualityAuditor
from pattern_analyzer import PatternAnalyzer

# Page configuration
st.set_page_config(
    page_title="SNOMED Mapping Analysis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the SNOMED dataset"""
    try:
        # Load the CSV file
        df = pd.read_csv('SNOMED_mappings_unscored.csv', delimiter=';')
        processor = DataProcessor()
        df_processed = processor.clean_data(df)
        return df_processed, processor
    except FileNotFoundError:
        st.error("SNOMED_mappings_unscored.csv file not found. Please ensure the file is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def main():
    st.title("🏥 SNOMED Mapping Analysis System")
    st.markdown("**Comprehensive analysis and classification of medical diagnosis mappings**")
    
    # Load data
    df, processor = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "📊 Data Overview",
            "🔍 Search & Browse",
            "🧠 Code Suggestions", 
            "📈 Pattern Analysis",
            "🔬 Data Quality Audit",
            "🤖 ML Classification",
            "📉 Confidence Scoring",
            "🎯 Semantic Retrieval"
        ]
    )
    
    # Dataset filters
    st.sidebar.header("Dataset Filters")
    datasets = ['CPSC', 'CPSC-Extra', 'StPetersburg', 'PTB', 'PTB-XL', 'Georgia']
    selected_datasets = st.sidebar.multiselect(
        "Select Datasets",
        datasets,
        default=datasets
    )
    
    # Filter data based on selection
    dataset_columns = [col.lower().replace('-', '_') for col in selected_datasets]
    
    # Main content based on page selection
    if "Data Overview" in page:
        show_data_overview(df, processor)
    elif "Search & Browse" in page:
        show_search_browse(df, processor, selected_datasets)
    elif "Code Suggestions" in page:
        show_code_suggestions(df, processor)
    elif "Pattern Analysis" in page:
        show_pattern_analysis(df, processor, selected_datasets)
    elif "Data Quality Audit" in page:
        show_quality_audit(df, processor)
    elif "ML Classification" in page:
        show_ml_classification(df, processor)
    elif "Confidence Scoring" in page:
        show_confidence_scoring(df, processor)
    elif "Semantic Retrieval" in page:
        show_semantic_retrieval(df, processor)

def show_data_overview(df, processor):
    """Display data overview and statistics"""
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Diagnoses", len(df))
    with col2:
        st.metric("Unique SNOMED Codes", df['snomed_ct_code'].nunique())
    with col3:
        st.metric("Total Cases", df['total'].sum())
    with col4:
        st.metric("Avg Cases per Diagnosis", f"{df['total'].mean():.1f}")
    
    # Dataset distribution
    st.subheader("Dataset Distribution")
    dataset_totals = {
        'CPSC': df['cpsc'].sum(),
        'CPSC-Extra': df['cpsc_extra'].sum(),
        'StPetersburg': df['st_petersburg'].sum(),
        'PTB': df['ptb'].sum(),
        'PTB-XL': df['ptb_xl'].sum(),
        'Georgia': df['georgia'].sum()
    }
    
    fig = px.pie(
        values=list(dataset_totals.values()),
        names=list(dataset_totals.keys()),
        title="Distribution of Cases by Dataset"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top diagnoses
    st.subheader("Top 20 Most Frequent Diagnoses")
    top_diagnoses = df.nlargest(20, 'total')[['dx', 'abbreviation', 'total']]
    
    fig = px.bar(
        top_diagnoses,
        x='total',
        y='dx',
        orientation='h',
        title="Most Frequent Diagnoses"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_search_browse(df, processor, selected_datasets):
    """Search and browse functionality"""
    st.header("🔍 Search & Browse")
    
    # Search functionality
    search_term = st.text_input("Search diagnoses, SNOMED codes, or abbreviations:")
    
    if search_term:
        mask = (
            df['dx'].str.contains(search_term, case=False, na=False) |
            df['snomed_ct_code'].astype(str).str.contains(search_term, case=False, na=False) |
            df['abbreviation'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Display options
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Number of results to show", 10, 100, 25)
    with col2:
        sort_by = st.selectbox("Sort by", ['total', 'dx', 'snomed_ct_code'])
    
    # Display results
    display_df = filtered_df.nlargest(top_n, 'total') if sort_by == 'total' else filtered_df.head(top_n)
    
    st.dataframe(
        display_df[['dx', 'snomed_ct_code', 'abbreviation', 'total'] + 
                  [col.lower().replace('-', '_') for col in selected_datasets]],
        use_container_width=True
    )
    
    # Visualization of search results
    if not filtered_df.empty and len(filtered_df) <= 50:
        st.subheader("Frequency Visualization")
        fig = px.bar(
            display_df.head(20),
            x='abbreviation',
            y='total',
            hover_data=['dx'],
            title=f"Top {min(20, len(display_df))} Results"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_code_suggestions(df, processor):
    """SNOMED code suggestion system"""
    st.header("🧠 SNOMED Code Suggestions")
    
    st.markdown("Enter a diagnosis description to get suggested SNOMED codes based on semantic similarity.")
    
    # Input for new diagnosis
    new_diagnosis = st.text_input("Enter diagnosis description:")
    
    if new_diagnosis:
        # Initialize semantic retrieval system
        retrieval_system = SemanticRetrieval(df)
        suggestions = retrieval_system.suggest_codes(new_diagnosis, top_k=5)
        
        st.subheader("Top 5 Suggested SNOMED Codes")
        
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"#{i} - {suggestion['abbreviation']} (Similarity: {suggestion['similarity']:.3f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Diagnosis:** {suggestion['dx']}")
                    st.write(f"**SNOMED Code:** {suggestion['snomed_ct_code']}")
                    st.write(f"**Abbreviation:** {suggestion['abbreviation']}")
                with col2:
                    st.write(f"**Total Cases:** {suggestion['total']}")
                    st.write(f"**Semantic Similarity:** {suggestion['similarity']:.3f}")
                    
                    # Dataset occurrence
                    datasets = ['cpsc', 'cpsc_extra', 'st_petersburg', 'ptb', 'ptb_xl', 'georgia']
                    occurrence = [suggestion[ds] for ds in datasets]
                    if sum(occurrence) > 0:
                        fig = px.bar(
                            x=['CPSC', 'CPSC-Extra', 'StPetersburg', 'PTB', 'PTB-XL', 'Georgia'],
                            y=occurrence,
                            title="Dataset Occurrence"
                        )
                        st.plotly_chart(fig, use_container_width=True)

def show_pattern_analysis(df, processor, selected_datasets):
    """Pattern analysis and clustering"""
    st.header("📈 Pattern Analysis")
    
    analyzer = PatternAnalyzer(df)
    
    # Statistical outliers
    st.subheader("Statistical Outliers")
    outliers = analyzer.find_outliers()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**High Frequency Outliers (Z-score > 2)**")
        high_outliers = outliers[outliers['z_score'] > 2].head(10)
        st.dataframe(high_outliers[['dx', 'total', 'z_score']])
    
    with col2:
        st.write("**Low Frequency Outliers (Z-score < -1)**")
        low_outliers = outliers[outliers['z_score'] < -1].head(10)
        st.dataframe(low_outliers[['dx', 'total', 'z_score']])
    
    # Dataset distribution patterns
    st.subheader("Dataset Distribution Patterns")
    patterns = analyzer.analyze_dataset_patterns()
    
    # Visualization of patterns
    fig = px.scatter(
        patterns,
        x='dominant_dataset_pct',
        y='total',
        color='dominant_dataset',
        hover_data=['dx'],
        title="Dataset Dominance vs Total Cases",
        log_y=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Clustering analysis
    st.subheader("Clustering Analysis")
    clusters = analyzer.perform_clustering(n_clusters=5)
    
    # PCA visualization
    pca_data = analyzer.get_pca_visualization(clusters)
    fig = px.scatter(
        pca_data,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['dx', 'total'],
        title="PCA Visualization of Diagnosis Clusters"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    cluster_stats = analyzer.get_cluster_stats(clusters)
    st.dataframe(cluster_stats)

def show_quality_audit(df, processor):
    """Data quality audit"""
    st.header("🔬 Data Quality Audit")
    
    auditor = QualityAuditor(df)
    audit_results = auditor.perform_audit()
    
    # Quality metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duplicate Diagnoses", len(audit_results['duplicates']))
    with col2:
        st.metric("Invalid SNOMED Codes", len(audit_results['invalid_codes']))
    with col3:
        st.metric("Zero Total Cases", len(audit_results['zero_totals']))
    with col4:
        st.metric("Ambiguous Abbreviations", len(audit_results['ambiguous_abbrev']))
    
    # Detailed audit results
    tab1, tab2, tab3, tab4 = st.tabs(["Duplicates", "Invalid Codes", "Zero Totals", "Ambiguous Abbreviations"])
    
    with tab1:
        if audit_results['duplicates']:
            st.dataframe(pd.DataFrame(audit_results['duplicates']))
        else:
            st.success("No duplicate diagnoses found!")
    
    with tab2:
        if audit_results['invalid_codes']:
            st.dataframe(pd.DataFrame(audit_results['invalid_codes']))
        else:
            st.success("All SNOMED codes are valid!")
    
    with tab3:
        if audit_results['zero_totals']:
            st.dataframe(pd.DataFrame(audit_results['zero_totals']))
        else:
            st.success("No diagnoses with zero total cases!")
    
    with tab4:
        if audit_results['ambiguous_abbrev']:
            st.dataframe(pd.DataFrame(audit_results['ambiguous_abbrev']))
        else:
            st.success("No ambiguous abbreviations found!")

def show_ml_classification(df, processor):
    """Machine learning classification models"""
    st.header("🤖 ML Classification Models")
    
    st.markdown("Build classification models to predict mapping acceptance and target SNOMED codes.")
    
    # Mapping Status Classifier
    st.subheader("Mapping Status Classifier")
    st.markdown("*Note: This is a demonstration with simulated labels since the dataset doesn't include mapping status.*")
    
    classifier = MappingClassifier(df)
    
    # Simulate training data (in real scenario, this would be labeled data)
    X, y_simulated = classifier.prepare_training_data()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Classification Model"):
            accuracy, report = classifier.train_model(X, y_simulated)
            st.success(f"Model trained with accuracy: {accuracy:.3f}")
            st.text("Classification Report:")
            st.text(report)
    
    with col2:
        # Feature importance
        if hasattr(classifier, 'model') and classifier.model is not None:
            importance = classifier.get_feature_importance()
            fig = px.bar(
                x=importance['importance'],
                y=importance['feature'],
                orientation='h',
                title="Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Multi-class SNOMED Code Classifier
    st.subheader("SNOMED Code Prediction")
    
    # Select common codes for demonstration
    common_codes = df['snomed_ct_code'].value_counts().head(10).index.tolist()
    
    target_code = st.selectbox("Select target SNOMED code to predict:", common_codes)
    
    if st.button("Train Code Prediction Model"):
        code_classifier = MappingClassifier(df)
        accuracy = code_classifier.train_code_classifier(target_code)
        st.success(f"Code prediction model trained with accuracy: {accuracy:.3f}")

def show_confidence_scoring(df, processor):
    """Confidence scoring for mappings"""
    st.header("📉 Confidence Scoring")
    
    estimator = ConfidenceEstimator(df)
    
    st.markdown("Estimate confidence scores for diagnosis-SNOMED code mappings.")
    
    # Select a diagnosis for confidence estimation
    diagnosis_options = df['dx'].unique()[:50]  # Limit for performance
    selected_diagnosis = st.selectbox("Select diagnosis:", diagnosis_options)
    
    if selected_diagnosis:
        confidence_scores = estimator.estimate_confidence(selected_diagnosis)
        
        st.subheader(f"Confidence Scores for: {selected_diagnosis}")
        
        # Display top confident mappings
        top_confident = confidence_scores.head(10)
        
        fig = px.bar(
            top_confident,
            x='confidence_score',
            y='snomed_ct_code',
            orientation='h',
            title="Top 10 Confident Mappings",
            hover_data=['dx']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(top_confident)

def show_semantic_retrieval(df, processor):
    """Semantic retrieval system"""
    st.header("🎯 Semantic Retrieval System")
    
    retrieval_system = SemanticRetrieval(df)
    
    st.markdown("Advanced semantic search using TF-IDF and cosine similarity.")
    
    # Query input
    query = st.text_input("Enter search query:")
    top_k = st.slider("Number of results:", 1, 20, 10)
    
    if query:
        results = retrieval_system.semantic_search(query, top_k=top_k)
        
        st.subheader(f"Top {top_k} Semantic Matches")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"#{i} - {result['abbreviation']} (Score: {result['similarity_score']:.3f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Diagnosis:** {result['dx']}")
                    st.write(f"**SNOMED Code:** {result['snomed_ct_code']}")
                    st.write(f"**Total Cases:** {result['total']}")
                with col2:
                    st.write(f"**Similarity Score:** {result['similarity_score']:.3f}")
                    
                    # Create a simple visualization of dataset distribution
                    datasets = ['CPSC', 'CPSC-Extra', 'StPetersburg', 'PTB', 'PTB-XL', 'Georgia']
                    values = [result['cpsc'], result['cpsc_extra'], result['st_petersburg'], 
                             result['ptb'], result['ptb_xl'], result['georgia']]
                    
                    if sum(values) > 0:
                        fig = px.bar(x=datasets, y=values, title="Dataset Distribution")
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
