# 🏥 SNOMED Medical NLP System

A comprehensive medical diagnosis mapping and analysis system for SNOMED CT codes using advanced NLP techniques.

## 📋 Overview

This system provides a complete suite of tools for analyzing, validating, and exploring SNOMED CT medical diagnosis mappings. It combines machine learning, natural language processing, and interactive visualizations to help medical professionals and researchers work with diagnostic data.

## ✨ Features

### 🔍 **Data Explorer**
- Interactive searchable SNOMED mapping table
- Dataset filtering and sorting capabilities
- Real-time search across diagnoses, codes, and abbreviations
- Comprehensive data statistics and metrics

### 🧠 **Semantic Search**
- Advanced NLP-powered diagnosis search
- BioBERT and clinical embeddings support
- Fuzzy string matching and similarity scoring
- Top-k semantic matches with confidence scores

### 📊 **Advanced Analytics**
- Statistical outlier detection using Z-scores
- Dataset distribution analysis and clustering
- PCA and t-SNE visualization for pattern discovery
- Correlation analysis between datasets

### 🤖 **Machine Learning Models**
- **Mapping Classifier**: Predicts mapping status (accepted/rejected/needs review)
- **Confidence Predictor**: Estimates mapping confidence scores
- **Multi-class Classifier**: Suggests appropriate SNOMED codes
- **Semantic Retriever**: Clinical embedding-based search system

### 🔍 **Quality Auditor**
- Comprehensive data quality assessment
- Duplicate and near-duplicate detection
- SNOMED code format validation
- Abbreviation consistency checking
- Missing data analysis
- Statistical anomaly detection

### 🎯 **Diagnosis Predictor**
- Real-time diagnosis code suggestion
- Top-3 SNOMED code recommendations
- Frequency-based matching
- Interactive prediction interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/snomed-medical-nlp.git
cd snomed-medical-nlp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Place your `SNOMED_mappings_unscored.csv` file in the root directory
   - Ensure the CSV follows the expected format (see Data Format section)

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start exploring your SNOMED data!

## 📁 Project Structure

```
snomed-medical-nlp/
├── streamlit_app.py          # Main Streamlit application
├── models.py                 # ML models and embeddings
├── data_processor.py         # Data processing utilities
├── quality_auditor.py        # Data quality assessment
├── utils.py                  # Utility functions and helpers
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── cache/                   # Model cache directory
│   └── embeddings/          # Cached embeddings
└── exports/                 # Exported files directory
```

## 📊 Data Format

Your CSV file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Dx` | Diagnosis description | "atrial fibrillation" |
| `SNOMED CT Code` | SNOMED CT identifier | "49436004" |
| `Abbreviation` | Standard abbreviation | "AF" |
| `CPSC` | Case count in CPSC dataset | "150" |
| `PTB` | Case count in PTB dataset | "89" |
| `PTB-XL` | Case count in PTB-XL dataset | "234" |
| `Georgia` | Case count in Georgia dataset | "67" |
| `Total` | Total case count | "540" |

## 🔧 Configuration

The system uses a comprehensive configuration system in `config.py`. Key settings include:

- **Model Settings**: Embedding models, similarity thresholds
- **Data Settings**: File paths, column mappings
- **UI Settings**: Visualization preferences, table pagination
- **Quality Settings**: Validation thresholds, audit parameters

## 🏗️ Architecture

### Core Components

1. **Data Layer** (`data_processor.py`)
   - Data loading and validation
   - Preprocessing and cleaning
   - Statistical analysis

2. **Model Layer** (`models.py`)
   - Machine learning classifiers
   - Embedding-based semantic search
   - Confidence prediction

3. **Quality Layer** (`quality_auditor.py`)
   - Comprehensive data validation
   - Anomaly detection
   - Consistency checking

4. **Presentation Layer** (`streamlit_app.py`)
   - Interactive web interface
   - Real-time visualizations
   - User interaction handling

5. **Utility Layer** (`utils.py`)
   - Helper functions
   - Visualization components
   - Export utilities

## 🤖 Machine Learning Models

### 1. Mapping Classifier
Predicts whether a source-target mapping will be:
- ✅ **Accepted**: High confidence mapping
- ❌ **Rejected**: Poor quality mapping  
- ⚠️ **Needs Review**: Uncertain mapping requiring expert review

### 2. Confidence Predictor
Estimates numerical confidence scores (0-1) for mappings using:
- Textual similarity features
- Ontology-based semantic distance
- Pre-trained biomedical language models

### 3. Semantic Retriever
Finds similar diagnoses using:
- Clinical embeddings (BioBERT, ClinicalBERT)
- TF-IDF vectorization
- Fuzzy string matching
- Cosine similarity scoring

### 4. Multi-class Classifier
Suggests the most appropriate SNOMED codes for new diagnoses:
- Top-k prediction with confidence scores
- Clinical domain-aware feature extraction
- Frequency-based weighting

## 📈 Quality Auditing

The quality auditor performs comprehensive checks:

### Data Integrity
- ✅ Duplicate diagnosis detection
- ✅ SNOMED code format validation
- ✅ Case count consistency verification
- ✅ Missing data identification

### Semantic Consistency
- ✅ Abbreviation-diagnosis alignment
- ✅ Medical keyword extraction
- ✅ Domain-specific validation
- ✅ Ontological relationship checking

### Statistical Analysis
- ✅ Outlier detection (Z-score based)
- ✅ Distribution analysis
- ✅ Cross-dataset correlation
- ✅ Anomaly pattern recognition

## 🎨 Visualizations

The system provides rich interactive visualizations:

- **📊 Bar Charts**: Dataset distributions and comparisons
- **🥧 Pie Charts**: Proportional breakdowns
- **🔥 Heatmaps**: Correlation matrices and patterns
- **☀️ Sunburst Charts**: Hierarchical data exploration
- **📈 Scatter Plots**: Relationship analysis
- **🗺️ Treemaps**: Nested data representation
- **🕸️ Radar Charts**: Multi-dimensional comparisons

## 🔍 Usage Examples

### Basic Data Exploration
```python
# Load and explore data
from data_processor import DataProcessor
processor = DataProcessor('SNOMED_mappings_unscored.csv')
summary = processor.get_summary_statistics()
```

### Semantic Search
```python
# Find similar diagnoses
from models import SemanticRetriever
retriever = SemanticRetriever(df)
results = retriever.search("chest pain", k=5)
```

### Quality Audit
```python
# Run comprehensive quality check
from quality_auditor import SNOMEDQualityAuditor
auditor = SNOMEDQualityAuditor(df)
audit_results = auditor.run_full_audit()
```

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with automatic updates

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
