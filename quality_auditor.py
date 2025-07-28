import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SNOMEDQualityAuditor:
    """
    Comprehensive data quality auditor for SNOMED mapping dataset
    Performs validation, consistency checks, and anomaly detection
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() # Ensure working on a copy
        self.audit_results = {}
        self.issues = []
        
    def run_full_audit(self) -> Dict:
        """Run complete quality audit and return comprehensive report"""
        print("🔍 Starting comprehensive SNOMED data quality audit...")
        
        # Run all audit checks
        self.check_duplicate_diagnoses()
        self.validate_snomed_codes()
        self.check_abbreviation_consistency()
        self.validate_case_counts()
        self.check_missing_data()
        self.detect_format_inconsistencies()
        self.analyze_distribution_anomalies()
        self.check_semantic_consistency()
        
        # Compile final report
        self.audit_results['total_issues'] = len(self.issues)
        self.audit_results['data_quality_score'] = self.calculate_quality_score()
        
        print(f"✅ Audit complete. Found {len(self.issues)} issues.")
        return self.audit_results
    
    def check_duplicate_diagnoses(self):
        """Check for duplicate or near-duplicate diagnosis descriptions"""
        print("Checking for duplicate diagnoses...")
        # Corrected column name: 'Dx' -> 'dx'
        duplicates = self.df[self.df.duplicated(['dx'], keep=False)].sort_values(by='dx')
        if not duplicates.empty:
            self.issues.append({
                'type': 'duplicate_diagnoses',
                'severity': 'high',
                'count': len(duplicates),
                'details': duplicates[['dx', 'snomed_ct_code', 'total']].to_dict(orient='records')
            })

    def validate_snomed_codes(self):
        """Validate SNOMED CT codes format and presence"""
        print("Validating SNOMED codes...")
        # Corrected column name: 'SNOMED CT Code' -> 'snomed_ct_code'
        invalid_codes = self.df[~self.df['snomed_ct_code'].apply(lambda x: isinstance(x, str) and x.isdigit())]
        if not invalid_codes.empty:
            self.issues.append({
                'type': 'invalid_snomed_codes',
                'severity': 'high',
                'count': len(invalid_codes),
                'details': invalid_codes[['dx', 'snomed_ct_code']].to_dict(orient='records')
            })

    def check_abbreviation_consistency(self):
        """Check for ambiguous abbreviations (same abbrev, different Dx)"""
        print("Checking abbreviation consistency...")
        # Corrected column names: 'Abbreviation' -> 'abbreviation', 'Dx' -> 'dx'
        abbrev_groups = self.df.groupby('abbreviation')['dx'].nunique()
        ambiguous_abbrevs = abbrev_groups[abbrev_groups > 1]

        if not ambiguous_abbrevs.empty:
            details = []
            for abbrev in ambiguous_abbrevs.index:
                diagnoses = self.df[self.df['abbreviation'] == abbrev]['dx'].unique().tolist()
                details.append({
                    'abbreviation': abbrev,
                    'diagnoses': diagnoses
                })
            self.issues.append({
                'type': 'ambiguous_abbreviations',
                'severity': 'medium',
                'count': len(ambiguous_abbrevs),
                'details': details
            })

    def validate_case_counts(self):
        """Validate total case counts and detect zero totals"""
        print("Validating case counts...")
        # Corrected column name: 'Total' -> 'total'
        zero_totals = self.df[self.df['total'] == 0]
        if not zero_totals.empty:
            self.issues.append({
                'type': 'zero_total_cases',
                'severity': 'low',
                'count': len(zero_totals),
                'details': zero_totals[['dx', 'snomed_ct_code']].to_dict(orient='records')
            })

    def check_missing_data(self):
        """Check for critical missing data points"""
        print("Checking for missing data...")
        # Corrected column names: 'Dx' -> 'dx', 'SNOMED CT Code' -> 'snomed_ct_code'
        missing_dx = self.df[self.df['dx'].isna() | (self.df['dx'] == '')]
        missing_snomed = self.df[self.df['snomed_ct_code'].isna() | (self.df['snomed_ct_code'] == '')]

        if not missing_dx.empty:
            self.issues.append({
                'type': 'missing_diagnosis_text',
                'severity': 'high',
                'count': len(missing_dx),
                'details': missing_dx[['dx', 'snomed_ct_code']].to_dict(orient='records')
            })
        if not missing_snomed.empty:
            self.issues.append({
                'type': 'missing_snomed_codes',
                'severity': 'high',
                'count': len(missing_snomed),
                'details': missing_snomed[['dx', 'snomed_ct_code']].to_dict(orient='records')
            })

    def detect_format_inconsistencies(self):
        """Detect formatting issues in diagnosis text (e.g., excessive special chars)"""
        print("Detecting format inconsistencies...")
        # This check relies on the 'dx' column being cleaned by DataProcessor already.
        # We can look for remnants of uncleaned text if DataProcessor isn't perfect,
        # or identify unusual patterns that might indicate issues.
        # For now, a simple check for very long words or unusual character sets
        
        # Corrected column name: 'Dx' -> 'dx'
        unusual_chars = self.df[self.df['dx'].apply(lambda x: bool(re.search(r'[^a-z0-9\s\-\(\)]', x)))]
        if not unusual_chars.empty:
            self.issues.append({
                'type': 'diagnosis_format_inconsistency',
                'severity': 'low',
                'count': len(unusual_chars),
                'details': unusual_chars[['dx', 'snomed_ct_code']].to_dict(orient='records')
            })

    def analyze_distribution_anomalies(self):
        """Analyze distribution of total cases and detect statistical anomalies"""
        print("Analyzing distribution anomalies...")
        # Corrected column name: 'Total' -> 'total'
        if 'total' in self.df.columns and not self.df['total'].empty:
            q1 = self.df['total'].quantile(0.25)
            q3 = self.df['total'].quantile(0.75)
            iqr = q3 - q1
            
            # Outliers definition (e.g., 1.5 * IQR rule)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Corrected column name: 'Total' -> 'total'
            anomalies = self.df[(self.df['total'] < lower_bound) | (self.df['total'] > upper_bound)]
            
            if not anomalies.empty:
                self.issues.append({
                    'type': 'case_count_anomalies',
                    'severity': 'medium',
                    'count': len(anomalies),
                    'details': anomalies[['dx', 'total']].to_dict(orient='records')
                })

    def check_semantic_consistency(self):
        """
        Perform basic semantic consistency checks, e.g., SNOMED codes that map to very different diagnoses.
        This is a more advanced check and might require external resources or more complex models.
        For now, a simple check for SNOMED codes mapped to a high number of unique diagnoses.
        """
        print("Checking semantic consistency...")
        # Corrected column names: 'SNOMED CT Code' -> 'snomed_ct_code', 'Dx' -> 'dx'
        snomed_dx_counts = self.df.groupby('snomed_ct_code')['dx'].nunique()
        # Identify SNOMED codes mapped to many different diagnosis texts (potential ambiguity)
        ambiguous_snomed_mappings = snomed_dx_counts[snomed_dx_counts > 5] # Threshold can be adjusted

        if not ambiguous_snomed_mappings.empty:
            details = []
            for snomed_code in ambiguous_snomed_mappings.index:
                diagnoses = self.df[self.df['snomed_ct_code'] == snomed_code]['dx'].unique().tolist()
                details.append({
                    'snomed_code': snomed_code,
                    'unique_diagnoses_count': len(diagnoses),
                    'diagnoses_examples': diagnoses[:5] # Show first 5 examples
                })
            self.issues.append({
                'type': 'ambiguous_snomed_mappings',
                'severity': 'medium',
                'count': len(ambiguous_snomed_mappings),
                'details': details
            })

    def calculate_quality_score(self) -> float:
        """Calculate an overall data quality score based on detected issues"""
        # Simple scoring: Deduct points based on severity and count of issues
        score = 100.0
        for issue in self.issues:
            if issue['severity'] == 'high':
                score -= issue['count'] * 2
            elif issue['severity'] == 'medium':
                score -= issue['count'] * 1
            else: # low severity
                score -= issue['count'] * 0.5
        return max(0, score) # Score cannot be less than 0

    def get_audit_report_summary(self) -> str:
        """Generate a human-readable summary of the audit findings"""
        report = ["=" * 60]
        report.append("SNOMED Data Quality Audit Report Summary")
        report.append("=" * 60)
        report.append(f"Total Issues Found: {len(self.issues)}")
        report.append(f"Overall Data Quality Score: {self.calculate_quality_score():.2f}/100")
        report.append("-" * 60)
        
        if not self.issues:
            report.append("🎉 No major data quality issues detected. Excellent!")
        else:
            issue_types = Counter(issue['type'] for issue in self.issues)
            report.append("Breakdown of Issues:")
            for issue_type, count in issue_types.items():
                report.append(f"- {issue_type.replace('_', ' ').title()}: {count} occurrences")
            
            report.append("\nRecommendations:")
            if 'duplicate_diagnoses' in issue_types:
                report.append("  1. Consolidate duplicate diagnosis entries.")
            if 'invalid_snomed_codes' in issue_types or 'missing_snomed_codes' in issue_types:
                report.append("  2. Verify and correct invalid or missing SNOMED CT codes.")
            if 'ambiguous_abbreviations' in issue_types:
                report.append("  3. Standardize abbreviations to ensure unique mapping.")
            if 'zero_total_cases' in issue_types or 'case_count_anomalies' in issue_types:
                report.append("  4. Investigate anomalies in case counts.")
            if 'diagnosis_format_inconsistency' in issue_types:
                report.append("  5. Review diagnosis text cleaning and standardization processes.")
            if 'ambiguous_snomed_mappings' in issue_types: # Added this check
                report.append("  6. Review SNOMED code mappings for consistency")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_issues_to_csv(self, filename: str = "snomed_quality_issues.csv"):
        """Export all issues to CSV for detailed review"""
        if not self.issues:
            print("No issues found to export.")
            return
        
        # Flatten issues for CSV export
        flattened_issues = []
        for issue in self.issues:
            base_info = {
                'issue_type': issue['type'],
                'severity': issue['severity'],
                'count': issue['count']
            }
            
            if 'details' in issue:
                if isinstance(issue['details'], list):
                    for detail in issue['details']:
                        row = base_info.copy()
                        if isinstance(detail, dict):
                            row.update(detail)
                        flattened_issues.append(row)
                else:
                    row = base_info.copy()
                    row.update(issue['details'])
                    flattened_issues.append(row)
            else:
                flattened_issues.append(base_info)
        
        issues_df = pd.DataFrame(flattened_issues)
        issues_df.to_csv(filename, index=False)
        print(f"Issues exported to {filename}")
        
        return issues_df
