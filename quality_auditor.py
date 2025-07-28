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
        self.df = df.copy()
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
        
        # Exact duplicates
        duplicates = self.df[self.df.duplicated(['Dx'], keep=False)]
        if not duplicates.empty:
            self.issues.append({
                'type': 'duplicate_diagnoses',
                'severity': 'high',
                'count': len(duplicates),
                'details': duplicates[['Dx', 'SNOMED CT Code']].to_dict('records')
            })
        
        # Near duplicates (fuzzy matching)
        near_duplicates = self.find_near_duplicate_diagnoses()
        if near_duplicates:
            self.issues.append({
                'type': 'near_duplicate_diagnoses',
                'severity': 'medium',
                'count': len(near_duplicates),
                'details': near_duplicates
            })
        
        self.audit_results['duplicate_check'] = {
            'exact_duplicates': len(duplicates),
            'near_duplicates': len(near_duplicates),
            'status': 'pass' if len(duplicates) == 0 and len(near_duplicates) == 0 else 'warning'
        }
    
    def find_near_duplicate_diagnoses(self, threshold: float = 0.9) -> List[Dict]:
        """Find diagnoses with high text similarity"""
        from fuzzywuzzy import fuzz
        
        diagnoses = self.df['Dx'].tolist()
        near_duplicates = []
        
        for i, dx1 in enumerate(diagnoses):
            for j, dx2 in enumerate(diagnoses[i+1:], i+1):
                similarity = fuzz.ratio(dx1.lower(), dx2.lower()) / 100.0
                if similarity >= threshold:
                    near_duplicates.append({
                        'diagnosis_1': dx1,
                        'diagnosis_2': dx2,
                        'similarity': similarity,
                        'snomed_1': self.df.iloc[i]['SNOMED CT Code'],
                        'snomed_2': self.df.iloc[j]['SNOMED CT Code']
                    })
        
        return near_duplicates
    
    def validate_snomed_codes(self):
        """Validate SNOMED CT code format and consistency"""
        print("Validating SNOMED CT codes...")
        
        invalid_codes = []
        duplicate_codes = []
        
        # Check format (should be numeric)
        for idx, code in enumerate(self.df['SNOMED CT Code']):
            if not str(code).isdigit():
                invalid_codes.append({
                    'index': idx,
                    'diagnosis': self.df.iloc[idx]['Dx'],
                    'invalid_code': code,
                    'reason': 'non_numeric'
                })
        
        # Check for duplicate SNOMED codes with different diagnoses
        code_to_diagnoses = defaultdict(list)
        for idx, row in self.df.iterrows():
            code_to_diagnoses[row['SNOMED CT Code']].append({
                'index': idx,
                'diagnosis': row['Dx']
            })
        
        for code, diagnoses in code_to_diagnoses.items():
            if len(diagnoses) > 1:
                unique_diagnoses = set(d['diagnosis'] for d in diagnoses)
                if len(unique_diagnoses) > 1:
                    duplicate_codes.append({
                        'snomed_code': code,
                        'diagnoses': list(unique_diagnoses),
                        'count': len(diagnoses)
                    })
        
        if invalid_codes:
            self.issues.append({
                'type': 'invalid_snomed_codes',
                'severity': 'high',
                'count': len(invalid_codes),
                'details': invalid_codes
            })
        
        if duplicate_codes:
            self.issues.append({
                'type': 'duplicate_snomed_codes',
                'severity': 'medium',
                'count': len(duplicate_codes),
                'details': duplicate_codes
            })
        
        self.audit_results['snomed_validation'] = {
            'invalid_format': len(invalid_codes),
            'duplicate_mappings': len(duplicate_codes),
            'total_unique_codes': self.df['SNOMED CT Code'].nunique(),
            'status': 'pass' if len(invalid_codes) == 0 and len(duplicate_codes) == 0 else 'fail'
        }
    
    def check_abbreviation_consistency(self):
        """Check abbreviation consistency and potential conflicts"""
        print("Checking abbreviation consistency...")
        
        # Check for abbreviations mapping to multiple SNOMED codes
        abbrev_conflicts = defaultdict(set)
        for _, row in self.df.iterrows():
            abbrev_conflicts[row['Abbreviation']].add(row['SNOMED CT Code'])
        
        conflicts = []
        for abbrev, codes in abbrev_conflicts.items():
            if len(codes) > 1:
                conflicts.append({
                    'abbreviation': abbrev,
                    'snomed_codes': list(codes),
                    'count': len(codes)
                })
        
        # Check for missing abbreviations
        missing_abbrev = self.df[self.df['Abbreviation'].isna() | (self.df['Abbreviation'] == '')]
        
        # Check abbreviation format consistency
        invalid_abbrev = []
        for idx, abbrev in enumerate(self.df['Abbreviation']):
            if pd.notna(abbrev):
                # Check for unusual characters or length
                if len(str(abbrev)) > 10 or re.search(r'[^A-Za-z0-9]', str(abbrev)):
                    invalid_abbrev.append({
                        'index': idx,
                        'abbreviation': abbrev,
                        'diagnosis': self.df.iloc[idx]['Dx']
                    })
        
        if conflicts:
            self.issues.append({
                'type': 'abbreviation_conflicts',
                'severity': 'medium',
                'count': len(conflicts),
                'details': conflicts
            })
        
        if not missing_abbrev.empty:
            self.issues.append({
                'type': 'missing_abbreviations',
                'severity': 'low',
                'count': len(missing_abbrev),
                'details': missing_abbrev[['Dx', 'SNOMED CT Code']].to_dict('records')
            })
        
        self.audit_results['abbreviation_check'] = {
            'conflicts': len(conflicts),
            'missing': len(missing_abbrev),
            'invalid_format': len(invalid_abbrev),
            'status': 'pass' if len(conflicts) == 0 else 'warning'
        }
    
    def validate_case_counts(self):
        """Validate case count data integrity"""
        print("Validating case counts...")
        
        dataset_cols = [col for col in self.df.columns 
                       if col not in ['Dx', 'SNOMED CT Code', 'Abbreviation', 'Total']]
        
        inconsistent_totals = []
        negative_counts = []
        extreme_outliers = []
        
        for idx, row in self.df.iterrows():
            # Check if Total equals sum of individual datasets
            calculated_total = sum(row[col] for col in dataset_cols if pd.notna(row[col]))
            if abs(calculated_total - row['Total']) > 0:
                inconsistent_totals.append({
                    'index': idx,
                    'diagnosis': row['Dx'],
                    'reported_total': row['Total'],
                    'calculated_total': calculated_total,
                    'difference': row['Total'] - calculated_total
                })
            
            # Check for negative counts
            for col in dataset_cols + ['Total']:
                if row[col] < 0:
                    negative_counts.append({
                        'index': idx,
                        'diagnosis': row['Dx'],
                        'column': col,
                        'value': row[col]
                    })
            
            # Check for extreme outliers (using IQR method)
            if row['Total'] > 0:
                q75, q25 = np.percentile(self.df['Total'], [75, 25])
                iqr = q75 - q25
                upper_bound = q75 + 3 * iqr
                
                if row['Total'] > upper_bound:
                    extreme_outliers.append({
                        'index': idx,
                        'diagnosis': row['Dx'],
                        'total': row['Total'],
                        'z_score': (row['Total'] - self.df['Total'].mean()) / self.df['Total'].std()
                    })
        
        if inconsistent_totals:
            self.issues.append({
                'type': 'inconsistent_totals',
                'severity': 'high',
                'count': len(inconsistent_totals),
                'details': inconsistent_totals
            })
        
        if negative_counts:
            self.issues.append({
                'type': 'negative_counts',
                'severity': 'high',
                'count': len(negative_counts),
                'details': negative_counts
            })
        
        if extreme_outliers:
            self.issues.append({
                'type': 'extreme_outliers',
                'severity': 'medium',
                'count': len(extreme_outliers),
                'details': extreme_outliers
            })
        
        self.audit_results['case_count_validation'] = {
            'inconsistent_totals': len(inconsistent_totals),
            'negative_counts': len(negative_counts),
            'extreme_outliers': len(extreme_outliers),
            'status': 'pass' if all(len(x) == 0 for x in [inconsistent_totals, negative_counts]) else 'fail'
        }
    
    def check_missing_data(self):
        """Check for missing or null data"""
        print("Checking for missing data...")
        
        missing_summary = {}
        critical_missing = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            missing_summary[col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            # Critical columns should not have missing data
            if col in ['Dx', 'SNOMED CT Code'] and missing_count > 0:
                critical_missing.append({
                    'column': col,
                    'missing_count': missing_count,
                    'percentage': missing_pct
                })
        
        if critical_missing:
            self.issues.append({
                'type': 'critical_missing_data',
                'severity': 'high',
                'count': len(critical_missing),
                'details': critical_missing
            })
        
        self.audit_results['missing_data'] = {
            'summary': missing_summary,
            'critical_missing': len(critical_missing),
            'status': 'pass' if len(critical_missing) == 0 else 'fail'
        }
    
    def detect_format_inconsistencies(self):
        """Detect format inconsistencies in text fields"""
        print("Detecting format inconsistencies...")
        
        format_issues = []
        
        # Check diagnosis text format
        for idx, dx in enumerate(self.df['Dx']):
            if pd.notna(dx):
                # Check for unusual patterns
                if re.search(r'^[^a-zA-Z]', str(dx)):  # Starts with non-letter
                    format_issues.append({
                        'index': idx,
                        'field': 'Dx',
                        'value': dx,
                        'issue': 'starts_with_non_letter'
                    })
                
                if len(str(dx)) < 3:  # Very short diagnosis
                    format_issues.append({
                        'index': idx,
                        'field': 'Dx',
                        'value': dx,
                        'issue': 'too_short'
                    })
                
                if str(dx).isupper() or str(dx).islower():  # All caps or all lowercase
                    format_issues.append({
                        'index': idx,
                        'field': 'Dx',
                        'value': dx,
                        'issue': 'case_consistency'
                    })
        
        if format_issues:
            self.issues.append({
                'type': 'format_inconsistencies',
                'severity': 'low',
                'count': len(format_issues),
                'details': format_issues
            })
        
        self.audit_results['format_consistency'] = {
            'issues_found': len(format_issues),
            'status': 'pass' if len(format_issues) == 0 else 'warning'
        }
    
    def analyze_distribution_anomalies(self):
        """Analyze statistical distribution anomalies"""
        print("Analyzing distribution anomalies...")
        
        dataset_cols = [col for col in self.df.columns 
                       if col not in ['Dx', 'SNOMED CT Code', 'Abbreviation', 'Total']]
        
        distribution_anomalies = []
        
        # Check for datasets with suspiciously low/high case counts
        for col in dataset_cols:
            col_sum = self.df[col].sum()
            col_mean = self.df[col].mean()
            col_std = self.df[col].std()
            
            # Check if dataset has extremely low contribution
            total_cases = self.df['Total'].sum()
            contribution_pct = (col_sum / total_cases) * 100
            
            if contribution_pct < 1 and col_sum > 0:  # Less than 1% contribution
                distribution_anomalies.append({
                    'dataset': col,
                    'issue': 'low_contribution',
                    'contribution_pct': contribution_pct,
                    'total_cases': col_sum
                })
            
            # Check for datasets with only a few diagnoses having all the cases
            non_zero_count = (self.df[col] > 0).sum()
            if non_zero_count < len(self.df) * 0.1 and col_sum > 100:  # Less than 10% of diagnoses
                distribution_anomalies.append({
                    'dataset': col,
                    'issue': 'concentrated_distribution',
                    'diagnoses_with_cases': non_zero_count,
                    'total_diagnoses': len(self.df),
                    'concentration_pct': (non_zero_count / len(self.df)) * 100
                })
        
        if distribution_anomalies:
            self.issues.append({
                'type': 'distribution_anomalies',
                'severity': 'medium',
                'count': len(distribution_anomalies),
                'details': distribution_anomalies
            })
        
        self.audit_results['distribution_analysis'] = {
            'anomalies_found': len(distribution_anomalies),
            'status': 'pass' if len(distribution_anomalies) == 0 else 'warning'
        }
    
    def check_semantic_consistency(self):
        """Check semantic consistency between diagnosis and SNOMED mapping"""
        print("Checking semantic consistency...")
        
        # This is a simplified semantic check
        # In a real implementation, you'd use medical ontologies
        
        semantic_issues = []
        
        # Basic keyword matching for obvious mismatches
        cardiac_keywords = ['heart', 'cardiac', 'myocardial', 'atrial', 'ventricular', 'av block']
        respiratory_keywords = ['lung', 'respiratory', 'pneumonia', 'asthma', 'copd']
        
        for idx, row in self.df.iterrows():
            dx_lower = str(row['Dx']).lower()
            
            # Simple heuristic checks
            if any(keyword in dx_lower for keyword in cardiac_keywords):
                # Should probably have cardiac-related SNOMED codes
                # This is a simplified check - real implementation would use SNOMED hierarchy
                pass
            
            # Check for abbreviation-diagnosis mismatch
            abbrev = str(row['Abbreviation']).lower()
            if abbrev and abbrev != 'nan':
                # Check if abbreviation makes sense for diagnosis
                dx_words = dx_lower.split()
                abbrev_chars = list(abbrev)
                
                # Simple check: first letters should somewhat match
                if len(dx_words) >= 2 and len(abbrev) >= 2:
                    first_letters = ''.join([word[0] for word in dx_words[:3]])
                    if not any(char in first_letters for char in abbrev_chars[:2]):
                        semantic_issues.append({
                            'index': idx,
                            'diagnosis': row['Dx'],
                            'abbreviation': abbrev,
                            'issue': 'abbreviation_mismatch'
                        })
        
        if semantic_issues:
            self.issues.append({
                'type': 'semantic_inconsistencies',
                'severity': 'low',
                'count': len(semantic_issues),
                'details': semantic_issues
            })
        
        self.audit_results['semantic_consistency'] = {
            'issues_found': len(semantic_issues),
            'status': 'pass' if len(semantic_issues) == 0 else 'warning'
        }
    
    def calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        total_records = len(self.df)
        
        # Weight different issue types
        weights = {
            'high': 10,
            'medium': 5,
            'low': 2
        }
        
        total_penalty = 0
        for issue in self.issues:
            penalty = issue['count'] * weights[issue['severity']]
            total_penalty += penalty
        
        # Calculate score (higher penalty = lower score)
        max_possible_penalty = total_records * weights['high']
        quality_score = max(0, 100 - (total_penalty / max_possible_penalty) * 100)
        
        return round(quality_score, 2)
    
    def get_quality_report(self) -> str:
        """Generate human-readable quality report"""
        if not self.audit_results:
            return "No audit results available. Run full_audit() first."
        
        report = []
        report.append("=" * 60)
        report.append("SNOMED DATA QUALITY AUDIT REPORT")
        report.append("=" * 60)
        report.append(f"Dataset Size: {len(self.df)} diagnoses")
        report.append(f"Overall Quality Score: {self.audit_results['data_quality_score']}/100")
        report.append(f"Total Issues Found: {self.audit_results['total_issues']}")
        report.append("")
        
        # Categorize issues by severity
        high_severity = [issue for issue in self.issues if issue['severity'] == 'high']
        medium_severity = [issue for issue in self.issues if issue['severity'] == 'medium']
        low_severity = [issue for issue in self.issues if issue['severity'] == 'low']
        
        if high_severity:
            report.append("🚨 HIGH SEVERITY ISSUES:")
            for issue in high_severity:
                report.append(f"  - {issue['type']}: {issue['count']} instances")
            report.append("")
        
        if medium_severity:
            report.append("⚠️  MEDIUM SEVERITY ISSUES:")
            for issue in medium_severity:
                report.append(f"  - {issue['type']}: {issue['count']} instances")
            report.append("")
        
        if low_severity:
            report.append("ℹ️  LOW SEVERITY ISSUES:")
            for issue in low_severity:
                report.append(f"  - {issue['type']}: {issue['count']} instances")
            report.append("")
        
        # Add specific recommendations
        report.append("RECOMMENDATIONS:")
        if high_severity:
            report.append("  1. Address high severity issues immediately")
        if any(issue['type'] == 'inconsistent_totals' for issue in self.issues):
            report.append("  2. Recalculate and verify total case counts")
        if any(issue['type'] == 'duplicate_snomed_codes' for issue in self.issues):
            report.append("  3. Review SNOMED code mappings for consistency")
        
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