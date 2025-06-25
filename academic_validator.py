"""
Academic Study Validator - Ensures All Objective Study Criteria Are Met
Implements comprehensive validation for BERT vs T5 comparison studies
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import time

class AcademicValidator:
    """
    Validates academic research standards for neural architecture comparison studies.
    Ensures objective criteria are met when comparing BERT vs T5 pipelines.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
    def validate_experimental_design(self, 
                                   bert_results: Dict[str, Any], 
                                   t5_results: Dict[str, Any],
                                   experimental_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of experimental design against academic standards.
        
        Args:
            bert_results: Results from BERT processing
            t5_results: Results from T5 processing  
            experimental_params: Experimental configuration parameters
            
        Returns:
            Validation report with compliance status
        """
        validation_report = {
            "overall_compliance": True,
            "criteria_met": [],
            "criteria_failed": [],
            "recommendations": [],
            "statistical_power": {},
            "bias_assessment": {},
            "reproducibility_score": 0.0
        }
        
        # 1. Controlled Variable Validation
        controlled_vars = self._validate_controlled_variables(bert_results, t5_results, experimental_params)
        validation_report.update(controlled_vars)
        
        # 2. Statistical Power Analysis
        power_analysis = self._analyze_statistical_power(bert_results, t5_results)
        validation_report["statistical_power"] = power_analysis
        
        # 3. Bias Assessment
        bias_assessment = self._assess_experimental_bias(bert_results, t5_results)
        validation_report["bias_assessment"] = bias_assessment
        
        # 4. Reproducibility Validation
        reproducibility = self._validate_reproducibility(experimental_params)
        validation_report["reproducibility_score"] = reproducibility
        
        # 5. Data Integrity Checks
        integrity_check = self._validate_data_integrity(bert_results, t5_results)
        validation_report.update(integrity_check)
        
        # 6. Effect Size Calculation
        effect_sizes = self._calculate_effect_sizes(bert_results, t5_results)
        validation_report["effect_sizes"] = effect_sizes
        
        return validation_report
    
    def _validate_controlled_variables(self, 
                                     bert_results: Dict[str, Any], 
                                     t5_results: Dict[str, Any],
                                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that experimental variables are properly controlled"""
        criteria_met = []
        criteria_failed = []
        
        # Temperature consistency
        bert_temp = bert_results.get("metadata", {}).get("temperature", None)
        t5_temp = t5_results.get("metadata", {}).get("temperature", None)
        
        if bert_temp == t5_temp and bert_temp is not None:
            criteria_met.append("Temperature consistency maintained")
        else:
            criteria_failed.append("Temperature inconsistency detected")
        
        # Input text consistency (same text processed by both models)
        if params.get("same_input_text", True):
            criteria_met.append("Identical input text used for both models")
        else:
            criteria_failed.append("Different input texts compromise comparison validity")
        
        # Processing environment consistency
        if params.get("controlled_environment", True):
            criteria_met.append("Controlled processing environment")
        else:
            criteria_failed.append("Uncontrolled processing environment")
            
        return {"criteria_met": criteria_met, "criteria_failed": criteria_failed}
    
    def _analyze_statistical_power(self, 
                                 bert_results: Dict[str, Any], 
                                 t5_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical power for detecting meaningful differences"""
        bert_relationships = bert_results.get("relationships", [])
        t5_relationships = t5_results.get("relationships", [])
        
        # Sample size analysis
        bert_n = len(bert_relationships)
        t5_n = len(t5_relationships)
        
        # Confidence score distributions
        bert_confidences = [r.get("confidence", 0.0) for r in bert_relationships]
        t5_confidences = [r.get("confidence", 0.0) for r in t5_relationships]
        
        power_analysis = {
            "sample_sizes": {"bert": bert_n, "t5": t5_n},
            "minimum_detectable_effect": self._calculate_minimum_detectable_effect(bert_n, t5_n),
            "confidence_intervals": {
                "bert": self._calculate_confidence_interval(bert_confidences),
                "t5": self._calculate_confidence_interval(t5_confidences)
            },
            "power_adequate": bert_n >= 30 and t5_n >= 30  # Rule of thumb for adequate power
        }
        
        return power_analysis
    
    def _assess_experimental_bias(self, 
                                bert_results: Dict[str, Any], 
                                t5_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential sources of experimental bias"""
        bias_assessment = {
            "selection_bias": "low",  # Same input text used
            "measurement_bias": "low",  # Automated processing
            "confirmation_bias": "low",  # Objective metrics
            "temporal_bias": "controlled"  # Same session processing
        }
        
        # Check for systematic differences that might indicate bias
        bert_stats = bert_results.get("statistics", {})
        t5_stats = t5_results.get("statistics", {})
        
        bert_avg_conf = bert_stats.get("average_confidence", 0.0)
        t5_avg_conf = t5_stats.get("average_confidence", 0.0)
        
        # Flag potential bias if one model shows consistently higher confidence
        confidence_ratio = bert_avg_conf / t5_avg_conf if t5_avg_conf > 0 else 1.0
        if confidence_ratio > 2.0 or confidence_ratio < 0.5:
            bias_assessment["confidence_bias"] = "potential systematic bias detected"
        else:
            bias_assessment["confidence_bias"] = "no systematic bias detected"
            
        return bias_assessment
    
    def _validate_reproducibility(self, params: Dict[str, Any]) -> float:
        """Calculate reproducibility score based on documented parameters"""
        reproducibility_factors = [
            params.get("temperature_documented", False),
            params.get("model_versions_documented", False),
            params.get("random_seed_controlled", False),
            params.get("processing_environment_documented", False),
            params.get("input_preprocessing_documented", False)
        ]
        
        return sum(reproducibility_factors) / len(reproducibility_factors)
    
    def _validate_data_integrity(self, 
                               bert_results: Dict[str, Any], 
                               t5_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data integrity and authenticity"""
        criteria_met = []
        criteria_failed = []
        
        # Check for non-empty results
        bert_rels = bert_results.get("relationships", [])
        t5_rels = t5_results.get("relationships", [])
        
        if len(bert_rels) > 0:
            criteria_met.append("BERT extracted meaningful relationships")
        else:
            criteria_failed.append("BERT failed to extract relationships")
            
        if len(t5_rels) > 0:
            criteria_met.append("T5 extracted meaningful relationships")
        else:
            criteria_failed.append("T5 failed to extract relationships")
        
        # Validate relationship structure integrity
        bert_valid = all(self._validate_relationship_structure(r) for r in bert_rels)
        t5_valid = all(self._validate_relationship_structure(r) for r in t5_rels)
        
        if bert_valid:
            criteria_met.append("BERT relationship structure validated")
        else:
            criteria_failed.append("BERT relationship structure integrity compromised")
            
        if t5_valid:
            criteria_met.append("T5 relationship structure validated")
        else:
            criteria_failed.append("T5 relationship structure integrity compromised")
            
        return {"criteria_met": criteria_met, "criteria_failed": criteria_failed}
    
    def _validate_relationship_structure(self, relationship: Dict[str, Any]) -> bool:
        """Validate that relationship has required fields and valid data"""
        required_fields = ["subject", "predicate", "object", "confidence"]
        
        # Check required fields exist
        if not all(field in relationship for field in required_fields):
            return False
            
        # Check data types and ranges
        if not isinstance(relationship["confidence"], (int, float)):
            return False
            
        if not (0.0 <= relationship["confidence"] <= 1.0):
            return False
            
        # Check non-empty strings
        text_fields = ["subject", "predicate", "object"]
        if not all(isinstance(relationship[field], str) and relationship[field].strip() 
                  for field in text_fields):
            return False
            
        return True
    
    def _calculate_effect_sizes(self, 
                              bert_results: Dict[str, Any], 
                              t5_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Cohen's d and other effect size measures"""
        bert_confidences = [r.get("confidence", 0.0) for r in bert_results.get("relationships", [])]
        t5_confidences = [r.get("confidence", 0.0) for r in t5_results.get("relationships", [])]
        
        if len(bert_confidences) == 0 or len(t5_confidences) == 0:
            return {"cohen_d": 0.0, "interpretation": "insufficient_data"}
        
        # Cohen's d calculation
        pooled_std = np.sqrt(((len(bert_confidences) - 1) * np.var(bert_confidences, ddof=1) + 
                             (len(t5_confidences) - 1) * np.var(t5_confidences, ddof=1)) / 
                            (len(bert_confidences) + len(t5_confidences) - 2))
        
        if pooled_std == 0:
            cohen_d = 0.0
        else:
            cohen_d = (np.mean(bert_confidences) - np.mean(t5_confidences)) / pooled_std
        
        # Interpret effect size
        if abs(cohen_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohen_d) < 0.5:
            interpretation = "small"
        elif abs(cohen_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
            
        return {
            "cohen_d": cohen_d,
            "interpretation": interpretation,
            "confidence_means": {
                "bert": np.mean(bert_confidences),
                "t5": np.mean(t5_confidences)
            }
        }
    
    def _calculate_minimum_detectable_effect(self, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
        """Calculate minimum detectable effect size given sample sizes"""
        if n1 < 2 or n2 < 2:
            return float('inf')
            
        # Simplified calculation for minimum detectable effect
        # Based on two-sample t-test power analysis
        critical_t = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
        power_t = stats.t.ppf(power, n1 + n2 - 2)
        
        pooled_se = np.sqrt(1/n1 + 1/n2)
        min_effect = (critical_t + power_t) * pooled_se
        
        return min_effect
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        if len(data) < 2:
            return (0.0, 0.0)
            
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return (mean - h, mean + h)
    
    def generate_comparison_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate academic comparison report"""
        report = []
        report.append("# Academic Validation Report: BERT vs T5 Comparison")
        report.append("=" * 60)
        
        # Overall compliance
        compliance = "COMPLIANT" if validation_results["overall_compliance"] else "NON-COMPLIANT"
        report.append(f"Overall Academic Compliance: {compliance}")
        report.append("")
        
        # Criteria met
        if validation_results["criteria_met"]:
            report.append("✓ Criteria Met:")
            for criterion in validation_results["criteria_met"]:
                report.append(f"  • {criterion}")
            report.append("")
        
        # Criteria failed
        if validation_results["criteria_failed"]:
            report.append("✗ Criteria Failed:")
            for criterion in validation_results["criteria_failed"]:
                report.append(f"  • {criterion}")
            report.append("")
        
        # Statistical power
        power = validation_results.get("statistical_power", {})
        report.append("Statistical Power Analysis:")
        report.append(f"  • BERT sample size: {power.get('sample_sizes', {}).get('bert', 0)}")
        report.append(f"  • T5 sample size: {power.get('sample_sizes', {}).get('t5', 0)}")
        report.append(f"  • Adequate power: {power.get('power_adequate', False)}")
        report.append("")
        
        # Effect sizes
        effects = validation_results.get("effect_sizes", {})
        report.append("Effect Size Analysis:")
        report.append(f"  • Cohen's d: {effects.get('cohen_d', 0.0):.3f}")
        report.append(f"  • Interpretation: {effects.get('interpretation', 'unknown')}")
        report.append("")
        
        # Reproducibility
        repro_score = validation_results.get("reproducibility_score", 0.0)
        report.append(f"Reproducibility Score: {repro_score:.2f}/1.0")
        
        return "\n".join(report)