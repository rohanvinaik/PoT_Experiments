"""
AI Risk Assessment Framework for PoT Governance
Comprehensive risk identification, assessment, and mitigation
"""

import json
import yaml
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import hashlib
import logging


class RiskCategory(Enum):
    """Risk categories for AI systems"""
    BIAS_DISCRIMINATION = "bias_discrimination"
    PRIVACY_BREACH = "privacy_breach"
    SAFETY_HARM = "safety_harm"
    SECURITY_VULNERABILITY = "security_vulnerability"
    TRANSPARENCY_DEFICIT = "transparency_deficit"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    OPERATIONAL = "operational"
    REPUTATIONAL = "reputational"
    FINANCIAL = "financial"
    # PoT-specific categories
    VERIFICATION_FAILURE = "verification_failure"
    CHALLENGE_COMPROMISE = "challenge_compromise"
    MODEL_DRIFT = "model_drift"
    ADVERSARIAL_ATTACK = "adversarial_attack"


class RiskLikelihood(Enum):
    """Risk likelihood levels"""
    RARE = (1, "rare", 0.05)              # <5% chance
    UNLIKELY = (2, "unlikely", 0.25)       # 5-25% chance
    POSSIBLE = (3, "possible", 0.50)       # 25-50% chance
    LIKELY = (4, "likely", 0.75)           # 50-75% chance
    ALMOST_CERTAIN = (5, "almost_certain", 0.95)  # >75% chance
    
    def __init__(self, level: int, label: str, probability: float):
        self.level = level
        self.label = label
        self.probability = probability


class RiskImpact(Enum):
    """Risk impact levels"""
    NEGLIGIBLE = (1, "negligible", 0.1)    # Minimal impact
    MINOR = (2, "minor", 0.3)              # Limited impact
    MODERATE = (3, "moderate", 0.5)        # Noticeable impact
    MAJOR = (4, "major", 0.7)              # Significant impact
    SEVERE = (5, "severe", 0.9)            # Critical impact
    
    def __init__(self, level: int, label: str, severity: float):
        self.level = level
        self.label = label
        self.severity = severity


class RiskAppetite(Enum):
    """Risk appetite levels"""
    AVERSE = "averse"          # Minimal risk tolerance
    MINIMAL = "minimal"        # Low risk tolerance
    CAUTIOUS = "cautious"      # Moderate risk tolerance
    OPEN = "open"              # Higher risk tolerance
    HUNGRY = "hungry"          # Seeking risk for reward


class RiskTreatment(Enum):
    """Risk treatment strategies"""
    AVOID = "avoid"            # Eliminate the risk
    MITIGATE = "mitigate"      # Reduce likelihood or impact
    TRANSFER = "transfer"      # Share or transfer risk
    ACCEPT = "accept"          # Accept the risk
    MONITOR = "monitor"        # Monitor without action


@dataclass
class Risk:
    """Individual risk"""
    risk_id: str
    category: RiskCategory
    title: str
    description: str
    likelihood: RiskLikelihood
    impact: RiskImpact
    inherent_score: float      # Before controls
    controls: List[str]        # Existing controls
    residual_likelihood: RiskLikelihood
    residual_impact: RiskImpact
    residual_score: float      # After controls
    owner: str
    identified_date: datetime
    review_date: Optional[datetime] = None
    status: str = "active"     # active, mitigated, closed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['category'] = self.category.value
        result['likelihood'] = self.likelihood.label
        result['impact'] = self.impact.label
        result['residual_likelihood'] = self.residual_likelihood.label
        result['residual_impact'] = self.residual_impact.label
        result['identified_date'] = self.identified_date.isoformat()
        if self.review_date:
            result['review_date'] = self.review_date.isoformat()
        return result


@dataclass
class Mitigation:
    """Risk mitigation strategy"""
    mitigation_id: str
    risk_id: str
    treatment: RiskTreatment
    description: str
    actions: List[str]
    cost_estimate: Optional[float]
    effort_estimate: str  # low, medium, high
    effectiveness: float  # 0-1 scale
    implementation_date: Optional[datetime]
    responsible_party: str
    status: str  # planned, in_progress, implemented, verified
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['treatment'] = self.treatment.value
        if self.implementation_date:
            result['implementation_date'] = self.implementation_date.isoformat()
        return result


@dataclass
class RiskProfile:
    """Risk profile for a system or component"""
    profile_id: str
    assessment_date: datetime
    system_name: str
    risks: List[Risk]
    overall_risk_score: float
    risk_distribution: Dict[str, int]  # By category
    high_priority_risks: List[Risk]
    mitigations: List[Mitigation]
    risk_appetite: RiskAppetite
    acceptable_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['assessment_date'] = self.assessment_date.isoformat()
        result['risks'] = [r.to_dict() for r in self.risks]
        result['high_priority_risks'] = [r.to_dict() for r in self.high_priority_risks]
        result['mitigations'] = [m.to_dict() for m in self.mitigations]
        result['risk_appetite'] = self.risk_appetite.value
        return result


@dataclass
class RiskIncident:
    """Risk incident record"""
    incident_id: str
    risk_id: str
    occurred_date: datetime
    description: str
    actual_impact: str
    response_actions: List[str]
    lessons_learned: List[str]
    cost_incurred: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['occurred_date'] = self.occurred_date.isoformat()
        return result


class RiskMatrix:
    """Risk assessment matrix"""
    
    def __init__(self, appetite: RiskAppetite = RiskAppetite.CAUTIOUS):
        self.appetite = appetite
        self.matrix = self._create_matrix()
        self.thresholds = self._set_thresholds()
    
    def _create_matrix(self) -> np.ndarray:
        """Create 5x5 risk matrix"""
        # Risk scores increase with likelihood (rows) and impact (columns)
        return np.array([
            [1, 2, 3, 4, 5],      # Rare
            [2, 4, 6, 8, 10],     # Unlikely
            [3, 6, 9, 12, 15],    # Possible
            [4, 8, 12, 16, 20],   # Likely
            [5, 10, 15, 20, 25]   # Almost Certain
        ])
    
    def _set_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Set risk thresholds based on appetite"""
        if self.appetite == RiskAppetite.AVERSE:
            return {
                "low": (0, 3),
                "medium": (4, 8),
                "high": (9, 15),
                "critical": (16, 25)
            }
        elif self.appetite == RiskAppetite.MINIMAL:
            return {
                "low": (0, 4),
                "medium": (5, 10),
                "high": (11, 17),
                "critical": (18, 25)
            }
        elif self.appetite == RiskAppetite.CAUTIOUS:
            return {
                "low": (0, 5),
                "medium": (6, 12),
                "high": (13, 19),
                "critical": (20, 25)
            }
        elif self.appetite == RiskAppetite.OPEN:
            return {
                "low": (0, 6),
                "medium": (7, 15),
                "high": (16, 22),
                "critical": (23, 25)
            }
        else:  # HUNGRY
            return {
                "low": (0, 8),
                "medium": (9, 17),
                "high": (18, 23),
                "critical": (24, 25)
            }
    
    def calculate_score(self, likelihood: RiskLikelihood, impact: RiskImpact) -> float:
        """Calculate risk score from likelihood and impact"""
        return self.matrix[likelihood.level - 1, impact.level - 1]
    
    def get_risk_level(self, score: float) -> str:
        """Get risk level from score"""
        for level, (min_score, max_score) in self.thresholds.items():
            if min_score <= score <= max_score:
                return level
        return "critical"


class AIRiskAssessment:
    """
    Comprehensive AI risk assessment framework
    """
    
    def __init__(self, risk_appetite: RiskAppetite = RiskAppetite.CAUTIOUS):
        """
        Initialize risk assessment framework
        
        Args:
            risk_appetite: Organization's risk appetite
        """
        self.risk_appetite = risk_appetite
        self.risk_matrix = RiskMatrix(risk_appetite)
        self.risk_register: List[Risk] = []
        self.mitigations: List[Mitigation] = []
        self.incidents: List[RiskIncident] = []
        self.logger = logging.getLogger(__name__)
        
        # Risk categories
        self.risk_categories = [
            RiskCategory.BIAS_DISCRIMINATION,
            RiskCategory.PRIVACY_BREACH,
            RiskCategory.SAFETY_HARM,
            RiskCategory.SECURITY_VULNERABILITY,
            RiskCategory.TRANSPARENCY_DEFICIT,
            RiskCategory.PERFORMANCE_DEGRADATION,
            RiskCategory.VERIFICATION_FAILURE,
            RiskCategory.CHALLENGE_COMPROMISE,
            RiskCategory.MODEL_DRIFT,
            RiskCategory.ADVERSARIAL_ATTACK
        ]
        
        # Risk catalog
        self.risk_catalog = self._load_risk_catalog()
    
    def _load_risk_catalog(self) -> Dict[RiskCategory, List[Dict[str, Any]]]:
        """Load predefined risk catalog"""
        catalog = {
            RiskCategory.BIAS_DISCRIMINATION: [
                {
                    "title": "Algorithmic Bias",
                    "description": "Model exhibits discriminatory behavior against protected groups",
                    "indicators": ["disparate impact", "unequal error rates", "representation gaps"]
                },
                {
                    "title": "Training Data Bias",
                    "description": "Training data contains historical biases",
                    "indicators": ["skewed demographics", "sampling bias", "label bias"]
                }
            ],
            RiskCategory.PRIVACY_BREACH: [
                {
                    "title": "Data Leakage",
                    "description": "Model memorizes and exposes training data",
                    "indicators": ["membership inference success", "data extraction", "PII exposure"]
                },
                {
                    "title": "Re-identification Risk",
                    "description": "Anonymized data can be re-identified",
                    "indicators": ["linkage attacks", "inference attacks", "auxiliary data"]
                }
            ],
            RiskCategory.SAFETY_HARM: [
                {
                    "title": "Unsafe Outputs",
                    "description": "Model generates harmful or dangerous content",
                    "indicators": ["toxic content", "dangerous advice", "self-harm content"]
                },
                {
                    "title": "Critical System Failure",
                    "description": "Model failure in safety-critical application",
                    "indicators": ["false negatives in medical", "autonomous vehicle errors"]
                }
            ],
            RiskCategory.SECURITY_VULNERABILITY: [
                {
                    "title": "Adversarial Attacks",
                    "description": "Model vulnerable to adversarial examples",
                    "indicators": ["low robustness score", "successful attacks", "transferability"]
                },
                {
                    "title": "Model Extraction",
                    "description": "Model can be stolen through queries",
                    "indicators": ["high fidelity copies", "functionality theft", "IP loss"]
                }
            ],
            RiskCategory.TRANSPARENCY_DEFICIT: [
                {
                    "title": "Black Box Decision Making",
                    "description": "Model decisions cannot be explained",
                    "indicators": ["no interpretability", "regulatory non-compliance", "user distrust"]
                },
                {
                    "title": "Insufficient Documentation",
                    "description": "Model lacks proper documentation",
                    "indicators": ["missing model cards", "no limitations documented", "unclear purpose"]
                }
            ],
            RiskCategory.PERFORMANCE_DEGRADATION: [
                {
                    "title": "Model Drift",
                    "description": "Model performance degrades over time",
                    "indicators": ["accuracy decline", "distribution shift", "concept drift"]
                },
                {
                    "title": "Edge Case Failures",
                    "description": "Model fails on rare but important cases",
                    "indicators": ["long tail errors", "OOD failures", "corner cases"]
                }
            ],
            RiskCategory.VERIFICATION_FAILURE: [
                {
                    "title": "Failed PoT Verification",
                    "description": "Model fails proof-of-training verification",
                    "indicators": ["low confidence score", "fingerprint mismatch", "challenge failure"]
                },
                {
                    "title": "Incomplete Verification Coverage",
                    "description": "Verification doesn't cover all aspects",
                    "indicators": ["untested behaviors", "limited challenges", "gaps in coverage"]
                }
            ],
            RiskCategory.CHALLENGE_COMPROMISE: [
                {
                    "title": "Challenge Leakage",
                    "description": "Verification challenges are compromised",
                    "indicators": ["repeated use", "public exposure", "predictable patterns"]
                },
                {
                    "title": "Challenge Gaming",
                    "description": "Model trained specifically to pass challenges",
                    "indicators": ["overfitting to challenges", "normal performance gap", "suspicious patterns"]
                }
            ],
            RiskCategory.MODEL_DRIFT: [
                {
                    "title": "Distribution Shift",
                    "description": "Input distribution changes from training",
                    "indicators": ["statistical tests", "performance metrics", "user reports"]
                },
                {
                    "title": "Concept Drift",
                    "description": "Underlying patterns change over time",
                    "indicators": ["prediction errors", "feedback loops", "temporal patterns"]
                }
            ],
            RiskCategory.ADVERSARIAL_ATTACK: [
                {
                    "title": "Targeted Manipulation",
                    "description": "Deliberate attacks to cause misclassification",
                    "indicators": ["attack success rate", "perturbation size", "detection rate"]
                },
                {
                    "title": "Poisoning Attacks",
                    "description": "Training data or model poisoned",
                    "indicators": ["backdoor triggers", "degraded performance", "suspicious patterns"]
                }
            ]
        }
        
        return catalog
    
    def assess_model_risks(self, model_config: Dict[str, Any]) -> RiskProfile:
        """
        Comprehensive model risk assessment
        
        Args:
            model_config: Model configuration and metadata
            
        Returns:
            RiskProfile with identified risks
        """
        risks = []
        
        # Assess each risk category
        for category in self.risk_categories:
            category_risks = self._assess_category_risks(category, model_config)
            risks.extend(category_risks)
        
        # Calculate overall risk score
        overall_score = self._calculate_overall_score(risks)
        
        # Identify high priority risks
        high_priority = [r for r in risks if r.residual_score >= 15]
        
        # Create risk distribution
        risk_distribution = Counter(r.category.value for r in risks)
        
        # Generate mitigations
        mitigations = self.recommend_mitigations(risks)
        
        profile = RiskProfile(
            profile_id=f"PROFILE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            assessment_date=datetime.now(),
            system_name=model_config.get("model_name", "unknown"),
            risks=risks,
            overall_risk_score=overall_score,
            risk_distribution=dict(risk_distribution),
            high_priority_risks=high_priority,
            mitigations=mitigations,
            risk_appetite=self.risk_appetite,
            acceptable_threshold=self._get_acceptable_threshold(),
            metadata=model_config
        )
        
        # Add to register
        self.risk_register.extend(risks)
        
        return profile
    
    def _assess_category_risks(
        self,
        category: RiskCategory,
        config: Dict[str, Any]
    ) -> List[Risk]:
        """Assess risks for a specific category"""
        risks = []
        
        if category not in self.risk_catalog:
            return risks
        
        for risk_template in self.risk_catalog[category]:
            # Assess likelihood and impact based on config
            likelihood = self._assess_likelihood(risk_template, config)
            impact = self._assess_impact(risk_template, config)
            
            # Calculate inherent risk
            inherent_score = self.risk_matrix.calculate_score(likelihood, impact)
            
            # Identify existing controls
            controls = self._identify_controls(category, config)
            
            # Calculate residual risk
            residual_likelihood, residual_impact = self._calculate_residual(
                likelihood, impact, controls
            )
            residual_score = self.risk_matrix.calculate_score(
                residual_likelihood, residual_impact
            )
            
            risk = Risk(
                risk_id=f"RISK-{category.value}-{len(self.risk_register):04d}",
                category=category,
                title=risk_template["title"],
                description=risk_template["description"],
                likelihood=likelihood,
                impact=impact,
                inherent_score=inherent_score,
                controls=controls,
                residual_likelihood=residual_likelihood,
                residual_impact=residual_impact,
                residual_score=residual_score,
                owner=config.get("risk_owner", "unassigned"),
                identified_date=datetime.now(),
                metadata={"indicators": risk_template.get("indicators", [])}
            )
            
            risks.append(risk)
        
        return risks
    
    def _assess_likelihood(
        self,
        risk_template: Dict[str, Any],
        config: Dict[str, Any]
    ) -> RiskLikelihood:
        """Assess likelihood of a risk"""
        # Base likelihood assessment on indicators and config
        indicators = risk_template.get("indicators", [])
        likelihood_score = 0
        
        # Check for risk indicators in config
        for indicator in indicators:
            if self._check_indicator(indicator, config):
                likelihood_score += 1
        
        # Map to likelihood level
        if likelihood_score == 0:
            return RiskLikelihood.RARE
        elif likelihood_score == 1:
            return RiskLikelihood.UNLIKELY
        elif likelihood_score == 2:
            return RiskLikelihood.POSSIBLE
        elif likelihood_score >= 3:
            return RiskLikelihood.LIKELY
        
        # Check specific conditions
        if "adversarial" in risk_template["title"].lower():
            if config.get("public_facing", False):
                return RiskLikelihood.LIKELY
        
        if "drift" in risk_template["title"].lower():
            if config.get("deployment_months", 0) > 6:
                return RiskLikelihood.LIKELY
        
        return RiskLikelihood.POSSIBLE
    
    def _assess_impact(
        self,
        risk_template: Dict[str, Any],
        config: Dict[str, Any]
    ) -> RiskImpact:
        """Assess impact of a risk"""
        # Base impact on risk type and system criticality
        criticality = config.get("criticality", "medium")
        risk_title = risk_template["title"].lower()
        
        # Safety-critical systems have higher impact
        if criticality == "critical":
            if "safety" in risk_title or "failure" in risk_title:
                return RiskImpact.SEVERE
            return RiskImpact.MAJOR
        
        # Privacy and security risks
        if "privacy" in risk_title or "breach" in risk_title:
            if config.get("handles_pii", False):
                return RiskImpact.SEVERE
            return RiskImpact.MAJOR
        
        # Bias and discrimination
        if "bias" in risk_title or "discrimination" in risk_title:
            if config.get("affects_individuals", True):
                return RiskImpact.MAJOR
            return RiskImpact.MODERATE
        
        # Performance degradation
        if "performance" in risk_title or "drift" in risk_title:
            return RiskImpact.MODERATE
        
        # Default assessment
        if criticality == "high":
            return RiskImpact.MAJOR
        elif criticality == "low":
            return RiskImpact.MINOR
        
        return RiskImpact.MODERATE
    
    def _check_indicator(self, indicator: str, config: Dict[str, Any]) -> bool:
        """Check if a risk indicator is present"""
        # Simple keyword matching - could be enhanced with more sophisticated checks
        indicator_lower = indicator.lower()
        
        # Check in config values
        for key, value in config.items():
            if isinstance(value, str) and indicator_lower in value.lower():
                return True
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and indicator_lower in item.lower():
                        return True
        
        # Check specific conditions
        if "disparate impact" in indicator_lower:
            return config.get("fairness_score", 1.0) < 0.8
        
        if "accuracy decline" in indicator_lower:
            return config.get("performance_trend", "stable") == "declining"
        
        return False
    
    def _identify_controls(
        self,
        category: RiskCategory,
        config: Dict[str, Any]
    ) -> List[str]:
        """Identify existing controls for a risk category"""
        controls = []
        
        # Common controls
        if config.get("monitoring_enabled", False):
            controls.append("Continuous monitoring")
        
        if config.get("audit_logging", False):
            controls.append("Audit logging")
        
        # Category-specific controls
        if category == RiskCategory.BIAS_DISCRIMINATION:
            if config.get("bias_testing", False):
                controls.append("Bias testing")
            if config.get("fairness_constraints", False):
                controls.append("Fairness constraints")
        
        elif category == RiskCategory.PRIVACY_BREACH:
            if config.get("differential_privacy", False):
                controls.append("Differential privacy")
            if config.get("encryption", False):
                controls.append("Data encryption")
        
        elif category == RiskCategory.SECURITY_VULNERABILITY:
            if config.get("adversarial_training", False):
                controls.append("Adversarial training")
            if config.get("input_validation", False):
                controls.append("Input validation")
        
        elif category == RiskCategory.PERFORMANCE_DEGRADATION:
            if config.get("drift_detection", False):
                controls.append("Drift detection")
            if config.get("automated_retraining", False):
                controls.append("Automated retraining")
        
        elif category == RiskCategory.VERIFICATION_FAILURE:
            if config.get("pot_verification", False):
                controls.append("PoT verification")
            if config.get("continuous_verification", False):
                controls.append("Continuous verification")
        
        return controls
    
    def _calculate_residual(
        self,
        likelihood: RiskLikelihood,
        impact: RiskImpact,
        controls: List[str]
    ) -> Tuple[RiskLikelihood, RiskImpact]:
        """Calculate residual risk after controls"""
        # Each control reduces likelihood or impact
        control_effectiveness = len(controls) * 0.2  # 20% reduction per control
        control_effectiveness = min(control_effectiveness, 0.8)  # Max 80% reduction
        
        # Reduce likelihood
        new_likelihood_level = max(1, likelihood.level - int(control_effectiveness * 2))
        residual_likelihood = list(RiskLikelihood)[new_likelihood_level - 1]
        
        # Reduce impact (less than likelihood)
        new_impact_level = max(1, impact.level - int(control_effectiveness))
        residual_impact = list(RiskImpact)[new_impact_level - 1]
        
        return residual_likelihood, residual_impact
    
    def assess_data_risks(self, data_profile: Dict[str, Any]) -> RiskProfile:
        """
        Data-related risk evaluation
        
        Args:
            data_profile: Data characteristics and metadata
            
        Returns:
            RiskProfile for data risks
        """
        risks = []
        
        # Data quality risks
        if data_profile.get("completeness", 1.0) < 0.9:
            risks.append(self._create_data_risk(
                "Incomplete Data",
                "Dataset has significant missing values",
                RiskLikelihood.LIKELY,
                RiskImpact.MODERATE
            ))
        
        # Data bias risks
        if data_profile.get("class_imbalance", 0) > 0.7:
            risks.append(self._create_data_risk(
                "Class Imbalance",
                "Severe class imbalance in training data",
                RiskLikelihood.ALMOST_CERTAIN,
                RiskImpact.MAJOR
            ))
        
        # Privacy risks
        if data_profile.get("contains_pii", False):
            risks.append(self._create_data_risk(
                "PII Exposure",
                "Dataset contains personally identifiable information",
                RiskLikelihood.POSSIBLE,
                RiskImpact.SEVERE
            ))
        
        # Data drift risks
        if data_profile.get("age_days", 0) > 180:
            risks.append(self._create_data_risk(
                "Stale Data",
                "Dataset is older than 6 months",
                RiskLikelihood.LIKELY,
                RiskImpact.MODERATE
            ))
        
        # Security risks
        if not data_profile.get("encrypted_at_rest", False):
            risks.append(self._create_data_risk(
                "Unencrypted Data",
                "Data not encrypted at rest",
                RiskLikelihood.POSSIBLE,
                RiskImpact.MAJOR
            ))
        
        # Calculate scores and create profile
        overall_score = self._calculate_overall_score(risks)
        high_priority = [r for r in risks if r.residual_score >= 12]
        
        profile = RiskProfile(
            profile_id=f"DATA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            assessment_date=datetime.now(),
            system_name=data_profile.get("dataset_name", "unknown"),
            risks=risks,
            overall_risk_score=overall_score,
            risk_distribution=Counter(r.category.value for r in risks),
            high_priority_risks=high_priority,
            mitigations=self.recommend_mitigations(risks),
            risk_appetite=self.risk_appetite,
            acceptable_threshold=self._get_acceptable_threshold(),
            metadata=data_profile
        )
        
        return profile
    
    def _create_data_risk(
        self,
        title: str,
        description: str,
        likelihood: RiskLikelihood,
        impact: RiskImpact
    ) -> Risk:
        """Create a data-related risk"""
        inherent_score = self.risk_matrix.calculate_score(likelihood, impact)
        
        return Risk(
            risk_id=f"DATA-RISK-{len(self.risk_register):04d}",
            category=RiskCategory.PRIVACY_BREACH,
            title=title,
            description=description,
            likelihood=likelihood,
            impact=impact,
            inherent_score=inherent_score,
            controls=[],
            residual_likelihood=likelihood,
            residual_impact=impact,
            residual_score=inherent_score,
            owner="data_team",
            identified_date=datetime.now()
        )
    
    def calculate_risk_score(self, likelihood: float, impact: float) -> float:
        """
        Risk scoring methodology
        
        Args:
            likelihood: Probability (0-1)
            impact: Severity (0-1)
            
        Returns:
            Risk score (0-25)
        """
        # Map to discrete levels
        likelihood_level = self._map_to_likelihood(likelihood)
        impact_level = self._map_to_impact(impact)
        
        # Use risk matrix
        return self.risk_matrix.calculate_score(likelihood_level, impact_level)
    
    def _map_to_likelihood(self, probability: float) -> RiskLikelihood:
        """Map probability to likelihood level"""
        if probability < 0.05:
            return RiskLikelihood.RARE
        elif probability < 0.25:
            return RiskLikelihood.UNLIKELY
        elif probability < 0.50:
            return RiskLikelihood.POSSIBLE
        elif probability < 0.75:
            return RiskLikelihood.LIKELY
        else:
            return RiskLikelihood.ALMOST_CERTAIN
    
    def _map_to_impact(self, severity: float) -> RiskImpact:
        """Map severity to impact level"""
        if severity < 0.2:
            return RiskImpact.NEGLIGIBLE
        elif severity < 0.4:
            return RiskImpact.MINOR
        elif severity < 0.6:
            return RiskImpact.MODERATE
        elif severity < 0.8:
            return RiskImpact.MAJOR
        else:
            return RiskImpact.SEVERE
    
    def recommend_mitigations(self, risks: List[Risk]) -> List[Mitigation]:
        """
        Suggest risk mitigation strategies
        
        Args:
            risks: List of identified risks
            
        Returns:
            List of mitigation recommendations
        """
        mitigations = []
        
        for risk in risks:
            # Only mitigate risks above threshold
            if risk.residual_score < self._get_mitigation_threshold():
                continue
            
            # Determine treatment strategy
            treatment = self._determine_treatment(risk)
            
            # Generate mitigation actions
            actions = self._generate_mitigation_actions(risk)
            
            mitigation = Mitigation(
                mitigation_id=f"MIT-{len(self.mitigations):04d}",
                risk_id=risk.risk_id,
                treatment=treatment,
                description=f"Mitigation for {risk.title}",
                actions=actions,
                cost_estimate=self._estimate_cost(actions),
                effort_estimate=self._estimate_effort(actions),
                effectiveness=self._estimate_effectiveness(risk, actions),
                implementation_date=None,
                responsible_party=risk.owner,
                status="planned"
            )
            
            mitigations.append(mitigation)
            self.mitigations.append(mitigation)
        
        return mitigations
    
    def _determine_treatment(self, risk: Risk) -> RiskTreatment:
        """Determine appropriate treatment strategy"""
        if risk.residual_score >= 20:
            return RiskTreatment.AVOID
        elif risk.residual_score >= 15:
            return RiskTreatment.MITIGATE
        elif risk.residual_score >= 10:
            return RiskTreatment.TRANSFER
        elif risk.residual_score >= 5:
            return RiskTreatment.MONITOR
        else:
            return RiskTreatment.ACCEPT
    
    def _generate_mitigation_actions(self, risk: Risk) -> List[str]:
        """Generate specific mitigation actions"""
        actions = []
        
        # Category-specific mitigations
        if risk.category == RiskCategory.BIAS_DISCRIMINATION:
            actions.extend([
                "Implement bias detection monitoring",
                "Apply fairness constraints in training",
                "Conduct regular fairness audits",
                "Diversify training data"
            ])
        
        elif risk.category == RiskCategory.PRIVACY_BREACH:
            actions.extend([
                "Implement differential privacy",
                "Apply data minimization",
                "Encrypt sensitive data",
                "Conduct privacy impact assessment"
            ])
        
        elif risk.category == RiskCategory.SECURITY_VULNERABILITY:
            actions.extend([
                "Implement adversarial training",
                "Add input validation layers",
                "Deploy anomaly detection",
                "Regular security testing"
            ])
        
        elif risk.category == RiskCategory.PERFORMANCE_DEGRADATION:
            actions.extend([
                "Implement drift detection",
                "Setup automated retraining",
                "Enhance monitoring coverage",
                "Define performance thresholds"
            ])
        
        elif risk.category == RiskCategory.VERIFICATION_FAILURE:
            actions.extend([
                "Enhance PoT verification coverage",
                "Increase challenge diversity",
                "Implement continuous verification",
                "Add redundant verification methods"
            ])
        
        elif risk.category == RiskCategory.CHALLENGE_COMPROMISE:
            actions.extend([
                "Rotate challenge keys regularly",
                "Implement challenge encryption",
                "Monitor challenge usage",
                "Use one-time challenges"
            ])
        
        # Select top actions based on risk score
        num_actions = min(len(actions), 1 + int(risk.residual_score / 5))
        return actions[:num_actions]
    
    def _estimate_cost(self, actions: List[str]) -> float:
        """Estimate mitigation cost"""
        # Simplified cost estimation
        cost_per_action = 10000  # Base cost
        complexity_multiplier = 1.5 ** (len(actions) - 1)
        return cost_per_action * len(actions) * complexity_multiplier
    
    def _estimate_effort(self, actions: List[str]) -> str:
        """Estimate implementation effort"""
        if len(actions) <= 2:
            return "low"
        elif len(actions) <= 4:
            return "medium"
        else:
            return "high"
    
    def _estimate_effectiveness(self, risk: Risk, actions: List[str]) -> float:
        """Estimate mitigation effectiveness"""
        # Base effectiveness on number and type of actions
        base_effectiveness = min(0.2 * len(actions), 0.8)
        
        # Adjust based on risk category
        if risk.category in [RiskCategory.BIAS_DISCRIMINATION, RiskCategory.PRIVACY_BREACH]:
            return min(base_effectiveness * 1.2, 0.9)
        elif risk.category in [RiskCategory.ADVERSARIAL_ATTACK, RiskCategory.SECURITY_VULNERABILITY]:
            return base_effectiveness * 0.8  # Harder to mitigate
        
        return base_effectiveness
    
    def _calculate_overall_score(self, risks: List[Risk]) -> float:
        """Calculate overall risk score"""
        if not risks:
            return 0
        
        # Weighted average based on risk scores
        total_score = sum(r.residual_score for r in risks)
        max_possible = 25 * len(risks)
        
        return (total_score / max_possible) * 100
    
    def _get_acceptable_threshold(self) -> float:
        """Get acceptable risk threshold based on appetite"""
        thresholds = {
            RiskAppetite.AVERSE: 20,
            RiskAppetite.MINIMAL: 30,
            RiskAppetite.CAUTIOUS: 40,
            RiskAppetite.OPEN: 50,
            RiskAppetite.HUNGRY: 60
        }
        return thresholds.get(self.risk_appetite, 40)
    
    def _get_mitigation_threshold(self) -> float:
        """Get threshold for requiring mitigation"""
        thresholds = {
            RiskAppetite.AVERSE: 5,
            RiskAppetite.MINIMAL: 8,
            RiskAppetite.CAUTIOUS: 10,
            RiskAppetite.OPEN: 15,
            RiskAppetite.HUNGRY: 20
        }
        return thresholds.get(self.risk_appetite, 10)
    
    def assess_pot_specific_risks(self, pot_config: Dict[str, Any]) -> RiskProfile:
        """
        Assess PoT-specific risks
        
        Args:
            pot_config: PoT system configuration
            
        Returns:
            RiskProfile for PoT risks
        """
        risks = []
        
        # Challenge generation risks
        if pot_config.get("challenge_reuse_count", 0) > 3:
            risks.append(Risk(
                risk_id=f"POT-{len(self.risk_register):04d}",
                category=RiskCategory.CHALLENGE_COMPROMISE,
                title="Challenge Reuse Risk",
                description="Challenges reused too many times",
                likelihood=RiskLikelihood.LIKELY,
                impact=RiskImpact.MAJOR,
                inherent_score=16,
                controls=["Challenge rotation"],
                residual_likelihood=RiskLikelihood.POSSIBLE,
                residual_impact=RiskImpact.MODERATE,
                residual_score=9,
                owner="security_team",
                identified_date=datetime.now()
            ))
        
        # Verification failure risks
        if pot_config.get("verification_confidence", 1.0) < 0.85:
            risks.append(Risk(
                risk_id=f"POT-{len(self.risk_register):04d}",
                category=RiskCategory.VERIFICATION_FAILURE,
                title="Low Verification Confidence",
                description="Verification confidence below threshold",
                likelihood=RiskLikelihood.LIKELY,
                impact=RiskImpact.MAJOR,
                inherent_score=16,
                controls=["Enhanced verification"],
                residual_likelihood=RiskLikelihood.POSSIBLE,
                residual_impact=RiskImpact.MODERATE,
                residual_score=9,
                owner="verification_team",
                identified_date=datetime.now()
            ))
        
        # Model drift risks
        if pot_config.get("drift_detected", False):
            risks.append(Risk(
                risk_id=f"POT-{len(self.risk_register):04d}",
                category=RiskCategory.MODEL_DRIFT,
                title="Model Drift Detected",
                description="Significant drift from training distribution",
                likelihood=RiskLikelihood.ALMOST_CERTAIN,
                impact=RiskImpact.MODERATE,
                inherent_score=15,
                controls=["Drift monitoring", "Automated retraining"],
                residual_likelihood=RiskLikelihood.LIKELY,
                residual_impact=RiskImpact.MINOR,
                residual_score=8,
                owner="ml_team",
                identified_date=datetime.now()
            ))
        
        # Adversarial attack risks
        if pot_config.get("public_api", False):
            risks.append(Risk(
                risk_id=f"POT-{len(self.risk_register):04d}",
                category=RiskCategory.ADVERSARIAL_ATTACK,
                title="Adversarial Attack Exposure",
                description="Public API vulnerable to adversarial attacks",
                likelihood=RiskLikelihood.LIKELY,
                impact=RiskImpact.MAJOR,
                inherent_score=16,
                controls=["Input validation", "Rate limiting", "Anomaly detection"],
                residual_likelihood=RiskLikelihood.POSSIBLE,
                residual_impact=RiskImpact.MODERATE,
                residual_score=9,
                owner="security_team",
                identified_date=datetime.now()
            ))
        
        # Create profile
        overall_score = self._calculate_overall_score(risks)
        
        return RiskProfile(
            profile_id=f"POT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            assessment_date=datetime.now(),
            system_name="PoT Verification System",
            risks=risks,
            overall_risk_score=overall_score,
            risk_distribution=Counter(r.category.value for r in risks),
            high_priority_risks=[r for r in risks if r.residual_score >= 12],
            mitigations=self.recommend_mitigations(risks),
            risk_appetite=self.risk_appetite,
            acceptable_threshold=self._get_acceptable_threshold(),
            metadata=pot_config
        )
    
    def update_risk_register(self, risk_id: str, updates: Dict[str, Any]):
        """Update a risk in the register"""
        for risk in self.risk_register:
            if risk.risk_id == risk_id:
                for key, value in updates.items():
                    if hasattr(risk, key):
                        setattr(risk, key, value)
                risk.review_date = datetime.now()
                break
    
    def record_incident(self, incident: RiskIncident):
        """Record a risk incident"""
        self.incidents.append(incident)
        
        # Update risk likelihood based on incident
        for risk in self.risk_register:
            if risk.risk_id == incident.risk_id:
                # Increase likelihood since risk materialized
                if risk.likelihood.level < 5:
                    new_level = min(5, risk.likelihood.level + 1)
                    risk.likelihood = list(RiskLikelihood)[new_level - 1]
                    risk.residual_score = self.risk_matrix.calculate_score(
                        risk.likelihood, risk.impact
                    )
                break
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        report = {
            "report_id": f"RISK-REPORT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_date": datetime.now().isoformat(),
            "risk_appetite": self.risk_appetite.value,
            "total_risks": len(self.risk_register),
            "risk_summary": {
                "critical": sum(1 for r in self.risk_register if r.residual_score >= 20),
                "high": sum(1 for r in self.risk_register if 15 <= r.residual_score < 20),
                "medium": sum(1 for r in self.risk_register if 10 <= r.residual_score < 15),
                "low": sum(1 for r in self.risk_register if r.residual_score < 10)
            },
            "by_category": {},
            "top_risks": [],
            "recent_incidents": [],
            "mitigation_status": {
                "planned": sum(1 for m in self.mitigations if m.status == "planned"),
                "in_progress": sum(1 for m in self.mitigations if m.status == "in_progress"),
                "implemented": sum(1 for m in self.mitigations if m.status == "implemented")
            },
            "risk_trend": self._calculate_risk_trend()
        }
        
        # Risks by category
        category_counts = Counter(r.category.value for r in self.risk_register)
        report["by_category"] = dict(category_counts)
        
        # Top risks
        top_risks = sorted(self.risk_register, key=lambda r: r.residual_score, reverse=True)[:10]
        report["top_risks"] = [r.to_dict() for r in top_risks]
        
        # Recent incidents
        recent_incidents = sorted(self.incidents, key=lambda i: i.occurred_date, reverse=True)[:5]
        report["recent_incidents"] = [i.to_dict() for i in recent_incidents]
        
        return report
    
    def _calculate_risk_trend(self) -> str:
        """Calculate overall risk trend"""
        if len(self.risk_register) < 2:
            return "stable"
        
        # Compare recent vs older risks
        recent_avg = np.mean([r.residual_score for r in self.risk_register[-10:]])
        older_avg = np.mean([r.residual_score for r in self.risk_register[:-10]])
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def export_risk_register(self, filepath: str, format: str = "json"):
        """Export risk register to file"""
        data = {
            "export_date": datetime.now().isoformat(),
            "risks": [r.to_dict() for r in self.risk_register],
            "mitigations": [m.to_dict() for m in self.mitigations],
            "incidents": [i.to_dict() for i in self.incidents]
        }
        
        path = Path(filepath)
        if format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def map_to_regulations(self) -> Dict[str, List[Risk]]:
        """Map risks to regulatory requirements"""
        regulatory_mapping = {
            "EU_AI_Act": [],
            "NIST_AI_RMF": [],
            "GDPR": [],
            "CCPA": [],
            "ISO_27001": []
        }
        
        for risk in self.risk_register:
            # EU AI Act mapping
            if risk.category in [
                RiskCategory.BIAS_DISCRIMINATION,
                RiskCategory.TRANSPARENCY_DEFICIT,
                RiskCategory.SAFETY_HARM
            ]:
                regulatory_mapping["EU_AI_Act"].append(risk)
            
            # NIST AI RMF mapping
            if risk.category in [
                RiskCategory.SECURITY_VULNERABILITY,
                RiskCategory.PERFORMANCE_DEGRADATION,
                RiskCategory.VERIFICATION_FAILURE
            ]:
                regulatory_mapping["NIST_AI_RMF"].append(risk)
            
            # GDPR mapping
            if risk.category == RiskCategory.PRIVACY_BREACH:
                regulatory_mapping["GDPR"].append(risk)
                regulatory_mapping["CCPA"].append(risk)
            
            # ISO 27001 mapping
            if risk.category in [
                RiskCategory.SECURITY_VULNERABILITY,
                RiskCategory.ADVERSARIAL_ATTACK
            ]:
                regulatory_mapping["ISO_27001"].append(risk)
        
        return regulatory_mapping