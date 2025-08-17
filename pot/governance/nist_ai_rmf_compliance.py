"""
NIST AI Risk Management Framework (AI RMF 1.0) Compliance Module
Implements the four functions: GOVERN, MAP, MEASURE, and MANAGE
Based on NIST AI 100-1 Framework
"""

import json
import yaml
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np


class RiskLevel(Enum):
    """Risk levels per NIST categorization"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class TrustworthinessCharacteristic(Enum):
    """NIST AI RMF Trustworthiness Characteristics"""
    VALID_RELIABLE = "valid_and_reliable"
    SAFE = "safe"
    SECURE_RESILIENT = "secure_and_resilient"
    ACCOUNTABLE_TRANSPARENT = "accountable_and_transparent"
    EXPLAINABLE_INTERPRETABLE = "explainable_and_interpretable"
    PRIVACY_ENHANCED = "privacy_enhanced"
    FAIR_BIAS_MANAGED = "fair_with_harmful_bias_managed"


class LifecyclePhase(Enum):
    """AI System Lifecycle Phases"""
    PLAN_DESIGN = "plan_and_design"
    DATA_COLLECTION = "data_collection_and_processing"
    MODEL_BUILDING = "model_building_and_validation"
    DEPLOYMENT = "deployment"
    OPERATION_MONITORING = "operation_and_monitoring"
    END_OF_LIFE = "end_of_life"


class Function(Enum):
    """NIST AI RMF Functions"""
    GOVERN = "govern"
    MAP = "map"
    MEASURE = "measure"
    MANAGE = "manage"


@dataclass
class AssessmentResult:
    """Result of a NIST AI RMF assessment"""
    function: Function
    category: str
    subcategory: str
    assessment_date: datetime
    status: str  # "implemented", "partially_implemented", "not_implemented", "not_applicable"
    maturity_level: int  # 1-5 scale
    findings: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['function'] = self.function.value
        result['assessment_date'] = self.assessment_date.isoformat()
        return result


@dataclass
class RiskMap:
    """AI Risk Mapping output"""
    map_id: str
    created_date: datetime
    context: Dict[str, Any]
    identified_risks: List[Dict[str, Any]]
    stakeholders: List[str]
    legal_requirements: List[str]
    ethical_considerations: List[str]
    risk_categories: Dict[str, List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['created_date'] = self.created_date.isoformat()
        return result


@dataclass
class MeasurementReport:
    """Risk Measurement Report"""
    report_id: str
    measurement_date: datetime
    metrics: Dict[str, Any]
    risk_scores: Dict[str, float]
    threshold_violations: List[str]
    trend_analysis: Dict[str, Any]
    confidence_levels: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['measurement_date'] = self.measurement_date.isoformat()
        return result


@dataclass
class ManagementPlan:
    """Risk Management Plan"""
    plan_id: str
    created_date: datetime
    risk_treatments: List[Dict[str, Any]]
    controls: Dict[str, Any]
    monitoring_procedures: List[str]
    response_plans: Dict[str, Any]
    success_criteria: Dict[str, Any]
    review_schedule: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['created_date'] = self.created_date.isoformat()
        return result


@dataclass
class TrustworthinessMetrics:
    """Metrics for AI trustworthiness characteristics"""
    characteristic: TrustworthinessCharacteristic
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    current_values: Dict[str, float]
    compliance_status: bool
    trend: str  # "improving", "stable", "degrading"
    
    def evaluate(self) -> bool:
        """Evaluate if metrics meet thresholds"""
        for metric, value in self.current_values.items():
            if metric in self.thresholds:
                if value < self.thresholds[metric]:
                    return False
        return True


@dataclass
class NISTDocumentation:
    """NIST-aligned documentation artifact"""
    doc_id: str
    doc_type: str
    version: str
    created_date: datetime
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        data = asdict(self)
        data['created_date'] = self.created_date.isoformat()
        return json.dumps(data, indent=2)
    
    def save(self, filepath: Path):
        """Save documentation to file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class NISTAIRMFCompliance:
    """
    NIST AI Risk Management Framework Compliance Module
    Implements NIST AI 100-1 Framework Version 1.0
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize NIST AI RMF compliance module"""
        self.framework_version = "1.0"
        self.functions = [Function.GOVERN, Function.MAP, Function.MEASURE, Function.MANAGE]
        self.assessments: List[AssessmentResult] = []
        self.risk_maps: List[RiskMap] = []
        self.measurements: List[MeasurementReport] = []
        self.management_plans: List[ManagementPlan] = []
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Initialize categories and subcategories
        self._init_framework_structure()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default NIST AI RMF configuration"""
        return {
            "organization": {
                "name": "",
                "sector": "",
                "size": "medium"
            },
            "risk_tolerance": {
                "safety": 0.01,
                "security": 0.05,
                "fairness": 0.10,
                "privacy": 0.05,
                "reliability": 0.15
            },
            "maturity_target": 3,  # Target maturity level (1-5)
            "assessment_frequency_days": 90,
            "continuous_monitoring": True
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _init_framework_structure(self):
        """Initialize NIST AI RMF categories and subcategories"""
        self.framework_structure = {
            Function.GOVERN: {
                "GOVERN 1": {
                    "name": "Policies, Processes, Procedures, and Practices",
                    "subcategories": [
                        "GOVERN 1.1: Legal and regulatory requirements",
                        "GOVERN 1.2: Organizational policies",
                        "GOVERN 1.3: AI risk management processes",
                        "GOVERN 1.4: Organizational risk tolerance",
                        "GOVERN 1.5: AI system impacts assessment",
                        "GOVERN 1.6: Roles and responsibilities",
                        "GOVERN 1.7: Organizational culture"
                    ]
                },
                "GOVERN 2": {
                    "name": "Accountability Structures",
                    "subcategories": [
                        "GOVERN 2.1: Roles and responsibilities assignment",
                        "GOVERN 2.2: Workforce AI risk literacy",
                        "GOVERN 2.3: Executive engagement"
                    ]
                },
                "GOVERN 3": {
                    "name": "Workforce Diversity and Capacity",
                    "subcategories": [
                        "GOVERN 3.1: Diverse perspectives",
                        "GOVERN 3.2: Workforce competency"
                    ]
                },
                "GOVERN 4": {
                    "name": "Organizational Culture",
                    "subcategories": [
                        "GOVERN 4.1: Risk awareness culture",
                        "GOVERN 4.2: Responsible AI practices",
                        "GOVERN 4.3: Continuous improvement"
                    ]
                }
            },
            Function.MAP: {
                "MAP 1": {
                    "name": "Context Establishment",
                    "subcategories": [
                        "MAP 1.1: Intended purpose and context",
                        "MAP 1.2: Interdependencies identification",
                        "MAP 1.3: Assumptions and limitations",
                        "MAP 1.4: Negative impacts assessment",
                        "MAP 1.5: Benefits assessment",
                        "MAP 1.6: Stakeholder engagement"
                    ]
                },
                "MAP 2": {
                    "name": "Categorization",
                    "subcategories": [
                        "MAP 2.1: AI system categorization",
                        "MAP 2.2: Task and output types",
                        "MAP 2.3: Risk level determination"
                    ]
                },
                "MAP 3": {
                    "name": "AI Capabilities and Limitations",
                    "subcategories": [
                        "MAP 3.1: Scientific understanding",
                        "MAP 3.2: Limitations documentation",
                        "MAP 3.3: Emergent properties",
                        "MAP 3.4: Performance boundaries"
                    ]
                },
                "MAP 4": {
                    "name": "AI Risks Identification",
                    "subcategories": [
                        "MAP 4.1: Risk sources identification",
                        "MAP 4.2: Risk likelihood and impact"
                    ]
                },
                "MAP 5": {
                    "name": "Impacts Characterization",
                    "subcategories": [
                        "MAP 5.1: Impact likelihood and magnitude",
                        "MAP 5.2: Affected stakeholders"
                    ]
                }
            },
            Function.MEASURE: {
                "MEASURE 1": {
                    "name": "Risk Assessment Methods",
                    "subcategories": [
                        "MEASURE 1.1: Appropriate methods identification",
                        "MEASURE 1.2: Testing and evaluation",
                        "MEASURE 1.3: Internal audit function"
                    ]
                },
                "MEASURE 2": {
                    "name": "AI System Performance",
                    "subcategories": [
                        "MEASURE 2.1: Test sets and metrics",
                        "MEASURE 2.2: Performance documentation",
                        "MEASURE 2.3: Data quality assessment",
                        "MEASURE 2.4: Effectiveness monitoring",
                        "MEASURE 2.5: Performance thresholds",
                        "MEASURE 2.6: Responsible parties identification",
                        "MEASURE 2.7: Domain expert involvement",
                        "MEASURE 2.8: Bias and fairness assessment",
                        "MEASURE 2.9: Privacy assessment",
                        "MEASURE 2.10: Security assessment",
                        "MEASURE 2.11: Explainability assessment",
                        "MEASURE 2.12: Safety assessment",
                        "MEASURE 2.13: Feedback mechanisms"
                    ]
                },
                "MEASURE 3": {
                    "name": "AI System Trustworthiness",
                    "subcategories": [
                        "MEASURE 3.1: Risk controls assessment",
                        "MEASURE 3.2: Residual risk evaluation",
                        "MEASURE 3.3: Incident tracking"
                    ]
                },
                "MEASURE 4": {
                    "name": "Risk Monitoring",
                    "subcategories": [
                        "MEASURE 4.1: Monitoring effectiveness",
                        "MEASURE 4.2: Compliance monitoring",
                        "MEASURE 4.3: Feedback analysis"
                    ]
                }
            },
            Function.MANAGE: {
                "MANAGE 1": {
                    "name": "Risk Response",
                    "subcategories": [
                        "MANAGE 1.1: Risk treatment decisions",
                        "MANAGE 1.2: Response plans",
                        "MANAGE 1.3: Transparency documentation",
                        "MANAGE 1.4: Impact documentation"
                    ]
                },
                "MANAGE 2": {
                    "name": "Risk Controls Implementation",
                    "subcategories": [
                        "MANAGE 2.1: Risk plans implementation",
                        "MANAGE 2.2: Treatment effectiveness",
                        "MANAGE 2.3: Ongoing monitoring",
                        "MANAGE 2.4: TEVV implementation"
                    ]
                },
                "MANAGE 3": {
                    "name": "Incident Response",
                    "subcategories": [
                        "MANAGE 3.1: Change management processes",
                        "MANAGE 3.2: Version control and rollback"
                    ]
                },
                "MANAGE 4": {
                    "name": "Third Party Risks",
                    "subcategories": [
                        "MANAGE 4.1: Supply chain risks",
                        "MANAGE 4.2: Third party monitoring",
                        "MANAGE 4.3: Dependency management"
                    ]
                }
            }
        }
    
    def assess_governance_function(self, org_context: Dict[str, Any]) -> AssessmentResult:
        """
        Assess the GOVERN function - governance structure and processes
        
        Args:
            org_context: Organizational context and governance information
            
        Returns:
            AssessmentResult for governance function
        """
        findings = []
        recommendations = []
        maturity_scores = []
        
        # GOVERN 1: Policies, Processes, Procedures
        govern1_score = self._assess_govern_1(org_context, findings, recommendations)
        maturity_scores.append(govern1_score)
        
        # GOVERN 2: Accountability Structures
        govern2_score = self._assess_govern_2(org_context, findings, recommendations)
        maturity_scores.append(govern2_score)
        
        # GOVERN 3: Workforce Diversity and Capacity
        govern3_score = self._assess_govern_3(org_context, findings, recommendations)
        maturity_scores.append(govern3_score)
        
        # GOVERN 4: Organizational Culture
        govern4_score = self._assess_govern_4(org_context, findings, recommendations)
        maturity_scores.append(govern4_score)
        
        # Calculate overall maturity
        avg_maturity = sum(maturity_scores) / len(maturity_scores) if maturity_scores else 1
        
        # Determine status
        if avg_maturity >= 4:
            status = "implemented"
        elif avg_maturity >= 2.5:
            status = "partially_implemented"
        else:
            status = "not_implemented"
        
        result = AssessmentResult(
            function=Function.GOVERN,
            category="Overall Governance",
            subcategory="All GOVERN categories",
            assessment_date=datetime.now(),
            status=status,
            maturity_level=int(avg_maturity),
            findings=findings,
            recommendations=recommendations,
            evidence={"org_context": org_context, "detailed_scores": maturity_scores}
        )
        
        self.assessments.append(result)
        return result
    
    def _assess_govern_1(self, context: Dict[str, Any], findings: List[str], recommendations: List[str]) -> int:
        """Assess GOVERN 1: Policies, Processes, Procedures"""
        score = 1
        
        # Check for legal/regulatory compliance documentation
        if context.get("legal_compliance_documented", False):
            score += 1
        else:
            findings.append("Legal and regulatory requirements not documented")
            recommendations.append("Document applicable legal and regulatory requirements")
        
        # Check for AI policies
        if context.get("ai_policies_exist", False):
            score += 1
            if context.get("ai_policies_current", False):
                score += 1
        else:
            findings.append("AI-specific policies not established")
            recommendations.append("Develop comprehensive AI governance policies")
        
        # Check for risk tolerance definition
        if context.get("risk_tolerance_defined", False):
            score += 1
        else:
            findings.append("Organizational risk tolerance not defined")
            recommendations.append("Define and document organizational risk tolerance levels")
        
        return min(score, 5)
    
    def _assess_govern_2(self, context: Dict[str, Any], findings: List[str], recommendations: List[str]) -> int:
        """Assess GOVERN 2: Accountability Structures"""
        score = 1
        
        # Check roles and responsibilities
        if context.get("roles_defined", False):
            score += 1
            if context.get("roles_communicated", False):
                score += 1
        else:
            findings.append("AI governance roles not clearly defined")
            recommendations.append("Define and assign AI governance roles and responsibilities")
        
        # Check workforce AI literacy
        if context.get("ai_training_program", False):
            score += 1
        else:
            findings.append("No AI risk literacy program for workforce")
            recommendations.append("Implement AI risk management training program")
        
        # Check executive engagement
        if context.get("executive_oversight", False):
            score += 1
        else:
            findings.append("Limited executive engagement in AI governance")
            recommendations.append("Establish executive-level AI oversight committee")
        
        return min(score, 5)
    
    def _assess_govern_3(self, context: Dict[str, Any], findings: List[str], recommendations: List[str]) -> int:
        """Assess GOVERN 3: Workforce Diversity and Capacity"""
        score = 1
        
        # Check for diverse perspectives
        if context.get("diverse_team", False):
            score += 2
        else:
            findings.append("Limited diversity in AI governance team")
            recommendations.append("Increase diversity of perspectives in AI governance")
        
        # Check workforce competency
        if context.get("competency_assessment", False):
            score += 1
            if context.get("competency_development", False):
                score += 1
        else:
            findings.append("No workforce competency assessment for AI risks")
            recommendations.append("Assess and develop workforce AI risk management competencies")
        
        return min(score, 5)
    
    def _assess_govern_4(self, context: Dict[str, Any], findings: List[str], recommendations: List[str]) -> int:
        """Assess GOVERN 4: Organizational Culture"""
        score = 1
        
        # Check risk awareness culture
        if context.get("risk_culture_assessment", False):
            score += 1
            if context.get("risk_culture_positive", False):
                score += 1
        else:
            findings.append("AI risk awareness culture not assessed")
            recommendations.append("Assess and strengthen AI risk awareness culture")
        
        # Check responsible AI practices
        if context.get("responsible_ai_principles", False):
            score += 1
        else:
            findings.append("Responsible AI principles not established")
            recommendations.append("Establish and communicate responsible AI principles")
        
        # Check continuous improvement
        if context.get("continuous_improvement_process", False):
            score += 1
        else:
            findings.append("No continuous improvement process for AI governance")
            recommendations.append("Implement continuous improvement process for AI risk management")
        
        return min(score, 5)
    
    def map_ai_risks(self, system_context: Dict[str, Any]) -> RiskMap:
        """
        MAP function - Context establishment and risk identification
        
        Args:
            system_context: AI system context information
            
        Returns:
            RiskMap with identified risks and context
        """
        # Initialize risk map
        risk_map = RiskMap(
            map_id=f"MAP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            created_date=datetime.now(),
            context=system_context,
            identified_risks=[],
            stakeholders=[],
            legal_requirements=[],
            ethical_considerations=[],
            risk_categories={}
        )
        
        # MAP 1: Establish context
        self._map_context(system_context, risk_map)
        
        # MAP 2: Categorize system
        self._map_categorization(system_context, risk_map)
        
        # MAP 3: Document capabilities and limitations
        self._map_capabilities(system_context, risk_map)
        
        # MAP 4: Identify risks
        self._map_risks(system_context, risk_map)
        
        # MAP 5: Characterize impacts
        self._map_impacts(system_context, risk_map)
        
        self.risk_maps.append(risk_map)
        return risk_map
    
    def _map_context(self, context: Dict[str, Any], risk_map: RiskMap):
        """MAP 1: Context Establishment"""
        # Extract stakeholders
        risk_map.stakeholders = context.get("stakeholders", [
            "developers", "operators", "users", "affected_parties"
        ])
        
        # Identify legal requirements
        risk_map.legal_requirements = context.get("legal_requirements", [])
        
        # Identify ethical considerations
        risk_map.ethical_considerations = context.get("ethical_considerations", [
            "fairness", "transparency", "accountability", "privacy"
        ])
    
    def _map_categorization(self, context: Dict[str, Any], risk_map: RiskMap):
        """MAP 2: System Categorization"""
        # Categorize by application domain
        domain = context.get("application_domain", "general")
        risk_level = self._determine_risk_level(domain)
        
        risk_map.risk_categories["application_risk"] = [risk_level.value]
        
        # Categorize by data sensitivity
        data_sensitivity = context.get("data_sensitivity", "medium")
        risk_map.risk_categories["data_risk"] = [data_sensitivity]
    
    def _map_capabilities(self, context: Dict[str, Any], risk_map: RiskMap):
        """MAP 3: Capabilities and Limitations"""
        capabilities = context.get("capabilities", {})
        limitations = context.get("limitations", {})
        
        # Document in context
        risk_map.context["documented_capabilities"] = capabilities
        risk_map.context["documented_limitations"] = limitations
        
        # Identify risks from limitations
        for limitation in limitations.get("known_limitations", []):
            risk_map.identified_risks.append({
                "source": "capability_limitation",
                "description": f"Limitation: {limitation}",
                "category": "technical"
            })
    
    def _map_risks(self, context: Dict[str, Any], risk_map: RiskMap):
        """MAP 4: Risk Identification"""
        # Technical risks
        technical_risks = [
            {"source": "model", "description": "Model accuracy degradation", "category": "technical"},
            {"source": "data", "description": "Data drift", "category": "technical"},
            {"source": "integration", "description": "System integration failures", "category": "technical"}
        ]
        
        # Ethical risks
        ethical_risks = [
            {"source": "bias", "description": "Algorithmic bias", "category": "ethical"},
            {"source": "fairness", "description": "Unfair treatment of groups", "category": "ethical"}
        ]
        
        # Security risks
        security_risks = [
            {"source": "adversarial", "description": "Adversarial attacks", "category": "security"},
            {"source": "data_breach", "description": "Training data extraction", "category": "security"}
        ]
        
        # Add identified risks
        risk_map.identified_risks.extend(technical_risks)
        risk_map.identified_risks.extend(ethical_risks)
        risk_map.identified_risks.extend(security_risks)
        
        # Categorize risks
        risk_map.risk_categories["technical"] = [r["description"] for r in technical_risks]
        risk_map.risk_categories["ethical"] = [r["description"] for r in ethical_risks]
        risk_map.risk_categories["security"] = [r["description"] for r in security_risks]
    
    def _map_impacts(self, context: Dict[str, Any], risk_map: RiskMap):
        """MAP 5: Impact Characterization"""
        for risk in risk_map.identified_risks:
            # Assess impact likelihood and magnitude
            risk["likelihood"] = self._assess_likelihood(risk, context)
            risk["impact"] = self._assess_impact(risk, context)
            risk["affected_stakeholders"] = self._identify_affected_stakeholders(risk)
    
    def _determine_risk_level(self, domain: str) -> RiskLevel:
        """Determine risk level based on domain"""
        high_risk_domains = ["healthcare", "finance", "criminal_justice", "employment"]
        moderate_risk_domains = ["education", "retail", "entertainment"]
        
        if domain in high_risk_domains:
            return RiskLevel.HIGH
        elif domain in moderate_risk_domains:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_likelihood(self, risk: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Assess likelihood of a risk occurring"""
        # Simplified likelihood assessment
        if "adversarial" in risk["description"]:
            return "high" if context.get("public_facing", False) else "medium"
        elif "drift" in risk["description"]:
            return "medium"
        else:
            return "low"
    
    def _assess_impact(self, risk: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Assess impact magnitude of a risk"""
        # Simplified impact assessment
        if risk["category"] == "security":
            return "high"
        elif risk["category"] == "ethical":
            return "medium" if context.get("affects_individuals", True) else "low"
        else:
            return "medium"
    
    def _identify_affected_stakeholders(self, risk: Dict[str, Any]) -> List[str]:
        """Identify stakeholders affected by a risk"""
        stakeholder_map = {
            "technical": ["developers", "operators"],
            "ethical": ["users", "affected_parties", "society"],
            "security": ["organization", "users", "partners"]
        }
        return stakeholder_map.get(risk["category"], ["all"])
    
    def measure_ai_risks(self, metrics: Dict[str, Any]) -> MeasurementReport:
        """
        MEASURE function - Risk assessment and analysis
        
        Args:
            metrics: Performance and risk metrics
            
        Returns:
            MeasurementReport with risk measurements
        """
        report = MeasurementReport(
            report_id=f"MEASURE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            measurement_date=datetime.now(),
            metrics=metrics,
            risk_scores={},
            threshold_violations=[],
            trend_analysis={},
            confidence_levels={}
        )
        
        # MEASURE 1: Apply assessment methods
        self._measure_assessment_methods(metrics, report)
        
        # MEASURE 2: Assess system performance
        self._measure_performance(metrics, report)
        
        # MEASURE 3: Assess trustworthiness
        self._measure_trustworthiness(metrics, report)
        
        # MEASURE 4: Monitor risks
        self._measure_monitoring(metrics, report)
        
        self.measurements.append(report)
        return report
    
    def _measure_assessment_methods(self, metrics: Dict[str, Any], report: MeasurementReport):
        """MEASURE 1: Assessment Methods"""
        # Verify appropriate methods are used
        if "test_methods" in metrics:
            report.confidence_levels["methodology"] = 0.8
        else:
            report.confidence_levels["methodology"] = 0.3
            report.threshold_violations.append("Inadequate testing methodology")
    
    def _measure_performance(self, metrics: Dict[str, Any], report: MeasurementReport):
        """MEASURE 2: System Performance"""
        performance_metrics = metrics.get("performance", {})
        
        # Measure accuracy
        accuracy = performance_metrics.get("accuracy", 0)
        report.risk_scores["accuracy_risk"] = 1.0 - accuracy
        
        # Measure robustness
        robustness = performance_metrics.get("robustness", 0)
        report.risk_scores["robustness_risk"] = 1.0 - robustness
        
        # Measure fairness
        fairness = performance_metrics.get("fairness", 0)
        report.risk_scores["fairness_risk"] = 1.0 - fairness
        
        # Check thresholds
        if accuracy < self.config["risk_tolerance"].get("reliability", 0.85):
            report.threshold_violations.append(f"Accuracy below threshold: {accuracy}")
        
        if fairness < self.config["risk_tolerance"].get("fairness", 0.90):
            report.threshold_violations.append(f"Fairness below threshold: {fairness}")
    
    def _measure_trustworthiness(self, metrics: Dict[str, Any], report: MeasurementReport):
        """MEASURE 3: Trustworthiness Assessment"""
        trustworthiness = metrics.get("trustworthiness", {})
        
        # Assess each characteristic
        for characteristic in TrustworthinessCharacteristic:
            char_metrics = trustworthiness.get(characteristic.value, {})
            if char_metrics:
                score = self._calculate_trustworthiness_score(char_metrics)
                report.risk_scores[f"{characteristic.value}_risk"] = 1.0 - score
                report.confidence_levels[characteristic.value] = char_metrics.get("confidence", 0.5)
    
    def _measure_monitoring(self, metrics: Dict[str, Any], report: MeasurementReport):
        """MEASURE 4: Risk Monitoring"""
        monitoring = metrics.get("monitoring", {})
        
        # Analyze trends
        if "historical_data" in monitoring:
            report.trend_analysis = self._analyze_trends(monitoring["historical_data"])
        
        # Check monitoring effectiveness
        if monitoring.get("coverage", 0) < 0.8:
            report.threshold_violations.append("Insufficient monitoring coverage")
    
    def _calculate_trustworthiness_score(self, char_metrics: Dict[str, Any]) -> float:
        """Calculate trustworthiness score for a characteristic"""
        scores = []
        for metric, value in char_metrics.items():
            if isinstance(value, (int, float)):
                scores.append(value)
        return sum(scores) / len(scores) if scores else 0.0
    
    def _analyze_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in historical data"""
        if not historical_data:
            return {"trend": "unknown", "direction": "stable"}
        
        # Simple trend analysis
        recent = historical_data[-5:] if len(historical_data) >= 5 else historical_data
        
        trend_analysis = {
            "period": len(recent),
            "direction": "stable",
            "rate_of_change": 0.0
        }
        
        # Calculate direction
        if len(recent) >= 2:
            first_avg = sum(r.get("score", 0) for r in recent[:len(recent)//2]) / (len(recent)//2)
            second_avg = sum(r.get("score", 0) for r in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            if second_avg > first_avg * 1.05:
                trend_analysis["direction"] = "improving"
            elif second_avg < first_avg * 0.95:
                trend_analysis["direction"] = "degrading"
            
            trend_analysis["rate_of_change"] = (second_avg - first_avg) / first_avg if first_avg != 0 else 0
        
        return trend_analysis
    
    def manage_ai_risks(self, controls: Dict[str, Any]) -> ManagementPlan:
        """
        MANAGE function - Risk treatment and monitoring
        
        Args:
            controls: Risk controls and treatments
            
        Returns:
            ManagementPlan with risk treatments
        """
        plan = ManagementPlan(
            plan_id=f"MANAGE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            created_date=datetime.now(),
            risk_treatments=[],
            controls=controls,
            monitoring_procedures=[],
            response_plans={},
            success_criteria={},
            review_schedule={}
        )
        
        # MANAGE 1: Determine risk responses
        self._manage_responses(controls, plan)
        
        # MANAGE 2: Implement controls
        self._manage_controls(controls, plan)
        
        # MANAGE 3: Incident response
        self._manage_incidents(controls, plan)
        
        # MANAGE 4: Third party risks
        self._manage_third_party(controls, plan)
        
        self.management_plans.append(plan)
        return plan
    
    def _manage_responses(self, controls: Dict[str, Any], plan: ManagementPlan):
        """MANAGE 1: Risk Response Planning"""
        # Define risk treatments for identified risks
        if self.risk_maps:
            latest_map = self.risk_maps[-1]
            for risk in latest_map.identified_risks:
                treatment = self._determine_treatment(risk, controls)
                plan.risk_treatments.append(treatment)
        
        # Document transparency requirements
        plan.response_plans["transparency"] = {
            "documentation": "Maintain comprehensive risk documentation",
            "communication": "Regular stakeholder updates",
            "reporting": "Periodic risk reports"
        }
    
    def _manage_controls(self, controls: Dict[str, Any], plan: ManagementPlan):
        """MANAGE 2: Control Implementation"""
        # Define monitoring procedures
        plan.monitoring_procedures = [
            "Continuous performance monitoring",
            "Regular security assessments",
            "Periodic fairness audits",
            "User feedback collection",
            "Incident tracking"
        ]
        
        # Set success criteria
        plan.success_criteria = {
            "risk_reduction": "Reduce high risks to moderate or below",
            "compliance": "Maintain regulatory compliance",
            "performance": "Meet or exceed performance thresholds",
            "incidents": "Reduce incident rate by 50%"
        }
    
    def _manage_incidents(self, controls: Dict[str, Any], plan: ManagementPlan):
        """MANAGE 3: Incident Response"""
        plan.response_plans["incident_response"] = {
            "detection": "Automated incident detection system",
            "assessment": "Rapid impact assessment process",
            "containment": "Immediate containment procedures",
            "recovery": "System recovery and rollback capabilities",
            "lessons_learned": "Post-incident review process"
        }
        
        # Version control and rollback
        plan.controls["version_control"] = {
            "enabled": True,
            "rollback_capability": True,
            "testing_required": True
        }
    
    def _manage_third_party(self, controls: Dict[str, Any], plan: ManagementPlan):
        """MANAGE 4: Third Party Risk Management"""
        plan.response_plans["third_party"] = {
            "assessment": "Vendor risk assessment process",
            "monitoring": "Continuous third-party monitoring",
            "contracts": "Risk allocation in contracts",
            "contingency": "Alternative supplier identification"
        }
        
        # Review schedule
        plan.review_schedule = {
            "risk_assessment": "Quarterly",
            "control_effectiveness": "Monthly",
            "third_party": "Semi-annually",
            "incident_response": "After each incident"
        }
    
    def _determine_treatment(self, risk: Dict[str, Any], controls: Dict[str, Any]) -> Dict[str, Any]:
        """Determine appropriate treatment for a risk"""
        treatment = {
            "risk": risk["description"],
            "strategy": "mitigate",  # avoid, mitigate, transfer, accept
            "controls": [],
            "residual_risk": "low"
        }
        
        # Determine strategy based on risk characteristics
        if risk.get("impact") == "high" and risk.get("likelihood") == "high":
            treatment["strategy"] = "avoid"
            treatment["controls"] = ["Redesign system to eliminate risk"]
        elif risk["category"] == "security":
            treatment["strategy"] = "mitigate"
            treatment["controls"] = ["Implement security controls", "Regular security testing"]
        elif risk["category"] == "ethical":
            treatment["strategy"] = "mitigate"
            treatment["controls"] = ["Bias mitigation", "Fairness monitoring"]
        else:
            treatment["strategy"] = "accept"
            treatment["controls"] = ["Monitor and review periodically"]
        
        return treatment
    
    def assess_trustworthiness_characteristics(
        self, 
        system_data: Dict[str, Any]
    ) -> Dict[TrustworthinessCharacteristic, TrustworthinessMetrics]:
        """
        Assess all trustworthiness characteristics
        
        Args:
            system_data: System performance and characteristic data
            
        Returns:
            Dictionary of trustworthiness metrics by characteristic
        """
        results = {}
        
        for characteristic in TrustworthinessCharacteristic:
            metrics = self._create_characteristic_metrics(characteristic, system_data)
            results[characteristic] = metrics
        
        return results
    
    def _create_characteristic_metrics(
        self, 
        characteristic: TrustworthinessCharacteristic,
        system_data: Dict[str, Any]
    ) -> TrustworthinessMetrics:
        """Create metrics for a specific trustworthiness characteristic"""
        char_data = system_data.get(characteristic.value, {})
        
        # Define default metrics and thresholds
        default_metrics = {
            TrustworthinessCharacteristic.VALID_RELIABLE: {
                "accuracy": 0.9,
                "precision": 0.85,
                "recall": 0.85,
                "consistency": 0.95
            },
            TrustworthinessCharacteristic.SAFE: {
                "safety_score": 0.95,
                "harm_prevention": 0.99,
                "fail_safe": 0.90
            },
            TrustworthinessCharacteristic.SECURE_RESILIENT: {
                "security_score": 0.90,
                "robustness": 0.85,
                "recovery_time": 0.80
            },
            TrustworthinessCharacteristic.ACCOUNTABLE_TRANSPARENT: {
                "accountability": 0.85,
                "transparency": 0.80,
                "auditability": 0.90
            },
            TrustworthinessCharacteristic.EXPLAINABLE_INTERPRETABLE: {
                "explainability": 0.75,
                "interpretability": 0.70,
                "user_understanding": 0.80
            },
            TrustworthinessCharacteristic.PRIVACY_ENHANCED: {
                "privacy_protection": 0.95,
                "data_minimization": 0.90,
                "consent_compliance": 1.00
            },
            TrustworthinessCharacteristic.FAIR_BIAS_MANAGED: {
                "fairness_score": 0.90,
                "bias_detection": 0.85,
                "disparate_impact": 0.80
            }
        }
        
        thresholds = default_metrics.get(characteristic, {})
        current_values = char_data.get("metrics", {})
        
        # Determine trend
        historical = char_data.get("historical", [])
        if len(historical) >= 2:
            if historical[-1] > historical[-2]:
                trend = "improving"
            elif historical[-1] < historical[-2]:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        metrics = TrustworthinessMetrics(
            characteristic=characteristic,
            metrics=thresholds,
            thresholds=thresholds,
            current_values=current_values,
            compliance_status=True,  # Will be evaluated
            trend=trend
        )
        
        # Evaluate compliance
        metrics.compliance_status = metrics.evaluate()
        
        return metrics
    
    def assess_lifecycle_phase(self, phase: LifecyclePhase, phase_data: Dict[str, Any]) -> AssessmentResult:
        """
        Assess risks for a specific lifecycle phase
        
        Args:
            phase: The lifecycle phase to assess
            phase_data: Data specific to the phase
            
        Returns:
            AssessmentResult for the phase
        """
        findings = []
        recommendations = []
        
        if phase == LifecyclePhase.PLAN_DESIGN:
            self._assess_plan_design(phase_data, findings, recommendations)
        elif phase == LifecyclePhase.DATA_COLLECTION:
            self._assess_data_collection(phase_data, findings, recommendations)
        elif phase == LifecyclePhase.MODEL_BUILDING:
            self._assess_model_building(phase_data, findings, recommendations)
        elif phase == LifecyclePhase.DEPLOYMENT:
            self._assess_deployment(phase_data, findings, recommendations)
        elif phase == LifecyclePhase.OPERATION_MONITORING:
            self._assess_operation(phase_data, findings, recommendations)
        elif phase == LifecyclePhase.END_OF_LIFE:
            self._assess_end_of_life(phase_data, findings, recommendations)
        
        # Determine status
        if not findings:
            status = "implemented"
            maturity = 4
        elif len(findings) <= 2:
            status = "partially_implemented"
            maturity = 3
        else:
            status = "not_implemented"
            maturity = 2
        
        result = AssessmentResult(
            function=Function.MAP,
            category=f"Lifecycle: {phase.value}",
            subcategory="Phase-specific assessment",
            assessment_date=datetime.now(),
            status=status,
            maturity_level=maturity,
            findings=findings,
            recommendations=recommendations,
            evidence={"phase_data": phase_data}
        )
        
        self.assessments.append(result)
        return result
    
    def _assess_plan_design(self, data: Dict[str, Any], findings: List[str], recommendations: List[str]):
        """Assess Plan and Design phase"""
        if not data.get("requirements_documented", False):
            findings.append("Requirements not fully documented")
            recommendations.append("Document all functional and non-functional requirements")
        
        if not data.get("risk_assessment_performed", False):
            findings.append("Initial risk assessment not performed")
            recommendations.append("Conduct comprehensive risk assessment during design")
        
        if not data.get("ethical_review", False):
            findings.append("Ethical review not conducted")
            recommendations.append("Perform ethical impact assessment")
    
    def _assess_data_collection(self, data: Dict[str, Any], findings: List[str], recommendations: List[str]):
        """Assess Data Collection and Processing phase"""
        if not data.get("data_quality_checks", False):
            findings.append("Data quality checks not implemented")
            recommendations.append("Implement comprehensive data quality validation")
        
        if not data.get("bias_assessment", False):
            findings.append("Data bias not assessed")
            recommendations.append("Assess and document potential data biases")
        
        if not data.get("privacy_compliance", False):
            findings.append("Privacy compliance not verified")
            recommendations.append("Ensure data collection meets privacy requirements")
    
    def _assess_model_building(self, data: Dict[str, Any], findings: List[str], recommendations: List[str]):
        """Assess Model Building and Validation phase"""
        if not data.get("validation_strategy", False):
            findings.append("Validation strategy not defined")
            recommendations.append("Define comprehensive validation strategy")
        
        if not data.get("performance_benchmarks", False):
            findings.append("Performance benchmarks not established")
            recommendations.append("Establish clear performance benchmarks")
        
        if not data.get("robustness_testing", False):
            findings.append("Robustness testing not performed")
            recommendations.append("Conduct robustness and stress testing")
    
    def _assess_deployment(self, data: Dict[str, Any], findings: List[str], recommendations: List[str]):
        """Assess Deployment phase"""
        if not data.get("deployment_validation", False):
            findings.append("Deployment validation incomplete")
            recommendations.append("Validate deployment configuration and environment")
        
        if not data.get("rollback_plan", False):
            findings.append("No rollback plan defined")
            recommendations.append("Define and test rollback procedures")
        
        if not data.get("monitoring_setup", False):
            findings.append("Monitoring not properly configured")
            recommendations.append("Setup comprehensive monitoring before deployment")
    
    def _assess_operation(self, data: Dict[str, Any], findings: List[str], recommendations: List[str]):
        """Assess Operation and Monitoring phase"""
        if not data.get("performance_monitoring", False):
            findings.append("Performance not continuously monitored")
            recommendations.append("Implement continuous performance monitoring")
        
        if not data.get("drift_detection", False):
            findings.append("Model/data drift detection not implemented")
            recommendations.append("Implement drift detection mechanisms")
        
        if not data.get("incident_response", False):
            findings.append("Incident response procedures not defined")
            recommendations.append("Define and practice incident response procedures")
    
    def _assess_end_of_life(self, data: Dict[str, Any], findings: List[str], recommendations: List[str]):
        """Assess End-of-Life phase"""
        if not data.get("decommission_plan", False):
            findings.append("Decommissioning plan not defined")
            recommendations.append("Define system decommissioning procedures")
        
        if not data.get("data_disposal", False):
            findings.append("Data disposal procedures not defined")
            recommendations.append("Define secure data disposal procedures")
        
        if not data.get("knowledge_transfer", False):
            findings.append("Knowledge transfer not planned")
            recommendations.append("Plan for knowledge transfer and documentation")
    
    def generate_risk_tolerance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Generate risk tolerance thresholds based on configuration
        
        Returns:
            Dictionary of thresholds by risk category
        """
        base_tolerance = self.config.get("risk_tolerance", {})
        
        thresholds = {
            "performance": {
                "accuracy": 1.0 - base_tolerance.get("reliability", 0.15),
                "latency": 1000,  # milliseconds
                "throughput": 100  # requests per second
            },
            "security": {
                "vulnerability_score": base_tolerance.get("security", 0.05),
                "incident_rate": 0.01,  # per day
                "patch_time": 24  # hours
            },
            "fairness": {
                "demographic_parity": base_tolerance.get("fairness", 0.10),
                "equal_opportunity": base_tolerance.get("fairness", 0.10),
                "disparate_impact": 0.8  # 80% rule
            },
            "privacy": {
                "data_minimization": 0.95,
                "consent_compliance": 1.00,
                "breach_probability": base_tolerance.get("privacy", 0.05)
            },
            "safety": {
                "harm_rate": base_tolerance.get("safety", 0.01),
                "failure_rate": 0.001,
                "recovery_time": 60  # seconds
            }
        }
        
        return thresholds
    
    def generate_nist_documentation(self, assessment_data: Dict[str, Any]) -> NISTDocumentation:
        """
        Generate NIST-aligned documentation
        
        Args:
            assessment_data: Complete assessment data
            
        Returns:
            NISTDocumentation artifact
        """
        content = {
            "framework_version": self.framework_version,
            "assessment_date": datetime.now().isoformat(),
            "organization": self.config.get("organization", {}),
            "executive_summary": self._generate_executive_summary(assessment_data),
            "governance_assessment": self._summarize_governance(),
            "risk_mapping": self._summarize_mapping(),
            "risk_measurement": self._summarize_measurement(),
            "risk_management": self._summarize_management(),
            "trustworthiness_profile": self._generate_trustworthiness_profile(assessment_data),
            "maturity_assessment": self._assess_overall_maturity(),
            "recommendations": self._generate_recommendations(),
            "action_plan": self._generate_action_plan()
        }
        
        doc = NISTDocumentation(
            doc_id=f"NIST-DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            doc_type="nist_ai_rmf_assessment",
            version="1.0",
            created_date=datetime.now(),
            content=content
        )
        
        return doc
    
    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary"""
        total_assessments = len(self.assessments)
        implemented = sum(1 for a in self.assessments if a.status == "implemented")
        
        return (
            f"NIST AI RMF Assessment completed with {total_assessments} assessments. "
            f"{implemented}/{total_assessments} fully implemented. "
            f"Overall maturity level: {self._assess_overall_maturity()}/5."
        )
    
    def _summarize_governance(self) -> Dict[str, Any]:
        """Summarize governance assessments"""
        govern_assessments = [a for a in self.assessments if a.function == Function.GOVERN]
        
        return {
            "assessments_completed": len(govern_assessments),
            "average_maturity": sum(a.maturity_level for a in govern_assessments) / len(govern_assessments) if govern_assessments else 0,
            "key_findings": [f for a in govern_assessments for f in a.findings[:2]],
            "priority_recommendations": [r for a in govern_assessments for r in a.recommendations[:2]]
        }
    
    def _summarize_mapping(self) -> Dict[str, Any]:
        """Summarize risk mapping"""
        if not self.risk_maps:
            return {"status": "not_performed"}
        
        latest_map = self.risk_maps[-1]
        return {
            "risks_identified": len(latest_map.identified_risks),
            "risk_categories": list(latest_map.risk_categories.keys()),
            "stakeholders": latest_map.stakeholders,
            "high_priority_risks": [r for r in latest_map.identified_risks if r.get("impact") == "high"]
        }
    
    def _summarize_measurement(self) -> Dict[str, Any]:
        """Summarize risk measurements"""
        if not self.measurements:
            return {"status": "not_performed"}
        
        latest = self.measurements[-1]
        return {
            "measurement_date": latest.measurement_date.isoformat(),
            "risk_scores": latest.risk_scores,
            "violations": latest.threshold_violations,
            "trends": latest.trend_analysis
        }
    
    def _summarize_management(self) -> Dict[str, Any]:
        """Summarize risk management"""
        if not self.management_plans:
            return {"status": "not_performed"}
        
        latest = self.management_plans[-1]
        return {
            "treatments_defined": len(latest.risk_treatments),
            "controls_implemented": len(latest.controls),
            "monitoring_procedures": latest.monitoring_procedures,
            "review_schedule": latest.review_schedule
        }
    
    def _generate_trustworthiness_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trustworthiness profile"""
        profile = {}
        
        for characteristic in TrustworthinessCharacteristic:
            char_data = data.get(characteristic.value, {})
            profile[characteristic.value] = {
                "status": "assessed" if char_data else "not_assessed",
                "score": char_data.get("score", 0),
                "trend": char_data.get("trend", "unknown")
            }
        
        return profile
    
    def _assess_overall_maturity(self) -> int:
        """Assess overall NIST AI RMF maturity"""
        if not self.assessments:
            return 1
        
        maturity_scores = [a.maturity_level for a in self.assessments]
        return int(sum(maturity_scores) / len(maturity_scores))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations"""
        all_recommendations = []
        
        for assessment in self.assessments:
            for rec in assessment.recommendations:
                all_recommendations.append({
                    "recommendation": rec,
                    "function": assessment.function.value,
                    "priority": "high" if assessment.maturity_level <= 2 else "medium"
                })
        
        # Sort by priority and return top 10
        high_priority = [r for r in all_recommendations if r["priority"] == "high"]
        medium_priority = [r for r in all_recommendations if r["priority"] == "medium"]
        
        return high_priority[:5] + medium_priority[:5]
    
    def _generate_action_plan(self) -> Dict[str, Any]:
        """Generate action plan based on assessments"""
        return {
            "immediate_actions": [
                "Address high-priority findings",
                "Implement missing governance structures",
                "Establish risk monitoring procedures"
            ],
            "short_term": [
                "Improve workforce AI literacy",
                "Enhance documentation",
                "Implement automated monitoring"
            ],
            "long_term": [
                "Achieve target maturity level",
                "Establish continuous improvement process",
                "Full framework implementation"
            ],
            "timeline": {
                "immediate": "0-30 days",
                "short_term": "1-3 months",
                "long_term": "3-12 months"
            }
        }
    
    def perform_complete_assessment(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete NIST AI RMF assessment
        
        Args:
            system_info: Complete system information
            
        Returns:
            Complete assessment results
        """
        results = {
            "assessment_id": f"NIST-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "framework_version": self.framework_version,
            "assessment_date": datetime.now().isoformat(),
            "functions": {}
        }
        
        # GOVERN
        governance_result = self.assess_governance_function(system_info.get("governance", {}))
        results["functions"]["govern"] = governance_result.to_dict()
        
        # MAP
        risk_map = self.map_ai_risks(system_info.get("system_context", {}))
        results["functions"]["map"] = risk_map.to_dict()
        
        # MEASURE
        measurement_report = self.measure_ai_risks(system_info.get("metrics", {}))
        results["functions"]["measure"] = measurement_report.to_dict()
        
        # MANAGE
        management_plan = self.manage_ai_risks(system_info.get("controls", {}))
        results["functions"]["manage"] = management_plan.to_dict()
        
        # Trustworthiness assessment
        trustworthiness = self.assess_trustworthiness_characteristics(system_info)
        results["trustworthiness"] = {
            char.value: metrics.to_dict() if hasattr(metrics, 'to_dict') else asdict(metrics)
            for char, metrics in trustworthiness.items()
        }
        
        # Lifecycle assessment
        if "lifecycle_phase" in system_info:
            phase = LifecyclePhase(system_info["lifecycle_phase"])
            phase_result = self.assess_lifecycle_phase(phase, system_info.get("phase_data", {}))
            results["lifecycle_assessment"] = phase_result.to_dict()
        
        # Generate documentation
        documentation = self.generate_nist_documentation(system_info)
        results["documentation"] = documentation.content
        
        # Overall summary
        results["summary"] = {
            "overall_maturity": self._assess_overall_maturity(),
            "total_findings": sum(len(a.findings) for a in self.assessments),
            "high_priority_items": len([a for a in self.assessments if a.maturity_level <= 2]),
            "compliance_percentage": (sum(1 for a in self.assessments if a.status == "implemented") / 
                                     len(self.assessments) * 100) if self.assessments else 0
        }
        
        return results
    
    def export_assessment_report(self, filepath: Path, format: str = "json") -> None:
        """
        Export assessment report to file
        
        Args:
            filepath: Output file path
            format: Output format (json or yaml)
        """
        report = {
            "report_id": f"NIST-RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_date": datetime.now().isoformat(),
            "framework_version": self.framework_version,
            "assessments": [a.to_dict() for a in self.assessments],
            "risk_maps": [r.to_dict() for r in self.risk_maps],
            "measurements": [m.to_dict() for m in self.measurements],
            "management_plans": [p.to_dict() for p in self.management_plans],
            "summary": {
                "functions_assessed": len(self.functions),
                "overall_maturity": self._assess_overall_maturity(),
                "total_assessments": len(self.assessments),
                "total_risks_identified": sum(len(r.identified_risks) for r in self.risk_maps)
            }
        }
        
        with open(filepath, 'w') as f:
            if format == "json":
                json.dump(report, f, indent=2)
            elif format == "yaml":
                yaml.dump(report, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")