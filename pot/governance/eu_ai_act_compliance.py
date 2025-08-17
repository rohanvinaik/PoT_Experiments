"""
EU AI Act Compliance Module for PoT Framework
Implements compliance checks, risk assessment, and documentation generation
per Regulation (EU) 2024/1689 (AI Act)
"""

import json
import yaml
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib


class RiskCategory(Enum):
    """EU AI Act Risk Categories (Article 6)"""
    MINIMAL = "minimal_risk"
    LIMITED = "limited_risk"
    HIGH = "high_risk"
    UNACCEPTABLE = "unacceptable_risk"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceCheck:
    """Individual compliance check result"""
    check_id: str
    check_name: str
    article_reference: str
    status: ComplianceStatus
    findings: List[str]
    recommendations: List[str]
    evidence: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class ComplianceArtifact:
    """Base class for compliance artifacts"""
    artifact_id: str
    artifact_type: str
    version: str
    created_date: datetime
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        data = asdict(self)
        data['created_date'] = self.created_date.isoformat()
        return json.dumps(data, indent=2)
    
    def save(self, filepath: Path):
        """Save artifact to file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())


@dataclass
class TechnicalDocumentation(ComplianceArtifact):
    """Technical documentation per Article 11 and Annex IV"""
    def __init__(self, system_info: Dict[str, Any]):
        super().__init__(
            artifact_id=f"TECH-DOC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            artifact_type="technical_documentation",
            version="1.0",
            created_date=datetime.now(),
            content=self._generate_content(system_info)
        )
    
    def _generate_content(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical documentation structure per Annex IV"""
        return {
            "1_general_description": {
                "system_name": system_info.get("name", ""),
                "intended_purpose": system_info.get("intended_purpose", ""),
                "developer": system_info.get("developer", ""),
                "version": system_info.get("version", ""),
                "release_date": system_info.get("release_date", "")
            },
            "2_detailed_description": {
                "design_specifications": system_info.get("design_specs", {}),
                "system_architecture": system_info.get("architecture", {}),
                "computational_resources": system_info.get("resources", {}),
                "software_dependencies": system_info.get("dependencies", [])
            },
            "3_information_on_data": {
                "training_data": system_info.get("training_data", {}),
                "validation_data": system_info.get("validation_data", {}),
                "data_governance": system_info.get("data_governance", {}),
                "data_annotation": system_info.get("annotation_process", {})
            },
            "4_training_methodology": {
                "training_process": system_info.get("training_process", {}),
                "model_architecture": system_info.get("model_architecture", {}),
                "hyperparameters": system_info.get("hyperparameters", {}),
                "optimization": system_info.get("optimization", {})
            },
            "5_performance_metrics": {
                "accuracy_metrics": system_info.get("accuracy", {}),
                "robustness_metrics": system_info.get("robustness", {}),
                "fairness_metrics": system_info.get("fairness", {}),
                "explainability_metrics": system_info.get("explainability", {})
            },
            "6_testing_validation": {
                "test_procedures": system_info.get("test_procedures", []),
                "validation_results": system_info.get("validation_results", {}),
                "limitations": system_info.get("limitations", []),
                "foreseeable_misuse": system_info.get("misuse_scenarios", [])
            },
            "7_cybersecurity_measures": {
                "security_measures": system_info.get("security", {}),
                "resilience_testing": system_info.get("resilience", {}),
                "vulnerability_assessment": system_info.get("vulnerabilities", {})
            },
            "8_risk_management": {
                "risk_assessment": system_info.get("risk_assessment", {}),
                "mitigation_measures": system_info.get("mitigations", {}),
                "residual_risks": system_info.get("residual_risks", [])
            },
            "9_human_oversight": {
                "oversight_measures": system_info.get("human_oversight", {}),
                "user_interface": system_info.get("ui_design", {}),
                "interpretability": system_info.get("interpretability", {})
            },
            "10_accuracy_levels": {
                "expected_accuracy": system_info.get("expected_accuracy", {}),
                "accuracy_monitoring": system_info.get("monitoring", {}),
                "degradation_handling": system_info.get("degradation", {})
            }
        }


@dataclass
class ConformityDeclaration(ComplianceArtifact):
    """EU Declaration of Conformity per Article 47"""
    def __init__(self, system_info: Dict[str, Any], compliance_results: Dict[str, Any]):
        super().__init__(
            artifact_id=f"DOC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            artifact_type="declaration_of_conformity",
            version="1.0",
            created_date=datetime.now(),
            content=self._generate_declaration(system_info, compliance_results)
        )
    
    def _generate_declaration(self, system_info: Dict[str, Any], compliance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Declaration of Conformity per Article 47"""
        return {
            "declaration_number": self.artifact_id,
            "manufacturer": {
                "name": system_info.get("manufacturer_name", ""),
                "address": system_info.get("manufacturer_address", ""),
                "contact": system_info.get("manufacturer_contact", "")
            },
            "authorized_representative": {
                "name": system_info.get("representative_name", ""),
                "address": system_info.get("representative_address", "")
            },
            "ai_system": {
                "name": system_info.get("name", ""),
                "type": system_info.get("type", ""),
                "version": system_info.get("version", ""),
                "unique_identifier": system_info.get("uid", "")
            },
            "declaration_statement": (
                "This AI system is in conformity with Regulation (EU) 2024/1689 "
                "and relevant harmonized standards."
            ),
            "applicable_requirements": [
                "Article 8 - Compliance with requirements",
                "Article 9 - Risk management system",
                "Article 10 - Data and data governance",
                "Article 11 - Technical documentation",
                "Article 12 - Record-keeping",
                "Article 13 - Transparency and provision of information",
                "Article 14 - Human oversight",
                "Article 15 - Accuracy, robustness and cybersecurity"
            ],
            "harmonized_standards": system_info.get("standards", []),
            "notified_body": {
                "name": system_info.get("notified_body_name", ""),
                "number": system_info.get("notified_body_number", ""),
                "certificate_number": system_info.get("certificate_number", "")
            },
            "place_and_date": {
                "place": system_info.get("declaration_place", ""),
                "date": datetime.now().isoformat()
            },
            "signature": {
                "name": system_info.get("signatory_name", ""),
                "position": system_info.get("signatory_position", "")
            },
            "compliance_summary": compliance_results
        }


@dataclass
class RiskAssessmentReport(ComplianceArtifact):
    """Risk Assessment Report per Article 9 and Annex III"""
    def __init__(self, risk_data: Dict[str, Any]):
        super().__init__(
            artifact_id=f"RISK-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            artifact_type="risk_assessment_report",
            version="1.0",
            created_date=datetime.now(),
            content=self._generate_report(risk_data)
        )
    
    def _generate_report(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment report per Article 9"""
        return {
            "executive_summary": risk_data.get("summary", ""),
            "risk_category": risk_data.get("category", ""),
            "identified_risks": risk_data.get("risks", []),
            "risk_analysis": risk_data.get("analysis", {}),
            "mitigation_measures": risk_data.get("mitigations", []),
            "residual_risks": risk_data.get("residual", []),
            "testing_results": risk_data.get("testing", {}),
            "monitoring_plan": risk_data.get("monitoring", {}),
            "review_schedule": risk_data.get("review", {}),
            "fundamental_rights_assessment": risk_data.get("rights_assessment", {})
        }


class EUAIActCompliance:
    """
    EU AI Act Compliance Module
    Implements compliance checks per Regulation (EU) 2024/1689
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize EU AI Act compliance module"""
        self.risk_category: Optional[RiskCategory] = None
        self.requirements: Dict[str, Any] = self.load_requirements()
        self.compliance_checks: List[ComplianceCheck] = []
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if provided
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default compliance configuration"""
        return {
            "strict_mode": True,
            "generate_artifacts": True,
            "monitoring_enabled": True,
            "notification_thresholds": {
                "high_risk": 0.95,
                "limited_risk": 0.85,
                "minimal_risk": 0.70
            }
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def load_requirements(self) -> Dict[str, Any]:
        """Load EU AI Act requirements by risk category"""
        return {
            RiskCategory.MINIMAL: {
                "transparency": ["Article 52 - Transparency obligations"],
                "voluntary_codes": ["Article 95 - Codes of conduct"]
            },
            RiskCategory.LIMITED: {
                "transparency": ["Article 52 - Transparency obligations for certain AI systems"],
                "notification": ["Article 52(1) - Inform persons of interaction with AI"],
                "emotion_recognition": ["Article 52(2) - Emotion recognition systems"],
                "biometric": ["Article 52(3) - Biometric categorization systems"],
                "deepfakes": ["Article 52(4) - Deep fake content"]
            },
            RiskCategory.HIGH: {
                "risk_management": ["Article 9 - Risk management system"],
                "data_governance": ["Article 10 - Data and data governance"],
                "technical_documentation": ["Article 11 - Technical documentation", "Annex IV"],
                "record_keeping": ["Article 12 - Record-keeping"],
                "transparency": ["Article 13 - Transparency and provision of information"],
                "human_oversight": ["Article 14 - Human oversight"],
                "accuracy_robustness": ["Article 15 - Accuracy, robustness and cybersecurity"],
                "quality_management": ["Article 17 - Quality management system"],
                "conformity_assessment": ["Article 43 - Conformity assessment"],
                "ce_marking": ["Article 48 - CE marking"],
                "registration": ["Article 49 - Registration"]
            },
            RiskCategory.UNACCEPTABLE: {
                "prohibition": ["Article 5 - Prohibited AI practices"]
            }
        }
    
    def assess_risk_category(self, system_config: Dict[str, Any]) -> RiskCategory:
        """
        Determine AI system risk level per EU AI Act Article 6
        
        Args:
            system_config: Configuration and characteristics of the AI system
            
        Returns:
            RiskCategory enum value
        """
        # Check for prohibited practices (Article 5)
        if self._check_prohibited_practices(system_config):
            self.risk_category = RiskCategory.UNACCEPTABLE
            self.logger.warning("System classified as UNACCEPTABLE risk - prohibited under Article 5")
            return RiskCategory.UNACCEPTABLE
        
        # Check for high-risk applications (Article 6 and Annex III)
        if self._check_high_risk_applications(system_config):
            self.risk_category = RiskCategory.HIGH
            self.logger.info("System classified as HIGH risk per Article 6 and Annex III")
            return RiskCategory.HIGH
        
        # Check for limited risk applications (Article 52)
        if self._check_limited_risk_applications(system_config):
            self.risk_category = RiskCategory.LIMITED
            self.logger.info("System classified as LIMITED risk per Article 52")
            return RiskCategory.LIMITED
        
        # Default to minimal risk
        self.risk_category = RiskCategory.MINIMAL
        self.logger.info("System classified as MINIMAL risk")
        return RiskCategory.MINIMAL
    
    def _check_prohibited_practices(self, config: Dict[str, Any]) -> bool:
        """Check for prohibited AI practices per Article 5"""
        prohibited_checks = [
            # Article 5(1)(a) - Subliminal techniques
            config.get("uses_subliminal_techniques", False),
            # Article 5(1)(b) - Exploits vulnerabilities
            config.get("exploits_vulnerabilities", False),
            # Article 5(1)(c) - Social scoring by public authorities
            config.get("social_scoring_public", False),
            # Article 5(1)(d) - Real-time biometric identification in public
            (config.get("biometric_identification", False) and 
             config.get("real_time", False) and 
             config.get("public_spaces", False)),
            # Article 5(1)(e) - Facial recognition databases
            config.get("facial_recognition_scraping", False),
            # Article 5(1)(f) - Emotion recognition in workplace/education
            (config.get("emotion_recognition", False) and 
             config.get("context", "") in ["workplace", "education"]),
            # Article 5(1)(g) - Biometric categorization (sensitive)
            (config.get("biometric_categorization", False) and 
             config.get("sensitive_attributes", False)),
            # Article 5(1)(h) - Individual risk assessment
            config.get("individual_criminal_risk", False)
        ]
        
        return any(prohibited_checks)
    
    def _check_high_risk_applications(self, config: Dict[str, Any]) -> bool:
        """Check for high-risk applications per Article 6 and Annex III"""
        # Annex III high-risk areas
        high_risk_areas = [
            # 1. Biometric identification
            (config.get("application_area", "") == "biometric_identification"),
            # 2. Critical infrastructure
            (config.get("application_area", "") == "critical_infrastructure"),
            # 3. Education and vocational training
            (config.get("application_area", "") == "education"),
            # 4. Employment
            (config.get("application_area", "") == "employment"),
            # 5. Essential services
            (config.get("application_area", "") in ["healthcare", "banking", "insurance"]),
            # 6. Law enforcement
            (config.get("application_area", "") == "law_enforcement"),
            # 7. Migration and border control
            (config.get("application_area", "") == "border_control"),
            # 8. Justice and democratic processes
            (config.get("application_area", "") in ["justice", "democratic_process"])
        ]
        
        # Also check if it's a safety component (Article 6(1))
        is_safety_component = config.get("safety_component", False)
        
        return any(high_risk_areas) or is_safety_component
    
    def _check_limited_risk_applications(self, config: Dict[str, Any]) -> bool:
        """Check for limited risk applications per Article 52"""
        limited_risk_checks = [
            config.get("ai_interaction", False),  # Systems interacting with humans
            config.get("emotion_recognition", False),  # Emotion recognition
            config.get("biometric_categorization", False),  # Biometric categorization
            config.get("content_generation", False),  # Deep fakes or synthetic content
            config.get("chatbot", False)  # Chatbot or conversational AI
        ]
        
        return any(limited_risk_checks)
    
    def check_transparency_requirements(self, model_info: Dict[str, Any]) -> ComplianceStatus:
        """
        Verify transparency obligations are met per Articles 13 and 52
        
        Args:
            model_info: Information about the model and its documentation
            
        Returns:
            ComplianceStatus indicating compliance level
        """
        findings = []
        recommendations = []
        
        # Article 13 requirements for high-risk systems
        if self.risk_category == RiskCategory.HIGH:
            required_info = [
                ("provider_identity", "Provider name and contact details"),
                ("system_capabilities", "Capabilities and limitations"),
                ("accuracy_levels", "Level of accuracy, robustness, and cybersecurity"),
                ("intended_purpose", "Intended purpose"),
                ("human_oversight", "Human oversight measures"),
                ("performance_metrics", "Performance metrics"),
                ("risks_and_mitigations", "Foreseeable risks and mitigation measures")
            ]
            
            for key, description in required_info:
                if key not in model_info or not model_info[key]:
                    findings.append(f"Missing: {description} (Article 13)")
                    recommendations.append(f"Add {description} to documentation")
        
        # Article 52 requirements for limited risk systems
        if self.risk_category in [RiskCategory.LIMITED, RiskCategory.HIGH]:
            if model_info.get("ai_interaction", False):
                if not model_info.get("ai_disclosure", False):
                    findings.append("AI system interaction not disclosed (Article 52(1))")
                    recommendations.append("Inform users they are interacting with an AI system")
            
            if model_info.get("emotion_recognition", False) or model_info.get("biometric_categorization", False):
                if not model_info.get("system_operation_disclosed", False):
                    findings.append("System operation not disclosed (Article 52(2))")
                    recommendations.append("Inform users of emotion recognition/biometric categorization")
            
            if model_info.get("synthetic_content", False):
                if not model_info.get("synthetic_content_marked", False):
                    findings.append("Synthetic content not marked (Article 52(4))")
                    recommendations.append("Mark artificially generated content appropriately")
        
        # Determine compliance status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 2:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        # Log compliance check
        check = ComplianceCheck(
            check_id=f"TRANS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Transparency Requirements",
            article_reference="Articles 13, 52",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"model_info": model_info},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def validate_human_oversight(self, governance_config: Dict[str, Any]) -> bool:
        """
        Ensure human oversight mechanisms are in place per Article 14
        
        Args:
            governance_config: Governance and oversight configuration
            
        Returns:
            True if human oversight requirements are met
        """
        required_capabilities = []
        findings = []
        
        # Article 14(3) - Human oversight capabilities
        oversight_requirements = [
            ("understand_capabilities", "Understand AI system capabilities and limitations"),
            ("monitor_operation", "Monitor system operation"),
            ("interpret_outputs", "Correctly interpret system outputs"),
            ("decide_not_use", "Decide not to use or disregard output"),
            ("intervene_operation", "Intervene or interrupt operation"),
            ("stop_button", "Stop system operation (kill switch)")
        ]
        
        for key, description in oversight_requirements:
            if not governance_config.get(key, False):
                findings.append(f"Missing capability: {description}")
        
        # Article 14(4) - Specific measures for high-risk systems
        if self.risk_category == RiskCategory.HIGH:
            if not governance_config.get("oversight_by_design", False):
                findings.append("Human oversight not implemented by design (Article 14(4)(a))")
            
            if not governance_config.get("oversight_training", False):
                findings.append("No training for human overseers (Article 14(4)(b))")
            
            if governance_config.get("automation_bias_measures", False) is False:
                findings.append("No measures against automation bias (Article 14(5))")
        
        # Log compliance check
        is_compliant = len(findings) == 0
        status = ComplianceStatus.COMPLIANT if is_compliant else ComplianceStatus.NON_COMPLIANT
        
        check = ComplianceCheck(
            check_id=f"HUMAN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Human Oversight Validation",
            article_reference="Article 14",
            status=status,
            findings=findings,
            recommendations=[f"Implement: {f}" for f in findings],
            evidence={"governance_config": governance_config},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return is_compliant
    
    def check_data_governance(self, data_policy: Dict[str, Any]) -> ComplianceStatus:
        """
        Verify data quality and governance measures per Article 10
        
        Args:
            data_policy: Data governance policies and practices
            
        Returns:
            ComplianceStatus indicating compliance level
        """
        findings = []
        recommendations = []
        
        # Article 10(2) - Training, validation and testing data requirements
        data_requirements = [
            ("relevant_data", "Data must be relevant", Article 10(2)"),
            ("representative_data", "Data must be representative (Article 10(2))"),
            ("error_free", "Data must be free of errors (Article 10(2))"),
            ("complete_data", "Data must be complete (Article 10(2))")
        ]
        
        for key, description in data_requirements:
            if not data_policy.get(key, False):
                findings.append(f"Data requirement not met: {description}")
        
        # Article 10(3) - Data governance and management
        if not data_policy.get("data_collection_process", False):
            findings.append("No documented data collection process (Article 10(3))")
            recommendations.append("Document data collection procedures")
        
        if not data_policy.get("data_preparation", False):
            findings.append("No data preparation procedures (Article 10(3))")
            recommendations.append("Document data cleaning and labeling procedures")
        
        if not data_policy.get("bias_assessment", False):
            findings.append("No bias assessment procedures (Article 10(3))")
            recommendations.append("Implement bias detection and mitigation")
        
        if not data_policy.get("data_relevance_assessment", False):
            findings.append("No data relevance assessment (Article 10(3))")
            recommendations.append("Assess data relevance for intended purpose")
        
        # Article 10(4) - Examination for biases
        if not data_policy.get("bias_monitoring", False):
            findings.append("No bias monitoring in place (Article 10(4))")
            recommendations.append("Implement continuous bias monitoring")
        
        # Article 10(5) - Special category data processing
        if data_policy.get("special_category_data", False):
            if not data_policy.get("special_data_safeguards", False):
                findings.append("Insufficient safeguards for special category data (Article 10(5))")
                recommendations.append("Implement appropriate safeguards for sensitive data")
        
        # Determine compliance status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 3:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        # Log compliance check
        check = ComplianceCheck(
            check_id=f"DATA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Data Governance Check",
            article_reference="Article 10",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"data_policy": data_policy},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def check_risk_management(self, risk_config: Dict[str, Any]) -> ComplianceStatus:
        """
        Verify risk management system per Article 9
        
        Args:
            risk_config: Risk management configuration
            
        Returns:
            ComplianceStatus
        """
        findings = []
        recommendations = []
        
        # Article 9(2) - Risk management process requirements
        risk_requirements = [
            ("risk_identification", "Risk identification and analysis (Article 9(2)(a))"),
            ("risk_estimation", "Risk estimation and evaluation (Article 9(2)(b))"),
            ("risk_mitigation", "Risk mitigation measures (Article 9(2)(c))"),
            ("residual_risk", "Residual risk assessment (Article 9(2)(d))")
        ]
        
        for key, description in risk_requirements:
            if not risk_config.get(key, False):
                findings.append(f"Missing: {description}")
                recommendations.append(f"Implement {description}")
        
        # Article 9(3) - Testing procedures
        if not risk_config.get("testing_procedures", False):
            findings.append("No testing procedures defined (Article 9(3))")
            recommendations.append("Define and document testing procedures")
        
        # Article 9(4) - Continuous improvement
        if not risk_config.get("continuous_improvement", False):
            findings.append("No continuous improvement process (Article 9(4))")
            recommendations.append("Implement iterative improvement process")
        
        # Article 9(5) - Foreseeable misuse
        if not risk_config.get("misuse_assessment", False):
            findings.append("No foreseeable misuse assessment (Article 9(5))")
            recommendations.append("Assess and document foreseeable misuse scenarios")
        
        # Article 9(7) - Regular updates
        if not risk_config.get("regular_updates", False):
            findings.append("No regular update schedule (Article 9(7))")
            recommendations.append("Establish regular risk assessment updates")
        
        # Determine status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 2:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        check = ComplianceCheck(
            check_id=f"RISK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Risk Management System",
            article_reference="Article 9",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"risk_config": risk_config},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def check_accuracy_robustness(self, performance_data: Dict[str, Any]) -> ComplianceStatus:
        """
        Check accuracy, robustness and cybersecurity per Article 15
        
        Args:
            performance_data: System performance and security data
            
        Returns:
            ComplianceStatus
        """
        findings = []
        recommendations = []
        
        # Article 15(1) - Accuracy levels
        if not performance_data.get("accuracy_metrics", False):
            findings.append("No accuracy metrics defined (Article 15(1))")
            recommendations.append("Define and measure accuracy metrics")
        
        if not performance_data.get("accuracy_declared", False):
            findings.append("Accuracy levels not declared (Article 15(1))")
            recommendations.append("Declare accuracy levels in documentation")
        
        # Article 15(2) - Robustness
        if not performance_data.get("robustness_testing", False):
            findings.append("No robustness testing performed (Article 15(2))")
            recommendations.append("Perform robustness and resilience testing")
        
        # Article 15(3) - Cybersecurity
        security_requirements = [
            ("security_by_design", "Security by design implementation"),
            ("vulnerability_assessment", "Vulnerability assessment"),
            ("security_testing", "Security testing"),
            ("incident_response", "Incident response plan")
        ]
        
        for key, description in security_requirements:
            if not performance_data.get(key, False):
                findings.append(f"Missing: {description} (Article 15(3))")
                recommendations.append(f"Implement {description}")
        
        # Article 15(4) - Technical redundancy
        if self.risk_category == RiskCategory.HIGH:
            if not performance_data.get("redundancy_measures", False):
                findings.append("No redundancy measures (Article 15(4))")
                recommendations.append("Implement technical redundancy solutions")
        
        # Determine status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 2:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        check = ComplianceCheck(
            check_id=f"PERF-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Accuracy, Robustness & Cybersecurity",
            article_reference="Article 15",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"performance_data": performance_data},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def check_biometric_safeguards(self, biometric_config: Dict[str, Any]) -> ComplianceStatus:
        """
        Check biometric identification safeguards for high-risk systems
        
        Args:
            biometric_config: Biometric system configuration
            
        Returns:
            ComplianceStatus
        """
        findings = []
        recommendations = []
        
        if not biometric_config.get("is_biometric", False):
            return ComplianceStatus.NOT_APPLICABLE
        
        # Special requirements for biometric systems
        if biometric_config.get("real_time", False):
            findings.append("Real-time biometric identification requires special authorization")
            recommendations.append("Obtain necessary authorizations per Article 5")
        
        # Article 10(5) - Special category data
        if not biometric_config.get("data_minimization", False):
            findings.append("No data minimization measures")
            recommendations.append("Implement data minimization per Article 10(5)")
        
        if not biometric_config.get("purpose_limitation", False):
            findings.append("No purpose limitation safeguards")
            recommendations.append("Implement strict purpose limitation")
        
        if not biometric_config.get("consent_mechanism", False):
            findings.append("No explicit consent mechanism")
            recommendations.append("Implement consent collection and management")
        
        # Logging and audit requirements
        if not biometric_config.get("access_logging", False):
            findings.append("No access logging for biometric data")
            recommendations.append("Implement comprehensive access logging")
        
        # Determine status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 2:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        check = ComplianceCheck(
            check_id=f"BIO-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Biometric Safeguards",
            article_reference="Articles 5, 10(5)",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"biometric_config": biometric_config},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def check_quality_management(self, qms_config: Dict[str, Any]) -> ComplianceStatus:
        """
        Verify quality management system for high-risk AI per Article 17
        
        Args:
            qms_config: Quality management system configuration
            
        Returns:
            ComplianceStatus
        """
        if self.risk_category != RiskCategory.HIGH:
            return ComplianceStatus.NOT_APPLICABLE
        
        findings = []
        recommendations = []
        
        # Article 17(1) - QMS requirements
        qms_requirements = [
            ("regulatory_compliance_strategy", "Regulatory compliance strategy"),
            ("design_control", "Design and development controls"),
            ("quality_control", "Quality control and assurance"),
            ("resource_management", "Resource management"),
            ("data_management", "Data management framework"),
            ("risk_management_integration", "Risk management integration"),
            ("monitoring_procedures", "Post-market monitoring"),
            ("incident_reporting", "Incident reporting procedures"),
            ("communication_procedures", "Communication with authorities"),
            ("document_control", "Document and record control")
        ]
        
        for key, description in qms_requirements:
            if not qms_config.get(key, False):
                findings.append(f"Missing QMS element: {description} (Article 17(1))")
                recommendations.append(f"Implement {description}")
        
        # Determine status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 3:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        check = ComplianceCheck(
            check_id=f"QMS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Quality Management System",
            article_reference="Article 17",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"qms_config": qms_config},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def check_post_market_monitoring(self, monitoring_config: Dict[str, Any]) -> ComplianceStatus:
        """
        Check post-market monitoring setup per Article 72
        
        Args:
            monitoring_config: Post-market monitoring configuration
            
        Returns:
            ComplianceStatus
        """
        findings = []
        recommendations = []
        
        # Article 72 requirements
        monitoring_requirements = [
            ("monitoring_plan", "Post-market monitoring plan"),
            ("performance_monitoring", "Performance monitoring system"),
            ("incident_detection", "Incident detection mechanism"),
            ("feedback_collection", "User feedback collection"),
            ("update_procedures", "Update and improvement procedures"),
            ("reporting_mechanism", "Reporting to authorities")
        ]
        
        for key, description in monitoring_requirements:
            if not monitoring_config.get(key, False):
                findings.append(f"Missing: {description} (Article 72)")
                recommendations.append(f"Implement {description}")
        
        # Determine status
        if not findings:
            status = ComplianceStatus.COMPLIANT
        elif len(findings) <= 2:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        check = ComplianceCheck(
            check_id=f"MON-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            check_name="Post-Market Monitoring",
            article_reference="Article 72",
            status=status,
            findings=findings,
            recommendations=recommendations,
            evidence={"monitoring_config": monitoring_config},
            timestamp=datetime.now()
        )
        
        self.compliance_checks.append(check)
        return status
    
    def perform_fundamental_rights_assessment(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform fundamental rights impact assessment per Article 27
        
        Args:
            system_info: System information and context
            
        Returns:
            Assessment results
        """
        assessment = {
            "assessment_id": f"FRA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "system_name": system_info.get("name", ""),
            "risk_category": self.risk_category.value if self.risk_category else "unknown",
            "rights_impacts": [],
            "mitigation_measures": [],
            "overall_risk": "low"
        }
        
        # Check impacts on fundamental rights
        rights_checks = [
            ("human_dignity", "Human dignity"),
            ("non_discrimination", "Non-discrimination"),
            ("data_protection", "Data protection and privacy"),
            ("freedom_expression", "Freedom of expression"),
            ("access_to_justice", "Access to justice"),
            ("consumer_protection", "Consumer protection"),
            ("workers_rights", "Workers' rights"),
            ("childrens_rights", "Children's rights")
        ]
        
        impact_count = 0
        for key, right in rights_checks:
            if system_info.get(f"impacts_{key}", False):
                assessment["rights_impacts"].append({
                    "right": right,
                    "impact_level": system_info.get(f"{key}_impact_level", "low"),
                    "description": system_info.get(f"{key}_impact_description", "")
                })
                impact_count += 1
        
        # Determine overall risk level
        if impact_count == 0:
            assessment["overall_risk"] = "minimal"
        elif impact_count <= 2:
            assessment["overall_risk"] = "low"
        elif impact_count <= 4:
            assessment["overall_risk"] = "medium"
        else:
            assessment["overall_risk"] = "high"
        
        # Add mitigation measures
        if assessment["overall_risk"] in ["medium", "high"]:
            assessment["mitigation_measures"] = [
                "Implement enhanced transparency measures",
                "Establish human review processes",
                "Regular bias audits",
                "Stakeholder consultation",
                "Impact monitoring and reporting"
            ]
        
        return assessment
    
    def generate_technical_documentation(self, system_info: Dict[str, Any]) -> TechnicalDocumentation:
        """
        Generate technical documentation per Article 11 and Annex IV
        
        Args:
            system_info: Complete system information
            
        Returns:
            TechnicalDocumentation artifact
        """
        return TechnicalDocumentation(system_info)
    
    def generate_conformity_declaration(
        self, 
        system_info: Dict[str, Any]
    ) -> ConformityDeclaration:
        """
        Generate EU Declaration of Conformity per Article 47
        
        Args:
            system_info: System information
            
        Returns:
            ConformityDeclaration artifact
        """
        # Compile compliance results
        compliance_summary = {
            "total_checks": len(self.compliance_checks),
            "compliant": sum(1 for c in self.compliance_checks if c.status == ComplianceStatus.COMPLIANT),
            "partially_compliant": sum(1 for c in self.compliance_checks if c.status == ComplianceStatus.PARTIALLY_COMPLIANT),
            "non_compliant": sum(1 for c in self.compliance_checks if c.status == ComplianceStatus.NON_COMPLIANT),
            "checks": [c.to_dict() for c in self.compliance_checks[-5:]]  # Last 5 checks
        }
        
        return ConformityDeclaration(system_info, compliance_summary)
    
    def generate_risk_assessment_report(self, risk_data: Dict[str, Any]) -> RiskAssessmentReport:
        """
        Generate risk assessment report per Article 9
        
        Args:
            risk_data: Risk assessment data
            
        Returns:
            RiskAssessmentReport artifact
        """
        # Add risk category
        risk_data["category"] = self.risk_category.value if self.risk_category else "unknown"
        
        # Add fundamental rights assessment
        if "system_info" in risk_data:
            risk_data["rights_assessment"] = self.perform_fundamental_rights_assessment(
                risk_data["system_info"]
            )
        
        return RiskAssessmentReport(risk_data)
    
    def perform_conformity_assessment(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform conformity assessment per Article 43
        
        Args:
            system_config: Complete system configuration
            
        Returns:
            Conformity assessment results
        """
        assessment_results = {
            "assessment_id": f"CA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "risk_category": self.risk_category.value if self.risk_category else "unknown",
            "assessments": {},
            "overall_conformity": True,
            "requires_notified_body": False
        }
        
        # Determine if notified body involvement is required (Article 43(1))
        if self.risk_category == RiskCategory.HIGH:
            if system_config.get("uses_annex_i_techniques", False):
                assessment_results["requires_notified_body"] = True
        
        # Perform individual assessments
        assessments = [
            ("transparency", self.check_transparency_requirements(system_config)),
            ("human_oversight", ComplianceStatus.COMPLIANT if self.validate_human_oversight(system_config) else ComplianceStatus.NON_COMPLIANT),
            ("data_governance", self.check_data_governance(system_config)),
            ("risk_management", self.check_risk_management(system_config)),
            ("accuracy_robustness", self.check_accuracy_robustness(system_config)),
            ("quality_management", self.check_quality_management(system_config)),
            ("post_market_monitoring", self.check_post_market_monitoring(system_config))
        ]
        
        for name, status in assessments:
            assessment_results["assessments"][name] = status.value
            if status not in [ComplianceStatus.COMPLIANT, ComplianceStatus.NOT_APPLICABLE]:
                assessment_results["overall_conformity"] = False
        
        return assessment_results
    
    def generate_compliance_package(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete compliance documentation package
        
        Args:
            system_info: Complete system information
            
        Returns:
            Dictionary containing all compliance artifacts
        """
        # First, assess risk category
        self.assess_risk_category(system_info)
        
        # Perform all compliance checks
        conformity_results = self.perform_conformity_assessment(system_info)
        
        # Generate artifacts
        package = {
            "package_id": f"PKG-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_date": datetime.now().isoformat(),
            "risk_category": self.risk_category.value,
            "conformity_assessment": conformity_results,
            "artifacts": {}
        }
        
        # Generate technical documentation for high-risk systems
        if self.risk_category == RiskCategory.HIGH:
            tech_doc = self.generate_technical_documentation(system_info)
            package["artifacts"]["technical_documentation"] = tech_doc.to_dict()
            
            # Generate conformity declaration
            conformity_dec = self.generate_conformity_declaration(system_info)
            package["artifacts"]["conformity_declaration"] = conformity_dec.to_dict()
        
        # Generate risk assessment report
        risk_report = self.generate_risk_assessment_report({
            "system_info": system_info,
            "risks": system_info.get("identified_risks", []),
            "mitigations": system_info.get("mitigation_measures", []),
            "testing": system_info.get("testing_results", {})
        })
        package["artifacts"]["risk_assessment"] = risk_report.to_dict()
        
        # Add fundamental rights assessment
        fra = self.perform_fundamental_rights_assessment(system_info)
        package["artifacts"]["fundamental_rights_assessment"] = fra
        
        # Add compliance check history
        package["compliance_checks"] = [check.to_dict() for check in self.compliance_checks]
        
        return package
    
    def export_compliance_report(self, filepath: Path, format: str = "json") -> None:
        """
        Export compliance report to file
        
        Args:
            filepath: Output file path
            format: Output format (json or yaml)
        """
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_date": datetime.now().isoformat(),
            "risk_category": self.risk_category.value if self.risk_category else "unknown",
            "compliance_checks": [check.to_dict() for check in self.compliance_checks],
            "summary": {
                "total_checks": len(self.compliance_checks),
                "compliant": sum(1 for c in self.compliance_checks if c.status == ComplianceStatus.COMPLIANT),
                "non_compliant": sum(1 for c in self.compliance_checks if c.status == ComplianceStatus.NON_COMPLIANT),
                "requires_review": sum(1 for c in self.compliance_checks if c.status == ComplianceStatus.REQUIRES_REVIEW)
            }
        }
        
        with open(filepath, 'w') as f:
            if format == "json":
                json.dump(report, f, indent=2)
            elif format == "yaml":
                yaml.dump(report, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")