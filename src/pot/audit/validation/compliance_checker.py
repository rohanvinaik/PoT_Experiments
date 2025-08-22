"""
Compliance Checker

Validates audit trails and systems against regulatory requirements and standards.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Supported compliance standards"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    FIPS = "fips"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    standard: ComplianceStandard
    requirement_id: str
    description: str
    category: str
    severity: str  # critical, high, medium, low
    validation_function: Optional[str] = None


@dataclass
class ComplianceViolation:
    """Detected compliance violation"""
    requirement: ComplianceRequirement
    violation_type: str
    description: str
    evidence: Dict[str, Any]
    remediation: str


@dataclass
class ComplianceReport:
    """Comprehensive compliance assessment report"""
    standards_checked: List[ComplianceStandard]
    is_compliant: bool
    compliance_scores: Dict[str, float]
    violations: List[ComplianceViolation]
    warnings: List[str]
    recommendations: List[str]
    audit_summary: Dict[str, Any]


class ComplianceChecker:
    """
    Validates systems against regulatory compliance requirements.
    
    Features:
    - Multi-standard compliance checking
    - Automated requirement validation
    - Violation detection and reporting
    - Remediation recommendations
    - Audit trail compliance verification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the compliance checker.
        
        Args:
            config: Configuration for compliance checking
        """
        self.config = config or {}
        self.standards = self.config.get('standards', [ComplianceStandard.SOC2])
        self.strict_mode = self.config.get('strict_mode', False)
        self.requirements = self._load_requirements()
        
    def _load_requirements(self) -> Dict[ComplianceStandard, List[ComplianceRequirement]]:
        """Load compliance requirements for each standard"""
        requirements = {
            ComplianceStandard.GDPR: self._gdpr_requirements(),
            ComplianceStandard.HIPAA: self._hipaa_requirements(),
            ComplianceStandard.SOC2: self._soc2_requirements(),
            ComplianceStandard.ISO27001: self._iso27001_requirements(),
            ComplianceStandard.NIST: self._nist_requirements(),
            ComplianceStandard.PCI_DSS: self._pci_dss_requirements(),
            ComplianceStandard.CCPA: self._ccpa_requirements(),
            ComplianceStandard.FIPS: self._fips_requirements()
        }
        return requirements
    
    def check_compliance(
        self,
        audit_data: Dict[str, Any],
        standards: Optional[List[ComplianceStandard]] = None
    ) -> ComplianceReport:
        """
        Check compliance against specified standards.
        
        Args:
            audit_data: Audit data to validate
            standards: Standards to check (uses configured if None)
            
        Returns:
            ComplianceReport with detailed findings
        """
        if standards is None:
            standards = self.standards
        
        violations = []
        warnings = []
        compliance_scores = {}
        
        for standard in standards:
            standard_violations, standard_score = self._check_standard(
                standard,
                audit_data
            )
            violations.extend(standard_violations)
            compliance_scores[standard.value] = standard_score
            
            # Generate warnings for near-violations
            if standard_score < 0.9 and standard_score > 0.7:
                warnings.append(f"{standard.value}: Score {standard_score:.2f} indicates potential issues")
        
        # Overall compliance determination
        is_compliant = len([v for v in violations if v.requirement.severity in ['critical', 'high']]) == 0
        
        if self.strict_mode:
            is_compliant = len(violations) == 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, compliance_scores)
        
        # Create audit summary
        audit_summary = self._create_audit_summary(audit_data, violations)
        
        return ComplianceReport(
            standards_checked=standards,
            is_compliant=is_compliant,
            compliance_scores=compliance_scores,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            audit_summary=audit_summary
        )
    
    def _check_standard(
        self,
        standard: ComplianceStandard,
        audit_data: Dict[str, Any]
    ) -> Tuple[List[ComplianceViolation], float]:
        """
        Check compliance for a specific standard.
        
        Args:
            standard: Compliance standard to check
            audit_data: Audit data to validate
            
        Returns:
            Tuple of (violations, compliance_score)
        """
        requirements = self.requirements.get(standard, [])
        violations = []
        passed = 0
        
        for req in requirements:
            if req.validation_function:
                validator = getattr(self, req.validation_function, None)
                if validator:
                    violation = validator(audit_data, req)
                    if violation:
                        violations.append(violation)
                    else:
                        passed += 1
        
        total = len(requirements)
        score = passed / total if total > 0 else 1.0
        
        return violations, score
    
    def _gdpr_requirements(self) -> List[ComplianceRequirement]:
        """Define GDPR compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="GDPR-1",
                description="Right to erasure (right to be forgotten)",
                category="data_rights",
                severity="critical",
                validation_function="_validate_data_erasure"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="GDPR-2",
                description="Data portability",
                category="data_rights",
                severity="high",
                validation_function="_validate_data_portability"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="GDPR-3",
                description="Consent management",
                category="consent",
                severity="critical",
                validation_function="_validate_consent_management"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="GDPR-4",
                description="Data breach notification (72 hours)",
                category="incident_response",
                severity="critical",
                validation_function="_validate_breach_notification"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.GDPR,
                requirement_id="GDPR-5",
                description="Privacy by design",
                category="privacy",
                severity="high",
                validation_function="_validate_privacy_by_design"
            )
        ]
    
    def _hipaa_requirements(self) -> List[ComplianceRequirement]:
        """Define HIPAA compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.HIPAA,
                requirement_id="HIPAA-1",
                description="PHI encryption at rest and in transit",
                category="encryption",
                severity="critical",
                validation_function="_validate_phi_encryption"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.HIPAA,
                requirement_id="HIPAA-2",
                description="Access controls and authentication",
                category="access_control",
                severity="critical",
                validation_function="_validate_access_controls"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.HIPAA,
                requirement_id="HIPAA-3",
                description="Audit logging for PHI access",
                category="audit",
                severity="critical",
                validation_function="_validate_phi_audit_logging"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.HIPAA,
                requirement_id="HIPAA-4",
                description="Business Associate Agreements",
                category="contracts",
                severity="high",
                validation_function="_validate_baa_compliance"
            )
        ]
    
    def _soc2_requirements(self) -> List[ComplianceRequirement]:
        """Define SOC2 compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.SOC2,
                requirement_id="SOC2-1",
                description="Security: Protection against unauthorized access",
                category="security",
                severity="critical",
                validation_function="_validate_unauthorized_access_protection"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.SOC2,
                requirement_id="SOC2-2",
                description="Availability: System uptime and performance",
                category="availability",
                severity="high",
                validation_function="_validate_system_availability"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.SOC2,
                requirement_id="SOC2-3",
                description="Processing Integrity: Complete and accurate processing",
                category="integrity",
                severity="high",
                validation_function="_validate_processing_integrity"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.SOC2,
                requirement_id="SOC2-4",
                description="Confidentiality: Protection of confidential information",
                category="confidentiality",
                severity="critical",
                validation_function="_validate_confidentiality"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.SOC2,
                requirement_id="SOC2-5",
                description="Privacy: Personal information handling",
                category="privacy",
                severity="high",
                validation_function="_validate_privacy_controls"
            )
        ]
    
    def _iso27001_requirements(self) -> List[ComplianceRequirement]:
        """Define ISO 27001 compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.ISO27001,
                requirement_id="ISO-1",
                description="Information security risk assessment",
                category="risk_management",
                severity="critical",
                validation_function="_validate_risk_assessment"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.ISO27001,
                requirement_id="ISO-2",
                description="Asset management and classification",
                category="asset_management",
                severity="high",
                validation_function="_validate_asset_management"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.ISO27001,
                requirement_id="ISO-3",
                description="Incident management procedures",
                category="incident_response",
                severity="high",
                validation_function="_validate_incident_management"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.ISO27001,
                requirement_id="ISO-4",
                description="Business continuity planning",
                category="continuity",
                severity="high",
                validation_function="_validate_business_continuity"
            )
        ]
    
    def _nist_requirements(self) -> List[ComplianceRequirement]:
        """Define NIST compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.NIST,
                requirement_id="NIST-1",
                description="Identify: Asset management",
                category="identify",
                severity="high",
                validation_function="_validate_nist_identify"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.NIST,
                requirement_id="NIST-2",
                description="Protect: Access control",
                category="protect",
                severity="critical",
                validation_function="_validate_nist_protect"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.NIST,
                requirement_id="NIST-3",
                description="Detect: Anomaly detection",
                category="detect",
                severity="high",
                validation_function="_validate_nist_detect"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.NIST,
                requirement_id="NIST-4",
                description="Respond: Incident response",
                category="respond",
                severity="high",
                validation_function="_validate_nist_respond"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.NIST,
                requirement_id="NIST-5",
                description="Recover: Recovery planning",
                category="recover",
                severity="medium",
                validation_function="_validate_nist_recover"
            )
        ]
    
    def _pci_dss_requirements(self) -> List[ComplianceRequirement]:
        """Define PCI-DSS compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.PCI_DSS,
                requirement_id="PCI-1",
                description="Cardholder data encryption",
                category="encryption",
                severity="critical",
                validation_function="_validate_cardholder_encryption"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.PCI_DSS,
                requirement_id="PCI-2",
                description="Network segmentation",
                category="network",
                severity="high",
                validation_function="_validate_network_segmentation"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.PCI_DSS,
                requirement_id="PCI-3",
                description="Regular security testing",
                category="testing",
                severity="high",
                validation_function="_validate_security_testing"
            )
        ]
    
    def _ccpa_requirements(self) -> List[ComplianceRequirement]:
        """Define CCPA compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.CCPA,
                requirement_id="CCPA-1",
                description="Right to know about personal information",
                category="transparency",
                severity="high",
                validation_function="_validate_ccpa_transparency"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.CCPA,
                requirement_id="CCPA-2",
                description="Right to delete personal information",
                category="data_rights",
                severity="critical",
                validation_function="_validate_ccpa_deletion"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.CCPA,
                requirement_id="CCPA-3",
                description="Right to opt-out of sale",
                category="consent",
                severity="high",
                validation_function="_validate_ccpa_opt_out"
            )
        ]
    
    def _fips_requirements(self) -> List[ComplianceRequirement]:
        """Define FIPS compliance requirements"""
        return [
            ComplianceRequirement(
                standard=ComplianceStandard.FIPS,
                requirement_id="FIPS-1",
                description="Approved cryptographic algorithms",
                category="cryptography",
                severity="critical",
                validation_function="_validate_fips_crypto"
            ),
            ComplianceRequirement(
                standard=ComplianceStandard.FIPS,
                requirement_id="FIPS-2",
                description="Key management procedures",
                category="key_management",
                severity="critical",
                validation_function="_validate_fips_key_management"
            )
        ]
    
    # Validation functions for requirements
    
    def _validate_data_erasure(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate data erasure capability"""
        erasure_logs = audit_data.get('erasure_logs', [])
        erasure_requests = audit_data.get('erasure_requests', [])
        
        if not erasure_logs and erasure_requests:
            return ComplianceViolation(
                requirement=requirement,
                violation_type="missing_capability",
                description="No data erasure logs found despite erasure requests",
                evidence={'requests': len(erasure_requests), 'logs': 0},
                remediation="Implement data erasure functionality with audit logging"
            )
        
        # Check erasure completeness
        for request in erasure_requests:
            if not any(log['request_id'] == request['id'] for log in erasure_logs):
                return ComplianceViolation(
                    requirement=requirement,
                    violation_type="incomplete_erasure",
                    description="Erasure request not completed",
                    evidence={'request_id': request['id']},
                    remediation="Ensure all erasure requests are processed and logged"
                )
        
        return None
    
    def _validate_consent_management(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate consent management"""
        consent_records = audit_data.get('consent_records', [])
        data_processing = audit_data.get('data_processing', [])
        
        for processing in data_processing:
            user_id = processing.get('user_id')
            if not any(c['user_id'] == user_id and c['granted'] for c in consent_records):
                return ComplianceViolation(
                    requirement=requirement,
                    violation_type="missing_consent",
                    description="Data processing without valid consent",
                    evidence={'user_id': user_id},
                    remediation="Obtain and record user consent before processing"
                )
        
        return None
    
    def _validate_phi_encryption(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate PHI encryption"""
        encryption_status = audit_data.get('encryption', {})
        
        if not encryption_status.get('at_rest', False):
            return ComplianceViolation(
                requirement=requirement,
                violation_type="missing_encryption",
                description="PHI not encrypted at rest",
                evidence={'at_rest': False},
                remediation="Implement AES-256 encryption for PHI at rest"
            )
        
        if not encryption_status.get('in_transit', False):
            return ComplianceViolation(
                requirement=requirement,
                violation_type="missing_encryption",
                description="PHI not encrypted in transit",
                evidence={'in_transit': False},
                remediation="Implement TLS 1.2+ for PHI in transit"
            )
        
        return None
    
    def _validate_access_controls(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate access control mechanisms"""
        access_logs = audit_data.get('access_logs', [])
        
        # Check for multi-factor authentication
        mfa_enabled = audit_data.get('security_config', {}).get('mfa_enabled', False)
        if not mfa_enabled:
            return ComplianceViolation(
                requirement=requirement,
                violation_type="weak_authentication",
                description="Multi-factor authentication not enabled",
                evidence={'mfa_enabled': False},
                remediation="Enable MFA for all user accounts"
            )
        
        # Check for unauthorized access attempts
        unauthorized = [log for log in access_logs if log.get('authorized') == False]
        if len(unauthorized) > 10:
            return ComplianceViolation(
                requirement=requirement,
                violation_type="excessive_unauthorized_attempts",
                description="High number of unauthorized access attempts",
                evidence={'count': len(unauthorized)},
                remediation="Review and strengthen access control policies"
            )
        
        return None
    
    def _validate_processing_integrity(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate processing integrity"""
        processing_errors = audit_data.get('processing_errors', [])
        total_operations = audit_data.get('total_operations', 0)
        
        if total_operations > 0:
            error_rate = len(processing_errors) / total_operations
            if error_rate > 0.01:  # 1% error threshold
                return ComplianceViolation(
                    requirement=requirement,
                    violation_type="high_error_rate",
                    description="Processing error rate exceeds acceptable threshold",
                    evidence={'error_rate': error_rate},
                    remediation="Investigate and fix processing errors"
                )
        
        return None
    
    def _validate_risk_assessment(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate risk assessment procedures"""
        last_assessment = audit_data.get('last_risk_assessment')
        
        if not last_assessment:
            return ComplianceViolation(
                requirement=requirement,
                violation_type="missing_assessment",
                description="No risk assessment found",
                evidence={},
                remediation="Conduct comprehensive risk assessment"
            )
        
        # Check if assessment is recent (within 1 year)
        assessment_date = datetime.fromisoformat(last_assessment)
        if datetime.now() - assessment_date > timedelta(days=365):
            return ComplianceViolation(
                requirement=requirement,
                violation_type="outdated_assessment",
                description="Risk assessment is outdated",
                evidence={'last_assessment': last_assessment},
                remediation="Update risk assessment (required annually)"
            )
        
        return None
    
    def _validate_fips_crypto(
        self,
        audit_data: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Optional[ComplianceViolation]:
        """Validate FIPS-approved cryptography"""
        crypto_config = audit_data.get('crypto_config', {})
        
        approved_algorithms = ['AES-256', 'SHA-256', 'SHA-384', 'SHA-512', 'RSA-2048', 'RSA-4096']
        used_algorithms = crypto_config.get('algorithms', [])
        
        for algo in used_algorithms:
            if algo not in approved_algorithms:
                return ComplianceViolation(
                    requirement=requirement,
                    violation_type="non_approved_crypto",
                    description=f"Non-FIPS-approved algorithm in use: {algo}",
                    evidence={'algorithm': algo},
                    remediation="Replace with FIPS-approved algorithm"
                )
        
        return None
    
    def _create_audit_summary(
        self,
        audit_data: Dict[str, Any],
        violations: List[ComplianceViolation]
    ) -> Dict[str, Any]:
        """Create audit summary"""
        return {
            'total_violations': len(violations),
            'critical_violations': len([v for v in violations if v.requirement.severity == 'critical']),
            'high_violations': len([v for v in violations if v.requirement.severity == 'high']),
            'medium_violations': len([v for v in violations if v.requirement.severity == 'medium']),
            'low_violations': len([v for v in violations if v.requirement.severity == 'low']),
            'audit_date': datetime.now().isoformat(),
            'data_points_analyzed': len(audit_data)
        }
    
    def _generate_recommendations(
        self,
        violations: List[ComplianceViolation],
        compliance_scores: Dict[str, float]
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Priority recommendations based on critical violations
        critical_violations = [v for v in violations if v.requirement.severity == 'critical']
        if critical_violations:
            recommendations.append("URGENT: Address critical compliance violations immediately")
            for v in critical_violations[:3]:  # Top 3 critical
                recommendations.append(f"- {v.remediation}")
        
        # Standard-specific recommendations
        for standard, score in compliance_scores.items():
            if score < 0.5:
                recommendations.append(f"Implement comprehensive {standard} compliance program")
            elif score < 0.8:
                recommendations.append(f"Strengthen {standard} compliance controls")
        
        # General recommendations
        if not violations:
            recommendations.append("Maintain current compliance posture")
            recommendations.append("Schedule regular compliance audits")
        
        return recommendations
    
    def export_report(
        self,
        report: ComplianceReport,
        filepath: Path,
        format: str = 'json'
    ) -> None:
        """
        Export compliance report to file.
        
        Args:
            report: Compliance report to export
            filepath: Output file path
            format: Export format (json, html, pdf)
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                report_dict = {
                    'standards_checked': [s.value for s in report.standards_checked],
                    'is_compliant': report.is_compliant,
                    'compliance_scores': report.compliance_scores,
                    'violations': [
                        {
                            'requirement_id': v.requirement.requirement_id,
                            'description': v.description,
                            'severity': v.requirement.severity,
                            'remediation': v.remediation
                        }
                        for v in report.violations
                    ],
                    'warnings': report.warnings,
                    'recommendations': report.recommendations,
                    'audit_summary': report.audit_summary
                }
                json.dump(report_dict, f, indent=2)
        
        elif format == 'html':
            # Generate HTML report
            html_content = self._generate_html_report(report)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, report: ComplianceReport) -> str:
        """Generate HTML compliance report"""
        html = f"""
        <html>
        <head>
            <title>Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .compliant {{ color: green; }}
                .non-compliant {{ color: red; }}
                .violation {{ background-color: #ffeeee; padding: 10px; margin: 10px 0; }}
                .critical {{ border-left: 5px solid red; }}
                .high {{ border-left: 5px solid orange; }}
                .medium {{ border-left: 5px solid yellow; }}
                .low {{ border-left: 5px solid gray; }}
            </style>
        </head>
        <body>
            <h1>Compliance Assessment Report</h1>
            <p>Date: {report.audit_summary.get('audit_date', 'N/A')}</p>
            
            <h2>Overall Status: <span class="{'compliant' if report.is_compliant else 'non-compliant'}">
                {'COMPLIANT' if report.is_compliant else 'NON-COMPLIANT'}
            </span></h2>
            
            <h2>Compliance Scores</h2>
            <ul>
        """
        
        for standard, score in report.compliance_scores.items():
            html += f"<li>{standard}: {score:.2%}</li>"
        
        html += """
            </ul>
            
            <h2>Violations</h2>
        """
        
        for violation in report.violations:
            html += f"""
            <div class="violation {violation.requirement.severity}">
                <h3>{violation.requirement.requirement_id}: {violation.requirement.description}</h3>
                <p><strong>Violation:</strong> {violation.description}</p>
                <p><strong>Remediation:</strong> {violation.remediation}</p>
            </div>
            """
        
        html += """
            <h2>Recommendations</h2>
            <ol>
        """
        
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
            </ol>
        </body>
        </html>
        """
        
        return html