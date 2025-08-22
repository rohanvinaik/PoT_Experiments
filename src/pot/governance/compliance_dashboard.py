"""
Compliance Dashboard Generator for PoT Governance Framework
Provides real-time monitoring, visualizations, and reporting
"""

import json
import csv
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time
from collections import defaultdict, Counter
import numpy as np

# Import governance components
try:
    from .governance import GovernanceFramework
    from .eu_ai_act_compliance import EUAIActCompliance, ComplianceStatus as EUComplianceStatus
    from .nist_ai_rmf_compliance import NISTAIRMFCompliance, RiskLevel
    from .policy_engine import PolicyEngine, PolicyStatus
    from .audit_logger import AuditLogger, LogCategory
except ImportError:
    # For standalone testing
    pass


@dataclass
class ComplianceMetrics:
    """Compliance metrics snapshot"""
    timestamp: datetime
    overall_score: float
    eu_ai_act_score: float
    nist_rmf_score: float
    policy_compliance_rate: float
    violations_count: int
    warnings_count: int
    critical_issues: int
    pending_actions: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class RiskItem:
    """Individual risk item"""
    risk_id: str
    category: str
    severity: str  # low, medium, high, critical
    description: str
    likelihood: float
    impact: float
    risk_score: float
    mitigation: str
    status: str  # identified, mitigating, mitigated, accepted
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionItem:
    """Remediation action item"""
    action_id: str
    priority: str  # low, medium, high, urgent
    category: str
    description: str
    assigned_to: str
    due_date: Optional[datetime]
    status: str  # pending, in_progress, completed, overdue
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.due_date:
            result['due_date'] = self.due_date.isoformat()
        return result


class ComplianceDashboard:
    """
    Comprehensive compliance monitoring dashboard generator
    """
    
    def __init__(
        self,
        governance_framework: 'GovernanceFramework',
        eu_compliance: Optional['EUAIActCompliance'] = None,
        nist_compliance: Optional['NISTAIRMFCompliance'] = None,
        policy_engine: Optional['PolicyEngine'] = None,
        audit_logger: Optional['AuditLogger'] = None
    ):
        """
        Initialize compliance dashboard
        
        Args:
            governance_framework: Main governance framework
            eu_compliance: EU AI Act compliance module
            nist_compliance: NIST AI RMF compliance module
            policy_engine: Policy engine
            audit_logger: Audit logger
        """
        self.framework = governance_framework
        self.eu_compliance = eu_compliance
        self.nist_compliance = nist_compliance
        self.policy_engine = policy_engine
        self.audit_logger = audit_logger
        
        # Metrics storage
        self.metrics = {}
        self.metrics_history = []
        self.risks = []
        self.action_items = []
        
        # Real-time monitoring
        self.monitoring_enabled = False
        self.monitoring_thread = None
        self.alerts = []
        self.alert_callbacks = []
        
        # Cache
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Initialize
        self._initialize_dashboard()
    
    def _initialize_dashboard(self):
        """Initialize dashboard components"""
        # Collect initial metrics
        self.collect_metrics()
        
        # Start monitoring if configured
        if self.monitoring_enabled:
            self.start_monitoring()
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Gather all compliance metrics
        
        Returns:
            Dictionary of current metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "compliance_scores": {},
            "policy_status": {},
            "audit_summary": {},
            "risk_assessment": {},
            "violations": [],
            "warnings": [],
            "action_items": []
        }
        
        # Overall compliance from governance framework
        if self.framework:
            compliance_report = self.framework.generate_compliance_report()
            metrics["compliance_scores"]["overall"] = compliance_report.get("compliance_rate", 0)
            metrics["violations"].extend(compliance_report.get("violations_summary", []))
        
        # EU AI Act compliance
        if self.eu_compliance:
            eu_metrics = self._collect_eu_metrics()
            metrics["compliance_scores"]["eu_ai_act"] = eu_metrics["score"]
            metrics["eu_ai_act"] = eu_metrics
        
        # NIST AI RMF compliance
        if self.nist_compliance:
            nist_metrics = self._collect_nist_metrics()
            metrics["compliance_scores"]["nist_rmf"] = nist_metrics["score"]
            metrics["nist_rmf"] = nist_metrics
        
        # Policy engine status
        if self.policy_engine:
            policy_metrics = self._collect_policy_metrics()
            metrics["compliance_scores"]["policy"] = policy_metrics["compliance_rate"]
            metrics["policy_status"] = policy_metrics
        
        # Audit log summary
        if self.audit_logger:
            audit_metrics = self._collect_audit_metrics()
            metrics["audit_summary"] = audit_metrics
        
        # Risk assessment
        metrics["risk_assessment"] = self._assess_risks()
        
        # Action items
        metrics["action_items"] = self._generate_action_items(metrics)
        
        # Create metrics snapshot
        snapshot = ComplianceMetrics(
            timestamp=datetime.now(),
            overall_score=metrics["compliance_scores"].get("overall", 0),
            eu_ai_act_score=metrics["compliance_scores"].get("eu_ai_act", 0),
            nist_rmf_score=metrics["compliance_scores"].get("nist_rmf", 0),
            policy_compliance_rate=metrics["compliance_scores"].get("policy", 0),
            violations_count=len(metrics["violations"]),
            warnings_count=len(metrics["warnings"]),
            critical_issues=sum(1 for r in self.risks if r.severity == "critical"),
            pending_actions=sum(1 for a in self.action_items if a.status == "pending")
        )
        
        # Store metrics
        self.metrics = metrics
        self.metrics_history.append(snapshot)
        
        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _collect_eu_metrics(self) -> Dict[str, Any]:
        """Collect EU AI Act compliance metrics"""
        metrics = {
            "score": 0.0,
            "risk_category": "unknown",
            "requirements_met": 0,
            "requirements_total": 0,
            "key_findings": []
        }
        
        if not self.eu_compliance:
            return metrics
        
        # Get compliance checks
        checks = self.eu_compliance.compliance_checks
        if checks:
            compliant = sum(1 for c in checks if c.status == EUComplianceStatus.COMPLIANT)
            total = len(checks)
            metrics["score"] = compliant / total if total > 0 else 0
            metrics["requirements_met"] = compliant
            metrics["requirements_total"] = total
            
            # Key findings
            for check in checks[-5:]:  # Last 5 checks
                if check.violations:
                    metrics["key_findings"].extend(check.violations[:2])
        
        # Risk category
        if self.eu_compliance.risk_category:
            metrics["risk_category"] = self.eu_compliance.risk_category.value
        
        return metrics
    
    def _collect_nist_metrics(self) -> Dict[str, Any]:
        """Collect NIST AI RMF metrics"""
        metrics = {
            "score": 0.0,
            "maturity_level": 1,
            "functions_assessed": {},
            "trustworthiness": {},
            "key_gaps": []
        }
        
        if not self.nist_compliance:
            return metrics
        
        # Get assessments
        assessments = self.nist_compliance.assessments
        if assessments:
            # Calculate average maturity
            maturity_scores = [a.maturity_level for a in assessments]
            metrics["maturity_level"] = int(np.mean(maturity_scores)) if maturity_scores else 1
            metrics["score"] = metrics["maturity_level"] / 5.0  # Normalize to 0-1
            
            # Functions assessed
            for func in ["GOVERN", "MAP", "MEASURE", "MANAGE"]:
                func_assessments = [a for a in assessments if a.function.value == func.lower()]
                if func_assessments:
                    metrics["functions_assessed"][func] = {
                        "assessed": True,
                        "maturity": int(np.mean([a.maturity_level for a in func_assessments]))
                    }
            
            # Key gaps
            for assessment in assessments[-3:]:  # Last 3 assessments
                if assessment.findings:
                    metrics["key_gaps"].extend(assessment.findings[:2])
        
        return metrics
    
    def _collect_policy_metrics(self) -> Dict[str, Any]:
        """Collect policy engine metrics"""
        metrics = {
            "compliance_rate": 0.0,
            "total_policies": 0,
            "enabled_policies": 0,
            "recent_violations": [],
            "policy_coverage": {}
        }
        
        if not self.policy_engine:
            return metrics
        
        # Get policy report
        report = self.policy_engine.generate_policy_report()
        
        metrics["total_policies"] = report["total_policies"]
        metrics["enabled_policies"] = report["policies_by_status"]["enabled"]
        
        # Calculate compliance rate from recent evaluations
        recent = report.get("recent_evaluations", [])
        if recent:
            passed = sum(1 for e in recent if e["status"] == "pass")
            metrics["compliance_rate"] = passed / len(recent) if recent else 0
        
        # Recent violations
        for eval in recent:
            if eval.get("violations", 0) > 0:
                metrics["recent_violations"].append({
                    "policy": eval["policy"],
                    "timestamp": eval["timestamp"],
                    "violations": eval["violations"]
                })
        
        # Policy coverage by type
        metrics["policy_coverage"] = report.get("policies_by_type", {})
        
        return metrics
    
    def _collect_audit_metrics(self) -> Dict[str, Any]:
        """Collect audit log metrics"""
        metrics = {
            "total_entries": 0,
            "entries_by_category": {},
            "recent_security_events": [],
            "anomalies": [],
            "integrity_status": "unknown"
        }
        
        if not self.audit_logger:
            return metrics
        
        # Get statistics
        stats = self.audit_logger.get_statistics()
        metrics["total_entries"] = stats.get("total_entries", 0)
        
        # Query recent logs
        recent_logs = self.audit_logger.query_logs(
            start_date=datetime.now() - timedelta(days=1),
            limit=1000
        )
        
        # Count by category
        category_counts = Counter(log.category.value for log in recent_logs)
        metrics["entries_by_category"] = dict(category_counts)
        
        # Recent security events
        security_events = [log for log in recent_logs if log.category == LogCategory.SECURITY_EVENT]
        metrics["recent_security_events"] = [
            {
                "timestamp": log.timestamp.isoformat(),
                "actor": log.actor,
                "action": log.action,
                "result": log.result
            }
            for log in security_events[:5]
        ]
        
        # Check integrity
        is_valid, issues = self.audit_logger.verify_integrity(
            start_date=datetime.now() - timedelta(days=7)
        )
        metrics["integrity_status"] = "valid" if is_valid else "compromised"
        if not is_valid:
            metrics["integrity_issues"] = issues[:5]
        
        return metrics
    
    def _assess_risks(self) -> Dict[str, Any]:
        """Assess current risks"""
        risks = []
        
        # Compliance risks
        if self.metrics.get("compliance_scores", {}).get("overall", 1) < 0.8:
            risks.append(RiskItem(
                risk_id="RISK-001",
                category="compliance",
                severity="high",
                description="Overall compliance below 80%",
                likelihood=0.8,
                impact=0.9,
                risk_score=0.72,
                mitigation="Review and update policies",
                status="identified"
            ))
        
        # EU AI Act risks
        if self.eu_compliance and self.eu_compliance.risk_category:
            if self.eu_compliance.risk_category.value in ["high_risk", "unacceptable_risk"]:
                risks.append(RiskItem(
                    risk_id="RISK-002",
                    category="regulatory",
                    severity="critical",
                    description=f"System classified as {self.eu_compliance.risk_category.value}",
                    likelihood=1.0,
                    impact=1.0,
                    risk_score=1.0,
                    mitigation="Implement required safeguards",
                    status="identified"
                ))
        
        # Policy violations
        if self.metrics.get("violations", []):
            risks.append(RiskItem(
                risk_id="RISK-003",
                category="policy",
                severity="medium",
                description=f"{len(self.metrics['violations'])} policy violations detected",
                likelihood=1.0,
                impact=0.6,
                risk_score=0.6,
                mitigation="Address policy violations",
                status="mitigating"
            ))
        
        # Audit integrity risks
        if self.metrics.get("audit_summary", {}).get("integrity_status") == "compromised":
            risks.append(RiskItem(
                risk_id="RISK-004",
                category="security",
                severity="critical",
                description="Audit log integrity compromised",
                likelihood=1.0,
                impact=1.0,
                risk_score=1.0,
                mitigation="Investigate and restore integrity",
                status="identified"
            ))
        
        self.risks = risks
        
        return {
            "total_risks": len(risks),
            "by_severity": Counter(r.severity for r in risks),
            "by_category": Counter(r.category for r in risks),
            "critical_risks": [r.to_dict() for r in risks if r.severity == "critical"],
            "average_risk_score": np.mean([r.risk_score for r in risks]) if risks else 0
        }
    
    def _generate_action_items(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action items based on metrics"""
        items = []
        
        # Low compliance score action
        if metrics["compliance_scores"].get("overall", 1) < 0.8:
            items.append(ActionItem(
                action_id="ACTION-001",
                priority="high",
                category="compliance",
                description="Improve overall compliance score above 80%",
                assigned_to="compliance_team",
                due_date=datetime.now() + timedelta(days=30),
                status="pending"
            ))
        
        # EU AI Act actions
        if metrics.get("eu_ai_act", {}).get("key_findings"):
            items.append(ActionItem(
                action_id="ACTION-002",
                priority="urgent",
                category="regulatory",
                description="Address EU AI Act compliance findings",
                assigned_to="legal_team",
                due_date=datetime.now() + timedelta(days=14),
                status="pending"
            ))
        
        # NIST maturity improvement
        if metrics.get("nist_rmf", {}).get("maturity_level", 1) < 3:
            items.append(ActionItem(
                action_id="ACTION-003",
                priority="medium",
                category="framework",
                description="Improve NIST AI RMF maturity to level 3",
                assigned_to="governance_team",
                due_date=datetime.now() + timedelta(days=60),
                status="pending"
            ))
        
        # Policy violations
        if metrics.get("policy_status", {}).get("recent_violations"):
            items.append(ActionItem(
                action_id="ACTION-004",
                priority="high",
                category="policy",
                description="Investigate and resolve policy violations",
                assigned_to="security_team",
                due_date=datetime.now() + timedelta(days=7),
                status="in_progress"
            ))
        
        self.action_items = items
        
        return [item.to_dict() for item in items]
    
    def generate_html_dashboard(self) -> str:
        """
        Create interactive HTML dashboard
        
        Returns:
            HTML string for dashboard
        """
        # Ensure metrics are current
        if not self.metrics or (datetime.now() - datetime.fromisoformat(self.metrics["timestamp"])).seconds > 300:
            self.collect_metrics()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PoT Compliance Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .header .timestamp {{
            color: #718096;
            font-size: 14px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            color: #718096;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .score-ring {{
            width: 100px;
            height: 100px;
            position: relative;
            margin: 0 auto;
        }}
        
        .score-ring svg {{
            transform: rotate(-90deg);
        }}
        
        .score-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            font-weight: bold;
        }}
        
        .status-good {{ color: #48bb78; }}
        .status-warning {{ color: #ed8936; }}
        .status-critical {{ color: #f56565; }}
        
        .section {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        
        .section h2 {{
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .risk-matrix {{
            display: grid;
            grid-template-columns: 50px repeat(3, 1fr);
            grid-template-rows: repeat(3, 100px) 50px;
            gap: 2px;
            margin: 20px 0;
        }}
        
        .matrix-cell {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: bold;
        }}
        
        .risk-low {{ background: #c6f6d5; }}
        .risk-medium {{ background: #fed7aa; }}
        .risk-high {{ background: #feb2b2; }}
        
        .action-list {{
            list-style: none;
        }}
        
        .action-item {{
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
            background: #f7fafc;
            border-radius: 5px;
        }}
        
        .priority-urgent {{ border-left-color: #f56565; }}
        .priority-high {{ border-left-color: #ed8936; }}
        .priority-medium {{ border-left-color: #ecc94b; }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        
        .timeline {{
            display: flex;
            overflow-x: auto;
            padding: 20px 0;
        }}
        
        .timeline-item {{
            min-width: 150px;
            padding: 10px;
            margin-right: 20px;
            background: #f7fafc;
            border-radius: 5px;
            position: relative;
        }}
        
        .timeline-item::after {{
            content: '';
            position: absolute;
            right: -20px;
            top: 50%;
            width: 20px;
            height: 2px;
            background: #cbd5e0;
        }}
        
        .alert-banner {{
            background: #feb2b2;
            color: #742a2a;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }}
        
        .alert-banner.warning {{
            background: #fed7aa;
            color: #744210;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🛡️ PoT Governance Compliance Dashboard</h1>
            <div class="timestamp">Last updated: {self.metrics['timestamp']}</div>
        </div>
        
        {self._generate_alerts_html()}
        
        <div class="metrics-grid">
            {self._generate_metric_cards_html()}
        </div>
        
        <div class="section">
            <h2>📊 Compliance Timeline</h2>
            <canvas id="complianceChart"></canvas>
        </div>
        
        <div class="section">
            <h2>⚠️ Risk Assessment</h2>
            {self._generate_risk_matrix_html()}
        </div>
        
        <div class="section">
            <h2>📋 Action Items</h2>
            {self._generate_action_items_html()}
        </div>
        
        <div class="section">
            <h2>🔍 Policy Coverage</h2>
            {self._generate_policy_coverage_html()}
        </div>
        
        <div class="section">
            <h2>📝 Audit Trail</h2>
            {self._generate_audit_summary_html()}
        </div>
    </div>
    
    <script>
        {self._generate_chart_scripts()}
    </script>
</body>
</html>
        """
        
        return html
    
    def _generate_alerts_html(self) -> str:
        """Generate alerts section"""
        html = ""
        
        # Critical issues alert
        critical_count = sum(1 for r in self.risks if r.severity == "critical")
        if critical_count > 0:
            html += f"""
            <div class="alert-banner">
                <span>⚠️ {critical_count} critical compliance issues require immediate attention</span>
            </div>
            """
        
        # Violations alert
        if self.metrics.get("violations"):
            html += f"""
            <div class="alert-banner warning">
                <span>📋 {len(self.metrics['violations'])} policy violations detected</span>
            </div>
            """
        
        return html
    
    def _generate_metric_cards_html(self) -> str:
        """Generate metric cards"""
        scores = self.metrics.get("compliance_scores", {})
        
        cards = []
        
        # Overall compliance score
        overall_score = scores.get("overall", 0) * 100
        cards.append(self._create_score_card("Overall Compliance", overall_score))
        
        # EU AI Act score
        if "eu_ai_act" in scores:
            eu_score = scores["eu_ai_act"] * 100
            cards.append(self._create_score_card("EU AI Act", eu_score))
        
        # NIST RMF score
        if "nist_rmf" in scores:
            nist_score = scores["nist_rmf"] * 100
            cards.append(self._create_score_card("NIST AI RMF", nist_score))
        
        # Policy compliance
        if "policy" in scores:
            policy_score = scores["policy"] * 100
            cards.append(self._create_score_card("Policy Compliance", policy_score))
        
        # Violations count
        violations = len(self.metrics.get("violations", []))
        cards.append(f"""
        <div class="metric-card">
            <div class="metric-label">Violations</div>
            <div class="metric-value {self._get_status_class(violations, reverse=True)}">{violations}</div>
        </div>
        """)
        
        # Action items
        pending_actions = sum(1 for a in self.action_items if a.status == "pending")
        cards.append(f"""
        <div class="metric-card">
            <div class="metric-label">Pending Actions</div>
            <div class="metric-value">{pending_actions}</div>
        </div>
        """)
        
        return "\n".join(cards)
    
    def _create_score_card(self, label: str, score: float) -> str:
        """Create a score card with ring visualization"""
        status_class = self._get_status_class(score)
        color = self._get_status_color(score)
        
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="score-ring">
                <svg width="100" height="100">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#e2e8f0" stroke-width="8"/>
                    <circle cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="8"
                            stroke-dasharray="{score * 2.51} 251" stroke-linecap="round"/>
                </svg>
                <div class="score-text {status_class}">{score:.0f}%</div>
            </div>
        </div>
        """
    
    def _get_status_class(self, value: float, reverse: bool = False) -> str:
        """Get CSS class based on status value"""
        if reverse:
            if value == 0:
                return "status-good"
            elif value <= 5:
                return "status-warning"
            else:
                return "status-critical"
        else:
            if value >= 80:
                return "status-good"
            elif value >= 60:
                return "status-warning"
            else:
                return "status-critical"
    
    def _get_status_color(self, value: float) -> str:
        """Get color based on status value"""
        if value >= 80:
            return "#48bb78"
        elif value >= 60:
            return "#ed8936"
        else:
            return "#f56565"
    
    def _generate_risk_matrix_html(self) -> str:
        """Generate risk matrix visualization"""
        # Create 3x3 risk matrix
        matrix = [[0 for _ in range(3)] for _ in range(3)]
        
        for risk in self.risks:
            # Map likelihood and impact to matrix position
            likelihood_idx = min(int(risk.likelihood * 3), 2)
            impact_idx = min(int(risk.impact * 3), 2)
            matrix[2 - impact_idx][likelihood_idx] += 1
        
        html = """
        <div class="risk-matrix">
            <div class="matrix-cell" style="grid-column: 1; grid-row: 1;">High</div>
        """
        
        # Generate matrix cells
        risk_levels = [
            ["risk-medium", "risk-high", "risk-high"],
            ["risk-low", "risk-medium", "risk-high"],
            ["risk-low", "risk-low", "risk-medium"]
        ]
        
        for i in range(3):
            for j in range(3):
                count = matrix[i][j]
                risk_class = risk_levels[i][j]
                html += f"""
                <div class="matrix-cell {risk_class}" style="grid-column: {j+2}; grid-row: {i+1};">
                    {count if count > 0 else ''}
                </div>
                """
        
        html += """
            <div class="matrix-cell" style="grid-column: 1; grid-row: 2;">Med</div>
            <div class="matrix-cell" style="grid-column: 1; grid-row: 3;">Low</div>
            <div class="matrix-cell" style="grid-column: 1; grid-row: 4;">Impact</div>
            <div class="matrix-cell" style="grid-column: 2; grid-row: 4;">Low</div>
            <div class="matrix-cell" style="grid-column: 3; grid-row: 4;">Med</div>
            <div class="matrix-cell" style="grid-column: 4; grid-row: 4;">High</div>
        </div>
        <div style="text-align: center; color: #718096; margin-top: 10px;">Likelihood →</div>
        """
        
        return html
    
    def _generate_action_items_html(self) -> str:
        """Generate action items list"""
        if not self.action_items:
            return "<p>No action items pending</p>"
        
        html = "<ul class='action-list'>"
        
        for item in self.action_items:
            priority_class = f"priority-{item.priority}"
            due_date = item.due_date.strftime("%Y-%m-%d") if item.due_date else "No due date"
            
            html += f"""
            <li class="action-item {priority_class}">
                <strong>{item.description}</strong><br>
                <small>Priority: {item.priority} | Assigned: {item.assigned_to} | Due: {due_date}</small>
            </li>
            """
        
        html += "</ul>"
        return html
    
    def _generate_policy_coverage_html(self) -> str:
        """Generate policy coverage visualization"""
        coverage = self.metrics.get("policy_status", {}).get("policy_coverage", {})
        
        if not coverage:
            return "<p>No policy coverage data available</p>"
        
        html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>"
        
        for policy_type, count in coverage.items():
            html += f"""
            <div style='padding: 15px; background: #f7fafc; border-radius: 5px;'>
                <div style='font-size: 24px; font-weight: bold; color: #4a5568;'>{count}</div>
                <div style='color: #718096; text-transform: capitalize;'>{policy_type.replace('_', ' ')}</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_audit_summary_html(self) -> str:
        """Generate audit summary"""
        audit = self.metrics.get("audit_summary", {})
        
        html = f"""
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div>
                <h3>Log Statistics</h3>
                <p>Total Entries: {audit.get('total_entries', 0):,}</p>
                <p>Integrity Status: <span class='{self._get_integrity_class(audit.get("integrity_status"))}'>{audit.get("integrity_status", "unknown")}</span></p>
            </div>
            <div>
                <h3>Recent Security Events</h3>
        """
        
        events = audit.get("recent_security_events", [])
        if events:
            html += "<ul>"
            for event in events[:3]:
                html += f"<li>{event['timestamp']}: {event['action']} by {event['actor']}</li>"
            html += "</ul>"
        else:
            html += "<p>No recent security events</p>"
        
        html += "</div></div>"
        return html
    
    def _get_integrity_class(self, status: str) -> str:
        """Get CSS class for integrity status"""
        if status == "valid":
            return "status-good"
        elif status == "compromised":
            return "status-critical"
        else:
            return "status-warning"
    
    def _generate_chart_scripts(self) -> str:
        """Generate Chart.js scripts"""
        # Prepare timeline data
        timeline_data = []
        timeline_labels = []
        
        for metric in self.metrics_history[-10:]:  # Last 10 data points
            timeline_labels.append(metric.timestamp.strftime("%H:%M"))
            timeline_data.append(metric.overall_score * 100)
        
        return f"""
        // Compliance Timeline Chart
        const ctx = document.getElementById('complianceChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(timeline_labels)},
                datasets: [{{
                    label: 'Overall Compliance',
                    data: {json.dumps(timeline_data)},
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        """
    
    def export_compliance_status(self, format: str = "json") -> bytes:
        """
        Export compliance status in various formats
        
        Args:
            format: Export format (json, csv, pdf)
            
        Returns:
            Exported data as bytes
        """
        if format == "json":
            return self._export_json()
        elif format == "csv":
            return self._export_csv()
        elif format == "pdf":
            return self._export_pdf()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self) -> bytes:
        """Export as JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "risks": [r.to_dict() for r in self.risks],
            "action_items": [a.to_dict() for a in self.action_items],
            "history": [m.to_dict() for m in self.metrics_history[-100:]]
        }
        
        return json.dumps(export_data, indent=2).encode()
    
    def _export_csv(self) -> bytes:
        """Export as CSV"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write metrics
        writer.writerow(["Metric", "Value"])
        for key, value in self.metrics.get("compliance_scores", {}).items():
            writer.writerow([key, f"{value*100:.2f}%"])
        
        writer.writerow([])
        
        # Write risks
        writer.writerow(["Risk ID", "Category", "Severity", "Description", "Status"])
        for risk in self.risks:
            writer.writerow([risk.risk_id, risk.category, risk.severity, risk.description, risk.status])
        
        writer.writerow([])
        
        # Write action items
        writer.writerow(["Action ID", "Priority", "Description", "Assigned", "Status"])
        for action in self.action_items:
            writer.writerow([
                action.action_id, action.priority, action.description,
                action.assigned_to, action.status
            ])
        
        return output.getvalue().encode()
    
    def _export_pdf(self) -> bytes:
        """Export as PDF (simplified - returns HTML for now)"""
        # In production, use a library like reportlab or weasyprint
        # For now, return HTML that can be converted to PDF
        html = self.generate_executive_summary()
        return html.encode()
    
    def generate_executive_summary(self) -> str:
        """
        Generate executive summary
        
        Returns:
            Executive summary as formatted text/HTML
        """
        summary = f"""
        <html>
        <head>
            <title>Compliance Executive Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 40px; }}
                h1 {{ color: #2d3748; }}
                h2 {{ color: #4a5568; margin-top: 30px; }}
                .metric {{ margin: 10px 0; }}
                .critical {{ color: #f56565; }}
                .warning {{ color: #ed8936; }}
                .good {{ color: #48bb78; }}
            </style>
        </head>
        <body>
            <h1>Compliance Executive Summary</h1>
            <p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <h2>Overall Compliance Status</h2>
            <div class="metric">
                <strong>Overall Score:</strong> 
                <span class="{self._get_summary_class(self.metrics['compliance_scores'].get('overall', 0))}">
                    {self.metrics['compliance_scores'].get('overall', 0)*100:.1f}%
                </span>
            </div>
            
            <h2>Key Metrics</h2>
            <ul>
                <li>EU AI Act Compliance: {self.metrics['compliance_scores'].get('eu_ai_act', 0)*100:.1f}%</li>
                <li>NIST AI RMF Maturity: Level {self.metrics.get('nist_rmf', {}).get('maturity_level', 1)}/5</li>
                <li>Policy Compliance Rate: {self.metrics['compliance_scores'].get('policy', 0)*100:.1f}%</li>
                <li>Active Violations: {len(self.metrics.get('violations', []))}</li>
            </ul>
            
            <h2>Critical Risks</h2>
        """
        
        critical_risks = [r for r in self.risks if r.severity == "critical"]
        if critical_risks:
            summary += "<ul>"
            for risk in critical_risks:
                summary += f"<li class='critical'>{risk.description}</li>"
            summary += "</ul>"
        else:
            summary += "<p class='good'>No critical risks identified</p>"
        
        summary += f"""
            <h2>Required Actions</h2>
            <ol>
        """
        
        urgent_actions = [a for a in self.action_items if a.priority in ["urgent", "high"]]
        for action in urgent_actions[:5]:
            summary += f"<li>{action.description} (Priority: {action.priority})</li>"
        
        summary += """
            </ol>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Continue monitoring compliance metrics daily</li>
                <li>Address critical risks within 7 days</li>
                <li>Schedule compliance review meeting</li>
                <li>Update policies based on recent violations</li>
            </ul>
        </body>
        </html>
        """
        
        return summary
    
    def _get_summary_class(self, score: float) -> str:
        """Get CSS class for summary"""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "warning"
        else:
            return "critical"
    
    def start_monitoring(self, interval: int = 60):
        """
        Start real-time monitoring
        
        Args:
            interval: Update interval in seconds
        """
        self.monitoring_enabled = True
        
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    # Collect metrics
                    old_metrics = self.metrics.copy() if self.metrics else {}
                    self.collect_metrics()
                    
                    # Check for alerts
                    self._check_alerts(old_metrics)
                    
                    # Sleep
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _check_alerts(self, old_metrics: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        
        # Check compliance score drop
        old_score = old_metrics.get("compliance_scores", {}).get("overall", 1)
        new_score = self.metrics.get("compliance_scores", {}).get("overall", 1)
        
        if new_score < old_score - 0.1:  # 10% drop
            alerts.append({
                "type": "compliance_drop",
                "severity": "high",
                "message": f"Compliance score dropped by {(old_score - new_score)*100:.1f}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check for new violations
        old_violations = len(old_metrics.get("violations", []))
        new_violations = len(self.metrics.get("violations", []))
        
        if new_violations > old_violations:
            alerts.append({
                "type": "new_violations",
                "severity": "medium",
                "message": f"{new_violations - old_violations} new violations detected",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check for critical risks
        critical_risks = sum(1 for r in self.risks if r.severity == "critical")
        if critical_risks > 0:
            alerts.append({
                "type": "critical_risk",
                "severity": "critical",
                "message": f"{critical_risks} critical risks require immediate attention",
                "timestamp": datetime.now().isoformat()
            })
        
        # Store and trigger callbacks
        self.alerts.extend(alerts)
        for alert in alerts:
            for callback in self.alert_callbacks:
                callback(alert)
    
    def register_alert_callback(self, callback):
        """
        Register callback for alerts
        
        Args:
            callback: Function to call on alert
        """
        self.alert_callbacks.append(callback)
    
    def get_predictive_risks(self) -> List[Dict[str, Any]]:
        """
        Predict future compliance risks based on trends
        
        Returns:
            List of predicted risks
        """
        predictions = []
        
        if len(self.metrics_history) < 5:
            return predictions
        
        # Analyze trends
        recent_scores = [m.overall_score for m in self.metrics_history[-5:]]
        
        # Check for downward trend
        if all(recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            predictions.append({
                "risk": "compliance_decline",
                "probability": 0.8,
                "timeframe": "7 days",
                "description": "Compliance score showing consistent decline",
                "recommendation": "Investigate root causes and implement corrective measures"
            })
        
        # Check violation trend
        recent_violations = [m.violations_count for m in self.metrics_history[-5:]]
        if sum(recent_violations) / len(recent_violations) > 5:
            predictions.append({
                "risk": "violation_increase",
                "probability": 0.7,
                "timeframe": "14 days",
                "description": "Policy violations trending upward",
                "recommendation": "Review and strengthen policy enforcement"
            })
        
        return predictions