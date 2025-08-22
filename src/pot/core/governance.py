import hashlib
import hmac
import secrets
from typing import Optional, List, Dict, Any, Tuple
import struct
import json
import yaml
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import threading


class PolicyType(Enum):
    """Types of governance policies"""
    DATA = "data_governance"
    MODEL = "model_governance"
    VERIFICATION = "verification_governance"
    AUDIT = "audit_requirements"
    SECURITY = "security_governance"
    COMPLIANCE = "regulatory_compliance"


class ComplianceStatus(Enum):
    """Compliance check results"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    CONDITIONAL = "conditional"
    PENDING_REVIEW = "pending_review"


class DecisionMaker(Enum):
    """Types of decision makers"""
    HUMAN = "human"
    AUTOMATED = "automated"
    HYBRID = "hybrid"


@dataclass
class ComplianceResult:
    """Result of a compliance check"""
    status: ComplianceStatus
    policy_type: PolicyType
    action: str
    violations: List[str]
    warnings: List[str]
    timestamp: datetime
    evidence: Dict[str, Any]
    recommendations: List[str]
    
    def is_compliant(self) -> bool:
        return self.status == ComplianceStatus.COMPLIANT
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['policy_type'] = self.policy_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class GovernanceDecision:
    """Record of a governance decision"""
    decision_id: str
    action: str
    decision_maker: DecisionMaker
    maker_id: str
    timestamp: datetime
    justification: str
    evidence: Dict[str, Any]
    policies_applied: List[str]
    compliance_results: List[ComplianceResult]
    approved: bool
    conditions: List[str]
    expiry: Optional[datetime] = None
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Convert to audit log format"""
        return {
            'decision_id': self.decision_id,
            'action': self.action,
            'decision_maker': self.decision_maker.value,
            'maker_id': self.maker_id,
            'timestamp': self.timestamp.isoformat(),
            'justification': self.justification,
            'evidence': self.evidence,
            'policies_applied': self.policies_applied,
            'compliance_results': [r.to_dict() for r in self.compliance_results],
            'approved': self.approved,
            'conditions': self.conditions,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'hash': self._compute_hash()
        }
    
    def _compute_hash(self) -> str:
        """Compute immutable hash of decision"""
        content = json.dumps({
            'decision_id': self.decision_id,
            'action': self.action,
            'timestamp': self.timestamp.isoformat(),
            'justification': self.justification,
            'approved': self.approved
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class GovernancePolicy:
    """Individual governance policy"""
    policy_id: str
    policy_type: PolicyType
    name: str
    description: str
    version: str
    active: bool
    rules: List[Dict[str, Any]]
    enforcement_level: str  # 'strict', 'warning', 'advisory'
    effective_date: datetime
    expiry_date: Optional[datetime]
    regulations: List[str]  # e.g., ['GDPR', 'CCPA', 'HIPAA']
    
    def evaluate(self, action: str, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Evaluate an action against this policy"""
        violations = []
        
        for rule in self.rules:
            if not self._check_rule(rule, action, context):
                violations.append(f"Rule {rule.get('id', 'unknown')}: {rule.get('description', '')}")
        
        return len(violations) == 0, violations
    
    def _check_rule(self, rule: Dict[str, Any], action: str, context: Dict[str, Any]) -> bool:
        """Check a single rule"""
        # Basic rule checking logic
        if 'required_fields' in rule:
            for field in rule['required_fields']:
                if field not in context:
                    return False
        
        if 'prohibited_actions' in rule:
            if action in rule['prohibited_actions']:
                return False
        
        if 'required_approvals' in rule:
            approvals = context.get('approvals', [])
            for required in rule['required_approvals']:
                if required not in approvals:
                    return False
        
        if 'max_retention_days' in rule and 'data_age_days' in context:
            if context['data_age_days'] > rule['max_retention_days']:
                return False
        
        return True


def kdf(master_key: bytes, label: str, context: bytes = b'') -> bytes:
    """
    Key Derivation Function (KDF) for cryptographic challenge derivation
    Based on HKDF-Extract-and-Expand (RFC 5869) simplified version
    
    From paper Section 6.2 Algorithm 3
    """
    # Use HMAC-SHA256 as the PRF
    info = label.encode() + context
    return hmac.new(master_key, info, hashlib.sha256).digest()

def derive_challenge_key(master_key_hex: str, epoch: int, session: str) -> bytes:
    """
    Cryptographic challenge key derivation with rotation
    Algorithm 3 from paper Section 6.2
    
    Args:
        master_key_hex: Master key in hexadecimal
        epoch: Epoch number for key rotation
        session: Session identifier
        
    Returns:
        Derived challenge key
    """
    master_key = bytes.fromhex(master_key_hex)
    
    # Step 1: k_epoch = KDF(k, "epoch" || e)
    epoch_bytes = struct.pack('>I', epoch)  # Big-endian 4-byte integer
    k_epoch = kdf(master_key, "epoch", epoch_bytes)
    
    # Step 2: k_session = KDF(k_epoch, "session" || s)
    k_session = kdf(k_epoch, "session", session.encode())
    
    # Step 3: seed = KDF(k_session, "challenge")
    seed = kdf(k_session, "challenge")
    
    return seed

def commit_reveal_protocol(challenge_id: str, salt_hex: str) -> dict:
    """
    Commit-reveal protocol for challenge governance
    
    Returns:
        dict with 'commitment' and 'reveal' phases
    """
    # Commitment phase: hash(challenge_id || salt)
    commitment = hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()
    
    # Reveal phase data
    reveal = {
        'challenge_id': challenge_id,
        'salt': salt_hex,
        'commitment': commitment
    }
    
    return {
        'commitment': commitment,
        'reveal': reveal
    }

def verify_commitment(commitment: str, challenge_id: str, salt_hex: str) -> bool:
    """
    Verify a commitment in the commit-reveal protocol
    
    Returns:
        True if the commitment is valid
    """
    expected = hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()
    return hmac.compare_digest(commitment, expected)

def rotate_epoch_key(master_key_hex: str, current_epoch: int) -> dict:
    """
    Implement epoch-based key rotation for leakage resilience
    From paper Section 6.2
    
    Returns:
        Dictionary with old and new epoch keys
    """
    master_key = bytes.fromhex(master_key_hex)
    
    old_epoch_key = kdf(master_key, "epoch", struct.pack('>I', current_epoch))
    new_epoch_key = kdf(master_key, "epoch", struct.pack('>I', current_epoch + 1))
    
    return {
        'old_epoch': current_epoch,
        'old_key': old_epoch_key.hex(),
        'new_epoch': current_epoch + 1,
        'new_key': new_epoch_key.hex()
    }

def commit_message(challenge_id: str, salt_hex: str) -> str:
    """Legacy function - use commit_reveal_protocol instead"""
    return hashlib.sha256((challenge_id + salt_hex).encode()).hexdigest()

def new_session_nonce() -> str:
    """Generate a new session nonce"""
    return secrets.token_hex(16)

class ChallengeGovernance:
    """
    Complete challenge governance system with cryptographic guarantees
    Implements leakage resilience from paper Section 6.2
    """
    
    def __init__(self, master_key_hex: str):
        self.master_key_hex = master_key_hex
        self.current_epoch = 0
        self.sessions = {}
        
    def new_epoch(self) -> int:
        """Start a new epoch for key rotation"""
        self.current_epoch += 1
        return self.current_epoch
    
    def new_session(self, session_id: Optional[str] = None) -> str:
        """Create a new verification session"""
        if session_id is None:
            session_id = secrets.token_hex(16)
        
        # Derive session key
        session_key = derive_challenge_key(
            self.master_key_hex, 
            self.current_epoch, 
            session_id
        )
        
        self.sessions[session_id] = {
            'epoch': self.current_epoch,
            'key': session_key.hex(),
            'challenges': []
        }
        
        return session_id
    
    def generate_challenge_seed(self, session_id: str) -> bytes:
        """Generate deterministic seed for challenge sampling"""
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        session = self.sessions[session_id]
        return bytes.fromhex(session['key'])
    
    def is_session_valid(self, session_id: str, max_age_epochs: int = 2) -> bool:
        """Check if a session is still valid (not too old)"""
        if session_id not in self.sessions:
            return False
        
        session_epoch = self.sessions[session_id]['epoch']
        return (self.current_epoch - session_epoch) <= max_age_epochs


class GovernanceFramework:
    """
    Comprehensive governance framework for PoT verification
    Manages policies, compliance checking, and audit trails
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the governance framework
        
        Args:
            config_path: Path to governance configuration file
        """
        self.config_path = Path(config_path)
        self.policies: Dict[str, GovernancePolicy] = {}
        self.decisions: List[GovernanceDecision] = []
        self.audit_log: List[Dict[str, Any]] = []
        self.challenge_governance: Optional[ChallengeGovernance] = None
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('governance_audit.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self._load_config()
        self._initialize_policies()
        self._setup_audit_logging()
    
    def _load_config(self):
        """Load governance configuration from file"""
        if not self.config_path.exists():
            # Create default configuration
            self.config = self._create_default_config()
            self._save_config()
        else:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize challenge governance if master key provided
        if 'master_key' in self.config:
            self.challenge_governance = ChallengeGovernance(
                self.config['master_key']
            )
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default governance configuration"""
        return {
            'version': '1.0.0',
            'master_key': secrets.token_hex(32),
            'policies': {
                'data_governance': {
                    'retention_days': 90,
                    'deletion_policy': 'secure_wipe',
                    'access_control': 'role_based',
                    'encryption_required': True,
                    'anonymization_required': True
                },
                'model_governance': {
                    'training_approval_required': True,
                    'deployment_approval_required': True,
                    'update_frequency_days': 30,
                    'version_control_required': True,
                    'testing_required': True
                },
                'verification_governance': {
                    'challenge_types': ['frequency', 'texture', 'template'],
                    'min_confidence_threshold': 0.85,
                    'max_challenge_reuse': 3,
                    'epoch_rotation_days': 7,
                    'sequential_testing_required': True
                },
                'audit_requirements': {
                    'logging_level': 'INFO',
                    'provenance_tracking': True,
                    'transparency_reports': True,
                    'immutable_logs': True,
                    'retention_years': 7
                }
            },
            'regulations': {
                'GDPR': {'enabled': True, 'jurisdiction': 'EU'},
                'CCPA': {'enabled': True, 'jurisdiction': 'California'},
                'HIPAA': {'enabled': False, 'jurisdiction': 'US Healthcare'}
            },
            'enforcement': {
                'strict_mode': False,
                'auto_approve_low_risk': True,
                'require_justification': True,
                'multi_approval_threshold': 'high_risk'
            }
        }
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _initialize_policies(self):
        """Initialize governance policies from configuration"""
        policy_configs = self.config.get('policies', {})
        
        # Data Governance Policy
        if 'data_governance' in policy_configs:
            self.policies['data_governance'] = GovernancePolicy(
                policy_id='POL-DATA-001',
                policy_type=PolicyType.DATA,
                name='Data Governance Policy',
                description='Controls data retention, deletion, and access',
                version='1.0.0',
                active=True,
                rules=self._create_data_rules(policy_configs['data_governance']),
                enforcement_level='strict',
                effective_date=datetime.now(),
                expiry_date=None,
                regulations=['GDPR', 'CCPA']
            )
        
        # Model Governance Policy
        if 'model_governance' in policy_configs:
            self.policies['model_governance'] = GovernancePolicy(
                policy_id='POL-MODEL-001',
                policy_type=PolicyType.MODEL,
                name='Model Governance Policy',
                description='Controls model training, deployment, and updates',
                version='1.0.0',
                active=True,
                rules=self._create_model_rules(policy_configs['model_governance']),
                enforcement_level='strict',
                effective_date=datetime.now(),
                expiry_date=None,
                regulations=[]
            )
        
        # Verification Governance Policy
        if 'verification_governance' in policy_configs:
            self.policies['verification_governance'] = GovernancePolicy(
                policy_id='POL-VERIFY-001',
                policy_type=PolicyType.VERIFICATION,
                name='Verification Governance Policy',
                description='Controls verification processes and thresholds',
                version='1.0.0',
                active=True,
                rules=self._create_verification_rules(policy_configs['verification_governance']),
                enforcement_level='warning',
                effective_date=datetime.now(),
                expiry_date=None,
                regulations=[]
            )
        
        # Audit Requirements Policy
        if 'audit_requirements' in policy_configs:
            self.policies['audit_requirements'] = GovernancePolicy(
                policy_id='POL-AUDIT-001',
                policy_type=PolicyType.AUDIT,
                name='Audit Requirements Policy',
                description='Defines audit logging and transparency requirements',
                version='1.0.0',
                active=True,
                rules=self._create_audit_rules(policy_configs['audit_requirements']),
                enforcement_level='strict',
                effective_date=datetime.now(),
                expiry_date=None,
                regulations=['SOX', 'ISO27001']
            )
    
    def _create_data_rules(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create data governance rules"""
        rules = []
        
        if 'retention_days' in config:
            rules.append({
                'id': 'DATA-R001',
                'description': f'Data retention limit: {config["retention_days"]} days',
                'type': 'retention',
                'max_retention_days': config['retention_days']
            })
        
        if config.get('encryption_required'):
            rules.append({
                'id': 'DATA-R002',
                'description': 'Encryption required for sensitive data',
                'type': 'security',
                'required_fields': ['encryption_method', 'key_id']
            })
        
        if config.get('anonymization_required'):
            rules.append({
                'id': 'DATA-R003',
                'description': 'Anonymization required for PII',
                'type': 'privacy',
                'required_fields': ['anonymization_method']
            })
        
        if 'access_control' in config:
            rules.append({
                'id': 'DATA-R004',
                'description': f'Access control: {config["access_control"]}',
                'type': 'access',
                'required_fields': ['user_role', 'access_level']
            })
        
        return rules
    
    def _create_model_rules(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create model governance rules"""
        rules = []
        
        if config.get('training_approval_required'):
            rules.append({
                'id': 'MODEL-R001',
                'description': 'Training requires approval',
                'type': 'approval',
                'required_approvals': ['model_owner', 'data_steward']
            })
        
        if config.get('deployment_approval_required'):
            rules.append({
                'id': 'MODEL-R002',
                'description': 'Deployment requires approval',
                'type': 'approval',
                'required_approvals': ['operations', 'security']
            })
        
        if config.get('testing_required'):
            rules.append({
                'id': 'MODEL-R003',
                'description': 'Testing required before deployment',
                'type': 'quality',
                'required_fields': ['test_results', 'accuracy_metrics']
            })
        
        if 'update_frequency_days' in config:
            rules.append({
                'id': 'MODEL-R004',
                'description': f'Model update frequency: {config["update_frequency_days"]} days',
                'type': 'maintenance',
                'update_frequency_days': config['update_frequency_days']
            })
        
        return rules
    
    def _create_verification_rules(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create verification governance rules"""
        rules = []
        
        if 'challenge_types' in config:
            rules.append({
                'id': 'VERIFY-R001',
                'description': f'Allowed challenge types: {config["challenge_types"]}',
                'type': 'challenge',
                'allowed_types': config['challenge_types']
            })
        
        if 'min_confidence_threshold' in config:
            rules.append({
                'id': 'VERIFY-R002',
                'description': f'Minimum confidence: {config["min_confidence_threshold"]}',
                'type': 'threshold',
                'min_confidence': config['min_confidence_threshold']
            })
        
        if 'max_challenge_reuse' in config:
            rules.append({
                'id': 'VERIFY-R003',
                'description': f'Maximum challenge reuse: {config["max_challenge_reuse"]}',
                'type': 'security',
                'max_reuse': config['max_challenge_reuse']
            })
        
        if config.get('sequential_testing_required'):
            rules.append({
                'id': 'VERIFY-R004',
                'description': 'Sequential testing required',
                'type': 'testing',
                'required_fields': ['sequential_config', 'early_stopping']
            })
        
        return rules
    
    def _create_audit_rules(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create audit requirement rules"""
        rules = []
        
        if config.get('immutable_logs'):
            rules.append({
                'id': 'AUDIT-R001',
                'description': 'Immutable audit logs required',
                'type': 'integrity',
                'required_fields': ['log_hash', 'timestamp']
            })
        
        if config.get('provenance_tracking'):
            rules.append({
                'id': 'AUDIT-R002',
                'description': 'Provenance tracking required',
                'type': 'traceability',
                'required_fields': ['origin', 'chain_of_custody']
            })
        
        if config.get('transparency_reports'):
            rules.append({
                'id': 'AUDIT-R003',
                'description': 'Transparency reports required',
                'type': 'reporting',
                'required_fields': ['report_type', 'period']
            })
        
        if 'retention_years' in config:
            rules.append({
                'id': 'AUDIT-R004',
                'description': f'Audit retention: {config["retention_years"]} years',
                'type': 'retention',
                'retention_years': config['retention_years']
            })
        
        return rules
    
    def _setup_audit_logging(self):
        """Setup immutable audit logging"""
        audit_file = Path('governance_audit.jsonl')
        if audit_file.exists():
            # Load existing audit log
            with open(audit_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.audit_log.append(json.loads(line))
    
    def check_compliance(self, action: str, context: Dict[str, Any]) -> ComplianceResult:
        """
        Check if an action complies with all active policies
        
        Args:
            action: The action to check
            context: Context information for the action
            
        Returns:
            ComplianceResult with status and details
        """
        violations = []
        warnings = []
        recommendations = []
        evidence = {}
        
        # Check each active policy
        for policy_id, policy in self.policies.items():
            if not policy.active:
                continue
            
            is_compliant, policy_violations = policy.evaluate(action, context)
            
            if not is_compliant:
                if policy.enforcement_level == 'strict':
                    violations.extend(policy_violations)
                elif policy.enforcement_level == 'warning':
                    warnings.extend(policy_violations)
                else:  # advisory
                    recommendations.extend(policy_violations)
            
            evidence[policy_id] = {
                'evaluated': True,
                'compliant': is_compliant,
                'violations': policy_violations
            }
        
        # Determine overall compliance status
        if violations:
            status = ComplianceStatus.NON_COMPLIANT
        elif warnings:
            status = ComplianceStatus.CONDITIONAL
        else:
            status = ComplianceStatus.COMPLIANT
        
        # Add regulatory compliance check
        regulatory_status = self._check_regulatory_compliance(action, context)
        if regulatory_status:
            evidence['regulatory'] = regulatory_status
            if regulatory_status.get('violations'):
                violations.extend(regulatory_status['violations'])
        
        result = ComplianceResult(
            status=status,
            policy_type=self._determine_policy_type(action),
            action=action,
            violations=violations,
            warnings=warnings,
            timestamp=datetime.now(),
            evidence=evidence,
            recommendations=recommendations
        )
        
        # Log the compliance check
        self._log_compliance_check(result)
        
        return result
    
    def _determine_policy_type(self, action: str) -> PolicyType:
        """Determine the primary policy type for an action"""
        action_lower = action.lower()
        
        if any(keyword in action_lower for keyword in ['data', 'retention', 'deletion', 'access']):
            return PolicyType.DATA
        elif any(keyword in action_lower for keyword in ['model', 'train', 'deploy', 'update']):
            return PolicyType.MODEL
        elif any(keyword in action_lower for keyword in ['verify', 'challenge', 'test']):
            return PolicyType.VERIFICATION
        elif any(keyword in action_lower for keyword in ['audit', 'log', 'report']):
            return PolicyType.AUDIT
        else:
            return PolicyType.COMPLIANCE
    
    def _check_regulatory_compliance(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with regulations"""
        result = {'violations': [], 'compliant': True}
        
        regulations = self.config.get('regulations', {})
        
        # GDPR compliance
        if regulations.get('GDPR', {}).get('enabled'):
            if 'personal_data' in context and context.get('personal_data'):
                if 'consent' not in context:
                    result['violations'].append('GDPR: Missing consent for personal data processing')
                    result['compliant'] = False
                if 'purpose' not in context:
                    result['violations'].append('GDPR: Missing purpose specification')
                    result['compliant'] = False
        
        # CCPA compliance
        if regulations.get('CCPA', {}).get('enabled'):
            if 'california_resident' in context and context.get('california_resident'):
                if action == 'data_sale' and not context.get('opt_out_honored'):
                    result['violations'].append('CCPA: Opt-out not honored')
                    result['compliant'] = False
        
        # HIPAA compliance
        if regulations.get('HIPAA', {}).get('enabled'):
            if 'health_data' in context and context.get('health_data'):
                if not context.get('encryption_enabled'):
                    result['violations'].append('HIPAA: Health data not encrypted')
                    result['compliant'] = False
        
        return result
    
    def log_decision(self, decision: GovernanceDecision):
        """
        Log a governance decision to the immutable audit trail
        
        Args:
            decision: The decision to log
        """
        with self._lock:
            # Add to decision history
            self.decisions.append(decision)
            
            # Create audit record
            audit_record = decision.to_audit_record()
            audit_record['sequence'] = len(self.audit_log)
            
            # Add previous hash for chain integrity
            if self.audit_log:
                prev_hash = self.audit_log[-1].get('hash', '')
                audit_record['prev_hash'] = prev_hash
            
            # Compute hash including previous hash
            content = json.dumps(audit_record, sort_keys=True)
            audit_record['hash'] = hashlib.sha256(content.encode()).hexdigest()
            
            # Append to audit log
            self.audit_log.append(audit_record)
            
            # Write to file
            with open('governance_audit.jsonl', 'a') as f:
                f.write(json.dumps(audit_record) + '\n')
            
            # Log to standard logger
            self.logger.info(f"Decision logged: {decision.decision_id} - {decision.action} - Approved: {decision.approved}")
    
    def _log_compliance_check(self, result: ComplianceResult):
        """Log a compliance check result"""
        log_entry = {
            'type': 'compliance_check',
            'timestamp': datetime.now().isoformat(),
            'result': result.to_dict()
        }
        
        with open('governance_audit.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.logger.info(f"Compliance check: {result.action} - Status: {result.status.value}")
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive compliance status report
        
        Returns:
            Dictionary containing compliance metrics and status
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'framework_version': self.config.get('version', '1.0.0'),
            'policies': {},
            'decisions': {},
            'compliance_rate': 0.0,
            'violations_summary': [],
            'recommendations': [],
            'regulatory_status': {}
        }
        
        # Policy status
        for policy_id, policy in self.policies.items():
            report['policies'][policy_id] = {
                'name': policy.name,
                'active': policy.active,
                'version': policy.version,
                'enforcement_level': policy.enforcement_level,
                'regulations': policy.regulations
            }
        
        # Decision statistics
        if self.decisions:
            approved = sum(1 for d in self.decisions if d.approved)
            report['decisions'] = {
                'total': len(self.decisions),
                'approved': approved,
                'rejected': len(self.decisions) - approved,
                'approval_rate': approved / len(self.decisions)
            }
            
            # Compliance rate from decisions
            compliant_decisions = sum(
                1 for d in self.decisions 
                if all(r.is_compliant() for r in d.compliance_results)
            )
            report['compliance_rate'] = compliant_decisions / len(self.decisions)
        
        # Violations summary
        violations_count = {}
        for decision in self.decisions:
            for result in decision.compliance_results:
                for violation in result.violations:
                    violations_count[violation] = violations_count.get(violation, 0) + 1
        
        report['violations_summary'] = [
            {'violation': v, 'count': c} 
            for v, c in sorted(violations_count.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Regulatory compliance status
        regulations = self.config.get('regulations', {})
        for reg_name, reg_config in regulations.items():
            if reg_config.get('enabled'):
                report['regulatory_status'][reg_name] = {
                    'enabled': True,
                    'jurisdiction': reg_config.get('jurisdiction'),
                    'compliant': True  # Would check actual compliance here
                }
        
        # Recommendations
        if report['compliance_rate'] < 0.8:
            report['recommendations'].append('Review and update policies to improve compliance rate')
        if len(report['violations_summary']) > 10:
            report['recommendations'].append('High number of violations detected - consider policy training')
        
        # Audit trail integrity
        report['audit_integrity'] = self._verify_audit_integrity()
        
        return report
    
    def _verify_audit_integrity(self) -> bool:
        """Verify the integrity of the audit trail"""
        if len(self.audit_log) < 2:
            return True
        
        for i in range(1, len(self.audit_log)):
            current = self.audit_log[i]
            previous = self.audit_log[i-1]
            
            # Check hash chain
            if current.get('prev_hash') != previous.get('hash'):
                self.logger.error(f"Audit integrity violation at sequence {i}")
                return False
        
        return True
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against regulatory requirements
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required sections
        required_sections = ['policies', 'regulations', 'enforcement']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate policy configurations
        if 'policies' in config:
            policy_issues = self._validate_policies(config['policies'])
            issues.extend(policy_issues)
        
        # Validate regulatory settings
        if 'regulations' in config:
            reg_issues = self._validate_regulations(config['regulations'])
            issues.extend(reg_issues)
        
        # Check security settings
        if 'master_key' in config:
            if len(config['master_key']) < 32:
                issues.append("Master key must be at least 32 characters")
        
        return len(issues) == 0, issues
    
    def _validate_policies(self, policies: Dict[str, Any]) -> List[str]:
        """Validate policy configurations"""
        issues = []
        
        # Data governance validation
        if 'data_governance' in policies:
            data_policy = policies['data_governance']
            if data_policy.get('retention_days', 0) > 365:
                issues.append("Data retention exceeds recommended 365 days")
            if not data_policy.get('encryption_required'):
                issues.append("Encryption should be required for data governance")
        
        # Model governance validation
        if 'model_governance' in policies:
            model_policy = policies['model_governance']
            if not model_policy.get('testing_required'):
                issues.append("Testing should be required before model deployment")
        
        # Verification governance validation
        if 'verification_governance' in policies:
            verify_policy = policies['verification_governance']
            if verify_policy.get('min_confidence_threshold', 1.0) < 0.7:
                issues.append("Minimum confidence threshold too low (< 0.7)")
        
        return issues
    
    def _validate_regulations(self, regulations: Dict[str, Any]) -> List[str]:
        """Validate regulatory configurations"""
        issues = []
        
        # Check for conflicting regulations
        if regulations.get('GDPR', {}).get('enabled') and regulations.get('CCPA', {}).get('enabled'):
            # This is actually fine, just note it
            pass
        
        # Check HIPAA requirements
        if regulations.get('HIPAA', {}).get('enabled'):
            # HIPAA has strict requirements
            policies = self.config.get('policies', {})
            if not policies.get('data_governance', {}).get('encryption_required'):
                issues.append("HIPAA requires encryption but it's not enabled")
        
        return issues
    
    def create_decision(
        self,
        action: str,
        context: Dict[str, Any],
        decision_maker: DecisionMaker = DecisionMaker.AUTOMATED,
        maker_id: str = "system",
        justification: str = ""
    ) -> GovernanceDecision:
        """
        Create and process a governance decision
        
        Args:
            action: The action requiring a decision
            context: Context for the decision
            decision_maker: Type of decision maker
            maker_id: ID of the decision maker
            justification: Justification for the decision
            
        Returns:
            GovernanceDecision object
        """
        # Check compliance
        compliance_result = self.check_compliance(action, context)
        
        # Determine if approved based on compliance and enforcement settings
        approved = compliance_result.is_compliant()
        conditions = []
        
        if compliance_result.status == ComplianceStatus.CONDITIONAL:
            # Check if auto-approval is allowed
            if self.config.get('enforcement', {}).get('auto_approve_low_risk'):
                approved = True
                conditions.append("Auto-approved with warnings")
            else:
                approved = False
        
        # Create decision
        decision = GovernanceDecision(
            decision_id=f"DEC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}",
            action=action,
            decision_maker=decision_maker,
            maker_id=maker_id,
            timestamp=datetime.now(),
            justification=justification or "Automated compliance check",
            evidence=context,
            policies_applied=list(self.policies.keys()),
            compliance_results=[compliance_result],
            approved=approved,
            conditions=conditions,
            expiry=datetime.now() + timedelta(days=30) if approved else None
        )
        
        # Log the decision
        self.log_decision(decision)
        
        return decision
    
    def get_active_session(self) -> Optional[str]:
        """Get or create an active challenge session"""
        if self.challenge_governance:
            # Check for valid existing session
            for session_id in self.challenge_governance.sessions:
                if self.challenge_governance.is_session_valid(session_id):
                    return session_id
            
            # Create new session
            return self.challenge_governance.new_session()
        return None
    
    def rotate_epoch(self) -> Dict[str, Any]:
        """Rotate to a new epoch for key management"""
        if self.challenge_governance:
            old_epoch = self.challenge_governance.current_epoch
            new_epoch = self.challenge_governance.new_epoch()
            
            # Rotate keys
            rotation = rotate_epoch_key(
                self.config['master_key'],
                old_epoch
            )
            
            # Log the rotation
            self.logger.info(f"Epoch rotated from {old_epoch} to {new_epoch}")
            
            # Create governance decision for the rotation
            self.create_decision(
                action="epoch_rotation",
                context={
                    'old_epoch': old_epoch,
                    'new_epoch': new_epoch,
                    'reason': 'scheduled_rotation'
                },
                justification="Scheduled epoch rotation for key management"
            )
            
            return rotation
        return {}