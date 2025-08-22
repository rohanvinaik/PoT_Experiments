"""
Policy Engine for PoT Governance Framework
Flexible policy evaluation, enforcement, and management system
"""

import json
import yaml
import logging
import re
import operator
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import copy
from collections import defaultdict


class PolicyType(Enum):
    """Types of policies"""
    DATA_RETENTION = "data_retention"
    MODEL_RETRAINING = "model_retraining"
    ACCESS_CONTROL = "access_control"
    VERIFICATION_THRESHOLD = "verification_threshold"
    AUDIT_LOGGING = "audit_logging"
    GOVERNANCE = "governance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class EnforcementMode(Enum):
    """Policy enforcement modes"""
    STRICT = "strict"       # Block if policy violation
    ADVISORY = "advisory"   # Warn but allow
    MONITOR = "monitor"     # Log only, no action
    DISABLED = "disabled"   # Policy not enforced


class PolicyStatus(Enum):
    """Policy evaluation status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    MOST_RESTRICTIVE = "most_restrictive"
    LEAST_RESTRICTIVE = "least_restrictive"
    PRIORITY_BASED = "priority_based"
    FIRST_MATCH = "first_match"
    MANUAL = "manual"


@dataclass
class PolicyRule:
    """Individual policy rule"""
    rule_id: str
    condition: str
    action: Optional[str] = None
    requirement: Optional[str] = None
    message: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition against context"""
        try:
            return evaluate_condition(self.condition, context)
        except Exception as e:
            logging.error(f"Error evaluating rule {self.rule_id}: {e}")
            return False


@dataclass
class PolicyResult:
    """Result of policy evaluation"""
    policy_name: str
    policy_type: PolicyType
    status: PolicyStatus
    timestamp: datetime
    rules_evaluated: int
    rules_passed: int
    rules_failed: int
    violations: List[str]
    warnings: List[str]
    actions_required: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_compliant(self) -> bool:
        return self.status in [PolicyStatus.PASS, PolicyStatus.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['policy_type'] = self.policy_type.value
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class EnforcementDecision:
    """Decision from policy enforcement"""
    action: str
    allowed: bool
    enforcement_mode: EnforcementMode
    timestamp: datetime
    policies_checked: List[str]
    violations: List[str]
    warnings: List[str]
    required_actions: List[str]
    override_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['enforcement_mode'] = self.enforcement_mode.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class Policy:
    """Policy definition"""
    name: str
    type: PolicyType
    version: str
    description: str
    enabled: bool
    enforcement_mode: EnforcementMode
    priority: int  # Higher number = higher priority
    rules: List[PolicyRule]
    metadata: Dict[str, Any]
    created_date: datetime
    modified_date: datetime
    author: str
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    
    def __hash__(self):
        """Hash for policy versioning"""
        content = f"{self.name}{self.version}{self.type.value}"
        return hash(content)
    
    def evaluate(self, context: Dict[str, Any]) -> PolicyResult:
        """Evaluate policy against context"""
        violations = []
        warnings = []
        actions = []
        rules_passed = 0
        rules_failed = 0
        
        for rule in self.rules:
            if rule.evaluate(context):
                rules_passed += 1
                if rule.action:
                    actions.append(rule.action)
            else:
                rules_failed += 1
                if rule.severity in ["high", "critical"]:
                    violations.append(f"{rule.rule_id}: {rule.message or rule.condition}")
                else:
                    warnings.append(f"{rule.rule_id}: {rule.message or rule.condition}")
        
        # Determine overall status
        if violations:
            status = PolicyStatus.FAIL
        elif warnings:
            status = PolicyStatus.WARNING
        elif rules_passed > 0:
            status = PolicyStatus.PASS
        else:
            status = PolicyStatus.NOT_APPLICABLE
        
        return PolicyResult(
            policy_name=self.name,
            policy_type=self.type,
            status=status,
            timestamp=datetime.now(),
            rules_evaluated=len(self.rules),
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            violations=violations,
            warnings=warnings,
            actions_required=actions,
            metadata={"version": self.version, "priority": self.priority}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            "name": self.name,
            "type": self.type.value,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "enforcement_mode": self.enforcement_mode.value,
            "priority": self.priority,
            "rules": [asdict(rule) for rule in self.rules],
            "metadata": self.metadata,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "author": self.author,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "conflicts_with": self.conflicts_with
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """Create policy from dictionary"""
        rules = [PolicyRule(**rule) for rule in data.get("rules", [])]
        
        return cls(
            name=data["name"],
            type=PolicyType(data["type"]),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            enforcement_mode=EnforcementMode(data.get("enforcement_mode", "advisory")),
            priority=data.get("priority", 50),
            rules=rules,
            metadata=data.get("metadata", {}),
            created_date=datetime.fromisoformat(data["created_date"]) if "created_date" in data else datetime.now(),
            modified_date=datetime.fromisoformat(data["modified_date"]) if "modified_date" in data else datetime.now(),
            author=data.get("author", "system"),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            conflicts_with=data.get("conflicts_with", [])
        )


class PolicyVersion:
    """Policy version management"""
    
    def __init__(self):
        self.versions: Dict[str, List[Policy]] = defaultdict(list)
        self.active_versions: Dict[str, str] = {}
    
    def add_version(self, policy: Policy):
        """Add a new version of a policy"""
        self.versions[policy.name].append(policy)
        self.versions[policy.name].sort(key=lambda p: p.version, reverse=True)
        
        # Set as active if no active version exists
        if policy.name not in self.active_versions:
            self.active_versions[policy.name] = policy.version
    
    def get_active(self, policy_name: str) -> Optional[Policy]:
        """Get active version of a policy"""
        if policy_name not in self.active_versions:
            return None
        
        active_version = self.active_versions[policy_name]
        for policy in self.versions[policy_name]:
            if policy.version == active_version:
                return policy
        return None
    
    def rollback(self, policy_name: str, version: str) -> bool:
        """Rollback to a specific version"""
        for policy in self.versions[policy_name]:
            if policy.version == version:
                self.active_versions[policy_name] = version
                return True
        return False
    
    def get_history(self, policy_name: str) -> List[Policy]:
        """Get version history of a policy"""
        return self.versions.get(policy_name, [])


class PolicyEngine:
    """
    Flexible policy engine for governance and compliance
    """
    
    def __init__(self, policy_dir: Optional[str] = None, enforcement_mode: str = "strict"):
        """
        Initialize policy engine
        
        Args:
            policy_dir: Directory containing policy files
            enforcement_mode: Default enforcement mode (strict/advisory/monitor)
        """
        self.policy_dir = Path(policy_dir) if policy_dir else None
        self.policies: Dict[str, Policy] = {}
        self.enforcement_mode = EnforcementMode(enforcement_mode)
        self.version_manager = PolicyVersion()
        self.evaluation_cache: Dict[str, PolicyResult] = {}
        self.conflict_resolution = ConflictResolution.MOST_RESTRICTIVE
        self.logger = logging.getLogger(__name__)
        
        # Load default policies
        self._load_default_policies()
        
        # Load policies from directory
        if self.policy_dir and self.policy_dir.exists():
            self.load_policies(str(self.policy_dir))
    
    def _load_default_policies(self):
        """Load default built-in policies"""
        # Data Retention Policy
        data_retention = Policy(
            name="data_retention_default",
            type=PolicyType.DATA_RETENTION,
            version="1.0.0",
            description="Default data retention policy",
            enabled=True,
            enforcement_mode=EnforcementMode.STRICT,
            priority=50,
            rules=[
                PolicyRule(
                    rule_id="DR001",
                    condition="data_age > 90",
                    action="archive",
                    message="Data older than 90 days should be archived"
                ),
                PolicyRule(
                    rule_id="DR002",
                    condition="data_age > 365",
                    action="delete",
                    message="Data older than 365 days should be deleted",
                    severity="high"
                ),
                PolicyRule(
                    rule_id="DR003",
                    condition="data_sensitivity == 'high' and not encryption_at_rest",
                    requirement="encryption_at_rest",
                    message="High sensitivity data must be encrypted at rest",
                    severity="critical"
                )
            ],
            metadata={"category": "compliance"},
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="system",
            tags=["data", "retention", "compliance"]
        )
        self.add_policy(data_retention)
        
        # Model Retraining Policy
        model_retraining = Policy(
            name="model_retraining_default",
            type=PolicyType.MODEL_RETRAINING,
            version="1.0.0",
            description="Default model retraining policy",
            enabled=True,
            enforcement_mode=EnforcementMode.ADVISORY,
            priority=40,
            rules=[
                PolicyRule(
                    rule_id="MR001",
                    condition="model_age_days > 30",
                    action="evaluate_retraining",
                    message="Model older than 30 days should be evaluated for retraining"
                ),
                PolicyRule(
                    rule_id="MR002",
                    condition="performance_degradation > 0.1",
                    action="retrain",
                    message="Model with >10% performance degradation must be retrained",
                    severity="high"
                ),
                PolicyRule(
                    rule_id="MR003",
                    condition="data_drift_detected and drift_severity == 'high'",
                    action="immediate_retrain",
                    message="High data drift requires immediate retraining",
                    severity="critical"
                )
            ],
            metadata={"category": "performance"},
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="system",
            tags=["model", "retraining", "performance"]
        )
        self.add_policy(model_retraining)
        
        # Access Control Policy
        access_control = Policy(
            name="access_control_default",
            type=PolicyType.ACCESS_CONTROL,
            version="1.0.0",
            description="Default access control policy",
            enabled=True,
            enforcement_mode=EnforcementMode.STRICT,
            priority=80,
            rules=[
                PolicyRule(
                    rule_id="AC001",
                    condition="not authenticated",
                    action="deny",
                    message="Authentication required",
                    severity="critical"
                ),
                PolicyRule(
                    rule_id="AC002",
                    condition="role not in allowed_roles",
                    action="deny",
                    message="Insufficient privileges",
                    severity="high"
                ),
                PolicyRule(
                    rule_id="AC003",
                    condition="accessing_sensitive_data and not mfa_enabled",
                    requirement="mfa",
                    message="MFA required for sensitive data access",
                    severity="high"
                )
            ],
            metadata={"category": "security"},
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="system",
            tags=["access", "security", "authentication"]
        )
        self.add_policy(access_control)
        
        # Verification Threshold Policy
        verification_threshold = Policy(
            name="verification_threshold_default",
            type=PolicyType.VERIFICATION_THRESHOLD,
            version="1.0.0",
            description="Default verification threshold policy",
            enabled=True,
            enforcement_mode=EnforcementMode.STRICT,
            priority=60,
            rules=[
                PolicyRule(
                    rule_id="VT001",
                    condition="confidence_score < 0.7",
                    action="reject",
                    message="Confidence score below minimum threshold",
                    severity="high"
                ),
                PolicyRule(
                    rule_id="VT002",
                    condition="confidence_score < 0.85 and risk_level == 'high'",
                    action="manual_review",
                    message="High-risk verification requires manual review",
                    severity="medium"
                ),
                PolicyRule(
                    rule_id="VT003",
                    condition="sequential_failures > 3",
                    action="block_verification",
                    message="Too many sequential verification failures",
                    severity="critical"
                )
            ],
            metadata={"category": "verification"},
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="system",
            tags=["verification", "threshold", "security"]
        )
        self.add_policy(verification_threshold)
        
        # Audit Logging Policy
        audit_logging = Policy(
            name="audit_logging_default",
            type=PolicyType.AUDIT_LOGGING,
            version="1.0.0",
            description="Default audit logging policy",
            enabled=True,
            enforcement_mode=EnforcementMode.STRICT,
            priority=70,
            rules=[
                PolicyRule(
                    rule_id="AL001",
                    condition="action_type in ['create', 'update', 'delete']",
                    action="log_detailed",
                    message="CRUD operations must be logged"
                ),
                PolicyRule(
                    rule_id="AL002",
                    condition="accessing_pii",
                    action="log_with_encryption",
                    message="PII access must be logged with encryption",
                    severity="high"
                ),
                PolicyRule(
                    rule_id="AL003",
                    condition="security_event",
                    action="immediate_alert",
                    message="Security events require immediate alerting",
                    severity="critical"
                )
            ],
            metadata={"category": "audit"},
            created_date=datetime.now(),
            modified_date=datetime.now(),
            author="system",
            tags=["audit", "logging", "compliance"]
        )
        self.add_policy(audit_logging)
    
    def load_policies(self, policy_dir: str) -> int:
        """
        Load policies from directory
        
        Args:
            policy_dir: Directory containing policy files (YAML/JSON)
            
        Returns:
            Number of policies loaded
        """
        policy_path = Path(policy_dir)
        if not policy_path.exists():
            self.logger.error(f"Policy directory {policy_dir} does not exist")
            return 0
        
        loaded = 0
        for file_path in policy_path.glob("*.{yaml,yml,json}"):
            try:
                policy = self._load_policy_file(file_path)
                if policy:
                    self.add_policy(policy)
                    loaded += 1
            except Exception as e:
                self.logger.error(f"Error loading policy from {file_path}: {e}")
        
        self.logger.info(f"Loaded {loaded} policies from {policy_dir}")
        return loaded
    
    def _load_policy_file(self, file_path: Path) -> Optional[Policy]:
        """Load a single policy file"""
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        if 'policy' in data:
            data = data['policy']
        
        return Policy.from_dict(data)
    
    def add_policy(self, policy: Policy) -> bool:
        """
        Add a policy with validation
        
        Args:
            policy: Policy to add
            
        Returns:
            True if added successfully
        """
        # Validate policy
        if not self._validate_policy(policy):
            self.logger.error(f"Policy {policy.name} validation failed")
            return False
        
        # Check for conflicts
        conflicts = self._check_conflicts(policy)
        if conflicts and self.conflict_resolution == ConflictResolution.MANUAL:
            self.logger.warning(f"Policy {policy.name} conflicts with: {conflicts}")
            return False
        
        # Add to policies
        self.policies[policy.name] = policy
        
        # Add to version manager
        self.version_manager.add_version(policy)
        
        # Clear evaluation cache
        self.evaluation_cache.clear()
        
        self.logger.info(f"Added policy {policy.name} v{policy.version}")
        return True
    
    def _validate_policy(self, policy: Policy) -> bool:
        """Validate policy structure and rules"""
        if not policy.name or not policy.rules:
            return False
        
        # Validate each rule
        for rule in policy.rules:
            try:
                # Test condition parsing
                test_context = {"test": True}
                evaluate_condition(rule.condition, test_context)
            except Exception as e:
                self.logger.error(f"Invalid rule condition in {policy.name}: {rule.condition}")
                return False
        
        # Check dependencies
        for dep in policy.dependencies:
            if dep not in self.policies:
                self.logger.warning(f"Policy {policy.name} has unmet dependency: {dep}")
        
        return True
    
    def _check_conflicts(self, policy: Policy) -> List[str]:
        """Check for policy conflicts"""
        conflicts = []
        
        for existing_name, existing_policy in self.policies.items():
            # Check explicit conflicts
            if existing_name in policy.conflicts_with:
                conflicts.append(existing_name)
            elif policy.name in existing_policy.conflicts_with:
                conflicts.append(existing_name)
            
            # Check for rule conflicts (simplified)
            if existing_policy.type == policy.type and existing_policy.priority == policy.priority:
                conflicts.append(existing_name)
        
        return conflicts
    
    def evaluate_policy(self, policy_name: str, context: Dict[str, Any]) -> PolicyResult:
        """
        Evaluate a single policy against context
        
        Args:
            policy_name: Name of policy to evaluate
            context: Context for evaluation
            
        Returns:
            PolicyResult
        """
        if policy_name not in self.policies:
            return PolicyResult(
                policy_name=policy_name,
                policy_type=PolicyType.CUSTOM,
                status=PolicyStatus.ERROR,
                timestamp=datetime.now(),
                rules_evaluated=0,
                rules_passed=0,
                rules_failed=0,
                violations=[f"Policy {policy_name} not found"],
                warnings=[],
                actions_required=[]
            )
        
        # Check cache
        cache_key = f"{policy_name}:{hash(str(context))}"
        if cache_key in self.evaluation_cache:
            cached_result = self.evaluation_cache[cache_key]
            # Check if cache is still valid (5 minutes)
            if (datetime.now() - cached_result.timestamp).seconds < 300:
                return cached_result
        
        # Evaluate policy
        policy = self.policies[policy_name]
        result = policy.evaluate(context)
        
        # Cache result
        self.evaluation_cache[cache_key] = result
        
        return result
    
    def enforce_policies(self, action: str, context: Dict[str, Any]) -> EnforcementDecision:
        """
        Check all relevant policies for an action
        
        Args:
            action: Action being performed
            context: Context for enforcement
            
        Returns:
            EnforcementDecision
        """
        # Add action to context
        context['action'] = action
        
        violations = []
        warnings = []
        required_actions = []
        policies_checked = []
        
        # Find relevant policies
        relevant_policies = self._find_relevant_policies(action, context)
        
        # Evaluate each policy
        policy_results = []
        for policy_name in relevant_policies:
            policy = self.policies[policy_name]
            
            # Skip disabled policies
            if not policy.enabled:
                continue
            
            result = self.evaluate_policy(policy_name, context)
            policy_results.append((policy, result))
            policies_checked.append(policy_name)
            
            # Collect violations and warnings
            if result.status == PolicyStatus.FAIL:
                violations.extend(result.violations)
            warnings.extend(result.warnings)
            required_actions.extend(result.actions_required)
        
        # Resolve conflicts if multiple policies apply
        final_decision = self._resolve_conflicts(policy_results)
        
        # Determine if action is allowed
        if self.enforcement_mode == EnforcementMode.STRICT:
            allowed = len(violations) == 0
        elif self.enforcement_mode == EnforcementMode.ADVISORY:
            allowed = True  # Allow but warn
        elif self.enforcement_mode == EnforcementMode.MONITOR:
            allowed = True  # Always allow, just log
        else:  # DISABLED
            allowed = True
        
        decision = EnforcementDecision(
            action=action,
            allowed=allowed,
            enforcement_mode=self.enforcement_mode,
            timestamp=datetime.now(),
            policies_checked=policies_checked,
            violations=violations,
            warnings=warnings,
            required_actions=list(set(required_actions)),
            metadata={"conflict_resolution": self.conflict_resolution.value}
        )
        
        # Log enforcement decision
        self._log_enforcement(decision)
        
        return decision
    
    def _find_relevant_policies(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Find policies relevant to an action"""
        relevant = []
        
        for name, policy in self.policies.items():
            # Check if policy type matches action category
            if self._is_policy_relevant(policy, action, context):
                relevant.append(name)
        
        # Sort by priority (higher priority first)
        relevant.sort(key=lambda n: self.policies[n].priority, reverse=True)
        
        return relevant
    
    def _is_policy_relevant(self, policy: Policy, action: str, context: Dict[str, Any]) -> bool:
        """Determine if a policy is relevant to an action"""
        # Check tags
        action_tags = context.get("tags", [])
        if any(tag in policy.tags for tag in action_tags):
            return True
        
        # Check policy type
        action_category = context.get("category", "")
        if policy.type.value in action_category:
            return True
        
        # Check metadata
        if "applies_to" in policy.metadata:
            applies_to = policy.metadata["applies_to"]
            if isinstance(applies_to, list) and action in applies_to:
                return True
        
        # Default relevance by type
        relevance_map = {
            PolicyType.ACCESS_CONTROL: ["access", "login", "authenticate"],
            PolicyType.DATA_RETENTION: ["store", "archive", "delete"],
            PolicyType.MODEL_RETRAINING: ["train", "retrain", "update_model"],
            PolicyType.VERIFICATION_THRESHOLD: ["verify", "validate", "check"],
            PolicyType.AUDIT_LOGGING: ["log", "audit", "record"]
        }
        
        if policy.type in relevance_map:
            keywords = relevance_map[policy.type]
            if any(keyword in action.lower() for keyword in keywords):
                return True
        
        return False
    
    def _resolve_conflicts(self, policy_results: List[Tuple[Policy, PolicyResult]]) -> str:
        """Resolve conflicts between multiple policies"""
        if not policy_results:
            return "allow"
        
        if self.conflict_resolution == ConflictResolution.MOST_RESTRICTIVE:
            # If any policy fails, deny
            for policy, result in policy_results:
                if result.status == PolicyStatus.FAIL:
                    return "deny"
            return "allow"
        
        elif self.conflict_resolution == ConflictResolution.LEAST_RESTRICTIVE:
            # If any policy passes, allow
            for policy, result in policy_results:
                if result.status == PolicyStatus.PASS:
                    return "allow"
            return "deny"
        
        elif self.conflict_resolution == ConflictResolution.PRIORITY_BASED:
            # Use highest priority policy
            if policy_results:
                highest_priority = policy_results[0]
                return "allow" if highest_priority[1].status == PolicyStatus.PASS else "deny"
        
        elif self.conflict_resolution == ConflictResolution.FIRST_MATCH:
            # Use first matching policy
            if policy_results:
                first_match = policy_results[0]
                return "allow" if first_match[1].status == PolicyStatus.PASS else "deny"
        
        return "deny"  # Default to deny
    
    def _log_enforcement(self, decision: EnforcementDecision):
        """Log enforcement decision"""
        log_level = logging.INFO
        if decision.violations:
            log_level = logging.WARNING if decision.allowed else logging.ERROR
        
        self.logger.log(
            log_level,
            f"Policy enforcement for '{decision.action}': "
            f"{'ALLOWED' if decision.allowed else 'DENIED'} "
            f"({len(decision.violations)} violations, {len(decision.warnings)} warnings)"
        )
    
    def update_policy(self, policy_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing policy
        
        Args:
            policy_name: Name of policy to update
            updates: Dictionary of updates
            
        Returns:
            True if updated successfully
        """
        if policy_name not in self.policies:
            return False
        
        old_policy = self.policies[policy_name]
        
        # Create new version
        new_policy_dict = old_policy.to_dict()
        new_policy_dict.update(updates)
        
        # Increment version
        version_parts = old_policy.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_policy_dict['version'] = '.'.join(version_parts)
        
        # Update modified date
        new_policy_dict['modified_date'] = datetime.now().isoformat()
        
        # Create new policy
        new_policy = Policy.from_dict(new_policy_dict)
        
        # Add new version
        return self.add_policy(new_policy)
    
    def remove_policy(self, policy_name: str) -> bool:
        """
        Remove a policy
        
        Args:
            policy_name: Name of policy to remove
            
        Returns:
            True if removed successfully
        """
        if policy_name not in self.policies:
            return False
        
        # Check if other policies depend on this one
        dependencies = []
        for name, policy in self.policies.items():
            if policy_name in policy.dependencies:
                dependencies.append(name)
        
        if dependencies:
            self.logger.warning(f"Cannot remove {policy_name}, required by: {dependencies}")
            return False
        
        del self.policies[policy_name]
        self.evaluation_cache.clear()
        
        self.logger.info(f"Removed policy {policy_name}")
        return True
    
    def rollback_policy(self, policy_name: str, version: str) -> bool:
        """
        Rollback a policy to a specific version
        
        Args:
            policy_name: Name of policy
            version: Version to rollback to
            
        Returns:
            True if rollback successful
        """
        if self.version_manager.rollback(policy_name, version):
            # Update active policy
            rolled_back = self.version_manager.get_active(policy_name)
            if rolled_back:
                self.policies[policy_name] = rolled_back
                self.evaluation_cache.clear()
                self.logger.info(f"Rolled back {policy_name} to version {version}")
                return True
        
        return False
    
    def generate_policy_report(self) -> Dict[str, Any]:
        """
        Generate summary of all active policies and compliance
        
        Returns:
            Policy report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_policies": len(self.policies),
            "enforcement_mode": self.enforcement_mode.value,
            "conflict_resolution": self.conflict_resolution.value,
            "policies_by_type": {},
            "policies_by_status": {
                "enabled": 0,
                "disabled": 0
            },
            "policies_by_mode": {
                "strict": 0,
                "advisory": 0,
                "monitor": 0,
                "disabled": 0
            },
            "policy_details": [],
            "version_summary": {},
            "recent_evaluations": []
        }
        
        # Count policies by type
        for policy_type in PolicyType:
            count = sum(1 for p in self.policies.values() if p.type == policy_type)
            if count > 0:
                report["policies_by_type"][policy_type.value] = count
        
        # Count by status and mode
        for policy in self.policies.values():
            if policy.enabled:
                report["policies_by_status"]["enabled"] += 1
            else:
                report["policies_by_status"]["disabled"] += 1
            
            report["policies_by_mode"][policy.enforcement_mode.value] += 1
        
        # Policy details
        for name, policy in self.policies.items():
            report["policy_details"].append({
                "name": name,
                "type": policy.type.value,
                "version": policy.version,
                "enabled": policy.enabled,
                "priority": policy.priority,
                "rules_count": len(policy.rules),
                "tags": policy.tags,
                "enforcement_mode": policy.enforcement_mode.value
            })
        
        # Version summary
        for policy_name in self.policies:
            history = self.version_manager.get_history(policy_name)
            if history:
                report["version_summary"][policy_name] = {
                    "versions": len(history),
                    "active": self.version_manager.active_versions.get(policy_name),
                    "latest": history[0].version if history else None
                }
        
        # Recent evaluations from cache
        recent = sorted(
            self.evaluation_cache.values(),
            key=lambda r: r.timestamp,
            reverse=True
        )[:10]
        
        for result in recent:
            report["recent_evaluations"].append({
                "policy": result.policy_name,
                "status": result.status.value,
                "timestamp": result.timestamp.isoformat(),
                "violations": len(result.violations),
                "warnings": len(result.warnings)
            })
        
        return report
    
    def export_policies(self, output_path: str, format: str = "yaml") -> bool:
        """
        Export all policies to file
        
        Args:
            output_path: Output file path
            format: Export format (yaml/json)
            
        Returns:
            True if exported successfully
        """
        policies_data = {
            "exported_at": datetime.now().isoformat(),
            "engine_config": {
                "enforcement_mode": self.enforcement_mode.value,
                "conflict_resolution": self.conflict_resolution.value
            },
            "policies": [policy.to_dict() for policy in self.policies.values()]
        }
        
        try:
            with open(output_path, 'w') as f:
                if format == "yaml":
                    yaml.dump(policies_data, f, default_flow_style=False)
                else:
                    json.dump(policies_data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.policies)} policies to {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting policies: {e}")
            return False
    
    def import_policies(self, input_path: str, replace: bool = False) -> int:
        """
        Import policies from file
        
        Args:
            input_path: Input file path
            replace: Replace existing policies
            
        Returns:
            Number of policies imported
        """
        try:
            with open(input_path, 'r') as f:
                if input_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            if replace:
                self.policies.clear()
                self.version_manager = PolicyVersion()
            
            imported = 0
            for policy_data in data.get("policies", []):
                policy = Policy.from_dict(policy_data)
                if self.add_policy(policy):
                    imported += 1
            
            self.logger.info(f"Imported {imported} policies from {input_path}")
            return imported
        
        except Exception as e:
            self.logger.error(f"Error importing policies: {e}")
            return 0


# Condition evaluation helpers
def evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
    """
    Evaluate a condition string against context
    
    Supports:
    - Comparisons: ==, !=, <, >, <=, >=
    - Logical: and, or, not
    - Membership: in, not in
    - Values: strings, numbers, booleans
    """
    try:
        # Replace context variables
        for key, value in context.items():
            # Handle different value types
            if isinstance(value, str):
                condition = condition.replace(key, f"'{value}'")
            elif isinstance(value, bool):
                condition = condition.replace(key, str(value))
            elif value is None:
                condition = condition.replace(key, "None")
            else:
                condition = condition.replace(key, str(value))
        
        # Safe evaluation
        # Note: In production, use a proper expression parser for security
        # This is simplified for demonstration
        result = eval(condition, {"__builtins__": {}}, {})
        return bool(result)
    
    except Exception as e:
        logging.error(f"Error evaluating condition '{condition}': {e}")
        return False


# Example policy file format
EXAMPLE_POLICY_YAML = """
policy:
  name: "advanced_data_governance"
  type: "data_retention"
  version: "2.0.0"
  description: "Advanced data governance policy with GDPR compliance"
  enabled: true
  enforcement_mode: "strict"
  priority: 100
  author: "compliance_team"
  tags:
    - "gdpr"
    - "data"
    - "compliance"
  dependencies:
    - "encryption_policy"
  conflicts_with:
    - "legacy_retention_policy"
  rules:
    - rule_id: "GDPR001"
      condition: "data_type == 'personal' and consent_withdrawn"
      action: "delete_immediately"
      message: "Personal data must be deleted when consent is withdrawn"
      severity: "critical"
    - rule_id: "GDPR002"
      condition: "data_type == 'personal' and data_age > 30"
      action: "pseudonymize"
      message: "Personal data older than 30 days should be pseudonymized"
      severity: "high"
    - rule_id: "GDPR003"
      condition: "data_location not in ['EU', 'EEA']"
      requirement: "data_transfer_agreement"
      message: "Data transfer agreement required for non-EU storage"
      severity: "high"
  metadata:
    regulation: "GDPR"
    last_audit: "2024-01-15"
    applies_to:
      - "store_data"
      - "transfer_data"
      - "process_data"
"""