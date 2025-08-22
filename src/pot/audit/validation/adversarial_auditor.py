"""
Adversarial Auditor

Tests audit systems under adversarial attack conditions to ensure robustness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import hashlib
import time
from enum import Enum

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks"""
    REPLAY = "replay"
    INJECTION = "injection"
    TAMPERING = "tampering"
    TIMING = "timing"
    EVASION = "evasion"
    POISONING = "poisoning"
    MODEL_EXTRACTION = "model_extraction"
    MEMBERSHIP_INFERENCE = "membership_inference"


@dataclass
class AttackResult:
    """Result of an adversarial attack attempt"""
    attack_type: AttackType
    success: bool
    detection_evaded: bool
    impact_score: float
    artifacts: Dict[str, Any]
    defense_effectiveness: float


@dataclass
class AuditRobustnessReport:
    """Report on audit system robustness against attacks"""
    overall_robustness: float
    successful_attacks: List[AttackResult]
    failed_attacks: List[AttackResult]
    vulnerabilities: List[str]
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]


class AdversarialAuditor:
    """
    Tests audit and verification systems under adversarial conditions.
    
    Features:
    - Multiple attack vector simulation
    - Adaptive attack strategies
    - Defense effectiveness measurement
    - Vulnerability assessment
    - Robustness scoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adversarial auditor.
        
        Args:
            config: Configuration for adversarial testing
        """
        self.config = config or {}
        self.attack_intensity = self.config.get('attack_intensity', 0.5)
        self.max_attempts = self.config.get('max_attempts', 100)
        self.detection_threshold = self.config.get('detection_threshold', 0.8)
        self.attack_strategies = self._initialize_strategies()
        
    def _initialize_strategies(self) -> Dict[AttackType, Callable]:
        """Initialize attack strategy functions"""
        return {
            AttackType.REPLAY: self._replay_attack,
            AttackType.INJECTION: self._injection_attack,
            AttackType.TAMPERING: self._tampering_attack,
            AttackType.TIMING: self._timing_attack,
            AttackType.EVASION: self._evasion_attack,
            AttackType.POISONING: self._poisoning_attack,
            AttackType.MODEL_EXTRACTION: self._model_extraction_attack,
            AttackType.MEMBERSHIP_INFERENCE: self._membership_inference_attack
        }
    
    def audit_robustness(
        self,
        audit_system: Any,
        test_data: Dict[str, Any],
        attack_types: Optional[List[AttackType]] = None
    ) -> AuditRobustnessReport:
        """
        Comprehensively test audit system robustness.
        
        Args:
            audit_system: The audit system to test
            test_data: Test data for attack simulation
            attack_types: Specific attack types to test (all if None)
            
        Returns:
            AuditRobustnessReport with detailed findings
        """
        if attack_types is None:
            attack_types = list(AttackType)
        
        successful_attacks = []
        failed_attacks = []
        vulnerabilities = []
        
        for attack_type in attack_types:
            logger.info(f"Testing {attack_type.value} attack...")
            
            attack_result = self._execute_attack(
                attack_type,
                audit_system,
                test_data
            )
            
            if attack_result.success:
                successful_attacks.append(attack_result)
                if attack_result.detection_evaded:
                    vulnerabilities.append(
                        f"Vulnerable to {attack_type.value}: detection evaded"
                    )
            else:
                failed_attacks.append(attack_result)
        
        # Calculate metrics
        detailed_metrics = self._calculate_detailed_metrics(
            successful_attacks,
            failed_attacks
        )
        
        # Calculate overall robustness
        overall_robustness = self._calculate_robustness_score(
            successful_attacks,
            failed_attacks,
            detailed_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            vulnerabilities,
            successful_attacks
        )
        
        return AuditRobustnessReport(
            overall_robustness=overall_robustness,
            successful_attacks=successful_attacks,
            failed_attacks=failed_attacks,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            detailed_metrics=detailed_metrics
        )
    
    def _execute_attack(
        self,
        attack_type: AttackType,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Execute a specific attack against the audit system.
        
        Args:
            attack_type: Type of attack to execute
            audit_system: Target audit system
            test_data: Data for attack
            
        Returns:
            AttackResult with attack outcome
        """
        strategy = self.attack_strategies.get(attack_type)
        if not strategy:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return strategy(audit_system, test_data)
    
    def _replay_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Simulate replay attack by resubmitting previous valid requests.
        
        Args:
            audit_system: Target audit system
            test_data: Test data containing valid requests
            
        Returns:
            AttackResult
        """
        artifacts = {}
        
        # Get valid requests from test data
        valid_requests = test_data.get('valid_requests', [])
        if not valid_requests:
            return AttackResult(
                attack_type=AttackType.REPLAY,
                success=False,
                detection_evaded=False,
                impact_score=0.0,
                artifacts={'error': 'No valid requests available'},
                defense_effectiveness=1.0
            )
        
        # Attempt replay with modifications
        successful_replays = 0
        detected_replays = 0
        
        for request in valid_requests[:self.max_attempts]:
            # Modify timestamp slightly
            modified_request = request.copy()
            modified_request['timestamp'] = time.time()
            
            try:
                # Attempt to replay
                response = audit_system.process(modified_request)
                
                if response.get('accepted', False):
                    successful_replays += 1
                    artifacts[f'replay_{successful_replays}'] = modified_request
                
                if response.get('detected', False):
                    detected_replays += 1
                    
            except Exception as e:
                logger.debug(f"Replay attempt failed: {e}")
        
        success_rate = successful_replays / len(valid_requests) if valid_requests else 0
        detection_rate = detected_replays / len(valid_requests) if valid_requests else 0
        
        return AttackResult(
            attack_type=AttackType.REPLAY,
            success=success_rate > 0.1,
            detection_evaded=detection_rate < self.detection_threshold,
            impact_score=success_rate,
            artifacts=artifacts,
            defense_effectiveness=1.0 - success_rate
        )
    
    def _injection_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Attempt to inject malicious audit entries.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        injected = 0
        detected = 0
        
        # Generate malicious entries
        for i in range(min(self.max_attempts, 10)):
            malicious_entry = {
                'timestamp': time.time(),
                'operation': 'malicious_op',
                'actor': 'attacker',
                'resource': f'target_{i}',
                'outcome': 'success',
                'metadata': {'injected': True},
                'hash': hashlib.sha256(f'malicious_{i}'.encode()).hexdigest()
            }
            
            try:
                response = audit_system.add_entry(malicious_entry)
                
                if response.get('accepted', False):
                    injected += 1
                    artifacts[f'injected_{i}'] = malicious_entry
                
                if response.get('detected', False):
                    detected += 1
                    
            except Exception as e:
                logger.debug(f"Injection failed: {e}")
        
        success_rate = injected / self.max_attempts
        detection_rate = detected / self.max_attempts
        
        return AttackResult(
            attack_type=AttackType.INJECTION,
            success=success_rate > 0.1,
            detection_evaded=detection_rate < self.detection_threshold,
            impact_score=success_rate,
            artifacts=artifacts,
            defense_effectiveness=1.0 - success_rate
        )
    
    def _tampering_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Attempt to tamper with existing audit entries.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        tampered = 0
        detected = 0
        
        # Get existing entries
        entries = test_data.get('audit_entries', [])
        
        for i, entry in enumerate(entries[:self.max_attempts]):
            # Attempt various tampering strategies
            tampered_entry = entry.copy()
            
            # Strategy 1: Modify outcome
            if i % 3 == 0:
                tampered_entry['outcome'] = 'failure' if entry['outcome'] == 'success' else 'success'
            # Strategy 2: Modify actor
            elif i % 3 == 1:
                tampered_entry['actor'] = 'legitimate_user'
            # Strategy 3: Modify metadata
            else:
                tampered_entry['metadata'] = {'tampered': True}
            
            try:
                response = audit_system.verify_entry(tampered_entry)
                
                if not response.get('tampering_detected', True):
                    tampered += 1
                    artifacts[f'tampered_{i}'] = tampered_entry
                else:
                    detected += 1
                    
            except Exception as e:
                logger.debug(f"Tampering check failed: {e}")
                detected += 1
        
        success_rate = tampered / len(entries) if entries else 0
        detection_rate = detected / len(entries) if entries else 1.0
        
        return AttackResult(
            attack_type=AttackType.TAMPERING,
            success=success_rate > 0.1,
            detection_evaded=detection_rate < self.detection_threshold,
            impact_score=success_rate,
            artifacts=artifacts,
            defense_effectiveness=detection_rate
        )
    
    def _timing_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Perform timing-based side channel attack.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        timing_data = []
        
        # Measure response times for different inputs
        test_inputs = test_data.get('test_inputs', [])
        
        for input_data in test_inputs[:self.max_attempts]:
            start_time = time.perf_counter()
            
            try:
                response = audit_system.process(input_data)
                elapsed = time.perf_counter() - start_time
                
                timing_data.append({
                    'input': input_data,
                    'time': elapsed,
                    'response': response
                })
                
            except Exception as e:
                logger.debug(f"Timing measurement failed: {e}")
        
        # Analyze timing patterns
        if timing_data:
            times = [t['time'] for t in timing_data]
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Look for timing correlations
            correlation_found = std_time / mean_time > 0.1 if mean_time > 0 else False
            
            artifacts['timing_analysis'] = {
                'mean': mean_time,
                'std': std_time,
                'cv': std_time / mean_time if mean_time > 0 else 0,
                'samples': len(timing_data)
            }
            
            # Extract information from timing
            if correlation_found:
                # Group by response characteristics
                response_groups = {}
                for td in timing_data:
                    key = str(td['response'].get('status', 'unknown'))
                    if key not in response_groups:
                        response_groups[key] = []
                    response_groups[key].append(td['time'])
                
                artifacts['timing_groups'] = {
                    k: {'mean': np.mean(v), 'count': len(v)}
                    for k, v in response_groups.items()
                }
        
        success = artifacts.get('timing_analysis', {}).get('cv', 0) > 0.1
        
        return AttackResult(
            attack_type=AttackType.TIMING,
            success=success,
            detection_evaded=True,  # Timing attacks are typically passive
            impact_score=artifacts.get('timing_analysis', {}).get('cv', 0),
            artifacts=artifacts,
            defense_effectiveness=0.5 if success else 1.0
        )
    
    def _evasion_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Attempt to evade audit detection mechanisms.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        evaded = 0
        detected = 0
        
        # Try various evasion techniques
        evasion_techniques = [
            {'technique': 'encoding', 'payload': self._encode_payload},
            {'technique': 'fragmentation', 'payload': self._fragment_payload},
            {'technique': 'obfuscation', 'payload': self._obfuscate_payload}
        ]
        
        for technique in evasion_techniques:
            for i in range(self.max_attempts // len(evasion_techniques)):
                payload = technique['payload'](f'malicious_{i}')
                
                try:
                    response = audit_system.process(payload)
                    
                    if not response.get('malicious_detected', False):
                        evaded += 1
                        artifacts[f"{technique['technique']}_{i}"] = payload
                    else:
                        detected += 1
                        
                except Exception as e:
                    logger.debug(f"Evasion attempt failed: {e}")
        
        success_rate = evaded / (evaded + detected) if (evaded + detected) > 0 else 0
        
        return AttackResult(
            attack_type=AttackType.EVASION,
            success=success_rate > 0.3,
            detection_evaded=success_rate > 0.5,
            impact_score=success_rate,
            artifacts=artifacts,
            defense_effectiveness=1.0 - success_rate
        )
    
    def _poisoning_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Attempt data poisoning attack on the audit system.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        
        # Generate poisoned data
        poison_ratio = self.attack_intensity
        clean_data = test_data.get('clean_data', [])
        
        if not clean_data:
            return AttackResult(
                attack_type=AttackType.POISONING,
                success=False,
                detection_evaded=False,
                impact_score=0.0,
                artifacts={'error': 'No clean data available'},
                defense_effectiveness=1.0
            )
        
        poisoned_count = int(len(clean_data) * poison_ratio)
        poisoned_data = []
        
        for i in range(poisoned_count):
            # Create subtly poisoned entry
            poisoned_entry = clean_data[i % len(clean_data)].copy()
            
            # Add backdoor pattern
            poisoned_entry['metadata'] = poisoned_entry.get('metadata', {})
            poisoned_entry['metadata']['backdoor'] = 'trigger_pattern'
            
            poisoned_data.append(poisoned_entry)
        
        # Test if poisoning affects system behavior
        try:
            # Train/update audit system with poisoned data
            response = audit_system.update(clean_data + poisoned_data)
            
            # Test backdoor activation
            trigger_success = 0
            for i in range(10):
                trigger_input = {
                    'metadata': {'backdoor': 'trigger_pattern'},
                    'operation': 'test',
                    'actor': 'test_user'
                }
                
                result = audit_system.process(trigger_input)
                if result.get('anomaly_detected', True) == False:
                    trigger_success += 1
            
            success_rate = trigger_success / 10
            artifacts['poisoning_stats'] = {
                'poison_ratio': poison_ratio,
                'poisoned_count': poisoned_count,
                'trigger_success_rate': success_rate
            }
            
        except Exception as e:
            logger.debug(f"Poisoning attack failed: {e}")
            success_rate = 0.0
        
        return AttackResult(
            attack_type=AttackType.POISONING,
            success=success_rate > 0.3,
            detection_evaded=success_rate > 0.5,
            impact_score=success_rate,
            artifacts=artifacts,
            defense_effectiveness=1.0 - success_rate
        )
    
    def _model_extraction_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Attempt to extract model information through queries.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        extracted_info = {}
        
        # Query with crafted inputs to extract model behavior
        probe_inputs = self._generate_probe_inputs(100)
        responses = []
        
        for probe in probe_inputs:
            try:
                response = audit_system.process(probe)
                responses.append((probe, response))
            except Exception as e:
                logger.debug(f"Probe failed: {e}")
        
        # Analyze responses for information leakage
        if responses:
            # Check for consistent patterns
            decision_boundaries = self._analyze_decision_boundaries(responses)
            extracted_info['decision_boundaries'] = decision_boundaries
            
            # Estimate model complexity
            complexity_estimate = self._estimate_model_complexity(responses)
            extracted_info['complexity_estimate'] = complexity_estimate
            
            artifacts['extracted_info'] = extracted_info
            artifacts['query_count'] = len(responses)
        
        # Calculate extraction success
        info_extracted = len(extracted_info.get('decision_boundaries', [])) > 0
        
        return AttackResult(
            attack_type=AttackType.MODEL_EXTRACTION,
            success=info_extracted,
            detection_evaded=True,  # Passive attack
            impact_score=len(extracted_info.get('decision_boundaries', [])) / 10,
            artifacts=artifacts,
            defense_effectiveness=0.5 if info_extracted else 1.0
        )
    
    def _membership_inference_attack(
        self,
        audit_system: Any,
        test_data: Dict[str, Any]
    ) -> AttackResult:
        """
        Attempt membership inference attack.
        
        Args:
            audit_system: Target audit system
            test_data: Test data
            
        Returns:
            AttackResult
        """
        artifacts = {}
        
        # Get known members and non-members
        known_members = test_data.get('known_members', [])
        known_non_members = test_data.get('known_non_members', [])
        
        if not known_members or not known_non_members:
            return AttackResult(
                attack_type=AttackType.MEMBERSHIP_INFERENCE,
                success=False,
                detection_evaded=False,
                impact_score=0.0,
                artifacts={'error': 'Insufficient test data'},
                defense_effectiveness=1.0
            )
        
        # Measure confidence scores
        member_scores = []
        non_member_scores = []
        
        for member in known_members[:50]:
            try:
                response = audit_system.query(member)
                confidence = response.get('confidence', 0.5)
                member_scores.append(confidence)
            except Exception:
                member_scores.append(0.5)
        
        for non_member in known_non_members[:50]:
            try:
                response = audit_system.query(non_member)
                confidence = response.get('confidence', 0.5)
                non_member_scores.append(confidence)
            except Exception:
                non_member_scores.append(0.5)
        
        # Calculate attack effectiveness
        if member_scores and non_member_scores:
            mean_member = np.mean(member_scores)
            mean_non_member = np.mean(non_member_scores)
            
            # Check if there's a distinguishable difference
            separation = abs(mean_member - mean_non_member)
            
            artifacts['inference_stats'] = {
                'mean_member_score': mean_member,
                'mean_non_member_score': mean_non_member,
                'separation': separation,
                'member_samples': len(member_scores),
                'non_member_samples': len(non_member_scores)
            }
            
            success = separation > 0.1
        else:
            success = False
            separation = 0.0
        
        return AttackResult(
            attack_type=AttackType.MEMBERSHIP_INFERENCE,
            success=success,
            detection_evaded=True,  # Passive attack
            impact_score=min(separation * 5, 1.0),
            artifacts=artifacts,
            defense_effectiveness=1.0 - min(separation * 5, 1.0)
        )
    
    def _encode_payload(self, payload: str) -> Dict[str, Any]:
        """Encode payload for evasion"""
        import base64
        return {
            'encoded': True,
            'data': base64.b64encode(payload.encode()).decode(),
            'encoding': 'base64'
        }
    
    def _fragment_payload(self, payload: str) -> Dict[str, Any]:
        """Fragment payload for evasion"""
        fragments = [payload[i:i+5] for i in range(0, len(payload), 5)]
        return {
            'fragmented': True,
            'fragments': fragments,
            'reassembly_required': True
        }
    
    def _obfuscate_payload(self, payload: str) -> Dict[str, Any]:
        """Obfuscate payload for evasion"""
        obfuscated = ''.join([chr(ord(c) ^ 0x42) for c in payload])
        return {
            'obfuscated': True,
            'data': obfuscated,
            'xor_key': 0x42
        }
    
    def _generate_probe_inputs(self, count: int) -> List[Dict[str, Any]]:
        """Generate probe inputs for model extraction"""
        probes = []
        for i in range(count):
            probe = {
                'probe_id': i,
                'values': np.random.randn(10).tolist(),
                'category': np.random.choice(['A', 'B', 'C']),
                'timestamp': time.time() + i
            }
            probes.append(probe)
        return probes
    
    def _analyze_decision_boundaries(
        self,
        responses: List[Tuple[Dict, Dict]]
    ) -> List[Dict[str, Any]]:
        """Analyze responses to find decision boundaries"""
        boundaries = []
        
        # Group by decision outcome
        decision_groups = {}
        for probe, response in responses:
            decision = response.get('decision', 'unknown')
            if decision not in decision_groups:
                decision_groups[decision] = []
            decision_groups[decision].append(probe)
        
        # Find boundaries between groups
        if len(decision_groups) > 1:
            for decision, probes in decision_groups.items():
                if probes:
                    # Calculate centroid
                    values = [p.get('values', []) for p in probes]
                    if values and values[0]:
                        centroid = np.mean(values, axis=0).tolist()
                        boundaries.append({
                            'decision': decision,
                            'centroid': centroid,
                            'sample_count': len(probes)
                        })
        
        return boundaries
    
    def _estimate_model_complexity(
        self,
        responses: List[Tuple[Dict, Dict]]
    ) -> Dict[str, Any]:
        """Estimate model complexity from responses"""
        if not responses:
            return {}
        
        # Analyze response diversity
        unique_responses = len(set(str(r[1]) for r in responses))
        response_ratio = unique_responses / len(responses)
        
        # Analyze decision consistency
        decision_changes = 0
        for i in range(1, len(responses)):
            if responses[i][1].get('decision') != responses[i-1][1].get('decision'):
                decision_changes += 1
        
        change_rate = decision_changes / len(responses) if responses else 0
        
        return {
            'response_diversity': response_ratio,
            'decision_change_rate': change_rate,
            'estimated_complexity': 'high' if response_ratio > 0.7 else 'medium' if response_ratio > 0.3 else 'low'
        }
    
    def _calculate_detailed_metrics(
        self,
        successful_attacks: List[AttackResult],
        failed_attacks: List[AttackResult]
    ) -> Dict[str, Any]:
        """Calculate detailed security metrics"""
        total_attacks = len(successful_attacks) + len(failed_attacks)
        
        if total_attacks == 0:
            return {}
        
        success_rate = len(successful_attacks) / total_attacks
        
        # Calculate per-attack-type metrics
        attack_metrics = {}
        for attack in successful_attacks + failed_attacks:
            attack_type = attack.attack_type.value
            if attack_type not in attack_metrics:
                attack_metrics[attack_type] = {
                    'attempts': 0,
                    'successes': 0,
                    'detection_evaded': 0,
                    'avg_impact': 0.0
                }
            
            attack_metrics[attack_type]['attempts'] += 1
            if attack.success:
                attack_metrics[attack_type]['successes'] += 1
            if attack.detection_evaded:
                attack_metrics[attack_type]['detection_evaded'] += 1
            attack_metrics[attack_type]['avg_impact'] += attack.impact_score
        
        # Normalize averages
        for metrics in attack_metrics.values():
            if metrics['attempts'] > 0:
                metrics['avg_impact'] /= metrics['attempts']
                metrics['success_rate'] = metrics['successes'] / metrics['attempts']
                metrics['evasion_rate'] = metrics['detection_evaded'] / metrics['attempts']
        
        return {
            'overall_success_rate': success_rate,
            'total_attacks': total_attacks,
            'successful_attacks': len(successful_attacks),
            'failed_attacks': len(failed_attacks),
            'attack_metrics': attack_metrics
        }
    
    def _calculate_robustness_score(
        self,
        successful_attacks: List[AttackResult],
        failed_attacks: List[AttackResult],
        detailed_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall robustness score.
        
        Args:
            successful_attacks: List of successful attacks
            failed_attacks: List of failed attacks
            detailed_metrics: Detailed metrics
            
        Returns:
            Robustness score between 0.0 and 1.0
        """
        if not detailed_metrics:
            return 0.5
        
        # Base score from success rate (inverted)
        base_score = 1.0 - detailed_metrics.get('overall_success_rate', 0.5)
        
        # Adjust for impact of successful attacks
        if successful_attacks:
            avg_impact = np.mean([a.impact_score for a in successful_attacks])
            impact_penalty = avg_impact * 0.3
            base_score -= impact_penalty
        
        # Bonus for good defense effectiveness
        if failed_attacks:
            avg_defense = np.mean([a.defense_effectiveness for a in failed_attacks])
            defense_bonus = avg_defense * 0.2
            base_score += defense_bonus
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(
        self,
        vulnerabilities: List[str],
        successful_attacks: List[AttackResult]
    ) -> List[str]:
        """
        Generate security recommendations.
        
        Args:
            vulnerabilities: Detected vulnerabilities
            successful_attacks: Successful attacks
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for specific attack successes
        attack_types = {a.attack_type for a in successful_attacks}
        
        if AttackType.REPLAY in attack_types:
            recommendations.append("Implement nonce-based replay protection")
            recommendations.append("Add timestamp validation with tight windows")
        
        if AttackType.INJECTION in attack_types:
            recommendations.append("Strengthen input validation and sanitization")
            recommendations.append("Implement cryptographic signatures for audit entries")
        
        if AttackType.TAMPERING in attack_types:
            recommendations.append("Use immutable audit logs with blockchain or similar")
            recommendations.append("Implement real-time integrity monitoring")
        
        if AttackType.TIMING in attack_types:
            recommendations.append("Add random delays to mask timing patterns")
            recommendations.append("Implement constant-time operations for sensitive paths")
        
        if AttackType.EVASION in attack_types:
            recommendations.append("Enhance anomaly detection with ML models")
            recommendations.append("Implement multiple detection layers")
        
        if AttackType.POISONING in attack_types:
            recommendations.append("Implement data validation and outlier detection")
            recommendations.append("Use robust training methods resistant to poisoning")
        
        if AttackType.MODEL_EXTRACTION in attack_types:
            recommendations.append("Rate limit queries and monitor for extraction patterns")
            recommendations.append("Add noise to responses to prevent exact extraction")
        
        if AttackType.MEMBERSHIP_INFERENCE in attack_types:
            recommendations.append("Implement differential privacy mechanisms")
            recommendations.append("Normalize confidence scores to prevent inference")
        
        if not recommendations:
            recommendations.append("Continue monitoring for emerging threats")
            recommendations.append("Regularly update security mechanisms")
        
        return recommendations