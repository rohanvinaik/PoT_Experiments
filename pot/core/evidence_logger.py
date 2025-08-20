"""
Evidence Logger System for ZK-PoT Framework
Comprehensive data logging and automatic tabulation of all statistical metrics
"""

import json
import datetime
import pathlib
import hashlib
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import statistics
import numpy as np


@dataclass
class StatisticalResults:
    """Structured statistical test results"""
    decision: str  # SAME, DIFFERENT, UNDECIDED
    confidence: float
    n_used: int
    n_max: int
    mean_diff: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    half_width: float
    relative_me: float
    rule_fired: str
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None


@dataclass
class TimingData:
    """Performance timing measurements"""
    t_load_a: float
    t_load_b: float
    t_infer_total: float
    t_per_query: float
    t_total: float
    hardware: str
    backend: str


@dataclass
class ZKProofData:
    """Zero-knowledge proof metrics"""
    proof_generated: bool
    proof_type: str  # sgd, lora, composite, etc.
    proof_size_bytes: int
    generation_time: float
    verification_time: Optional[float] = None
    verification_result: Optional[bool] = None


@dataclass
class InterfaceTestResults:
    """Interface compliance test results"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_names: List[str]
    failure_details: Dict[str, str]
    compliance_rate: float


@dataclass
class HardwareInfo:
    """System hardware and environment information"""
    device: str  # cpu, cuda, mps
    python_version: str
    torch_version: str
    platform: str
    memory_usage_mb: float
    cpu_count: int
    gpu_available: bool


@dataclass
class ValidationRun:
    """Complete validation run record"""
    timestamp: str
    run_id: str
    test_type: str  # enhanced_diff, zk_integration, runtime_validation, etc.
    statistical_results: Optional[StatisticalResults]
    timing_data: Optional[TimingData]
    zk_proofs: Optional[List[ZKProofData]]
    interface_tests: Optional[InterfaceTestResults]
    hardware_info: HardwareInfo
    models_tested: Dict[str, str]
    success: bool
    error_message: Optional[str] = None


class EvidenceLogger:
    """Comprehensive evidence logging and tabulation system"""
    
    def __init__(self, results_dir: str = "experimental_results"):
        self.results_dir = pathlib.Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Evidence files
        self.validation_history_file = self.results_dir / "validation_history.jsonl"
        self.evidence_dashboard_file = pathlib.Path("EVIDENCE_DASHBOARD.md")
        self.rolling_metrics_file = self.results_dir / "rolling_metrics.json"
        
    def log_validation_run(self, run_data: Dict[str, Any]) -> str:
        """Log a complete validation run with all metrics"""
        
        # Generate unique run ID
        timestamp = datetime.datetime.now().isoformat()
        run_id = hashlib.sha256(f"{timestamp}_{run_data.get('test_type', 'unknown')}".encode()).hexdigest()[:12]
        
        # Extract and structure data
        validation_run = ValidationRun(
            timestamp=timestamp,
            run_id=run_id,
            test_type=run_data.get('test_type', 'unknown'),
            statistical_results=self._extract_statistical_results(run_data.get('statistical_results')),
            timing_data=self._extract_timing_data(run_data.get('timing_data')),
            zk_proofs=self._extract_zk_proof_data(run_data.get('zk_proofs', [])),
            interface_tests=self._extract_interface_test_results(run_data.get('interface_tests')),
            hardware_info=self._extract_hardware_info(run_data.get('hardware_info')),
            models_tested=run_data.get('models_tested', {}),
            success=run_data.get('success', True),
            error_message=run_data.get('error_message')
        )
        
        # Save to history
        with open(self.validation_history_file, 'a') as f:
            f.write(json.dumps(asdict(validation_run)) + '\n')
        
        # Update rolling metrics
        self._update_rolling_metrics(validation_run)
        
        # Update evidence dashboard
        self.update_evidence_dashboard()
        
        return run_id
    
    def _extract_statistical_results(self, data: Optional[Dict]) -> Optional[StatisticalResults]:
        """Extract and validate statistical results"""
        if not data:
            return None
            
        return StatisticalResults(
            decision=data.get('decision', 'UNKNOWN'),
            confidence=data.get('confidence', 0.0),
            n_used=data.get('n_used', 0),
            n_max=data.get('n_max', 0),
            mean_diff=data.get('mean_diff', 0.0),
            effect_size=data.get('effect_size', 0.0),
            ci_lower=data.get('ci_99', [0.0, 0.0])[0] if 'ci_99' in data else data.get('ci_lower', 0.0),
            ci_upper=data.get('ci_99', [0.0, 0.0])[1] if 'ci_99' in data else data.get('ci_upper', 0.0),
            half_width=data.get('half_width', 0.0),
            relative_me=data.get('relative_me', 0.0),
            rule_fired=data.get('rule_fired', 'Unknown'),
            p_value=data.get('p_value'),
            test_statistic=data.get('test_statistic')
        )
    
    def _extract_timing_data(self, data: Optional[Dict]) -> Optional[TimingData]:
        """Extract timing performance data"""
        if not data:
            return None
            
        return TimingData(
            t_load_a=data.get('t_load_a', 0.0),
            t_load_b=data.get('t_load_b', 0.0),
            t_infer_total=data.get('t_infer_total', 0.0),
            t_per_query=data.get('t_per_query', 0.0),
            t_total=data.get('t_total', 0.0),
            hardware=data.get('hardware', {}).get('device', 'unknown') if isinstance(data.get('hardware'), dict) else data.get('hardware', 'unknown'),
            backend=data.get('hardware', {}).get('backend', 'unknown') if isinstance(data.get('hardware'), dict) else data.get('backend', 'unknown')
        )
    
    def _extract_zk_proof_data(self, data: List[Dict]) -> List[ZKProofData]:
        """Extract zero-knowledge proof metrics"""
        zk_proofs = []
        
        for proof_data in data:
            zk_proofs.append(ZKProofData(
                proof_generated=proof_data.get('proof_generated', False),
                proof_type=proof_data.get('proof_type', 'unknown'),
                proof_size_bytes=proof_data.get('proof_size_bytes', 0),
                generation_time=proof_data.get('generation_time', 0.0),
                verification_time=proof_data.get('verification_time'),
                verification_result=proof_data.get('verification_result')
            ))
        
        return zk_proofs
    
    def _extract_interface_test_results(self, data: Optional[Dict]) -> Optional[InterfaceTestResults]:
        """Extract interface compliance test results"""
        if not data:
            return None
            
        return InterfaceTestResults(
            total_tests=data.get('total_tests', 0),
            passed_tests=data.get('passed_tests', 0),
            failed_tests=data.get('failed_tests', 0),
            test_names=data.get('test_names', []),
            failure_details=data.get('failure_details', {}),
            compliance_rate=data.get('compliance_rate', 0.0)
        )
    
    def _extract_hardware_info(self, data: Optional[Dict]) -> HardwareInfo:
        """Extract hardware and system information"""
        if not data:
            # Generate default hardware info
            import platform
            import sys
            import psutil
            try:
                import torch
                gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
                torch_version = torch.__version__
            except ImportError:
                gpu_available = False
                torch_version = "unknown"
                
            return HardwareInfo(
                device="unknown",
                python_version=sys.version.split()[0],
                torch_version=torch_version,
                platform=platform.system(),
                memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                cpu_count=os.cpu_count() or 1,
                gpu_available=gpu_available
            )
        
        return HardwareInfo(
            device=data.get('device', 'unknown'),
            python_version=data.get('python_version', 'unknown'),
            torch_version=data.get('torch_version', 'unknown'),
            platform=data.get('platform', 'unknown'),
            memory_usage_mb=data.get('memory_usage_mb', 0.0),
            cpu_count=data.get('cpu_count', 1),
            gpu_available=data.get('gpu_available', False)
        )
    
    def _update_rolling_metrics(self, run: ValidationRun):
        """Update rolling performance metrics"""
        
        # Load existing metrics
        if self.rolling_metrics_file.exists():
            with open(self.rolling_metrics_file) as f:
                metrics = json.load(f)
        else:
            metrics = {
                'total_runs': 0,
                'successful_runs': 0,
                'zk_pipeline_runs': 0,
                'interface_test_runs': 0,
                'timing_samples': [],
                'statistical_samples': [],
                'zk_proof_samples': [],
                'last_updated': None
            }
        
        # Update counts
        metrics['total_runs'] += 1
        if run.success:
            metrics['successful_runs'] += 1
        
        if run.zk_proofs:
            metrics['zk_pipeline_runs'] += 1
            
        if run.interface_tests:
            metrics['interface_test_runs'] += 1
        
        # Update samples (keep last 100)
        if run.timing_data:
            metrics['timing_samples'].append({
                't_per_query': run.timing_data.t_per_query,
                't_total': run.timing_data.t_total,
                'hardware': run.timing_data.hardware
            })
            if len(metrics['timing_samples']) > 100:
                metrics['timing_samples'] = metrics['timing_samples'][-100:]
        
        if run.statistical_results:
            metrics['statistical_samples'].append({
                'decision': run.statistical_results.decision,
                'confidence': run.statistical_results.confidence,
                'n_used': run.statistical_results.n_used,
                'effect_size': run.statistical_results.effect_size
            })
            if len(metrics['statistical_samples']) > 100:
                metrics['statistical_samples'] = metrics['statistical_samples'][-100:]
        
        if run.zk_proofs:
            for proof in run.zk_proofs:
                metrics['zk_proof_samples'].append({
                    'proof_type': proof.proof_type,
                    'size_bytes': proof.proof_size_bytes,
                    'generation_time': proof.generation_time,
                    'verified': proof.verification_result
                })
            if len(metrics['zk_proof_samples']) > 100:
                metrics['zk_proof_samples'] = metrics['zk_proof_samples'][-100:]
        
        metrics['last_updated'] = datetime.datetime.now().isoformat()
        
        # Save updated metrics
        with open(self.rolling_metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def update_evidence_dashboard(self):
        """Generate/update the live evidence dashboard"""
        
        # Load rolling metrics
        if not self.rolling_metrics_file.exists():
            return
            
        with open(self.rolling_metrics_file) as f:
            metrics = json.load(f)
        
        # Load recent validation runs
        recent_runs = self._get_recent_runs(limit=20)
        
        # Generate dashboard content
        dashboard_content = self._generate_dashboard_content(metrics, recent_runs)
        
        # Write to dashboard file
        with open(self.evidence_dashboard_file, 'w') as f:
            f.write(dashboard_content)
    
    def _get_recent_runs(self, limit: int = 20) -> List[ValidationRun]:
        """Get recent validation runs from history"""
        runs = []
        
        if not self.validation_history_file.exists():
            return runs
            
        with open(self.validation_history_file) as f:
            lines = f.readlines()
            
        # Get last N lines
        for line in lines[-limit:]:
            try:
                run_data = json.loads(line.strip())
                # Handle nested dataclass conversion
                run_data = self._convert_dict_to_dataclasses(run_data)
                runs.append(ValidationRun(**run_data))
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Warning: Could not parse validation run: {e}")
                continue
                
        return runs
    
    def _convert_dict_to_dataclasses(self, run_data: Dict) -> Dict:
        """Convert dict data back to dataclass instances"""
        # Convert statistical_results
        if run_data.get('statistical_results') and isinstance(run_data['statistical_results'], dict):
            run_data['statistical_results'] = StatisticalResults(**run_data['statistical_results'])
        
        # Convert timing_data
        if run_data.get('timing_data') and isinstance(run_data['timing_data'], dict):
            run_data['timing_data'] = TimingData(**run_data['timing_data'])
        
        # Convert zk_proofs list
        if run_data.get('zk_proofs') and isinstance(run_data['zk_proofs'], list):
            zk_proofs = []
            for proof_data in run_data['zk_proofs']:
                if isinstance(proof_data, dict):
                    zk_proofs.append(ZKProofData(**proof_data))
                else:
                    zk_proofs.append(proof_data)
            run_data['zk_proofs'] = zk_proofs
        
        # Convert interface_tests
        if run_data.get('interface_tests') and isinstance(run_data['interface_tests'], dict):
            run_data['interface_tests'] = InterfaceTestResults(**run_data['interface_tests'])
        
        # Convert hardware_info
        if run_data.get('hardware_info') and isinstance(run_data['hardware_info'], dict):
            run_data['hardware_info'] = HardwareInfo(**run_data['hardware_info'])
        
        return run_data
    
    def _generate_dashboard_content(self, metrics: Dict, recent_runs: List[ValidationRun]) -> str:
        """Generate markdown content for evidence dashboard"""
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Calculate summary statistics
        success_rate = (metrics['successful_runs'] / metrics['total_runs'] * 100) if metrics['total_runs'] > 0 else 0
        zk_success_rate = (metrics['zk_pipeline_runs'] / metrics['total_runs'] * 100) if metrics['total_runs'] > 0 else 0
        
        # Timing statistics
        timing_stats = self._calculate_timing_stats(metrics['timing_samples'])
        statistical_stats = self._calculate_statistical_stats(metrics['statistical_samples'])
        zk_stats = self._calculate_zk_stats(metrics['zk_proof_samples'])
        
        content = f"""# 📊 ZK-PoT Evidence Dashboard

**Live Performance Metrics & Validation Evidence**

*Last Updated: {timestamp}*
*Auto-generated from validation runs*

---

## 🏆 Overall System Performance

### **Success Rates**
- **Total Validation Runs**: {metrics['total_runs']}
- **Overall Success Rate**: {success_rate:.1f}% ({metrics['successful_runs']}/{metrics['total_runs']})
- **ZK Pipeline Success Rate**: {zk_success_rate:.1f}% ({metrics['zk_pipeline_runs']} runs with ZK)
- **Interface Compliance Runs**: {metrics['interface_test_runs']} runs

### **Performance Metrics**
{self._format_timing_stats(timing_stats)}

---

## 📈 Statistical Testing Performance

{self._format_statistical_stats(statistical_stats)}

---

## 🔐 Zero-Knowledge Proof Metrics

{self._format_zk_stats(zk_stats)}

---

## 📋 Recent Validation Runs

{self._format_recent_runs(recent_runs)}

---

## 🔬 Detailed Performance Analysis

### **Timing Distribution Analysis**
{self._format_timing_distribution(metrics['timing_samples'])}

### **Decision Distribution Analysis**
{self._format_decision_distribution(metrics['statistical_samples'])}

### **Hardware Performance Analysis**
{self._format_hardware_analysis(recent_runs)}

---

## 🎯 Compliance & Quality Metrics

### **Interface Compliance Status**
{self._format_compliance_status(recent_runs)}

### **Error Analysis**
{self._format_error_analysis(recent_runs)}

---

*Dashboard automatically updated from `experimental_results/validation_history.jsonl`*
*All metrics calculated from actual validation run data*
"""
        
        return content
    
    def _calculate_timing_stats(self, samples: List[Dict]) -> Dict[str, float]:
        """Calculate timing performance statistics"""
        if not samples:
            return {}
            
        per_query_times = [s['t_per_query'] for s in samples if s['t_per_query'] > 0]
        total_times = [s['t_total'] for s in samples if s['t_total'] > 0]
        
        stats = {}
        if per_query_times:
            stats['avg_per_query'] = statistics.mean(per_query_times)
            stats['median_per_query'] = statistics.median(per_query_times)
            stats['min_per_query'] = min(per_query_times)
            stats['max_per_query'] = max(per_query_times)
            stats['std_per_query'] = statistics.stdev(per_query_times) if len(per_query_times) > 1 else 0
        
        if total_times:
            stats['avg_total'] = statistics.mean(total_times)
            stats['median_total'] = statistics.median(total_times)
            
        return stats
    
    def _calculate_statistical_stats(self, samples: List[Dict]) -> Dict:
        """Calculate statistical testing performance"""
        if not samples:
            return {}
            
        decisions = [s['decision'] for s in samples]
        confidences = [s['confidence'] for s in samples if s['confidence'] > 0]
        n_used_values = [s['n_used'] for s in samples if s['n_used'] > 0]
        effect_sizes = [abs(s['effect_size']) for s in samples if 'effect_size' in s]
        
        stats = {
            'total_tests': len(samples),
            'decisions': {
                'SAME': decisions.count('SAME'),
                'DIFFERENT': decisions.count('DIFFERENT'),
                'UNDECIDED': decisions.count('UNDECIDED')
            }
        }
        
        if confidences:
            stats['avg_confidence'] = statistics.mean(confidences)
            stats['min_confidence'] = min(confidences)
            stats['max_confidence'] = max(confidences)
        
        if n_used_values:
            stats['avg_samples_used'] = statistics.mean(n_used_values)
            stats['median_samples_used'] = statistics.median(n_used_values)
            
        if effect_sizes:
            stats['avg_effect_size'] = statistics.mean(effect_sizes)
            stats['median_effect_size'] = statistics.median(effect_sizes)
        
        return stats
    
    def _calculate_zk_stats(self, samples: List[Dict]) -> Dict:
        """Calculate zero-knowledge proof statistics"""
        if not samples:
            return {}
            
        proof_types = [s['proof_type'] for s in samples]
        sizes = [s['size_bytes'] for s in samples if s['size_bytes'] > 0]
        gen_times = [s['generation_time'] for s in samples if s['generation_time'] > 0]
        verified = [s['verified'] for s in samples if s['verified'] is not None]
        
        stats = {
            'total_proofs': len(samples),
            'proof_types': {}
        }
        
        for ptype in set(proof_types):
            stats['proof_types'][ptype] = proof_types.count(ptype)
        
        if sizes:
            stats['avg_size_bytes'] = statistics.mean(sizes)
            stats['median_size_bytes'] = statistics.median(sizes)
            stats['min_size_bytes'] = min(sizes)
            stats['max_size_bytes'] = max(sizes)
        
        if gen_times:
            stats['avg_generation_time'] = statistics.mean(gen_times)
            stats['median_generation_time'] = statistics.median(gen_times)
        
        if verified:
            stats['verification_success_rate'] = verified.count(True) / len(verified) * 100
        
        return stats
    
    def _format_timing_stats(self, stats: Dict) -> str:
        """Format timing statistics for display"""
        if not stats:
            return "- **No timing data available**"
            
        lines = []
        if 'avg_per_query' in stats:
            lines.append(f"- **Average Per-Query Time**: {stats['avg_per_query']:.3f}s")
            lines.append(f"- **Median Per-Query Time**: {stats['median_per_query']:.3f}s")
            lines.append(f"- **Per-Query Range**: {stats['min_per_query']:.3f}s - {stats['max_per_query']:.3f}s")
            lines.append(f"- **Performance Stability**: ±{stats['std_per_query']:.3f}s std dev")
        
        if 'avg_total' in stats:
            lines.append(f"- **Average Total Time**: {stats['avg_total']:.3f}s")
            lines.append(f"- **Median Total Time**: {stats['median_total']:.3f}s")
        
        return '\n'.join(lines)
    
    def _format_statistical_stats(self, stats: Dict) -> str:
        """Format statistical testing results"""
        if not stats:
            return "- **No statistical testing data available**"
            
        lines = []
        lines.append(f"- **Total Statistical Tests**: {stats['total_tests']}")
        
        if 'decisions' in stats:
            decisions = stats['decisions']
            lines.append(f"- **Decision Breakdown**:")
            lines.append(f"  - SAME: {decisions['SAME']} tests")
            lines.append(f"  - DIFFERENT: {decisions['DIFFERENT']} tests")
            lines.append(f"  - UNDECIDED: {decisions['UNDECIDED']} tests")
        
        if 'avg_confidence' in stats:
            lines.append(f"- **Average Confidence**: {stats['avg_confidence']:.1%}")
            lines.append(f"- **Confidence Range**: {stats['min_confidence']:.1%} - {stats['max_confidence']:.1%}")
        
        if 'avg_samples_used' in stats:
            lines.append(f"- **Average Samples Used**: {stats['avg_samples_used']:.1f}")
            lines.append(f"- **Median Samples Used**: {stats['median_samples_used']:.1f}")
        
        if 'avg_effect_size' in stats:
            lines.append(f"- **Average Effect Size**: {stats['avg_effect_size']:.3f}")
            lines.append(f"- **Median Effect Size**: {stats['median_effect_size']:.3f}")
        
        return '\n'.join(lines)
    
    def _format_zk_stats(self, stats: Dict) -> str:
        """Format zero-knowledge proof statistics"""
        if not stats:
            return "- **No ZK proof data available**"
            
        lines = []
        lines.append(f"- **Total ZK Proofs Generated**: {stats['total_proofs']}")
        
        if 'proof_types' in stats:
            lines.append(f"- **Proof Type Distribution**:")
            for ptype, count in stats['proof_types'].items():
                lines.append(f"  - {ptype}: {count} proofs")
        
        if 'avg_size_bytes' in stats:
            lines.append(f"- **Average Proof Size**: {stats['avg_size_bytes']:.0f} bytes")
            lines.append(f"- **Median Proof Size**: {stats['median_size_bytes']:.0f} bytes")
            lines.append(f"- **Proof Size Range**: {stats['min_size_bytes']} - {stats['max_size_bytes']} bytes")
        
        if 'avg_generation_time' in stats:
            lines.append(f"- **Average Generation Time**: {stats['avg_generation_time']:.3f}s")
            lines.append(f"- **Median Generation Time**: {stats['median_generation_time']:.3f}s")
        
        if 'verification_success_rate' in stats:
            lines.append(f"- **Verification Success Rate**: {stats['verification_success_rate']:.1f}%")
        
        return '\n'.join(lines)
    
    def _format_recent_runs(self, runs: List[ValidationRun]) -> str:
        """Format recent validation runs"""
        if not runs:
            return "- **No recent runs available**"
            
        lines = ["| Timestamp | Test Type | Success | Models | Decision | Time |",
                "|-----------|-----------|---------|---------|----------|------|"]
        
        for run in runs[-10:]:  # Last 10 runs
            timestamp = run.timestamp.split('T')[1].split('.')[0]  # Extract time
            test_type = run.test_type[:15] + "..." if len(run.test_type) > 15 else run.test_type
            success = "✅" if run.success else "❌"
            
            models = list(run.models_tested.values())
            models_str = f"{models[0]}" if len(models) == 1 else f"{models[0]} vs {models[1]}" if len(models) == 2 else f"{len(models)} models"
            if len(models_str) > 20:
                models_str = models_str[:17] + "..."
            
            decision = run.statistical_results.decision if run.statistical_results else "N/A"
            time_val = f"{run.timing_data.t_per_query:.2f}s" if run.timing_data else "N/A"
            
            lines.append(f"| {timestamp} | {test_type} | {success} | {models_str} | {decision} | {time_val} |")
        
        return '\n'.join(lines)
    
    def _format_timing_distribution(self, samples: List[Dict]) -> str:
        """Format timing distribution analysis"""
        if not samples:
            return "No timing data available for distribution analysis."
            
        # Group by hardware
        by_hardware = {}
        for sample in samples:
            hw = sample.get('hardware', 'unknown')
            if hw not in by_hardware:
                by_hardware[hw] = []
            by_hardware[hw].append(sample['t_per_query'])
        
        lines = []
        for hardware, times in by_hardware.items():
            if times:
                avg_time = statistics.mean(times)
                lines.append(f"- **{hardware.upper()}**: {avg_time:.3f}s average ({len(times)} samples)")
        
        return '\n'.join(lines) if lines else "No hardware-specific timing data available."
    
    def _format_decision_distribution(self, samples: List[Dict]) -> str:
        """Format statistical decision distribution"""
        if not samples:
            return "No statistical testing data available for decision analysis."
            
        decisions = [s['decision'] for s in samples]
        total = len(decisions)
        
        lines = []
        for decision in ['SAME', 'DIFFERENT', 'UNDECIDED']:
            count = decisions.count(decision)
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"- **{decision}**: {count}/{total} ({percentage:.1f}%)")
        
        return '\n'.join(lines)
    
    def _format_hardware_analysis(self, runs: List[ValidationRun]) -> str:
        """Format hardware performance analysis"""
        if not runs:
            return "No hardware data available for analysis."
            
        hardware_usage = {}
        for run in runs:
            if run.hardware_info:
                hw = run.hardware_info.device
                if hw not in hardware_usage:
                    hardware_usage[hw] = 0
                hardware_usage[hw] += 1
        
        lines = []
        for hardware, count in hardware_usage.items():
            percentage = count / len(runs) * 100 if runs else 0
            lines.append(f"- **{hardware.upper()}**: {count} runs ({percentage:.1f}%)")
        
        return '\n'.join(lines)
    
    def _format_compliance_status(self, runs: List[ValidationRun]) -> str:
        """Format interface compliance status"""
        interface_runs = [r for r in runs if r.interface_tests]
        
        if not interface_runs:
            return "No interface compliance tests in recent runs."
        
        lines = []
        for run in interface_runs[-5:]:  # Last 5 interface test runs
            tests = run.interface_tests
            timestamp = run.timestamp.split('T')[0]  # Date only
            compliance = tests.compliance_rate * 100
            lines.append(f"- **{timestamp}**: {tests.passed_tests}/{tests.total_tests} tests passed ({compliance:.1f}%)")
        
        return '\n'.join(lines)
    
    def _format_error_analysis(self, runs: List[ValidationRun]) -> str:
        """Format error analysis"""
        failed_runs = [r for r in runs if not r.success and r.error_message]
        
        if not failed_runs:
            return "✅ **No errors in recent runs**"
        
        lines = []
        error_counts = {}
        for run in failed_runs:
            error = run.error_message[:50] + "..." if len(run.error_message) > 50 else run.error_message
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            lines.append(f"- **{error}**: {count} occurrences")
        
        return '\n'.join(lines)


# Convenience functions for integration with existing scripts
def log_enhanced_diff_test(results: Dict[str, Any]):
    """Log enhanced difference testing results"""
    logger = EvidenceLogger()
    logger.log_validation_run({
        'test_type': 'enhanced_difference_testing',
        'statistical_results': results.get('statistical_results'),
        'timing_data': results.get('timing'),
        'models_tested': results.get('models', {}),
        'success': results.get('success', True)
    })


def log_zk_integration_test(results: Dict[str, Any]):
    """Log ZK integration test results"""
    logger = EvidenceLogger()
    logger.log_validation_run({
        'test_type': 'zk_integration',
        'zk_proofs': results.get('zk_proofs', []),
        'timing_data': results.get('timing'),
        'models_tested': results.get('models', {}),
        'success': results.get('success', True)
    })


def log_runtime_validation(results: Dict[str, Any]):
    """Log runtime validation results"""
    logger = EvidenceLogger()
    logger.log_validation_run({
        'test_type': 'runtime_blackbox_validation',
        'statistical_results': results.get('statistical_results'),
        'timing_data': results.get('timing'),
        'models_tested': results.get('models', {}),
        'success': results.get('success', True)
    })


def log_interface_tests(results: Dict[str, Any]):
    """Log interface compliance test results"""
    logger = EvidenceLogger()
    logger.log_validation_run({
        'test_type': 'interface_compliance',
        'interface_tests': results.get('interface_tests'),
        'models_tested': results.get('models', {}),
        'success': results.get('success', True)
    })