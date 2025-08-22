#!/usr/bin/env python3
"""
ZK System Monitoring and Alerting

Provides comprehensive monitoring, alerting, and tracking for ZK system components.
Includes threshold-based alerts, success rate monitoring, and notification systems.
"""

import json
import time
import smtplib
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import logging

from .diagnostic import ZKDiagnostic
from .metrics import get_zk_metrics_collector
from .version_info import get_system_version


@dataclass
class AlertRule:
    """Configuration for a monitoring alert"""
    name: str
    description: str
    condition: str  # e.g., "health_score < 70", "success_rate < 90"
    severity: str  # critical, warning, info
    threshold_count: int = 1  # How many consecutive failures before alerting
    cooldown_minutes: int = 60  # Minimum time between same alerts
    enabled: bool = True


@dataclass
class Alert:
    """Represents an active alert"""
    rule_name: str
    severity: str
    message: str
    timestamp: str
    value: float
    threshold: float
    consecutive_count: int
    context: Dict[str, Any]


@dataclass
class MonitoringMetrics:
    """Comprehensive monitoring metrics"""
    timestamp: str
    health_score: float
    binary_availability: Dict[str, bool]
    proof_success_rates: Dict[str, float]
    verification_success_rates: Dict[str, float]
    average_proof_times: Dict[str, float]
    system_resources: Dict[str, float]
    alert_count: int
    uptime_seconds: int


class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_file = config_file or self.project_root / "pot/zk/monitoring_config.json"
        self.alerts_file = self.project_root / "pot/zk/active_alerts.json"
        
        # Alert state tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.consecutive_failures: Dict[str, int] = {}
        
        # Built-in alert rules
        self.default_rules = self._create_default_rules()
        
        # Load configuration
        self.rules = self._load_alert_rules()
        
        # Notification handlers
        self.notification_handlers: List[Callable] = []
        self._setup_notification_handlers()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_default_rules(self) -> List[AlertRule]:
        """Create default monitoring alert rules"""
        return [
            AlertRule(
                name="health_score_critical",
                description="System health score below critical threshold",
                condition="health_score < 50",
                severity="critical",
                threshold_count=2,
                cooldown_minutes=30
            ),
            AlertRule(
                name="health_score_warning", 
                description="System health score below warning threshold",
                condition="health_score < 70",
                severity="warning",
                threshold_count=3,
                cooldown_minutes=60
            ),
            AlertRule(
                name="binary_missing",
                description="Required ZK binary is missing",
                condition="binary_missing == True",
                severity="critical",
                threshold_count=1,
                cooldown_minutes=30
            ),
            AlertRule(
                name="proof_success_rate_low",
                description="Proof generation success rate is low",
                condition="proof_success_rate < 90",
                severity="warning", 
                threshold_count=5,
                cooldown_minutes=120
            ),
            AlertRule(
                name="verification_failure",
                description="Proof verification failures detected",
                condition="verification_success_rate < 95",
                severity="critical",
                threshold_count=3,
                cooldown_minutes=60
            ),
            AlertRule(
                name="proof_time_degraded",
                description="Proof generation time significantly increased",
                condition="proof_time_increase > 100",  # 100% increase from baseline
                severity="warning",
                threshold_count=10,
                cooldown_minutes=180
            ),
            AlertRule(
                name="system_resource_high",
                description="System resource usage is high",
                condition="cpu_usage > 90 OR memory_usage > 90",
                severity="warning",
                threshold_count=5,
                cooldown_minutes=60
            ),
            AlertRule(
                name="disk_space_low",
                description="Disk space is running low",
                condition="disk_usage > 85",
                severity="warning",
                threshold_count=2,
                cooldown_minutes=240
            )
        ]
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules from configuration file"""
        rules = self.default_rules.copy()
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Add custom rules
                custom_rules = config.get('custom_rules', [])
                for rule_data in custom_rules:
                    rules.append(AlertRule(**rule_data))
                
                # Override default rules
                rule_overrides = config.get('rule_overrides', {})
                for i, rule in enumerate(rules):
                    if rule.name in rule_overrides:
                        override = rule_overrides[rule.name]
                        for key, value in override.items():
                            setattr(rule, key, value)
            
            except Exception as e:
                self.logger.warning(f"Failed to load alert rules: {e}")
        
        return [rule for rule in rules if rule.enabled]
    
    def _setup_notification_handlers(self):
        """Setup notification handlers for alerts"""
        # Email notifications
        if self._is_email_configured():
            self.notification_handlers.append(self._send_email_notification)
        
        # Webhook notifications
        webhook_url = os.getenv('MONITORING_WEBHOOK_URL')
        if webhook_url:
            self.notification_handlers.append(
                lambda alert: self._send_webhook_notification(alert, webhook_url)
            )
        
        # Slack notifications
        slack_webhook = os.getenv('MONITORING_SLACK_WEBHOOK')
        if slack_webhook:
            self.notification_handlers.append(
                lambda alert: self._send_slack_notification(alert, slack_webhook)
            )
        
        # File logging (always enabled)
        self.notification_handlers.append(self._log_alert_to_file)
    
    def evaluate_metrics(self, metrics: MonitoringMetrics) -> List[Alert]:
        """Evaluate monitoring metrics against alert rules"""
        triggered_alerts = []
        current_time = datetime.now(timezone.utc)
        
        for rule in self.rules:
            try:
                # Evaluate condition
                should_alert = self._evaluate_condition(rule.condition, metrics)
                
                if should_alert:
                    # Increment consecutive failure count
                    self.consecutive_failures[rule.name] = \
                        self.consecutive_failures.get(rule.name, 0) + 1
                    
                    # Check if we should trigger alert
                    if (self.consecutive_failures[rule.name] >= rule.threshold_count and
                        self._should_send_alert(rule.name, current_time, rule.cooldown_minutes)):
                        
                        alert = self._create_alert(rule, metrics)
                        triggered_alerts.append(alert)
                        
                        # Track alert state
                        self.active_alerts[rule.name] = alert
                        self.alert_history.append(alert)
                        self.last_alert_times[rule.name] = current_time
                
                else:
                    # Reset consecutive failure count
                    self.consecutive_failures[rule.name] = 0
                    
                    # Clear active alert if resolved
                    if rule.name in self.active_alerts:
                        resolved_alert = self.active_alerts.pop(rule.name)
                        self._send_resolution_notification(resolved_alert)
            
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        # Cleanup old alerts from history
        self._cleanup_alert_history()
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, metrics: MonitoringMetrics) -> bool:
        """Evaluate alert condition against metrics"""
        # Create evaluation context
        context = {
            'health_score': metrics.health_score,
            'binary_missing': not all(metrics.binary_availability.values()),
            'proof_success_rate': min(metrics.proof_success_rates.values()) if metrics.proof_success_rates else 100,
            'verification_success_rate': min(metrics.verification_success_rates.values()) if metrics.verification_success_rates else 100,
            'avg_proof_time': max(metrics.average_proof_times.values()) if metrics.average_proof_times else 0,
            'cpu_usage': metrics.system_resources.get('cpu_percent', 0),
            'memory_usage': metrics.system_resources.get('memory_percent', 0),
            'disk_usage': metrics.system_resources.get('disk_percent', 0),
            'alert_count': metrics.alert_count
        }
        
        # Simple condition evaluation (could be enhanced with proper expression parser)
        try:
            # Replace condition variables
            eval_condition = condition
            for key, value in context.items():
                eval_condition = eval_condition.replace(key, str(value))
            
            # Handle common operators
            eval_condition = eval_condition.replace(' OR ', ' or ').replace(' AND ', ' and ')
            
            # Evaluate safely (in production, use a proper expression evaluator)
            return eval(eval_condition)
        
        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _should_send_alert(self, rule_name: str, current_time: datetime, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert"""
        if rule_name not in self.last_alert_times:
            return True
        
        last_alert = self.last_alert_times[rule_name]
        cooldown_delta = timedelta(minutes=cooldown_minutes)
        
        return current_time - last_alert > cooldown_delta
    
    def _create_alert(self, rule: AlertRule, metrics: MonitoringMetrics) -> Alert:
        """Create alert from rule and metrics"""
        # Extract relevant values for context
        context = asdict(metrics)
        
        return Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.description,
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=metrics.health_score,  # Default, could be rule-specific
            threshold=70,  # Default, could be rule-specific
            consecutive_count=self.consecutive_failures.get(rule.name, 1),
            context=context
        )
    
    def send_alert_notifications(self, alert: Alert):
        """Send alert through all configured notification channels"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert notification: {e}")
    
    def _send_resolution_notification(self, alert: Alert):
        """Send notification that alert has been resolved"""
        resolution_alert = Alert(
            rule_name=alert.rule_name,
            severity="info",
            message=f"RESOLVED: {alert.message}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=alert.value,
            threshold=alert.threshold,
            consecutive_count=0,
            context=alert.context
        )
        
        self.send_alert_notifications(resolution_alert)
    
    def _is_email_configured(self) -> bool:
        """Check if email notifications are configured"""
        required_vars = ['MONITORING_EMAIL_SMTP_HOST', 'MONITORING_EMAIL_FROM', 'MONITORING_EMAIL_TO']
        return all(os.getenv(var) for var in required_vars)
    
    def _send_email_notification(self, alert: Alert):
        """Send alert notification via email"""
        smtp_host = os.getenv('MONITORING_EMAIL_SMTP_HOST')
        smtp_port = int(os.getenv('MONITORING_EMAIL_SMTP_PORT', '587'))
        email_from = os.getenv('MONITORING_EMAIL_FROM')
        email_to = os.getenv('MONITORING_EMAIL_TO')
        email_user = os.getenv('MONITORING_EMAIL_USER', email_from)
        email_pass = os.getenv('MONITORING_EMAIL_PASS')
        
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to
        msg['Subject'] = f"ZK System Alert [{alert.severity.upper()}]: {alert.rule_name}"
        
        body = f"""
ZK System Monitoring Alert

Alert: {alert.rule_name}
Severity: {alert.severity.upper()}
Time: {alert.timestamp}
Message: {alert.message}

Details:
- Value: {alert.value}
- Threshold: {alert.threshold}
- Consecutive Count: {alert.consecutive_count}

System Context:
- Health Score: {alert.context.get('health_score', 'unknown')}
- Uptime: {alert.context.get('uptime_seconds', 0)} seconds
- Binary Availability: {alert.context.get('binary_availability', {})}

This is an automated message from ZK System Monitoring.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
            if email_pass:
                server.login(email_user, email_pass)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert.rule_name}")
        
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_notification(self, alert: Alert, webhook_url: str):
        """Send alert notification via webhook"""
        import requests
        
        payload = {
            'alert_type': 'zk_system_monitoring',
            'rule_name': alert.rule_name,
            'severity': alert.severity,
            'message': alert.message,
            'timestamp': alert.timestamp,
            'value': alert.value,
            'context': alert.context
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            self.logger.info(f"Webhook alert sent for {alert.rule_name}")
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_slack_notification(self, alert: Alert, slack_webhook: str):
        """Send alert notification to Slack"""
        import requests
        
        color = {
            'critical': '#FF0000',
            'warning': '#FFA500', 
            'info': '#00FF00'
        }.get(alert.severity, '#808080')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"ZK System Alert: {alert.rule_name}",
                'text': alert.message,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.upper(), 'short': True},
                    {'title': 'Time', 'value': alert.timestamp, 'short': True},
                    {'title': 'Health Score', 'value': str(alert.context.get('health_score', 'unknown')), 'short': True},
                    {'title': 'Consecutive Count', 'value': str(alert.consecutive_count), 'short': True}
                ],
                'footer': 'ZK System Monitoring',
                'ts': int(time.time())
            }]
        }
        
        try:
            response = requests.post(slack_webhook, json=payload, timeout=30)
            response.raise_for_status()
            self.logger.info(f"Slack alert sent for {alert.rule_name}")
        
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    def _log_alert_to_file(self, alert: Alert):
        """Log alert to file"""
        log_file = self.project_root / "pot/zk/alerts.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'a') as f:
                log_entry = {
                    'timestamp': alert.timestamp,
                    'rule': alert.rule_name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'value': alert.value,
                    'consecutive_count': alert.consecutive_count
                }
                f.write(json.dumps(log_entry) + '\n')
            
        except Exception as e:
            self.logger.error(f"Failed to log alert to file: {e}")
    
    def _cleanup_alert_history(self):
        """Remove old alerts from history"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00')) > cutoff_time
        ]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        active_count = len(self.active_alerts)
        severity_counts = {}
        
        for alert in self.active_alerts.values():
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00')) > 
               datetime.now(timezone.utc) - timedelta(hours=24)
        ]
        
        return {
            'active_alerts': active_count,
            'severity_breakdown': severity_counts,
            'alerts_last_24h': len(recent_alerts),
            'rules_configured': len(self.rules),
            'notification_channels': len(self.notification_handlers)
        }


class ZKSystemMonitor:
    """Main monitoring service for ZK system"""
    
    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval
        self.start_time = time.time()
        self.diagnostic_runner = ZKDiagnostic()
        self.alert_manager = AlertManager()
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        
        # Metrics history
        self.metrics_history: List[MonitoringMetrics] = []
        self.baseline_metrics: Optional[MonitoringMetrics] = None
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            print(f"ZK system monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
            print("ZK system monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Evaluate alerts
                alerts = self.alert_manager.evaluate_metrics(metrics)
                
                # Send notifications for new alerts
                for alert in alerts:
                    self.alert_manager.send_alert_notifications(alert)
                
                # Cleanup old metrics
                self._cleanup_metrics_history()
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                logging.error(f"Monitoring loop error: {e}")
            
            # Wait for next check
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _collect_metrics(self) -> MonitoringMetrics:
        """Collect comprehensive monitoring metrics"""
        current_time = time.time()
        
        # Run diagnostic
        diagnostic_results = self.diagnostic_runner.diagnose_zk_system()
        
        # Get ZK metrics
        metrics_collector = get_zk_metrics_collector()
        zk_metrics = metrics_collector.generate_report()
        
        # System resource info
        system_resources = self._get_system_resources()
        
        # Binary availability
        binary_availability = self._check_binary_availability()
        
        # Calculate success rates
        proof_success_rates = self._calculate_success_rates(zk_metrics.get('sgd_proofs', []) + 
                                                           zk_metrics.get('lora_proofs', []), 'success')
        verification_success_rates = self._calculate_success_rates(zk_metrics.get('verification_times', []), 'success')
        
        # Calculate average times
        average_proof_times = self._calculate_average_times(zk_metrics)
        
        return MonitoringMetrics(
            timestamp=datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
            health_score=diagnostic_results.get('health_score', 0),
            binary_availability=binary_availability,
            proof_success_rates=proof_success_rates,
            verification_success_rates=verification_success_rates,
            average_proof_times=average_proof_times,
            system_resources=system_resources,
            alert_count=len(self.alert_manager.get_active_alerts()),
            uptime_seconds=int(current_time - self.start_time)
        )
    
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_percent': 0,
                'memory_percent': 0, 
                'disk_percent': 0
            }
    
    def _check_binary_availability(self) -> Dict[str, bool]:
        """Check if required binaries are available"""
        prover_dir = Path(__file__).parent.parent.parent / "pot/zk/prover_halo2"
        required_binaries = [
            'target/release/prove_sgd_stdin',
            'target/release/verify_sgd_stdin',
            'target/release/prove_lora_stdin', 
            'target/release/verify_lora_stdin'
        ]
        
        availability = {}
        for binary in required_binaries:
            binary_path = prover_dir / binary
            availability[binary.split('/')[-1]] = binary_path.exists() and binary_path.is_file()
        
        return availability
    
    def _calculate_success_rates(self, metrics: List[Dict], success_field: str) -> Dict[str, float]:
        """Calculate success rates from metrics"""
        if not metrics:
            return {}
        
        total = len(metrics)
        successful = sum(1 for m in metrics if m.get(success_field, False))
        
        return {
            'overall': (successful / total * 100) if total > 0 else 100
        }
    
    def _calculate_average_times(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate average operation times"""
        averages = {}
        
        for metric_type in ['sgd_proofs', 'lora_proofs']:
            operations = metrics.get(metric_type, [])
            if operations:
                times = [op.get('duration', 0) for op in operations]
                averages[metric_type] = sum(times) / len(times) if times else 0
        
        return averages
    
    def _cleanup_metrics_history(self):
        """Remove old metrics from history"""
        # Keep last 1000 metrics (adjust based on interval)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.running,
            'uptime_seconds': int(time.time() - self.start_time),
            'check_interval': self.check_interval,
            'metrics_collected': len(self.metrics_history),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'last_check': self.metrics_history[-1].timestamp if self.metrics_history else None
        }


def main():
    """Command-line interface for monitoring system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ZK System Monitoring')
    parser.add_argument('command', choices=['start', 'status', 'alerts', 'test-alert'],
                       help='Command to execute')
    parser.add_argument('--interval', type=int, default=300,
                       help='Monitoring interval in seconds')
    parser.add_argument('--config', help='Alert configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        monitor = ZKSystemMonitor(check_interval=args.interval)
        try:
            monitor.start_monitoring()
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            monitor.stop_monitoring()
    
    elif args.command == 'status':
        monitor = ZKSystemMonitor()
        status = monitor.get_monitoring_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == 'alerts':
        alert_manager = AlertManager()
        alerts = alert_manager.get_active_alerts()
        summary = alert_manager.get_alert_summary()
        
        print("Alert Summary:")
        print(json.dumps(summary, indent=2))
        print(f"\nActive Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"- {alert.rule_name} [{alert.severity}]: {alert.message}")
    
    elif args.command == 'test-alert':
        # Send test alert
        alert_manager = AlertManager()
        test_alert = Alert(
            rule_name='test_alert',
            severity='info',
            message='This is a test alert from ZK monitoring system',
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=100,
            threshold=50,
            consecutive_count=1,
            context={'test': True}
        )
        
        alert_manager.send_alert_notifications(test_alert)
        print("Test alert sent through all configured channels")


if __name__ == "__main__":
    main()