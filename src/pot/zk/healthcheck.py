#!/usr/bin/env python3
"""
ZK System Health Check Endpoint

Provides HTTP endpoint for monitoring ZK system health and status.
Returns JSON status reports for integration with monitoring systems.
"""

import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

from .diagnostic import ZKDiagnostic
from .version_info import get_system_version
from .metrics import get_zk_metrics_collector


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints"""
    
    def __init__(self, *args, health_monitor=None, **kwargs):
        self.health_monitor = health_monitor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if path == '/health':
            self._handle_health_check(query)
        elif path == '/health/detailed':
            self._handle_detailed_health(query)
        elif path == '/health/metrics':
            self._handle_metrics(query)
        elif path == '/health/version':
            self._handle_version(query)
        elif path == '/':
            self._handle_root()
        else:
            self._send_error(404, "Endpoint not found")
    
    def do_HEAD(self):
        """Handle HEAD requests"""
        if self.path == '/health':
            self._send_response(200, {})
        else:
            self._send_error(404, "Endpoint not found")
    
    def _handle_health_check(self, query: Dict[str, Any]):
        """Handle basic health check"""
        try:
            if self.health_monitor:
                status = self.health_monitor.get_quick_status()
            else:
                # Fallback basic check
                status = {
                    'status': 'healthy',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'uptime_seconds': 0,
                    'checks_passed': 0,
                    'checks_failed': 0
                }
            
            status_code = 200 if status['status'] == 'healthy' else 503
            self._send_response(status_code, status)
        
        except Exception as e:
            self._send_error(500, f"Health check failed: {str(e)}")
    
    def _handle_detailed_health(self, query: Dict[str, Any]):
        """Handle detailed health check with full diagnostics"""
        try:
            if self.health_monitor:
                status = self.health_monitor.get_detailed_status()
            else:
                # Fallback detailed check
                diagnostic_runner = ZKDiagnostic()
                results = diagnostic_runner.diagnose_zk_system()
                status = {
                    'status': 'healthy' if results['overall_health'] == 'pass' else 'unhealthy',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'diagnostic_results': results,
                    'system_info': {
                        'python_version': results.get('python_version', 'unknown'),
                        'platform': results.get('platform_info', {}).get('system', 'unknown')
                    }
                }
            
            status_code = 200 if status['status'] == 'healthy' else 503
            self._send_response(status_code, status)
        
        except Exception as e:
            self._send_error(500, f"Detailed health check failed: {str(e)}")
    
    def _handle_metrics(self, query: Dict[str, Any]):
        """Handle metrics endpoint"""
        try:
            collector = get_zk_metrics_collector()
            metrics = collector.generate_report()
            
            response = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': metrics
            }
            
            self._send_response(200, response)
        
        except Exception as e:
            self._send_error(500, f"Metrics retrieval failed: {str(e)}")
    
    def _handle_version(self, query: Dict[str, Any]):
        """Handle version information endpoint"""
        try:
            version_info = get_system_version()
            response = asdict(version_info)
            
            self._send_response(200, response)
        
        except Exception as e:
            self._send_error(500, f"Version info retrieval failed: {str(e)}")
    
    def _handle_root(self):
        """Handle root endpoint with available endpoints"""
        endpoints = {
            'message': 'ZK System Health Check Service',
            'version': '1.0.0',
            'endpoints': {
                '/health': 'Basic health check (quick)',
                '/health/detailed': 'Detailed health check with full diagnostics',
                '/health/metrics': 'ZK system metrics',
                '/health/version': 'System version information'
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self._send_response(200, endpoints)
    
    def _send_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        response_data = json.dumps(data, indent=2, default=str)
        self.wfile.write(response_data.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response"""
        error_data = {
            'error': message,
            'status_code': status_code,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        response_data = json.dumps(error_data, indent=2)
        self.wfile.write(response_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override log message to reduce noise"""
        return  # Disable default logging


class HealthMonitor:
    """Monitors ZK system health and maintains status cache"""
    
    def __init__(self, check_interval: int = 300):  # 5 minutes default
        self.check_interval = check_interval
        self.start_time = time.time()
        self.diagnostic_runner = ZKDiagnostic()
        
        # Cached status
        self.last_check_time = 0
        self.cached_status = None
        self.cached_detailed_status = None
        
        # Statistics
        self.total_checks = 0
        self.failed_checks = 0
        self.check_history = []  # Last 100 checks
        
        # Background thread for periodic checks
        self.running = False
        self.check_thread = None
        
    def start_monitoring(self):
        """Start background health monitoring"""
        if not self.running:
            self.running = True
            self.check_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.check_thread.start()
            print(f"Health monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
            print("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self._perform_health_check()
            except Exception as e:
                print(f"Health check error: {e}")
                self.failed_checks += 1
            
            # Wait for next check
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _perform_health_check(self):
        """Perform periodic health check"""
        current_time = time.time()
        
        try:
            # Run diagnostic
            results = self.diagnostic_runner.diagnose_zk_system()
            
            # Update statistics
            self.total_checks += 1
            self.last_check_time = current_time
            
            status = 'healthy' if results['overall_health'] == 'pass' else 'unhealthy'
            
            if status != 'healthy':
                self.failed_checks += 1
            
            # Update cached status
            self.cached_status = {
                'status': status,
                'timestamp': datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
                'uptime_seconds': int(current_time - self.start_time),
                'checks_passed': self.total_checks - self.failed_checks,
                'checks_failed': self.failed_checks,
                'last_check_duration': results.get('total_duration', 0),
                'health_score': results.get('health_score', 0)
            }
            
            # Keep check history
            self.check_history.append({
                'timestamp': current_time,
                'status': status,
                'health_score': results.get('health_score', 0),
                'duration': results.get('total_duration', 0)
            })
            
            # Keep only last 100 checks
            if len(self.check_history) > 100:
                self.check_history = self.check_history[-100:]
        
        except Exception as e:
            self.failed_checks += 1
            self.cached_status = {
                'status': 'unhealthy',
                'timestamp': datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
                'error': str(e),
                'uptime_seconds': int(current_time - self.start_time),
                'checks_passed': self.total_checks - self.failed_checks,
                'checks_failed': self.failed_checks
            }
    
    def get_quick_status(self) -> Dict[str, Any]:
        """Get quick cached status"""
        if self.cached_status is None:
            # First time - run quick check
            self._perform_health_check()
        
        return self.cached_status or {
            'status': 'unknown',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': 'No cached status available'
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status with full diagnostics"""
        current_time = time.time()
        
        # Run fresh detailed diagnostic
        try:
            results = self.diagnostic_runner.diagnose_zk_system()
            
            status = {
                'status': 'healthy' if results['overall_health'] == 'pass' else 'unhealthy',
                'timestamp': datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
                'uptime_seconds': int(current_time - self.start_time),
                'diagnostic_results': results,
                'monitoring_stats': {
                    'total_checks': self.total_checks,
                    'failed_checks': self.failed_checks,
                    'success_rate': (self.total_checks - self.failed_checks) / max(1, self.total_checks) * 100,
                    'last_check_time': self.last_check_time,
                    'check_interval': self.check_interval
                },
                'recent_checks': self.check_history[-10:] if self.check_history else []
            }
            
            return status
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'timestamp': datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
                'error': f"Detailed diagnostic failed: {str(e)}",
                'uptime_seconds': int(current_time - self.start_time)
            }


def create_health_check_handler(health_monitor: Optional[HealthMonitor] = None):
    """Factory function to create health check handler with monitor"""
    def handler_factory(*args, **kwargs):
        return HealthCheckHandler(*args, health_monitor=health_monitor, **kwargs)
    return handler_factory


class HealthCheckServer:
    """HTTP server for health check endpoints"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080, 
                 monitor_interval: int = 300):
        self.host = host
        self.port = port
        self.server = None
        self.health_monitor = HealthMonitor(check_interval=monitor_interval)
        
    def start(self):
        """Start the health check server"""
        handler = create_health_check_handler(self.health_monitor)
        self.server = HTTPServer((self.host, self.port), handler)
        
        # Start background monitoring
        self.health_monitor.start_monitoring()
        
        print(f"Health check server starting on http://{self.host}:{self.port}")
        print("Available endpoints:")
        print(f"  http://{self.host}:{self.port}/health - Basic health check")
        print(f"  http://{self.host}:{self.port}/health/detailed - Detailed diagnostics")
        print(f"  http://{self.host}:{self.port}/health/metrics - System metrics")
        print(f"  http://{self.host}:{self.port}/health/version - Version info")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the health check server"""
        if self.server:
            print("Stopping health check server...")
            self.server.shutdown()
            self.server.server_close()
        
        self.health_monitor.stop_monitoring()
        print("Health check server stopped")


def main():
    """Command-line interface for health check server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ZK System Health Check Server')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port to bind to (default: 8080)')
    parser.add_argument('--monitor-interval', type=int, default=300,
                       help='Health check interval in seconds (default: 300)')
    parser.add_argument('--test-endpoint', 
                       help='Test a specific endpoint and exit')
    
    args = parser.parse_args()
    
    if args.test_endpoint:
        # Test mode - just run one check and output result
        import requests
        import sys
        
        try:
            url = f"http://{args.host}:{args.port}{args.test_endpoint}"
            response = requests.get(url, timeout=30)
            print(f"Status: {response.status_code}")
            print(f"Response:")
            print(json.dumps(response.json(), indent=2))
            sys.exit(0 if response.status_code == 200 else 1)
        
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Test failed: {e}")
            sys.exit(1)
    
    else:
        # Server mode
        server = HealthCheckServer(
            host=args.host, 
            port=args.port, 
            monitor_interval=args.monitor_interval
        )
        server.start()


if __name__ == "__main__":
    main()