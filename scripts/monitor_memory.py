#!/usr/bin/env python3
"""
Real-time memory monitoring for large model testing.
Run this in a separate terminal while testing large models.
"""

import time
import psutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json

class MemoryWatchdog:
    """Monitor and alert on memory usage"""
    
    def __init__(self, warning_threshold=80, critical_threshold=90, kill_threshold=95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.kill_threshold = kill_threshold
        self.peak_memory = 0
        self.alerts = []
        
    def get_memory_status(self):
        """Get current memory statistics"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'ram': {
                'total_gb': mem.total / (1024**3),
                'used_gb': mem.used / (1024**3),
                'available_gb': mem.available / (1024**3),
                'percent': mem.percent
            },
            'swap': {
                'total_gb': swap.total / (1024**3),
                'used_gb': swap.used / (1024**3),
                'percent': swap.percent
            }
        }
    
    def find_python_processes(self):
        """Find Python processes consuming significant memory"""
        python_procs = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    mem_percent = proc.info['memory_percent']
                    if mem_percent > 1.0:  # Only show processes using >1% memory
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if len(cmdline) > 100:
                            cmdline = cmdline[:97] + "..."
                        
                        python_procs.append({
                            'pid': proc.info['pid'],
                            'memory_percent': mem_percent,
                            'memory_gb': (psutil.virtual_memory().total * mem_percent / 100) / (1024**3),
                            'cmdline': cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return sorted(python_procs, key=lambda x: x['memory_percent'], reverse=True)
    
    def check_alerts(self, status):
        """Check if any thresholds are exceeded"""
        mem_percent = status['ram']['percent']
        
        alert_level = None
        if mem_percent >= self.kill_threshold:
            alert_level = 'CRITICAL - KILL THRESHOLD'
        elif mem_percent >= self.critical_threshold:
            alert_level = 'CRITICAL'
        elif mem_percent >= self.warning_threshold:
            alert_level = 'WARNING'
        
        if alert_level:
            alert = {
                'timestamp': status['timestamp'],
                'level': alert_level,
                'memory_percent': mem_percent,
                'memory_gb': status['ram']['used_gb']
            }
            self.alerts.append(alert)
            return alert
        
        return None
    
    def emergency_kill_largest(self):
        """Kill the largest Python process to prevent system crash"""
        python_procs = self.find_python_processes()
        
        if python_procs:
            largest = python_procs[0]
            print(f"\n🚨 EMERGENCY: Killing PID {largest['pid']} using {largest['memory_gb']:.1f}GB")
            
            try:
                proc = psutil.Process(largest['pid'])
                proc.terminate()
                time.sleep(2)
                if proc.is_running():
                    proc.kill()
                print("   Process terminated")
                return True
            except Exception as e:
                print(f"   Failed to kill process: {e}")
                return False
        
        return False
    
    def monitor_loop(self, interval=2):
        """Main monitoring loop"""
        print("="*60)
        print("🔍 MEMORY WATCHDOG STARTED")
        print("="*60)
        print(f"Thresholds:")
        print(f"  Warning: {self.warning_threshold}%")
        print(f"  Critical: {self.critical_threshold}%")
        print(f"  Kill: {self.kill_threshold}%")
        print("="*60)
        print("\nPress Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                status = self.get_memory_status()
                self.peak_memory = max(self.peak_memory, status['ram']['used_gb'])
                
                # Clear screen for clean display
                print("\033[2J\033[H")  # Clear screen and move cursor to top
                
                # Display header
                print("="*60)
                print(f"💾 MEMORY MONITOR - {datetime.now().strftime('%H:%M:%S')}")
                print("="*60)
                
                # RAM status
                ram = status['ram']
                bar_length = 40
                filled = int(bar_length * ram['percent'] / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Color based on usage
                if ram['percent'] >= self.critical_threshold:
                    color = '\033[91m'  # Red
                elif ram['percent'] >= self.warning_threshold:
                    color = '\033[93m'  # Yellow
                else:
                    color = '\033[92m'  # Green
                reset = '\033[0m'
                
                print(f"\nRAM Usage: {color}{bar}{reset} {ram['percent']:.1f}%")
                print(f"Used: {ram['used_gb']:.1f}GB / {ram['total_gb']:.1f}GB")
                print(f"Free: {ram['available_gb']:.1f}GB")
                print(f"Peak: {self.peak_memory:.1f}GB")
                
                # Swap status
                if status['swap']['total_gb'] > 0:
                    swap = status['swap']
                    print(f"\nSwap: {swap['used_gb']:.1f}GB / {swap['total_gb']:.1f}GB ({swap['percent']:.1f}%)")
                    if swap['percent'] > 50:
                        print("  ⚠️ High swap usage - system may be slow")
                
                # Python processes
                python_procs = self.find_python_processes()
                if python_procs:
                    print("\n📊 Top Python Processes:")
                    for i, proc in enumerate(python_procs[:5]):
                        print(f"  {i+1}. PID {proc['pid']}: {proc['memory_gb']:.1f}GB ({proc['memory_percent']:.1f}%)")
                        if proc['cmdline']:
                            print(f"     {proc['cmdline']}")
                
                # Check for alerts
                alert = self.check_alerts(status)
                if alert:
                    print(f"\n{'='*60}")
                    if 'KILL' in alert['level']:
                        print(f"🚨🚨🚨 {alert['level']} 🚨🚨🚨")
                        print("SYSTEM ABOUT TO CRASH - TAKING EMERGENCY ACTION")
                        if self.emergency_kill_largest():
                            print("Emergency kill successful - memory should recover")
                        else:
                            print("MANUAL INTERVENTION REQUIRED!")
                    else:
                        print(f"⚠️ {alert['level']}: Memory at {alert['memory_percent']:.1f}%")
                    print(f"{'='*60}")
                
                # Show recent alerts
                if self.alerts:
                    recent_alerts = self.alerts[-3:]
                    if len(recent_alerts) > 0:
                        print("\n📝 Recent Alerts:")
                        for a in recent_alerts:
                            time_str = datetime.fromisoformat(a['timestamp']).strftime('%H:%M:%S')
                            print(f"  {time_str}: {a['level']} ({a['memory_percent']:.1f}%)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self.save_report()
    
    def save_report(self):
        """Save monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"experimental_results/memory_monitor_{timestamp}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'peak_memory_gb': self.peak_memory,
            'total_alerts': len(self.alerts),
            'alerts': self.alerts[-20:] if self.alerts else [],  # Last 20 alerts
            'final_status': self.get_memory_status()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📁 Report saved to: {report_path}")
        print(f"Peak memory usage: {self.peak_memory:.1f}GB")
        print(f"Total alerts: {len(self.alerts)}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory monitoring for large model testing")
    parser.add_argument('--warning', type=int, default=80,
                       help='Warning threshold (default: 80%%)')
    parser.add_argument('--critical', type=int, default=90,
                       help='Critical threshold (default: 90%%)')
    parser.add_argument('--kill', type=int, default=95,
                       help='Emergency kill threshold (default: 95%%)')
    parser.add_argument('--interval', type=int, default=2,
                       help='Update interval in seconds (default: 2)')
    
    args = parser.parse_args()
    
    watchdog = MemoryWatchdog(
        warning_threshold=args.warning,
        critical_threshold=args.critical,
        kill_threshold=args.kill
    )
    
    watchdog.monitor_loop(interval=args.interval)

if __name__ == "__main__":
    main()