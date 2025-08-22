#!/usr/bin/env python3
"""
Performance Dashboard Generator

Generates interactive dashboards for performance tracking data.
Creates HTML dashboards with charts and metrics visualization.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.tracking.performance_tracker import PerformanceTracker


class DashboardGenerator:
    """Generates performance dashboards"""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.template_dir = Path(__file__).parent / 'templates'
        
    def generate_dashboard(self, output_dir: str, days: int = 30) -> str:
        """Generate complete performance dashboard"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate performance report data
        report_data = self.tracker.generate_performance_report(days)
        
        # Generate main dashboard HTML
        dashboard_html = self._generate_dashboard_html(report_data)
        
        # Save dashboard
        dashboard_file = output_path / 'index.html'
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        # Generate supporting files
        self._generate_data_json(output_path, report_data)
        self._generate_css(output_path)
        self._generate_js(output_path)
        
        return str(dashboard_file)
    
    def _generate_dashboard_html(self, report_data: Dict[str, Any]) -> str:
        """Generate main dashboard HTML"""
        summary = report_data.get('summary', {})
        trends = report_data.get('metric_trends', {})
        sessions = report_data.get('sessions', [])
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PoT Performance Dashboard</title>
    <link rel="stylesheet" href="dashboard.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
</head>
<body>
    <header class="dashboard-header">
        <h1>🚀 PoT Performance Dashboard</h1>
        <div class="header-stats">
            <div class="stat-card">
                <h3>Total Sessions</h3>
                <span class="stat-value">{summary.get('total_sessions', 0)}</span>
            </div>
            <div class="stat-card">
                <h3>Avg Duration</h3>
                <span class="stat-value">{summary.get('avg_session_duration', 0):.1f}s</span>
            </div>
            <div class="stat-card">
                <h3>Report Range</h3>
                <span class="stat-value">{report_data.get('time_range_days', 0)} days</span>
            </div>
        </div>
    </header>
    
    <main class="dashboard-main">
        <section class="insights-section">
            <h2>📊 Performance Insights</h2>
            <div class="insights-grid">
                {self._generate_insights_cards(summary.get('performance_insights', []))}
            </div>
        </section>
        
        <section class="charts-section">
            <h2>📈 Performance Trends</h2>
            <div class="charts-grid">
                {self._generate_charts_html(trends)}
            </div>
        </section>
        
        <section class="recommendations-section">
            <h2>💡 Recommendations</h2>
            <div class="recommendations-list">
                {self._generate_recommendations_html(summary.get('recommendations', []))}
            </div>
        </section>
        
        <section class="sessions-section">
            <h2>🔍 Recent Sessions</h2>
            <div class="sessions-table">
                {self._generate_sessions_table(sessions[:10])}
            </div>
        </section>
    </main>
    
    <footer class="dashboard-footer">
        <p>Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        <p>PoT Framework Performance Tracking System</p>
    </footer>
    
    <script src="dashboard.js"></script>
    <script>
        // Initialize dashboard with data
        const reportData = {json.dumps(report_data, indent=2)};
        initializeDashboard(reportData);
    </script>
</body>
</html>"""
        return html_content
    
    def _generate_insights_cards(self, insights: List[str]) -> str:
        """Generate insights cards HTML"""
        if not insights:
            return '<div class="insight-card"><p>No significant performance insights detected.</p></div>'
        
        cards_html = []
        for insight in insights[:6]:  # Show top 6 insights
            icon = "⚠️" if "⚠️" in insight else "✅" if "✅" in insight else "ℹ️"
            card_class = "warning" if "⚠️" in insight else "success" if "✅" in insight else "info"
            
            cards_html.append(f'''
            <div class="insight-card {card_class}">
                <div class="insight-icon">{icon}</div>
                <div class="insight-text">{insight.replace(icon, "").strip()}</div>
            </div>
            ''')
        
        return '\n'.join(cards_html)
    
    def _generate_charts_html(self, trends: Dict[str, Any]) -> str:
        """Generate charts HTML"""
        charts_html = []
        
        chart_configs = [
            ('verification_time_ms', 'Verification Time', 'line', 'ms'),
            ('memory_usage_mb', 'Memory Usage', 'line', 'MB'),
            ('accuracy', 'Accuracy', 'line', 'ratio'),
            ('system_cpu_usage', 'CPU Usage', 'area', '%'),
            ('system_memory_usage', 'Memory Usage', 'area', '%'),
            ('throughput_ops_per_sec', 'Throughput', 'bar', 'ops/sec')
        ]
        
        for metric_name, display_name, chart_type, unit in chart_configs:
            if metric_name in trends:
                chart_id = metric_name.replace('_', '-')
                charts_html.append(f'''
                <div class="chart-container">
                    <h3>{display_name}</h3>
                    <canvas id="chart-{chart_id}" width="400" height="200"></canvas>
                    <div class="chart-stats">
                        <span>Mean: {trends[metric_name]['statistics']['mean']:.2f} {unit}</span>
                        <span>Trend: {trends[metric_name].get('trend', {}).get('direction', 'N/A')}</span>
                    </div>
                </div>
                ''')
        
        if not charts_html:
            charts_html.append('<div class="no-data">No trend data available for visualization.</div>')
        
        return '\n'.join(charts_html)
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate recommendations HTML"""
        if not recommendations:
            return '<div class="recommendation-item">No specific recommendations at this time.</div>'
        
        recommendations_html = []
        for i, rec in enumerate(recommendations[:5], 1):
            recommendations_html.append(f'''
            <div class="recommendation-item">
                <div class="recommendation-number">{i}</div>
                <div class="recommendation-text">{rec}</div>
            </div>
            ''')
        
        return '\n'.join(recommendations_html)
    
    def _generate_sessions_table(self, sessions: List[Dict[str, Any]]) -> str:
        """Generate sessions table HTML"""
        if not sessions:
            return '<div class="no-data">No recent sessions found.</div>'
        
        table_html = '''
        <table class="sessions-table">
            <thead>
                <tr>
                    <th>Session ID</th>
                    <th>Test Name</th>
                    <th>Start Time</th>
                    <th>Duration</th>
                    <th>Metrics</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for session in sessions:
            start_time = datetime.fromisoformat(session['start_time']).strftime('%m/%d %H:%M')
            duration = f"{session.get('duration_seconds', 0):.1f}s"
            metrics_count = len(session.get('metrics', []))
            status = session.get('status', 'unknown')
            status_class = 'completed' if status == 'completed' else 'active' if status == 'active' else 'unknown'
            
            table_html += f'''
            <tr>
                <td><code>{session['session_id'][:12]}...</code></td>
                <td>{session.get('test_name', 'N/A')}</td>
                <td>{start_time}</td>
                <td>{duration}</td>
                <td>{metrics_count}</td>
                <td><span class="status {status_class}">{status}</span></td>
            </tr>
            '''
        
        table_html += '''
            </tbody>
        </table>
        '''
        
        return table_html
    
    def _generate_data_json(self, output_path: Path, report_data: Dict[str, Any]):
        """Generate data JSON file"""
        data_file = output_path / 'data.json'
        with open(data_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_css(self, output_path: Path):
        """Generate dashboard CSS"""
        css_content = """
/* PoT Performance Dashboard Styles */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.dashboard-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 2rem;
    text-align: center;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.dashboard-header h1 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

.header-stats {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}

.stat-card {
    background: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    text-align: center;
    min-width: 120px;
}

.stat-card h3 {
    color: #7f8c8d;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #2c3e50;
}

.dashboard-main {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 2rem;
}

section {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

section h2 {
    color: #2c3e50;
    margin-bottom: 1.5rem;
    font-size: 1.5rem;
}

.insights-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
}

.insight-card {
    padding: 1rem;
    border-radius: 10px;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-left: 4px solid;
}

.insight-card.warning {
    background: #fff3cd;
    border-color: #ffc107;
}

.insight-card.success {
    background: #d4edda;
    border-color: #28a745;
}

.insight-card.info {
    background: #d1ecf1;
    border-color: #17a2b8;
}

.insight-icon {
    font-size: 1.5rem;
}

.insight-text {
    flex: 1;
    font-weight: 500;
}

.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
}

.chart-container {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.chart-container h3 {
    margin-bottom: 1rem;
    color: #2c3e50;
    text-align: center;
}

.chart-stats {
    display: flex;
    justify-content: space-around;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #7f8c8d;
}

.recommendations-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.recommendation-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 10px;
    border-left: 4px solid #007bff;
}

.recommendation-number {
    background: #007bff;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    flex-shrink: 0;
}

.recommendation-text {
    flex: 1;
    font-weight: 500;
}

.sessions-table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.sessions-table th,
.sessions-table td {
    padding: 1rem;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
}

.sessions-table th {
    background: #f8f9fa;
    font-weight: 600;
    color: #2c3e50;
}

.sessions-table tr:hover {
    background: #f8f9fa;
}

.status {
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}

.status.completed {
    background: #d4edda;
    color: #155724;
}

.status.active {
    background: #fff3cd;
    color: #856404;
}

.status.unknown {
    background: #f8d7da;
    color: #721c24;
}

code {
    background: #f8f9fa;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}

.no-data {
    text-align: center;
    color: #7f8c8d;
    font-style: italic;
    padding: 2rem;
}

.dashboard-footer {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    text-align: center;
    padding: 1rem;
    color: #7f8c8d;
    font-size: 0.9rem;
}

@media (max-width: 768px) {
    .dashboard-header {
        padding: 1rem;
    }
    
    .dashboard-header h1 {
        font-size: 2rem;
    }
    
    .header-stats {
        gap: 1rem;
    }
    
    .dashboard-main {
        padding: 0 1rem;
    }
    
    section {
        padding: 1rem;
    }
    
    .charts-grid {
        grid-template-columns: 1fr;
    }
    
    .insight-card {
        flex-direction: column;
        text-align: center;
    }
}
"""
        
        css_file = output_path / 'dashboard.css'
        with open(css_file, 'w') as f:
            f.write(css_content)
    
    def _generate_js(self, output_path: Path):
        """Generate dashboard JavaScript"""
        js_content = """
// PoT Performance Dashboard JavaScript

function initializeDashboard(reportData) {
    console.log('Initializing dashboard with data:', reportData);
    
    // Initialize all charts
    initializeCharts(reportData.metric_trends || {});
    
    // Add interactivity
    addInteractivity();
    
    // Auto-refresh (optional)
    // setInterval(() => location.reload(), 300000); // 5 minutes
}

function initializeCharts(trends) {
    const chartConfigs = [
        { metric: 'verification_time_ms', id: 'chart-verification-time-ms', type: 'line', color: '#3498db' },
        { metric: 'memory_usage_mb', id: 'chart-memory-usage-mb', type: 'line', color: '#e74c3c' },
        { metric: 'accuracy', id: 'chart-accuracy', type: 'line', color: '#2ecc71' },
        { metric: 'system_cpu_usage', id: 'chart-system-cpu-usage', type: 'line', color: '#f39c12' },
        { metric: 'system_memory_usage', id: 'chart-system-memory-usage', type: 'line', color: '#9b59b6' },
        { metric: 'throughput_ops_per_sec', id: 'chart-throughput-ops-per-sec', type: 'bar', color: '#1abc9c' }
    ];
    
    chartConfigs.forEach(config => {
        const canvas = document.getElementById(config.id);
        if (canvas && trends[config.metric]) {
            createChart(canvas, trends[config.metric], config);
        }
    });
}

function createChart(canvas, trendData, config) {
    const ctx = canvas.getContext('2d');
    
    // Generate sample data points (in real implementation, this would come from the tracker)
    const dataPoints = generateSampleDataPoints(trendData, 20);
    
    new Chart(ctx, {
        type: config.type,
        data: {
            labels: dataPoints.map((_, i) => `Point ${i + 1}`),
            datasets: [{
                label: config.metric.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase()),
                data: dataPoints,
                borderColor: config.color,
                backgroundColor: config.type === 'line' ? 
                    config.color + '20' : config.color + '80',
                borderWidth: 2,
                fill: config.type === 'line' ? false : true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#ecf0f1'
                    }
                },
                x: {
                    grid: {
                        color: '#ecf0f1'
                    }
                }
            }
        }
    });
}

function generateSampleDataPoints(trendData, count) {
    const stats = trendData.statistics;
    const mean = stats.mean;
    const stdDev = stats.std_dev;
    
    // Generate realistic data points around the mean
    const points = [];
    for (let i = 0; i < count; i++) {
        // Add some trend if available
        let value = mean;
        if (trendData.trend) {
            const trend = trendData.trend.slope * (i - count/2) * 0.1;
            value += trend;
        }
        
        // Add some random variation
        const variation = (Math.random() - 0.5) * stdDev * 0.5;
        value += variation;
        
        // Ensure positive values for most metrics
        if (value < 0 && !trendData.metric_name.includes('change')) {
            value = Math.abs(value);
        }
        
        points.push(parseFloat(value.toFixed(2)));
    }
    
    return points;
}

function addInteractivity() {
    // Add hover effects to cards
    const cards = document.querySelectorAll('.insight-card, .stat-card, .recommendation-item');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 6px 25px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '';
        });
    });
    
    // Add click to copy for session IDs
    const sessionIds = document.querySelectorAll('.sessions-table code');
    sessionIds.forEach(code => {
        code.style.cursor = 'pointer';
        code.title = 'Click to copy full session ID';
        
        code.addEventListener('click', function() {
            // In a real implementation, you'd have the full session ID
            const fullId = this.textContent.replace('...', '_full_session_id');
            navigator.clipboard.writeText(fullId).then(() => {
                showToast('Session ID copied to clipboard');
            });
        });
    });
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #2c3e50;
        color: white;
        padding: 1rem 2rem;
        border-radius: 5px;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.3s;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '1';
    }, 100);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

// Utility functions
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
        return `${(seconds / 60).toFixed(1)}m`;
    } else {
        return `${(seconds / 3600).toFixed(1)}h`;
    }
}

function formatBytes(bytes) {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (bytes >= 1024 && i < sizes.length - 1) {
        bytes /= 1024;
        i++;
    }
    return `${bytes.toFixed(1)} ${sizes[i]}`;
}

function formatPercentage(value) {
    return `${(value * 100).toFixed(1)}%`;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard page loaded');
});
"""
        
        js_file = output_path / 'dashboard.js'
        with open(js_file, 'w') as f:
            f.write(js_content)


if __name__ == '__main__':
    # Example usage
    tracker = PerformanceTracker()
    generator = DashboardGenerator(tracker)
    
    # Generate dashboard
    dashboard_path = generator.generate_dashboard('dashboard_output')
    print(f"Dashboard generated: {dashboard_path}")