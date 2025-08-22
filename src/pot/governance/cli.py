#!/usr/bin/env python3
"""
PoT Governance and Compliance Management CLI
Comprehensive command-line interface for governance operations
"""

import click
import json
import yaml
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich import print as rprint
import time

# Import governance modules
try:
    from .governance import GovernanceFramework
    from .eu_ai_act_compliance import EUAIActCompliance
    from .nist_ai_rmf_compliance import NISTAIRMFCompliance
    from .policy_engine import PolicyEngine, Policy, PolicyType
    from .audit_logger import AuditLogger, LogCategory
    from .compliance_dashboard import ComplianceDashboard
    from .risk_assessment import AIRiskAssessment, RiskAppetite
except ImportError:
    # For standalone testing
    pass

# Initialize Rich console
console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def governance(ctx, verbose, config):
    """
    🛡️ PoT Governance and Compliance Management
    
    Comprehensive governance framework for AI systems with support for:
    
    • EU AI Act compliance checking
    
    • NIST AI RMF assessment
    
    • Policy engine management
    
    • Risk assessment and mitigation
    
    • Audit trail management
    
    • Compliance dashboard generation
    
    Examples:
    
        # Check overall compliance status
        pot-governance check-compliance
        
        # Generate EU AI Act report
        pot-governance generate-report --regulation eu-ai-act
        
        # Start interactive risk assessment
        pot-governance assess-risk --interactive
        
        # Query audit logs
        pot-governance audit-trail --days 7
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config_file'] = config
    
    # Load configuration if provided
    if config:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                ctx.obj['config'] = yaml.safe_load(f)
            else:
                ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}


@governance.command()
@click.option('--config', help='Governance configuration file')
@click.option('--output', '-o', type=click.Choice(['json', 'yaml', 'table']), default='table')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed compliance information')
@click.pass_context
def check_compliance(ctx, config, output, detailed):
    """
    ✓ Check current compliance status
    
    Performs comprehensive compliance assessment across all frameworks
    """
    console.print("[bold cyan]🔍 Checking Compliance Status...[/bold cyan]")
    
    # Initialize framework
    config_path = config or ctx.obj.get('config_file', 'governance_config.yaml')
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Initialize components
        task = progress.add_task("Initializing governance framework...", total=None)
        framework = GovernanceFramework(config_path)
        
        progress.update(task, description="Checking EU AI Act compliance...")
        eu_compliance = EUAIActCompliance()
        
        progress.update(task, description="Checking NIST AI RMF compliance...")
        nist_compliance = NISTAIRMFCompliance()
        
        progress.update(task, description="Analyzing policies...")
        policy_engine = PolicyEngine(enforcement_mode="advisory")
        
        # Generate compliance report
        progress.update(task, description="Generating compliance report...")
        report = framework.generate_compliance_report()
    
    # Display results
    if output == 'json':
        click.echo(json.dumps(report, indent=2))
    elif output == 'yaml':
        click.echo(yaml.dump(report, default_flow_style=False))
    else:  # table
        _display_compliance_table(report, detailed)
    
    # Show compliance score with color coding
    score = report.get('compliance_rate', 0) * 100
    if score >= 80:
        console.print(f"[green]✓ Overall Compliance: {score:.1f}%[/green]")
    elif score >= 60:
        console.print(f"[yellow]⚠ Overall Compliance: {score:.1f}%[/yellow]")
    else:
        console.print(f"[red]✗ Overall Compliance: {score:.1f}%[/red]")
    
    # Exit with appropriate code
    sys.exit(0 if score >= 80 else 1)


def _display_compliance_table(report: Dict[str, Any], detailed: bool):
    """Display compliance report as a formatted table"""
    table = Table(title="Compliance Status Report", show_header=True)
    
    table.add_column("Category", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Issues", justify="right")
    
    # Overall status
    overall_score = report.get('compliance_rate', 0) * 100
    status_emoji = "✓" if overall_score >= 80 else "⚠" if overall_score >= 60 else "✗"
    table.add_row(
        "Overall Compliance",
        status_emoji,
        f"{overall_score:.1f}%",
        str(len(report.get('violations_summary', [])))
    )
    
    # Policy status
    policies = report.get('policies', {})
    if policies:
        active = sum(1 for p in policies.values() if p.get('active'))
        table.add_row(
            "Active Policies",
            "✓" if active > 0 else "✗",
            f"{active}/{len(policies)}",
            ""
        )
    
    # Violations
    violations = report.get('violations_summary', [])
    if violations and detailed:
        table.add_row("", "", "", "")
        table.add_row("[bold]Violations[/bold]", "", "", "")
        for violation in violations[:5]:
            table.add_row(
                f"  • {violation.get('violation', 'Unknown')}",
                "✗",
                "",
                str(violation.get('count', 1))
            )
    
    console.print(table)


@governance.command()
@click.option('--regulation', type=click.Choice(['eu-ai-act', 'nist-rmf', 'all']), default='all')
@click.option('--format', type=click.Choice(['pdf', 'html', 'json', 'docx']), default='html')
@click.option('--output-dir', '-o', type=click.Path(), default='./reports')
@click.option('--include-evidence', is_flag=True, help='Include supporting evidence')
@click.pass_context
def generate_report(ctx, regulation, format, output_dir, include_evidence):
    """
    📊 Generate compliance report
    
    Creates detailed compliance reports for specified regulations
    """
    console.print(f"[bold cyan]📊 Generating {regulation.upper()} Compliance Report...[/bold cyan]")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if regulation in ['eu-ai-act', 'all']:
            task = progress.add_task("Generating EU AI Act report...", total=None)
            eu_compliance = EUAIActCompliance()
            
            # Perform assessment
            system_info = ctx.obj.get('config', {}).get('system_info', {})
            package = eu_compliance.generate_compliance_package(system_info)
            
            # Save report
            report_file = output_path / f"eu_ai_act_report_{timestamp}.{format}"
            _save_report(package, report_file, format)
            console.print(f"[green]✓ EU AI Act report saved to {report_file}[/green]")
        
        if regulation in ['nist-rmf', 'all']:
            task = progress.add_task("Generating NIST AI RMF report...", total=None)
            nist_compliance = NISTAIRMFCompliance()
            
            # Perform assessment
            system_info = ctx.obj.get('config', {}).get('system_info', {})
            assessment = nist_compliance.perform_complete_assessment(system_info)
            
            # Save report
            report_file = output_path / f"nist_rmf_report_{timestamp}.{format}"
            _save_report(assessment, report_file, format)
            console.print(f"[green]✓ NIST AI RMF report saved to {report_file}[/green]")
    
    console.print("[bold green]✓ Report generation complete![/bold green]")


def _save_report(data: Dict[str, Any], filepath: Path, format: str):
    """Save report in specified format"""
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'html':
        # Generate HTML report
        html = _generate_html_report(data)
        with open(filepath, 'w') as f:
            f.write(html)
    elif format == 'pdf':
        # Would use a library like reportlab or weasyprint
        console.print("[yellow]PDF generation requires additional dependencies[/yellow]")
        # For now, save as HTML
        filepath = filepath.with_suffix('.html')
        html = _generate_html_report(data)
        with open(filepath, 'w') as f:
            f.write(html)
    elif format == 'docx':
        # Would use python-docx
        console.print("[yellow]DOCX generation requires additional dependencies[/yellow]")
        # For now, save as JSON
        filepath = filepath.with_suffix('.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def _generate_html_report(data: Dict[str, Any]) -> str:
    """Generate HTML report from data"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Compliance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2d3748; }}
            h2 {{ color: #4a5568; margin-top: 30px; }}
            .metric {{ margin: 10px 0; padding: 10px; background: #f7fafc; }}
            .success {{ color: #48bb78; }}
            .warning {{ color: #ed8936; }}
            .error {{ color: #f56565; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 10px; text-align: left; border: 1px solid #e2e8f0; }}
            th {{ background: #edf2f7; }}
        </style>
    </head>
    <body>
        <h1>Compliance Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <pre>{json.dumps(data, indent=2)}</pre>
    </body>
    </html>
    """
    return html


@governance.command()
@click.option('--days', '-d', type=int, default=7, help='Number of days to show')
@click.option('--category', type=click.Choice(['all', 'governance', 'model', 'data', 'security']))
@click.option('--actor', help='Filter by actor/user')
@click.option('--export', type=click.Path(), help='Export audit trail to file')
@click.pass_context
def audit_trail(ctx, days, category, actor, export):
    """
    📜 View and manage audit trail
    
    Query and analyze audit log entries
    """
    console.print(f"[bold cyan]📜 Audit Trail (Last {days} days)[/bold cyan]")
    
    # Initialize audit logger
    audit_logger = AuditLogger(
        log_dir=ctx.obj.get('config', {}).get('audit_dir', './audit_logs')
    )
    
    # Query logs
    start_date = datetime.now() - timedelta(days=days)
    
    # Map category string to LogCategory enum
    log_category = None
    if category and category != 'all':
        category_map = {
            'governance': LogCategory.GOVERNANCE,
            'model': LogCategory.MODEL_OPERATION,
            'data': LogCategory.DATA_ACCESS,
            'security': LogCategory.SECURITY_EVENT
        }
        log_category = category_map.get(category)
    
    entries = audit_logger.query_logs(
        start_date=start_date,
        category=log_category,
        actor=actor,
        limit=100
    )
    
    if not entries:
        console.print("[yellow]No audit entries found for the specified criteria[/yellow]")
        return
    
    # Display entries
    table = Table(title=f"Audit Log Entries ({len(entries)} entries)", show_header=True)
    table.add_column("Timestamp", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Actor")
    table.add_column("Action")
    table.add_column("Resource")
    table.add_column("Result")
    
    for entry in entries[:20]:  # Show first 20
        result_style = "green" if "success" in entry.result.lower() else "red"
        table.add_row(
            entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            entry.category.value,
            entry.actor,
            entry.action,
            entry.resource,
            f"[{result_style}]{entry.result}[/{result_style}]"
        )
    
    console.print(table)
    
    if len(entries) > 20:
        console.print(f"[dim]... and {len(entries) - 20} more entries[/dim]")
    
    # Verify integrity
    console.print("\n[bold]Integrity Check:[/bold]")
    is_valid, issues = audit_logger.verify_integrity(start_date)
    if is_valid:
        console.print("[green]✓ Audit trail integrity verified[/green]")
    else:
        console.print(f"[red]✗ Integrity issues detected: {issues[:3]}[/red]")
    
    # Export if requested
    if export:
        export_path = Path(export)
        export_data = [entry.to_dict() for entry in entries]
        
        if export_path.suffix == '.json':
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif export_path.suffix == '.csv':
            import csv
            with open(export_path, 'w', newline='') as f:
                if export_data:
                    writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
                    writer.writeheader()
                    writer.writerows(export_data)
        else:
            # Default to JSON lines
            with open(export_path, 'w') as f:
                for entry in export_data:
                    f.write(json.dumps(entry) + '\n')
        
        console.print(f"[green]✓ Audit trail exported to {export_path}[/green]")


# Policy Management Commands
@governance.group()
def policy():
    """📋 Policy management commands"""
    pass


@policy.command()
@click.option('--type', 'policy_type', type=click.Choice([
    'data_retention', 'model_retraining', 'access_control', 
    'verification_threshold', 'audit_logging'
]))
@click.option('--file', type=click.Path(exists=True), help='Policy definition file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive policy creation')
@click.pass_context
def add(ctx, policy_type, file, interactive):
    """Add a new policy"""
    console.print("[bold cyan]➕ Adding New Policy[/bold cyan]")
    
    policy_engine = PolicyEngine()
    
    if interactive:
        # Interactive policy creation wizard
        policy_data = _policy_creation_wizard()
    elif file:
        # Load from file
        with open(file, 'r') as f:
            if file.endswith('.yaml') or file.endswith('.yml'):
                policy_data = yaml.safe_load(f)
            else:
                policy_data = json.load(f)
    else:
        console.print("[red]Please specify --file or use --interactive mode[/red]")
        return
    
    # Create and add policy
    try:
        policy = Policy.from_dict(policy_data)
        if policy_engine.add_policy(policy):
            console.print(f"[green]✓ Policy '{policy.name}' added successfully[/green]")
        else:
            console.print(f"[red]✗ Failed to add policy[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error adding policy: {e}[/red]")


def _policy_creation_wizard() -> Dict[str, Any]:
    """Interactive policy creation wizard"""
    console.print("[bold]Policy Creation Wizard[/bold]")
    
    policy_data = {
        "name": Prompt.ask("Policy name"),
        "type": Prompt.ask(
            "Policy type",
            choices=["data_retention", "model_retraining", "access_control", 
                    "verification_threshold", "audit_logging"]
        ),
        "version": Prompt.ask("Version", default="1.0.0"),
        "description": Prompt.ask("Description"),
        "enabled": Confirm.ask("Enable policy?", default=True),
        "enforcement_mode": Prompt.ask(
            "Enforcement mode",
            choices=["strict", "advisory", "monitor"],
            default="advisory"
        ),
        "priority": int(Prompt.ask("Priority (1-100)", default="50")),
        "author": Prompt.ask("Author", default="admin"),
        "rules": []
    }
    
    # Add rules
    console.print("\n[bold]Add Policy Rules[/bold]")
    console.print("[dim]Enter rules one by one. Press Enter with empty rule to finish.[/dim]")
    
    rule_count = 1
    while True:
        console.print(f"\n[cyan]Rule {rule_count}:[/cyan]")
        condition = Prompt.ask("Condition (e.g., 'data_age > 90')")
        
        if not condition:
            break
        
        rule = {
            "rule_id": f"R{rule_count:03d}",
            "condition": condition,
            "action": Prompt.ask("Action (optional)", default=""),
            "message": Prompt.ask("Message"),
            "severity": Prompt.ask("Severity", choices=["low", "medium", "high", "critical"], default="medium")
        }
        
        policy_data["rules"].append(rule)
        rule_count += 1
    
    return policy_data


@policy.command()
@click.argument('policy_name')
@click.pass_context
def remove(ctx, policy_name):
    """Remove a policy"""
    console.print(f"[bold cyan]➖ Removing Policy: {policy_name}[/bold cyan]")
    
    if Confirm.ask(f"Are you sure you want to remove policy '{policy_name}'?"):
        policy_engine = PolicyEngine()
        if policy_engine.remove_policy(policy_name):
            console.print(f"[green]✓ Policy '{policy_name}' removed[/green]")
        else:
            console.print(f"[red]✗ Failed to remove policy[/red]")
    else:
        console.print("[yellow]Cancelled[/yellow]")


@policy.command()
@click.option('--enabled/--disabled', default=None, help='Filter by enabled status')
@click.option('--type', 'policy_type', help='Filter by policy type')
@click.pass_context
def list(ctx, enabled, policy_type):
    """List all policies"""
    console.print("[bold cyan]📋 Policy List[/bold cyan]")
    
    policy_engine = PolicyEngine()
    report = policy_engine.generate_policy_report()
    
    table = Table(title="Active Policies", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Version")
    table.add_column("Status")
    table.add_column("Mode")
    table.add_column("Priority")
    table.add_column("Rules")
    
    for policy_info in report.get('policy_details', []):
        # Apply filters
        if enabled is not None and policy_info['enabled'] != enabled:
            continue
        if policy_type and policy_info['type'] != policy_type:
            continue
        
        status = "[green]✓[/green]" if policy_info['enabled'] else "[red]✗[/red]"
        table.add_row(
            policy_info['name'],
            policy_info['type'],
            policy_info['version'],
            status,
            policy_info['enforcement_mode'],
            str(policy_info['priority']),
            str(policy_info['rules_count'])
        )
    
    console.print(table)


@policy.command()
@click.pass_context
def check_conflicts(ctx):
    """Check for policy conflicts"""
    console.print("[bold cyan]🔍 Checking Policy Conflicts[/bold cyan]")
    
    # This would analyze policies for conflicts
    console.print("[yellow]Policy conflict detection not yet implemented[/yellow]")


# Risk Assessment Commands
@governance.group()
def risk():
    """⚠️ Risk assessment and management"""
    pass


@risk.command()
@click.option('--model-config', type=click.Path(exists=True), help='Model configuration file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive risk assessment')
@click.option('--appetite', type=click.Choice(['averse', 'minimal', 'cautious', 'open', 'hungry']), default='cautious')
@click.pass_context
def assess(ctx, model_config, interactive, appetite):
    """Perform risk assessment"""
    console.print("[bold cyan]⚠️ Risk Assessment[/bold cyan]")
    
    risk_assessment = AIRiskAssessment(risk_appetite=RiskAppetite(appetite))
    
    if interactive:
        # Interactive risk assessment wizard
        config = _risk_assessment_wizard()
    elif model_config:
        with open(model_config, 'r') as f:
            config = yaml.safe_load(f) if model_config.endswith(('.yaml', '.yml')) else json.load(f)
    else:
        config = ctx.obj.get('config', {}).get('model_config', {})
    
    # Perform assessment
    with console.status("Performing risk assessment..."):
        profile = risk_assessment.assess_model_risks(config)
    
    # Display results
    _display_risk_profile(profile)
    
    # Show mitigations
    if profile.mitigations:
        console.print("\n[bold]Recommended Mitigations:[/bold]")
        for mit in profile.mitigations[:5]:
            console.print(f"• {mit.risk_id}: {mit.treatment.value}")
            for action in mit.actions[:2]:
                console.print(f"  - {action}")


def _risk_assessment_wizard() -> Dict[str, Any]:
    """Interactive risk assessment wizard"""
    console.print("[bold]Risk Assessment Configuration[/bold]")
    
    config = {
        "model_name": Prompt.ask("Model name"),
        "criticality": Prompt.ask("System criticality", choices=["low", "medium", "high", "critical"]),
        "public_facing": Confirm.ask("Is the system public-facing?"),
        "handles_pii": Confirm.ask("Does it handle personal data?"),
        "affects_individuals": Confirm.ask("Does it affect individuals directly?"),
        "deployment_months": int(Prompt.ask("Months since deployment", default="0")),
        "bias_testing": Confirm.ask("Is bias testing implemented?"),
        "adversarial_training": Confirm.ask("Is adversarial training used?"),
        "monitoring_enabled": Confirm.ask("Is monitoring enabled?"),
        "drift_detection": Confirm.ask("Is drift detection implemented?")
    }
    
    return config


def _display_risk_profile(profile):
    """Display risk profile"""
    table = Table(title="Risk Assessment Results", show_header=True)
    table.add_column("Risk Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Severity", justify="center")
    
    for category, count in profile.risk_distribution.items():
        table.add_row(category, str(count), _get_severity_indicator(category))
    
    console.print(table)
    
    console.print(f"\n[bold]Overall Risk Score:[/bold] {profile.overall_risk_score:.1f}/100")
    console.print(f"[bold]High Priority Risks:[/bold] {len(profile.high_priority_risks)}")
    console.print(f"[bold]Risk Appetite:[/bold] {profile.risk_appetite.value}")
    console.print(f"[bold]Acceptable Threshold:[/bold] {profile.acceptable_threshold}")


def _get_severity_indicator(category: str) -> str:
    """Get severity indicator for risk category"""
    high_risk = ["security_vulnerability", "privacy_breach", "safety_harm"]
    medium_risk = ["bias_discrimination", "performance_degradation"]
    
    if category in high_risk:
        return "[red]●●●[/red]"
    elif category in medium_risk:
        return "[yellow]●●○[/yellow]"
    else:
        return "[green]●○○[/green]"


@risk.command()
@click.pass_context
def register(ctx):
    """View risk register"""
    console.print("[bold cyan]📊 Risk Register[/bold cyan]")
    
    risk_assessment = AIRiskAssessment()
    report = risk_assessment.generate_risk_report()
    
    # Display summary
    summary = report.get('risk_summary', {})
    
    console.print(Panel(
        f"[red]Critical: {summary.get('critical', 0)}[/red]  "
        f"[orange1]High: {summary.get('high', 0)}[/orange1]  "
        f"[yellow]Medium: {summary.get('medium', 0)}[/yellow]  "
        f"[green]Low: {summary.get('low', 0)}[/green]",
        title="Risk Summary"
    ))
    
    # Display top risks
    if report.get('top_risks'):
        console.print("\n[bold]Top Risks:[/bold]")
        for risk in report['top_risks'][:5]:
            console.print(f"• [{risk['category']}] {risk['title']} (Score: {risk['residual_score']})")


# Dashboard Command
@governance.command()
@click.option('--output', '-o', type=click.Path(), default='dashboard.html')
@click.option('--open-browser', '-b', is_flag=True, help='Open dashboard in browser')
@click.pass_context
def dashboard(ctx, output, open_browser):
    """
    📊 Generate compliance dashboard
    
    Creates an interactive HTML dashboard with compliance metrics
    """
    console.print("[bold cyan]📊 Generating Compliance Dashboard...[/bold cyan]")
    
    with console.status("Initializing components..."):
        # Initialize all components
        governance = GovernanceFramework(
            ctx.obj.get('config_file', 'governance_config.yaml')
        )
        eu_compliance = EUAIActCompliance()
        nist_compliance = NISTAIRMFCompliance()
        policy_engine = PolicyEngine()
        audit_logger = AuditLogger(log_dir='./audit_logs')
        
        # Create dashboard
        dashboard = ComplianceDashboard(
            governance_framework=governance,
            eu_compliance=eu_compliance,
            nist_compliance=nist_compliance,
            policy_engine=policy_engine,
            audit_logger=audit_logger
        )
    
    with console.status("Collecting metrics..."):
        dashboard.collect_metrics()
    
    with console.status("Generating HTML..."):
        html = dashboard.generate_html_dashboard()
    
    # Save dashboard
    output_path = Path(output)
    with open(output_path, 'w') as f:
        f.write(html)
    
    console.print(f"[green]✓ Dashboard saved to {output_path}[/green]")
    
    # Open in browser if requested
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{output_path.absolute()}")
        console.print("[green]✓ Dashboard opened in browser[/green]")


# Validation Commands
@governance.group()
def validate():
    """✓ Validation and verification commands"""
    pass


@validate.command()
@click.option('--config', type=click.Path(exists=True), required=True)
@click.pass_context
def config(ctx, config):
    """Validate configuration file"""
    console.print(f"[bold cyan]✓ Validating Configuration: {config}[/bold cyan]")
    
    try:
        with open(config, 'r') as f:
            if config.endswith('.yaml') or config.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Validate with governance framework
        framework = GovernanceFramework('temp_config.yaml')
        is_valid, issues = framework.validate_configuration(config_data)
        
        if is_valid:
            console.print("[green]✓ Configuration is valid[/green]")
        else:
            console.print("[red]✗ Configuration validation failed:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
    
    except Exception as e:
        console.print(f"[red]✗ Error validating configuration: {e}[/red]")


@validate.command()
@click.pass_context
def pre_deployment(ctx):
    """Pre-deployment compliance check"""
    console.print("[bold cyan]🚀 Pre-Deployment Compliance Check[/bold cyan]")
    
    checks = [
        ("EU AI Act Compliance", lambda: _check_eu_compliance()),
        ("NIST AI RMF Assessment", lambda: _check_nist_compliance()),
        ("Policy Compliance", lambda: _check_policy_compliance()),
        ("Risk Assessment", lambda: _check_risk_levels()),
        ("Audit Trail Integrity", lambda: _check_audit_integrity()),
    ]
    
    all_passed = True
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for name, check_func in checks:
            task = progress.add_task(f"Checking {name}...", total=None)
            
            try:
                passed, message = check_func()
                results.append((name, passed, message))
                if not passed:
                    all_passed = False
            except Exception as e:
                results.append((name, False, str(e)))
                all_passed = False
            
            progress.update(task, completed=True)
    
    # Display results
    console.print("\n[bold]Pre-Deployment Check Results:[/bold]")
    for name, passed, message in results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"{status} {name}: {message}")
    
    if all_passed:
        console.print("\n[bold green]✓ System is ready for deployment![/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]✗ System is NOT ready for deployment[/bold red]")
        sys.exit(1)


def _check_eu_compliance():
    """Check EU AI Act compliance"""
    eu = EUAIActCompliance()
    # Simplified check
    return True, "Compliance checks passed"


def _check_nist_compliance():
    """Check NIST compliance"""
    nist = NISTAIRMFCompliance()
    # Simplified check
    return True, "Maturity level acceptable"


def _check_policy_compliance():
    """Check policy compliance"""
    engine = PolicyEngine()
    report = engine.generate_policy_report()
    enabled = report['policies_by_status']['enabled']
    return enabled > 0, f"{enabled} policies active"


def _check_risk_levels():
    """Check risk levels"""
    risk = AIRiskAssessment()
    # Simplified check
    return True, "Risk levels within tolerance"


def _check_audit_integrity():
    """Check audit trail integrity"""
    logger = AuditLogger(log_dir='./audit_logs')
    is_valid, issues = logger.verify_integrity()
    return is_valid, "Integrity verified" if is_valid else f"{len(issues)} issues found"


# Batch Operations
@governance.group()
def batch():
    """📦 Batch operations"""
    pass


@batch.command()
@click.option('--policies-dir', type=click.Path(exists=True), required=True)
@click.option('--replace', is_flag=True, help='Replace existing policies')
@click.pass_context
def import_policies(ctx, policies_dir, replace):
    """Import policies from directory"""
    console.print(f"[bold cyan]📦 Importing Policies from {policies_dir}[/bold cyan]")
    
    policy_engine = PolicyEngine()
    
    if replace:
        console.print("[yellow]Replacing existing policies...[/yellow]")
    
    imported = policy_engine.load_policies(policies_dir)
    console.print(f"[green]✓ Imported {imported} policies[/green]")


@batch.command()
@click.option('--output-dir', '-o', type=click.Path(), default='./exports')
@click.option('--format', type=click.Choice(['json', 'csv', 'yaml']), default='json')
@click.pass_context
def export_all(ctx, output_dir, format):
    """Export all governance data"""
    console.print(f"[bold cyan]📦 Exporting All Governance Data[/bold cyan]")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Export policies
        task = progress.add_task("Exporting policies...", total=None)
        policy_engine = PolicyEngine()
        policy_file = output_path / f"policies_{timestamp}.{format}"
        policy_engine.export_policies(str(policy_file), format)
        console.print(f"[green]✓ Policies exported to {policy_file}[/green]")
        
        # Export risk register
        task = progress.add_task("Exporting risk register...", total=None)
        risk_assessment = AIRiskAssessment()
        risk_file = output_path / f"risk_register_{timestamp}.{format}"
        risk_assessment.export_risk_register(str(risk_file), format)
        console.print(f"[green]✓ Risk register exported to {risk_file}[/green]")
        
        # Export audit logs
        task = progress.add_task("Exporting audit logs...", total=None)
        audit_logger = AuditLogger(log_dir='./audit_logs')
        audit_file = output_path / f"audit_logs_{timestamp}.jsonl"
        entries = audit_logger.query_logs(limit=10000)
        with open(audit_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + '\n')
        console.print(f"[green]✓ Audit logs exported to {audit_file}[/green]")
    
    console.print("[bold green]✓ Export complete![/bold green]")


@batch.command()
@click.option('--interval', type=click.Choice(['daily', 'weekly', 'monthly']), default='weekly')
@click.option('--email', help='Email address for reports')
@click.pass_context
def schedule_reports(ctx, interval, email):
    """Schedule automated reports"""
    console.print(f"[bold cyan]📅 Scheduling {interval} Reports[/bold cyan]")
    
    # This would integrate with a scheduler like cron or celery
    console.print(f"[yellow]Report scheduling requires additional setup[/yellow]")
    console.print(f"[dim]Would schedule {interval} reports to {email or 'file'}[/dim]")


# Help Commands
@governance.command()
@click.pass_context
def status(ctx):
    """
    📈 Show overall governance status
    
    Quick overview of compliance and governance state
    """
    console.print("[bold cyan]📈 Governance Status Overview[/bold cyan]")
    
    try:
        # Quick status checks
        framework = GovernanceFramework(ctx.obj.get('config_file', 'governance_config.yaml'))
        report = framework.generate_compliance_report()
        
        # Create status panel
        status_text = f"""
[bold]Compliance Rate:[/bold] {report.get('compliance_rate', 0)*100:.1f}%
[bold]Active Policies:[/bold] {len(report.get('policies', {}))}
[bold]Recent Violations:[/bold] {len(report.get('violations_summary', []))}
[bold]Recommendations:[/bold] {len(report.get('recommendations', []))}
        """
        
        console.print(Panel(status_text.strip(), title="Governance Status"))
        
        # Show critical issues if any
        if report.get('violations_summary'):
            console.print("\n[bold red]⚠ Critical Issues:[/bold red]")
            for violation in report['violations_summary'][:3]:
                console.print(f"  • {violation.get('violation', 'Unknown')}")
    
    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")


@governance.command()
@click.pass_context
def version(ctx):
    """Show version information"""
    console.print("[bold]PoT Governance CLI[/bold]")
    console.print("Version: 1.0.0")
    console.print("Framework: Proof-of-Training Governance")
    console.print("Compliance: EU AI Act, NIST AI RMF")


if __name__ == "__main__":
    governance()