"""
Common CLI utilities shared across the PoT framework.
Provides formatters, argument parsers, and output helpers.
"""

import json
import yaml
import click
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
import sys
import os
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class OutputFormat(Enum):
    """Output format options"""
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    MARKDOWN = "markdown"


@dataclass
class CLIConfig:
    """Configuration for CLI utilities"""
    output_format: OutputFormat = OutputFormat.TEXT
    color: bool = True
    verbose: bool = False
    quiet: bool = False
    log_level: str = "INFO"
    output_file: Optional[Path] = None
    timestamp: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['output_format'] = self.output_format.value
        return data


class CLIFormatter:
    """
    Formatter for CLI output.
    Consolidates formatting logic from multiple CLI modules.
    """
    
    def __init__(self, config: Optional[CLIConfig] = None):
        """
        Initialize CLI formatter.
        
        Args:
            config: CLI configuration
        """
        self.config = config or CLIConfig()
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if self.config.timestamp else '%(name)s - %(levelname)s - %(message)s'
        )
    
    def format_output(self, data: Any, format: Optional[OutputFormat] = None) -> str:
        """
        Format data for output.
        
        Args:
            data: Data to format
            format: Output format (uses config default if None)
            
        Returns:
            Formatted string
        """
        format = format or self.config.output_format
        
        if format == OutputFormat.JSON:
            return json.dumps(data, indent=2, default=str)
        
        elif format == OutputFormat.YAML:
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        
        elif format == OutputFormat.CSV:
            return self._format_csv(data)
        
        elif format == OutputFormat.MARKDOWN:
            return self._format_markdown(data)
        
        else:  # TEXT
            return self._format_text(data)
    
    def _format_text(self, data: Any) -> str:
        """Format as plain text"""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    lines.append(f"{key}:")
                    lines.append(self._indent(self._format_text(value)))
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
        
        elif isinstance(data, list):
            return "\n".join(f"- {self._format_text(item)}" for item in data)
        
        else:
            return str(data)
    
    def _format_csv(self, data: Any) -> str:
        """Format as CSV"""
        import csv
        import io
        
        if not isinstance(data, list):
            data = [data]
        
        if not data:
            return ""
        
        # Get headers from first item
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
        else:
            headers = ["value"]
            data = [{"value": item} for item in data]
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def _format_markdown(self, data: Any) -> str:
        """Format as Markdown"""
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    lines.append(f"**{key}:**")
                    lines.append("")
                    lines.append(self._format_markdown(value))
                else:
                    lines.append(f"- **{key}:** {value}")
            return "\n".join(lines)
        
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                # Format as table
                headers = list(data[0].keys())
                lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |"
                ]
                for item in data:
                    row = [str(item.get(h, "")) for h in headers]
                    lines.append("| " + " | ".join(row) + " |")
                return "\n".join(lines)
            else:
                return "\n".join(f"- {item}" for item in data)
        
        else:
            return str(data)
    
    def _indent(self, text: str, level: int = 2) -> str:
        """Indent text"""
        indent = " " * level
        return "\n".join(indent + line for line in text.split("\n"))
    
    def print(self, data: Any, format: Optional[OutputFormat] = None):
        """
        Print formatted data.
        
        Args:
            data: Data to print
            format: Output format
        """
        if self.config.quiet:
            return
        
        output = self.format_output(data, format)
        
        if self.config.output_file:
            with open(self.config.output_file, 'w') as f:
                f.write(output)
        else:
            if self.config.color and sys.stdout.isatty():
                output = self._colorize(output)
            print(output)
    
    def _colorize(self, text: str) -> str:
        """Add color to text output"""
        # Simple colorization for common patterns
        text = text.replace("SUCCESS", click.style("SUCCESS", fg="green", bold=True))
        text = text.replace("FAILED", click.style("FAILED", fg="red", bold=True))
        text = text.replace("WARNING", click.style("WARNING", fg="yellow", bold=True))
        text = text.replace("INFO", click.style("INFO", fg="blue"))
        
        # Colorize numbers
        import re
        text = re.sub(r'\b(\d+\.?\d*)\b', lambda m: click.style(m.group(1), fg="cyan"), text)
        
        return text
    
    def success(self, message: str):
        """Print success message"""
        if not self.config.quiet:
            click.echo(click.style(f"✓ {message}", fg="green"))
    
    def error(self, message: str):
        """Print error message"""
        click.echo(click.style(f"✗ {message}", fg="red"), err=True)
    
    def warning(self, message: str):
        """Print warning message"""
        if not self.config.quiet:
            click.echo(click.style(f"⚠ {message}", fg="yellow"))
    
    def info(self, message: str):
        """Print info message"""
        if self.config.verbose and not self.config.quiet:
            click.echo(click.style(f"ℹ {message}", fg="blue"))


def add_common_arguments(func: Callable) -> Callable:
    """
    Decorator to add common CLI arguments to a Click command.
    
    Args:
        func: Click command function
        
    Returns:
        Decorated function
    """
    decorators = [
        click.option('--output-format', '-f',
                    type=click.Choice(['text', 'json', 'yaml', 'csv', 'markdown']),
                    default='text',
                    help='Output format'),
        click.option('--output', '-o',
                    type=click.Path(),
                    help='Output file path'),
        click.option('--verbose', '-v',
                    is_flag=True,
                    help='Verbose output'),
        click.option('--quiet', '-q',
                    is_flag=True,
                    help='Suppress output'),
        click.option('--no-color',
                    is_flag=True,
                    help='Disable colored output'),
        click.option('--config', '-c',
                    type=click.Path(exists=True),
                    help='Configuration file'),
        click.option('--log-level',
                    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
                    default='INFO',
                    help='Logging level')
    ]
    
    for decorator in reversed(decorators):
        func = decorator(func)
    
    return func


def format_results(results: Dict[str, Any], 
                  format: OutputFormat = OutputFormat.TEXT,
                  include_metadata: bool = True) -> str:
    """
    Format verification/experiment results.
    
    Args:
        results: Results dictionary
        format: Output format
        include_metadata: Whether to include metadata
        
    Returns:
        Formatted results string
    """
    # Prepare output data
    output = {}
    
    # Main results
    if 'verdict' in results:
        output['Verdict'] = results['verdict']
    if 'confidence' in results:
        output['Confidence'] = f"{results['confidence']:.2%}"
    if 'elapsed_time' in results:
        output['Execution Time'] = f"{results['elapsed_time']:.3f}s"
    
    # Statistics
    if 'statistics' in results:
        stats = results['statistics']
        output['Statistics'] = {
            'Samples': stats.get('n_samples', 'N/A'),
            'Mean Difference': f"{stats.get('mean_diff', 0):.6f}",
            'Std Deviation': f"{stats.get('std_diff', 0):.6f}",
            'Effect Size': f"{stats.get('effect_size', 0):.4f}"
        }
    
    # Metadata
    if include_metadata and 'metadata' in results:
        output['Metadata'] = results['metadata']
    
    # Errors or warnings
    if 'errors' in results:
        output['Errors'] = results['errors']
    if 'warnings' in results:
        output['Warnings'] = results['warnings']
    
    # Format output
    formatter = CLIFormatter()
    return formatter.format_output(output, format)


def create_progress_bar(total: int, 
                       description: str = "Processing",
                       unit: str = "items",
                       disable: bool = False) -> 'ProgressBar':
    """
    Create a progress bar for CLI operations.
    
    Args:
        total: Total number of items
        description: Progress bar description
        unit: Unit name for items
        disable: Whether to disable the progress bar
        
    Returns:
        Progress bar instance
    """
    return ProgressBar(total, description, unit, disable)


class ProgressBar:
    """
    Progress bar for CLI operations.
    Wraps click.progressbar with additional features.
    """
    
    def __init__(self, total: int, description: str = "Processing", 
                unit: str = "items", disable: bool = False):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items
            description: Progress bar description
            unit: Unit name
            disable: Whether to disable
        """
        self.total = total
        self.description = description
        self.unit = unit
        self.disable = disable
        self.current = 0
        
        if not disable:
            self.bar = click.progressbar(
                length=total,
                label=description,
                show_eta=True,
                show_percent=True,
                show_pos=True,
                item_show_func=lambda x: f"{x} {unit}" if x else ""
            )
            self.bar.__enter__()
    
    def update(self, n: int = 1):
        """Update progress by n items"""
        self.current += n
        if not self.disable:
            self.bar.update(n)
    
    def set_description(self, description: str):
        """Update progress bar description"""
        self.description = description
        if not self.disable and hasattr(self.bar, 'label'):
            self.bar.label = description
    
    def close(self):
        """Close progress bar"""
        if not self.disable:
            self.bar.__exit__(None, None, None)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def load_config(path: Optional[Path] = None, 
               defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        path: Configuration file path
        defaults: Default values
        
    Returns:
        Configuration dictionary
    """
    config = defaults or {}
    
    if path:
        path = Path(path)
        if path.suffix in ['.json', '.jsonc']:
            with open(path) as f:
                file_config = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path) as f:
                file_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Merge with defaults
        config.update(file_config)
    
    # Check environment variables
    for key in config.keys():
        env_key = f"POT_{key.upper()}"
        if env_key in os.environ:
            config[key] = os.environ[env_key]
    
    return config


def validate_arguments(**kwargs) -> Dict[str, Any]:
    """
    Validate and normalize CLI arguments.
    
    Args:
        **kwargs: Arguments to validate
        
    Returns:
        Validated arguments
        
    Raises:
        click.BadParameter: If validation fails
    """
    validated = {}
    
    for key, value in kwargs.items():
        if value is None:
            continue
        
        # Validate file paths
        if key.endswith('_file') or key.endswith('_path'):
            path = Path(value)
            if key.startswith('output'):
                # Create parent directory for output files
                path.parent.mkdir(parents=True, exist_ok=True)
            elif not path.exists():
                raise click.BadParameter(f"File not found: {path}")
            validated[key] = path
        
        # Validate numeric ranges
        elif key in ['confidence', 'threshold']:
            if not 0 <= value <= 1:
                raise click.BadParameter(f"{key} must be between 0 and 1")
            validated[key] = value
        
        elif key in ['samples', 'iterations', 'epochs']:
            if value <= 0:
                raise click.BadParameter(f"{key} must be positive")
            validated[key] = value
        
        # Validate device
        elif key == 'device':
            import torch
            if value != 'cpu' and not torch.cuda.is_available():
                click.echo(f"Warning: CUDA not available, using CPU instead", err=True)
                validated[key] = 'cpu'
            else:
                validated[key] = value
        
        else:
            validated[key] = value
    
    return validated


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.
    
    Args:
        message: Confirmation message
        default: Default response
        
    Returns:
        User's response
    """
    return click.confirm(message, default=default)


def select_option(options: List[str], 
                 prompt: str = "Select an option",
                 default: Optional[int] = None) -> str:
    """
    Present options and let user select.
    
    Args:
        options: List of options
        prompt: Selection prompt
        default: Default option index
        
    Returns:
        Selected option
    """
    for i, option in enumerate(options, 1):
        click.echo(f"{i}. {option}")
    
    while True:
        try:
            choice = click.prompt(prompt, type=int, default=default)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                click.echo(f"Please select a number between 1 and {len(options)}")
        except (ValueError, click.Abort):
            if default is not None:
                return options[default - 1]
            raise


def format_table(headers: List[str], 
                rows: List[List[Any]],
                align: Optional[List[str]] = None) -> str:
    """
    Format data as an ASCII table.
    
    Args:
        headers: Column headers
        rows: Data rows
        align: Column alignments ('l', 'r', 'c')
        
    Returns:
        Formatted table string
    """
    if not rows:
        return "No data"
    
    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Default alignment
    if align is None:
        align = ['l'] * len(headers)
    
    # Format functions based on alignment
    formatters = []
    for i, a in enumerate(align):
        if a == 'r':
            formatters.append(lambda x, w=widths[i]: str(x).rjust(w))
        elif a == 'c':
            formatters.append(lambda x, w=widths[i]: str(x).center(w))
        else:
            formatters.append(lambda x, w=widths[i]: str(x).ljust(w))
    
    # Build table
    lines = []
    
    # Header
    header_row = " | ".join(formatters[i](h) for i, h in enumerate(headers))
    lines.append(header_row)
    lines.append("-" * len(header_row))
    
    # Data rows
    for row in rows:
        lines.append(" | ".join(formatters[i](cell) for i, cell in enumerate(row)))
    
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"