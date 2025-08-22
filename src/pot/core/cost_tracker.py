#!/usr/bin/env python3
"""
Cost Tracking Module for PoT Experiments
Tracks computational costs, API usage, and resource consumption
"""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from enum import Enum


class ResourceType(Enum):
    """Types of computational resources"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALL = "api_call"


class CostModel(Enum):
    """Predefined cost models for different cloud providers"""
    AWS_ON_DEMAND = "aws_on_demand"
    AWS_SPOT = "aws_spot"
    GCP_STANDARD = "gcp_standard"
    AZURE_STANDARD = "azure_standard"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ResourceUsage:
    """Record of resource usage"""
    resource_type: ResourceType
    amount: float
    unit: str
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class APICost:
    """Cost for API calls"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    num_calls: int
    total_cost: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ComputeCost:
    """Cost for computational resources"""
    resource_type: ResourceType
    instance_type: str
    hours_used: float
    cost_per_hour: float
    total_cost: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CostTracker:
    """
    Comprehensive cost tracking for PoT experiments
    Tracks both API costs and computational resource costs
    """
    
    # Default cost models ($/hour)
    DEFAULT_COSTS = {
        CostModel.AWS_ON_DEMAND: {
            "cpu": {"t3.medium": 0.0416, "c5.xlarge": 0.17, "c5.2xlarge": 0.34},
            "gpu": {"p3.2xlarge": 3.06, "p3.8xlarge": 12.24, "v100": 3.06, "a100": 4.10}
        },
        CostModel.GCP_STANDARD: {
            "cpu": {"n1-standard-4": 0.19, "n1-standard-8": 0.38},
            "gpu": {"nvidia-tesla-v100": 2.48, "nvidia-tesla-a100": 3.67}
        },
        CostModel.LOCAL: {
            "cpu": {"local": 0.01},  # Minimal cost for local compute
            "gpu": {"local": 0.05}    # Estimated electricity cost
        }
    }
    
    # API pricing (per 1K tokens)
    API_PRICING = {
        "openai": {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        },
        "anthropic": {
            "claude-2": {"input": 0.008, "output": 0.024},
            "claude-instant": {"input": 0.0008, "output": 0.0024}
        },
        "google": {
            "palm-2": {"input": 0.0005, "output": 0.0015},
            "gemini-pro": {"input": 0.00025, "output": 0.0005}
        }
    }
    
    def __init__(self, 
                 cost_model: CostModel = CostModel.LOCAL,
                 custom_costs: Optional[Dict] = None,
                 storage_path: str = "cost_tracking.db",
                 budget_limit: Optional[float] = None):
        """
        Initialize cost tracker
        
        Args:
            cost_model: Predefined cost model to use
            custom_costs: Custom cost definitions
            storage_path: Path to SQLite database for cost tracking
            budget_limit: Optional budget limit to enforce
        """
        self.cost_model = cost_model
        self.custom_costs = custom_costs or {}
        self.storage_path = Path(storage_path)
        self.budget_limit = budget_limit
        
        # Current session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        self.total_cost = 0.0
        self.api_costs = []
        self.compute_costs = []
        self.resource_usage = []
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for cost tracking"""
        with sqlite3.connect(self.storage_path) as conn:
            # Resource usage table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    resource_type TEXT,
                    amount REAL,
                    unit TEXT,
                    duration_seconds REAL,
                    timestamp TEXT
                )
            """)
            
            # API costs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    provider TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    num_calls INTEGER,
                    total_cost REAL,
                    timestamp TEXT
                )
            """)
            
            # Compute costs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compute_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    resource_type TEXT,
                    instance_type TEXT,
                    hours_used REAL,
                    cost_per_hour REAL,
                    total_cost REAL,
                    timestamp TEXT
                )
            """)
            
            # Session summary table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summary (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_cost REAL,
                    api_cost REAL,
                    compute_cost REAL,
                    peak_memory_gb REAL,
                    total_gpu_hours REAL,
                    total_cpu_hours REAL
                )
            """)
    
    def track_resource_usage(self, 
                           resource_type: ResourceType,
                           amount: float,
                           unit: str = "unit",
                           duration_seconds: float = 0):
        """Track resource usage"""
        usage = ResourceUsage(
            resource_type=resource_type,
            amount=amount,
            unit=unit,
            duration_seconds=duration_seconds
        )
        self.resource_usage.append(usage)
        
        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT INTO resource_usage 
                (session_id, resource_type, amount, unit, duration_seconds, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                resource_type.value,
                amount,
                unit,
                duration_seconds,
                usage.timestamp
            ))
    
    def track_api_call(self,
                      provider: str,
                      model: str,
                      input_tokens: int,
                      output_tokens: int,
                      num_calls: int = 1) -> float:
        """
        Track API call and calculate cost
        
        Returns:
            Total cost for the API call
        """
        # Calculate cost
        if provider in self.API_PRICING and model in self.API_PRICING[provider]:
            pricing = self.API_PRICING[provider][model]
            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]
            total_cost = (input_cost + output_cost) * num_calls
        else:
            # Use custom pricing or default
            total_cost = self.custom_costs.get(f"{provider}_{model}", 0.001) * num_calls
        
        api_cost = APICost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            num_calls=num_calls,
            total_cost=total_cost
        )
        
        self.api_costs.append(api_cost)
        self.total_cost += total_cost
        
        # Check budget
        if self.budget_limit and self.total_cost > self.budget_limit:
            raise BudgetExceededException(
                f"Budget limit of ${self.budget_limit:.2f} exceeded. "
                f"Current total: ${self.total_cost:.2f}"
            )
        
        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT INTO api_costs
                (session_id, provider, model, input_tokens, output_tokens, 
                 num_calls, total_cost, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                provider,
                model,
                input_tokens,
                output_tokens,
                num_calls,
                total_cost,
                api_cost.timestamp
            ))
        
        return total_cost
    
    def track_compute_time(self,
                          resource_type: ResourceType,
                          instance_type: str,
                          duration_seconds: float) -> float:
        """
        Track computational resource usage and calculate cost
        
        Returns:
            Total cost for the compute time
        """
        hours_used = duration_seconds / 3600
        
        # Get cost per hour
        if self.cost_model in self.DEFAULT_COSTS:
            cost_model = self.DEFAULT_COSTS[self.cost_model]
            resource_costs = cost_model.get(resource_type.value, {})
            cost_per_hour = resource_costs.get(instance_type, 0.01)
        else:
            cost_per_hour = self.custom_costs.get(f"{resource_type.value}_{instance_type}", 0.01)
        
        total_cost = hours_used * cost_per_hour
        
        compute_cost = ComputeCost(
            resource_type=resource_type,
            instance_type=instance_type,
            hours_used=hours_used,
            cost_per_hour=cost_per_hour,
            total_cost=total_cost
        )
        
        self.compute_costs.append(compute_cost)
        self.total_cost += total_cost
        
        # Check budget
        if self.budget_limit and self.total_cost > self.budget_limit:
            raise BudgetExceededException(
                f"Budget limit of ${self.budget_limit:.2f} exceeded. "
                f"Current total: ${self.total_cost:.2f}"
            )
        
        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT INTO compute_costs
                (session_id, resource_type, instance_type, hours_used, 
                 cost_per_hour, total_cost, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                resource_type.value,
                instance_type,
                hours_used,
                cost_per_hour,
                total_cost,
                compute_cost.timestamp
            ))
        
        return total_cost
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session costs"""
        api_total = sum(c.total_cost for c in self.api_costs)
        compute_total = sum(c.total_cost for c in self.compute_costs)
        
        # Calculate resource totals
        gpu_hours = sum(
            c.hours_used for c in self.compute_costs 
            if c.resource_type == ResourceType.GPU
        )
        cpu_hours = sum(
            c.hours_used for c in self.compute_costs
            if c.resource_type == ResourceType.CPU
        )
        
        # Peak memory from resource usage
        memory_usage = [
            u.amount for u in self.resource_usage
            if u.resource_type == ResourceType.MEMORY
        ]
        peak_memory_gb = max(memory_usage) / 1024 if memory_usage else 0
        
        duration = time.time() - self.session_start
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "total_cost": self.total_cost,
            "api_cost": api_total,
            "compute_cost": compute_total,
            "num_api_calls": sum(c.num_calls for c in self.api_costs),
            "total_tokens": sum(c.input_tokens + c.output_tokens for c in self.api_costs),
            "gpu_hours": gpu_hours,
            "cpu_hours": cpu_hours,
            "peak_memory_gb": peak_memory_gb,
            "cost_per_hour": self.total_cost / (duration / 3600) if duration > 0 else 0,
            "budget_remaining": self.budget_limit - self.total_cost if self.budget_limit else None
        }
    
    def save_session_summary(self):
        """Save session summary to database"""
        summary = self.get_session_summary()
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO session_summary
                (session_id, start_time, end_time, total_cost, api_cost, 
                 compute_cost, peak_memory_gb, total_gpu_hours, total_cpu_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                datetime.fromtimestamp(self.session_start).isoformat(),
                datetime.now().isoformat(),
                summary["total_cost"],
                summary["api_cost"],
                summary["compute_cost"],
                summary["peak_memory_gb"],
                summary["gpu_hours"],
                summary["cpu_hours"]
            ))
    
    def get_historical_costs(self, 
                           days_back: int = 30) -> Dict[str, Any]:
        """Get historical cost data"""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.storage_path) as conn:
            # Get session summaries
            cursor = conn.execute("""
                SELECT * FROM session_summary
                WHERE start_time >= ?
                ORDER BY start_time DESC
            """, (cutoff_date,))
            
            sessions = []
            for row in cursor:
                sessions.append(dict(zip([col[0] for col in cursor.description], row)))
            
            # Calculate aggregates
            if sessions:
                total_cost = sum(s["total_cost"] for s in sessions)
                total_api = sum(s["api_cost"] for s in sessions)
                total_compute = sum(s["compute_cost"] for s in sessions)
                total_gpu_hours = sum(s["total_gpu_hours"] for s in sessions)
                total_cpu_hours = sum(s["total_cpu_hours"] for s in sessions)
            else:
                total_cost = total_api = total_compute = 0
                total_gpu_hours = total_cpu_hours = 0
            
            return {
                "period_days": days_back,
                "num_sessions": len(sessions),
                "total_cost": total_cost,
                "api_cost": total_api,
                "compute_cost": total_compute,
                "total_gpu_hours": total_gpu_hours,
                "total_cpu_hours": total_cpu_hours,
                "avg_cost_per_session": total_cost / len(sessions) if sessions else 0,
                "sessions": sessions
            }
    
    def generate_cost_report(self, output_path: str = "cost_report.json"):
        """Generate comprehensive cost report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "current_session": self.get_session_summary(),
            "last_30_days": self.get_historical_costs(30),
            "last_7_days": self.get_historical_costs(7),
            "cost_breakdown": {
                "by_provider": self._get_cost_by_provider(),
                "by_model": self._get_cost_by_model(),
                "by_resource": self._get_cost_by_resource()
            },
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _get_cost_by_provider(self) -> Dict[str, float]:
        """Get cost breakdown by API provider"""
        costs = {}
        for api_cost in self.api_costs:
            if api_cost.provider not in costs:
                costs[api_cost.provider] = 0
            costs[api_cost.provider] += api_cost.total_cost
        return costs
    
    def _get_cost_by_model(self) -> Dict[str, float]:
        """Get cost breakdown by model"""
        costs = {}
        for api_cost in self.api_costs:
            model_key = f"{api_cost.provider}/{api_cost.model}"
            if model_key not in costs:
                costs[model_key] = 0
            costs[model_key] += api_cost.total_cost
        return costs
    
    def _get_cost_by_resource(self) -> Dict[str, float]:
        """Get cost breakdown by resource type"""
        costs = {}
        for compute_cost in self.compute_costs:
            resource = compute_cost.resource_type.value
            if resource not in costs:
                costs[resource] = 0
            costs[resource] += compute_cost.total_cost
        return costs
    
    def _generate_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check API usage patterns
        if self.api_costs:
            avg_tokens = sum(c.input_tokens + c.output_tokens for c in self.api_costs) / len(self.api_costs)
            if avg_tokens > 1000:
                recommendations.append(
                    "Consider using shorter prompts or caching responses for repeated queries"
                )
            
            # Check for expensive models
            expensive_models = [c for c in self.api_costs if "gpt-4" in c.model]
            if expensive_models:
                recommendations.append(
                    f"Consider using GPT-3.5 instead of GPT-4 where possible "
                    f"(potential savings: ${sum(c.total_cost * 0.9 for c in expensive_models):.2f})"
                )
        
        # Check compute usage
        if self.compute_costs:
            gpu_costs = [c for c in self.compute_costs if c.resource_type == ResourceType.GPU]
            if gpu_costs:
                gpu_total = sum(c.total_cost for c in gpu_costs)
                if gpu_total > self.total_cost * 0.7:
                    recommendations.append(
                        "GPU costs dominate (>70%). Consider using spot instances or "
                        "optimizing batch sizes for better GPU utilization"
                    )
        
        # Check for inefficient resource usage
        if self.resource_usage:
            memory_usage = [u for u in self.resource_usage if u.resource_type == ResourceType.MEMORY]
            if memory_usage:
                avg_memory = sum(u.amount for u in memory_usage) / len(memory_usage)
                peak_memory = max(u.amount for u in memory_usage)
                if peak_memory > avg_memory * 2:
                    recommendations.append(
                        "Memory usage is spiky. Consider optimizing memory allocation "
                        "or using gradient checkpointing"
                    )
        
        if not recommendations:
            recommendations.append("Cost usage appears optimal")
        
        return recommendations
    
    def export_to_csv(self, output_dir: str = "."):
        """Export cost data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export API costs
        if self.api_costs:
            import csv
            with open(output_path / "api_costs.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.api_costs[0]).keys())
                writer.writeheader()
                for cost in self.api_costs:
                    writer.writerow(asdict(cost))
        
        # Export compute costs
        if self.compute_costs:
            import csv
            with open(output_path / "compute_costs.csv", 'w', newline='') as f:
                fieldnames = ["resource_type", "instance_type", "hours_used", 
                             "cost_per_hour", "total_cost", "timestamp"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for cost in self.compute_costs:
                    row = asdict(cost)
                    row["resource_type"] = row["resource_type"].value
                    writer.writerow(row)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save session summary"""
        self.save_session_summary()


class BudgetExceededException(Exception):
    """Exception raised when budget limit is exceeded"""
    pass


# Convenience functions
def estimate_experiment_cost(
    num_models: int,
    num_challenges: int,
    avg_tokens_per_challenge: int,
    model_type: str = "gpt-3.5-turbo",
    include_compute: bool = True,
    gpu_hours: float = 0
) -> Dict[str, float]:
    """
    Estimate cost for an experiment
    
    Returns:
        Dictionary with cost breakdown
    """
    tracker = CostTracker()
    
    # API costs
    api_cost = 0
    if model_type in tracker.API_PRICING.get("openai", {}):
        pricing = tracker.API_PRICING["openai"][model_type]
        tokens_total = num_models * num_challenges * avg_tokens_per_challenge
        api_cost = (tokens_total / 1000) * pricing["output"]
    
    # Compute costs
    compute_cost = 0
    if include_compute:
        if gpu_hours > 0:
            # Assume V100 GPU
            compute_cost = gpu_hours * tracker.DEFAULT_COSTS[CostModel.AWS_ON_DEMAND]["gpu"]["v100"]
        else:
            # Assume CPU instance
            cpu_hours = num_models * num_challenges * 0.001  # ~3.6s per challenge
            compute_cost = cpu_hours * tracker.DEFAULT_COSTS[CostModel.AWS_ON_DEMAND]["cpu"]["c5.xlarge"]
    
    return {
        "api_cost": api_cost,
        "compute_cost": compute_cost,
        "total_cost": api_cost + compute_cost,
        "cost_per_model": (api_cost + compute_cost) / num_models if num_models > 0 else 0,
        "cost_per_challenge": (api_cost + compute_cost) / (num_models * num_challenges) 
                              if num_models * num_challenges > 0 else 0
    }