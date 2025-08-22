"""
Artifact Manager for ZK Benchmarks

Manages storage, retrieval, and comparison of benchmark artifacts.
"""

import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BenchmarkArtifact:
    """Container for benchmark artifacts"""
    artifact_id: str
    timestamp: float
    benchmark_type: str
    circuit_size: str
    results: Dict[str, Any]
    system_info: Dict[str, Any]
    git_commit: Optional[str] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkArtifact':
        return cls(**data)


class ArtifactManager:
    """Manage benchmark artifacts and history"""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize artifact manager"""
        self.storage_dir = storage_dir or Path("benchmarks/artifacts")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_dir / "benchmark_history.db"
        self._init_database()
        
        self.comparison_dir = self.storage_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                artifact_id TEXT PRIMARY KEY,
                timestamp REAL,
                benchmark_type TEXT,
                circuit_size TEXT,
                mean_time REAL,
                peak_memory_mb REAL,
                throughput REAL,
                git_commit TEXT,
                tags TEXT,
                results_json TEXT,
                system_info_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmarks(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_type_size ON benchmarks(benchmark_type, circuit_size)
        """)
        
        conn.commit()
        conn.close()
    
    def store_artifact(
        self,
        results: List[Dict[str, Any]],
        git_commit: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store benchmark results as artifact"""
        
        # Generate artifact ID
        artifact_id = self._generate_artifact_id()
        timestamp = time.time()
        
        # Get git commit if not provided
        if git_commit is None:
            git_commit = self._get_git_commit()
        
        # Store raw results file
        artifact_file = self.storage_dir / f"artifact_{artifact_id}.json"
        artifact_data = {
            'artifact_id': artifact_id,
            'timestamp': timestamp,
            'git_commit': git_commit,
            'tags': tags or [],
            'results': results
        }
        
        with open(artifact_file, 'w') as f:
            json.dump(artifact_data, f, indent=2)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute("""
                INSERT INTO benchmarks (
                    artifact_id, timestamp, benchmark_type, circuit_size,
                    mean_time, peak_memory_mb, throughput, git_commit, tags,
                    results_json, system_info_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact_id,
                timestamp,
                result.get('benchmark_type', ''),
                result.get('circuit_size', ''),
                result.get('mean_time', 0),
                result.get('peak_memory_mb', 0),
                result.get('throughput_proofs_per_hour', 0),
                git_commit,
                json.dumps(tags or []),
                json.dumps(result),
                json.dumps(result.get('system_info', {}))
            ))
        
        conn.commit()
        conn.close()
        
        print(f"Artifact stored: {artifact_id}")
        return artifact_id
    
    def retrieve_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact by ID"""
        artifact_file = self.storage_dir / f"artifact_{artifact_id}.json"
        
        if artifact_file.exists():
            with open(artifact_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_artifacts(
        self,
        benchmark_type: Optional[str] = None,
        circuit_size: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List available artifacts"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT DISTINCT artifact_id, timestamp, benchmark_type, circuit_size, git_commit, tags FROM benchmarks"
        conditions = []
        params = []
        
        if benchmark_type:
            conditions.append("benchmark_type = ?")
            params.append(benchmark_type)
        
        if circuit_size:
            conditions.append("circuit_size = ?")
            params.append(circuit_size)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df.to_dict('records')
    
    def compare_artifacts(
        self,
        artifact_id1: str,
        artifact_id2: str,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """Compare two benchmark artifacts"""
        
        # Retrieve artifacts
        artifact1 = self.retrieve_artifact(artifact_id1)
        artifact2 = self.retrieve_artifact(artifact_id2)
        
        if not artifact1 or not artifact2:
            raise ValueError("One or both artifacts not found")
        
        comparison = {
            'artifact1': artifact_id1,
            'artifact2': artifact_id2,
            'timestamp1': artifact1['timestamp'],
            'timestamp2': artifact2['timestamp'],
            'improvements': {},
            'regressions': {}
        }
        
        # Compare results by type and size
        results1_map = self._create_results_map(artifact1['results'])
        results2_map = self._create_results_map(artifact2['results'])
        
        for key in results1_map:
            if key in results2_map:
                r1 = results1_map[key]
                r2 = results2_map[key]
                
                # Calculate performance changes
                time_change = (r2['mean_time'] - r1['mean_time']) / r1['mean_time'] * 100
                memory_change = (r2['peak_memory_mb'] - r1['peak_memory_mb']) / r1['peak_memory_mb'] * 100
                
                change_data = {
                    'time_change_pct': time_change,
                    'memory_change_pct': memory_change,
                    'old_time': r1['mean_time'],
                    'new_time': r2['mean_time'],
                    'old_memory': r1['peak_memory_mb'],
                    'new_memory': r2['peak_memory_mb']
                }
                
                if time_change < -5:  # 5% improvement threshold
                    comparison['improvements'][key] = change_data
                elif time_change > 5:  # 5% regression threshold
                    comparison['regressions'][key] = change_data
        
        # Generate comparison plots if requested
        if save_plots:
            self._generate_comparison_plots(artifact1, artifact2, comparison)
        
        return comparison
    
    def get_performance_history(
        self,
        benchmark_type: str,
        circuit_size: str,
        metric: str = 'mean_time',
        days: int = 30
    ) -> pd.DataFrame:
        """Get performance history for specific benchmark"""
        
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = time.time() - (days * 86400)
        
        query = f"""
            SELECT timestamp, {metric}, git_commit
            FROM benchmarks
            WHERE benchmark_type = ? AND circuit_size = ? AND timestamp > ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(benchmark_type, circuit_size, cutoff_time)
        )
        
        conn.close()
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def detect_regressions(
        self,
        threshold_pct: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Detect performance regressions in recent benchmarks"""
        
        regressions = []
        
        # Get unique benchmark types and sizes
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT benchmark_type, circuit_size
            FROM benchmarks
            WHERE timestamp > ?
        """, (time.time() - 7 * 86400,))  # Last 7 days
        
        for benchmark_type, circuit_size in cursor.fetchall():
            # Get recent history
            history = self.get_performance_history(
                benchmark_type,
                circuit_size,
                days=7
            )
            
            if len(history) < 2:
                continue
            
            # Check for regression
            baseline = history.iloc[:-1]['mean_time'].mean()
            latest = history.iloc[-1]['mean_time']
            
            change_pct = (latest - baseline) / baseline * 100
            
            if change_pct > threshold_pct:
                regressions.append({
                    'benchmark_type': benchmark_type,
                    'circuit_size': circuit_size,
                    'baseline_time': baseline,
                    'latest_time': latest,
                    'regression_pct': change_pct,
                    'git_commit': history.iloc[-1]['git_commit']
                })
        
        conn.close()
        return regressions
    
    def _generate_artifact_id(self) -> str:
        """Generate unique artifact ID"""
        timestamp = str(time.time())
        random_part = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{random_part}"
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()[:8]
        except:
            return None
    
    def _create_results_map(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create map of results by type and size"""
        results_map = {}
        for result in results:
            key = f"{result.get('benchmark_type', '')}_{result.get('circuit_size', '')}"
            results_map[key] = result
        return results_map
    
    def _generate_comparison_plots(
        self,
        artifact1: Dict[str, Any],
        artifact2: Dict[str, Any],
        comparison: Dict[str, Any]
    ):
        """Generate comparison visualization plots"""
        
        # Prepare data
        results1 = artifact1['results']
        results2 = artifact2['results']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Time comparison
        ax = axes[0, 0]
        circuit_sizes = []
        times1 = []
        times2 = []
        
        for r1, r2 in zip(results1, results2):
            if r1.get('benchmark_type') == 'proof_generation':
                circuit_sizes.append(r1.get('circuit_size', ''))
                times1.append(r1.get('mean_time', 0))
                times2.append(r2.get('mean_time', 0))
        
        if circuit_sizes:
            x = range(len(circuit_sizes))
            width = 0.35
            ax.bar([i - width/2 for i in x], times1, width, label='Artifact 1', alpha=0.8)
            ax.bar([i + width/2 for i in x], times2, width, label='Artifact 2', alpha=0.8)
            ax.set_xlabel('Circuit Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Proof Generation Time Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(circuit_sizes)
            ax.legend()
        
        # Plot 2: Memory comparison
        ax = axes[0, 1]
        memory1 = [r.get('peak_memory_mb', 0) for r in results1 if r.get('benchmark_type') == 'proof_generation']
        memory2 = [r.get('peak_memory_mb', 0) for r in results2 if r.get('benchmark_type') == 'proof_generation']
        
        if memory1 and memory2:
            ax.plot(memory1, 'o-', label='Artifact 1', alpha=0.8)
            ax.plot(memory2, 's-', label='Artifact 2', alpha=0.8)
            ax.set_xlabel('Test Index')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title('Memory Usage Comparison')
            ax.legend()
        
        # Plot 3: Improvements
        ax = axes[1, 0]
        if comparison['improvements']:
            improvements = list(comparison['improvements'].values())
            improvements_pct = [abs(i['time_change_pct']) for i in improvements]
            ax.bar(range(len(improvements_pct)), improvements_pct, color='green', alpha=0.7)
            ax.set_xlabel('Benchmark')
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Performance Improvements')
        
        # Plot 4: Regressions
        ax = axes[1, 1]
        if comparison['regressions']:
            regressions = list(comparison['regressions'].values())
            regressions_pct = [r['time_change_pct'] for r in regressions]
            ax.bar(range(len(regressions_pct)), regressions_pct, color='red', alpha=0.7)
            ax.set_xlabel('Benchmark')
            ax.set_ylabel('Regression (%)')
            ax.set_title('Performance Regressions')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.comparison_dir / f"comparison_{artifact1['artifact_id']}_{artifact2['artifact_id']}.png"
        plt.savefig(plot_file, dpi=100)
        plt.close()
        
        print(f"Comparison plot saved: {plot_file}")