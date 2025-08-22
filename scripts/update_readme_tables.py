#!/usr/bin/env python3
"""
README Table Updater

Automatically updates tables in README.md with live experimental data from successful pipeline runs.
"""

import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class ReadmeTableUpdater:
    """Updates README.md tables with live experimental data"""
    
    def __init__(self, repo_root: str = None):
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).parent.parent
        self.readme_path = self.repo_root / "README.md"
        self.experimental_results_dir = self.repo_root / "experimental_results"
        self.validation_reports_dir = self.repo_root / "outputs" / "validation_reports"
        self.rolling_metrics_path = self.experimental_results_dir / "rolling_metrics.json"
        
    def get_recent_successful_runs(self, days: int = 7) -> List[Dict]:
        """Get recent successful validation runs from multiple sources"""
        runs = []
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Check validation reports directory
        if self.validation_reports_dir.exists():
            for results_file in self.validation_reports_dir.glob("pipeline_results_*.json"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = results_file.stem.split('_')[-2:]  # e.g., ['20250822', '000850']
                    if len(timestamp_str) == 2:
                        timestamp = datetime.strptime('_'.join(timestamp_str), '%Y%m%d_%H%M%S')
                        
                        if timestamp >= cutoff_time:
                            with open(results_file, 'r') as f:
                                data = json.load(f)
                                if data.get('success'):
                                    data['timestamp'] = timestamp
                                    data['source'] = 'validation_reports'
                                    runs.append(data)
                except Exception as e:
                    logger.debug(f"Could not parse {results_file}: {e}")
        
        # Check experimental results for additional data
        if self.experimental_results_dir.exists():
            for results_file in self.experimental_results_dir.glob("*_validation_*.json"):
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and data.get('success'):
                            # Try to extract timestamp from filename or data
                            if 'timestamp' in data:
                                timestamp = datetime.fromisoformat(data['timestamp'])
                            else:
                                # Extract from filename pattern
                                match = re.search(r'(\d{8}_\d{6})', results_file.name)
                                if match:
                                    timestamp = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
                                else:
                                    timestamp = datetime.fromtimestamp(results_file.stat().st_mtime)
                            
                            if timestamp >= cutoff_time:
                                data['timestamp'] = timestamp
                                data['source'] = 'experimental_results'
                                runs.append(data)
                except Exception as e:
                    logger.debug(f"Could not parse {results_file}: {e}")
        
        # Sort by timestamp, newest first
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        return runs
    
    def extract_model_pair_info(self, run_data: Dict) -> Tuple[str, str, str]:
        """Extract model pair information from run data"""
        ref_model = "Unknown"
        cand_model = "Unknown"
        notes = ""
        
        # Try to extract from stage metrics
        if 'stage_metrics' in run_data and 'model_loading' in run_data['stage_metrics']:
            metadata = run_data['stage_metrics']['model_loading'].get('metadata', {})
            ref_model = metadata.get('ref_model', ref_model)
            cand_model = metadata.get('cand_model', cand_model)
        
        # Try to extract from run_id or other fields
        if 'run_id' in run_data:
            run_id = run_data['run_id']
            if 'gpt2' in run_id.lower() and 'distil' in run_id.lower():
                ref_model = "GPT-2"
                cand_model = "DistilGPT-2"
                notes = "Distillation"
            elif 'pythia' in run_id.lower():
                if '70m' in run_id.lower():
                    ref_model = "Pythia-70M"
                    cand_model = "Pythia-70M"
                    notes = "Self-consistency"
        
        # Clean up model names
        ref_model = self.format_model_name(ref_model)
        cand_model = self.format_model_name(cand_model)
        
        # Auto-detect notes based on model pair
        if not notes:
            notes = self.infer_validation_type(ref_model, cand_model, run_data)
        
        return ref_model, cand_model, notes
    
    def format_model_name(self, model_name: str) -> str:
        """Format model name for display"""
        if not model_name or model_name == "Unknown":
            return model_name
        
        # Handle common model name patterns
        model_name = model_name.replace('/', ' ')
        if model_name.startswith('gpt2'):
            return "GPT-2" if model_name == 'gpt2' else f"GPT-2-{model_name[4:]}"
        elif 'distilgpt2' in model_name.lower():
            return "DistilGPT-2"
        elif 'pythia' in model_name.lower():
            return model_name.replace('EleutherAI/', '').replace('pythia-', 'Pythia-')
        elif 'llama' in model_name.lower():
            return model_name.replace('meta-llama/', '').replace('Meta-Llama-', 'Llama-')
        elif 'mistral' in model_name.lower():
            return model_name.replace('mistralai/', '').replace('Mistral-', 'Mistral-')
        elif 'yi-' in model_name.lower():
            return model_name.replace('01-ai/', '')
        else:
            return model_name
    
    def infer_validation_type(self, ref_model: str, cand_model: str, run_data: Dict) -> str:
        """Infer the type of validation based on model pair and results"""
        decision = run_data.get('decision', 'UNKNOWN')
        
        if ref_model == cand_model:
            return "Self-consistency"
        elif 'distil' in cand_model.lower() or 'distil' in ref_model.lower():
            return "Distillation"
        elif 'chat' in cand_model.lower() or 'instruct' in cand_model.lower():
            return "Instruction tuning"
        elif 'yi' in ref_model.lower() and 'yi' in cand_model.lower():
            return "Sharded (34B-class)" if run_data.get('total_duration', 0) > 100 else "Architecture change"
        elif any(x in ref_model.lower() for x in ['llama', 'mistral', 'qwen']) and \
             any(x in cand_model.lower() for x in ['llama', 'mistral', 'qwen']):
            return "Architecture change"
        elif decision == "DIFFERENT":
            return "Behavioral difference"
        else:
            return "Model comparison"
    
    def generate_table_rows(self, runs: List[Dict], max_rows: int = 10) -> List[str]:
        """Generate table rows from run data"""
        rows = []
        seen_pairs = set()
        
        for run in runs[:max_rows * 2]:  # Get extra to account for filtering
            if len(rows) >= max_rows:
                break
                
            try:
                ref_model, cand_model, notes = self.extract_model_pair_info(run)
                
                # Skip duplicate pairs (keep most recent)
                pair_key = f"{ref_model}|{cand_model}"
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                # Extract metrics
                decision = run.get('decision', 'UNKNOWN')
                queries = run.get('n_queries', 0)
                total_time = run.get('total_duration', 0)
                per_query_time = total_time / queries if queries > 0 else 0
                
                # Format model pair for display
                if decision == "DIFFERENT":
                    model_pair = f"{ref_model} vs **{cand_model}**"
                else:
                    model_pair = f"**{ref_model}** vs **{cand_model}**"
                
                # Format timing
                if total_time > 60:
                    time_str = f"~{total_time:.0f} s"
                else:
                    time_str = f"~{total_time:.1f} s"
                
                per_query_str = f"~{per_query_time:.1f} s" if per_query_time > 0.1 else f"~{per_query_time:.2f} s"
                
                # Determine mode based on confidence and queries
                confidence = run.get('confidence', 0)
                if confidence >= 0.99:
                    mode = "Audit-grade"
                elif confidence >= 0.975:
                    mode = "Quick-gate"
                else:
                    mode = "Local-weights"
                
                row = f"| {model_pair} | {mode} | {decision} | {queries} | {time_str} | {per_query_str} | {notes} |"
                rows.append(row)
                
            except Exception as e:
                logger.debug(f"Could not process run: {e}")
                continue
        
        return rows
    
    def update_example_runs_table(self) -> bool:
        """Update the example runs table in README.md"""
        try:
            # Read current README
            with open(self.readme_path, 'r') as f:
                content = f.read()
            
            # Get recent runs
            runs = self.get_recent_successful_runs(days=30)  # Look back further for examples
            if not runs:
                logger.warning("No recent successful runs found")
                return False
            
            # Generate new table rows
            new_rows = self.generate_table_rows(runs, max_rows=8)
            if not new_rows:
                logger.warning("Could not generate any table rows")
                return False
            
            # Find the table in README
            table_pattern = re.compile(
                r'(\| Pair\s+\| Mode\s+\| Decision\s+\| Queries\s+\| Total Time\s+\| Per-Query\s+\| Notes\s+\|\s*\n'
                r'\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n)'
                r'((?:\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\n)+)',
                re.MULTILINE
            )
            
            match = table_pattern.search(content)
            if not match:
                logger.error("Could not find example runs table in README")
                return False
            
            # Build new table
            header = match.group(1)
            new_table_body = '\n'.join(new_rows) + '\n'
            
            # Add update timestamp comment
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            update_comment = f"\n<!-- Table auto-updated: {timestamp} -->\n"
            
            # Replace table
            new_content = content[:match.start()] + header + new_table_body + update_comment + content[match.end():]
            
            # Write back to README
            with open(self.readme_path, 'w') as f:
                f.write(new_content)
            
            logger.info(f"Updated README table with {len(new_rows)} recent runs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update README table: {e}")
            return False
    
    def update_rolling_metrics_summary(self) -> bool:
        """Update rolling metrics summary in README if it exists"""
        try:
            if not self.rolling_metrics_path.exists():
                return False
            
            with open(self.rolling_metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Calculate summary stats
            total_runs = metrics.get('total_runs', 0)
            successful_runs = metrics.get('successful_runs', 0)
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
            
            # Calculate average timing from recent samples
            timing_samples = metrics.get('timing_samples', [])
            if timing_samples:
                recent_samples = timing_samples[-10:]  # Last 10 runs
                avg_per_query = sum(s.get('t_per_query', 0) for s in recent_samples) / len(recent_samples)
                avg_total = sum(s.get('t_total', 0) for s in recent_samples) / len(recent_samples)
            else:
                avg_per_query = avg_total = 0
            
            # Update README with stats (if summary section exists)
            with open(self.readme_path, 'r') as f:
                content = f.read()
            
            # Look for a metrics summary section to update
            summary_pattern = re.compile(
                r'(<!-- METRICS_SUMMARY_START -->.*?<!-- METRICS_SUMMARY_END -->)',
                re.DOTALL
            )
            
            new_summary = f"""<!-- METRICS_SUMMARY_START -->
**Recent Performance Summary:**
- Total validation runs: {total_runs}
- Success rate: {success_rate:.1f}% ({successful_runs}/{total_runs})
- Average per-query time: {avg_per_query:.2f}s
- Average total time: {avg_total:.1f}s
- Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<!-- METRICS_SUMMARY_END -->"""
            
            if summary_pattern.search(content):
                new_content = summary_pattern.sub(new_summary, content)
                with open(self.readme_path, 'w') as f:
                    f.write(new_content)
                logger.info("Updated rolling metrics summary in README")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update rolling metrics summary: {e}")
            return False
    
    def add_recent_run_to_metrics(self, run_data: Dict) -> None:
        """Add a recent run to rolling metrics"""
        try:
            # Load existing metrics
            if self.rolling_metrics_path.exists():
                with open(self.rolling_metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "timing_samples": [],
                    "statistical_samples": [],
                    "zk_proof_samples": []
                }
            
            # Update counters
            metrics["total_runs"] += 1
            if run_data.get('success'):
                metrics["successful_runs"] += 1
            
            # Add timing sample
            if 'total_duration' in run_data and 'n_queries' in run_data:
                timing_sample = {
                    "t_per_query": run_data['total_duration'] / max(run_data['n_queries'], 1),
                    "t_total": run_data['total_duration'],
                    "hardware": "unknown"  # Could be enhanced to detect hardware
                }
                metrics["timing_samples"].append(timing_sample)
                
                # Keep only last 50 samples
                if len(metrics["timing_samples"]) > 50:
                    metrics["timing_samples"] = metrics["timing_samples"][-50:]
            
            # Add statistical sample
            if all(k in run_data for k in ['decision', 'confidence', 'n_queries']):
                stat_sample = {
                    "decision": run_data['decision'],
                    "confidence": run_data['confidence'],
                    "n_used": run_data['n_queries'],
                    "effect_size": run_data.get('effect_size', 0.0)
                }
                metrics["statistical_samples"].append(stat_sample)
                
                # Keep only last 50 samples
                if len(metrics["statistical_samples"]) > 50:
                    metrics["statistical_samples"] = metrics["statistical_samples"][-50:]
            
            # Update timestamp
            metrics["last_updated"] = datetime.now().isoformat()
            
            # Save back
            with open(self.rolling_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Added run data to rolling metrics")
            
        except Exception as e:
            logger.error(f"Failed to update rolling metrics: {e}")
    
    def update_all_tables(self) -> bool:
        """Update all README tables with latest data"""
        success = True
        
        logger.info("Updating README tables with latest experimental data...")
        
        # Update main example runs table
        if not self.update_example_runs_table():
            success = False
        
        # Update rolling metrics summary if present
        self.update_rolling_metrics_summary()
        
        return success


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update README tables with experimental data')
    parser.add_argument('--repo-root', type=str, help='Repository root directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--add-run', type=str, help='Add specific run file to metrics')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    updater = ReadmeTableUpdater(args.repo_root)
    
    # Add specific run if requested
    if args.add_run:
        try:
            with open(args.add_run, 'r') as f:
                run_data = json.load(f)
            updater.add_recent_run_to_metrics(run_data)
            logger.info(f"Added {args.add_run} to rolling metrics")
        except Exception as e:
            logger.error(f"Failed to add run {args.add_run}: {e}")
            return 1
    
    # Update all tables
    if updater.update_all_tables():
        logger.info("Successfully updated README tables")
        return 0
    else:
        logger.error("Failed to update some README tables")
        return 1


if __name__ == '__main__':
    sys.exit(main())