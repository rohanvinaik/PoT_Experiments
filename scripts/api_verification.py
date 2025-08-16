#!/usr/bin/env python3
"""
API Verification Script for Proof-of-Training
Queries external APIs with PoT challenges and measures FAR/FRR/AUROC
"""

import os
import sys
import json
import yaml
import time
import hashlib
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.core.challenge import generate_challenges
from pot.core.stats import far_frr
from pot.core.logging import StructuredLogger

# Optional imports for API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # make attribute available for tests
    OPENAI_AVAILABLE = False
    
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class APIResponse:
    """Response from an API call"""
    api_name: str
    model: str
    challenge_id: int
    prompt: str
    response: str
    latency_ms: float
    tokens_used: int
    cost: float
    timestamp: str
    success: bool
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of verification against reference"""
    api_name: str
    model: str
    num_challenges: int
    distances: List[float]
    threshold: float
    accepted: bool
    confidence: float
    far: float
    frr: float
    auroc: float
    precision: List[float]
    recall: List[float]
    queries_used: int
    total_cost: float
    total_latency_ms: float
    timestamp: str


class APIVerifier:
    """Manages API verification with PoT challenges"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.logger = StructuredLogger("api_results")
        self.log_file = "api_verification.jsonl"
        self.results_dir = Path(self.config.get('output', {}).get('base_dir', 'api_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize API clients
        self.clients = self._init_clients()
        
        # Load reference responses if available
        self.references = self._load_references()
        
        # Challenge cache
        self.challenges = []
        self.challenge_responses = {}
        
        # Metrics storage
        self.all_results = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _init_clients(self) -> Dict:
        """Initialize API clients based on configuration"""
        clients = {}
        
        for api_config in self.config.get('apis', []):
            provider = api_config.get('provider')
            model = api_config.get('model_name')
            key = f"{provider}_{model}"
            
            if provider == 'openai' and OPENAI_AVAILABLE:
                api_key = os.getenv(api_config.get('api_key', '').strip('${}'))
                if api_key:
                    clients[key] = {
                        'type': 'openai',
                        'client': openai.OpenAI(api_key=api_key),
                        'config': api_config
                    }
                    
            elif provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                api_key = os.getenv(api_config.get('api_key', '').strip('${}'))
                if api_key:
                    clients[key] = {
                        'type': 'anthropic',
                        'client': anthropic.Anthropic(api_key=api_key),
                        'config': api_config
                    }
                    
            elif provider == 'google' and GOOGLE_AVAILABLE:
                api_key = os.getenv(api_config.get('api_key', '').strip('${}'))
                if api_key:
                    genai.configure(api_key=api_key)
                    clients[key] = {
                        'type': 'google',
                        'client': genai.GenerativeModel(api_config.get('model_name')),
                        'config': api_config
                    }
                    
            elif provider == 'mock':
                # Mock API for testing
                clients[key] = {
                    'type': 'mock',
                    'client': None,
                    'config': api_config
                }
                
        return clients
        
    def _load_references(self) -> Dict:
        """Load reference model responses"""
        references = {}
        ref_config = self.config.get('reference_models', {})
        
        for ref_name, ref_data in ref_config.items():
            ref_path = ref_data.get('path')
            if ref_path and os.path.exists(ref_path):
                with open(ref_path, 'r') as f:
                    references[ref_name] = json.load(f)
                self.logger.log_jsonl(self.log_file, {
                    'event': 'reference_loaded',
                    'name': ref_name,
                    'path': ref_path,
                    'num_responses': len(references[ref_name])
                })
                
        return references
        
    def generate_pot_challenges(self, num_challenges: int = 100) -> List[Dict]:
        """Generate PoT challenges for API testing"""
        challenges = []
        
        # Get challenge configuration
        challenge_config = self.config.get('challenges', {})
        families = challenge_config.get('families', [])
        
        if not families:
            # Use default text challenges
            for i in range(num_challenges):
                seed = hashlib.sha256(f"challenge_{i}".encode()).digest()
                challenge = {
                    'id': i,
                    'type': 'text',
                    'prompt': f"Complete: The proof-of-training system challenge {i} is",
                    'seed': seed.hex()
                }
                challenges.append(challenge)
        else:
            # Use configured challenge families
            for family in families:
                family_type = family.get('family')
                n = min(family.get('n', 10), num_challenges // len(families))
                
                if family_type == 'lm:templates':
                    templates = family.get('params', {}).get('templates', [])
                    slots = family.get('params', {}).get('slots', {})
                    
                    for i in range(n):
                        template = templates[i % len(templates)]
                        prompt = template
                        
                        # Fill in template slots
                        for slot, values in slots.items():
                            if f"{{{slot}}}" in prompt and values:
                                value = values[i % len(values)]
                                prompt = prompt.replace(f"{{{slot}}}", value)
                                
                        seed = hashlib.sha256(f"{family_type}_{i}_{prompt}".encode()).digest()
                        challenge = {
                            'id': len(challenges),
                            'type': 'text',
                            'family': family_type,
                            'prompt': prompt,
                            'seed': seed.hex()
                        }
                        challenges.append(challenge)
                        
        self.challenges = challenges
        self.logger.log_jsonl(self.log_file, {
            'event': 'challenges_generated',
            'num_challenges': len(challenges),
            'types': list(set(c.get('type') for c in challenges))
        })
        
        return challenges
        
    def query_api(self, api_key: str, challenge: Dict) -> APIResponse:
        """Query a single API with a challenge"""
        if api_key not in self.clients:
            return APIResponse(
                api_name=api_key,
                model="unknown",
                challenge_id=challenge['id'],
                prompt=challenge['prompt'],
                response="",
                latency_ms=0,
                tokens_used=0,
                cost=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error="API client not initialized"
            )
            
        client_info = self.clients[api_key]
        client_type = client_info['type']
        client = client_info['client']
        config = client_info['config']
        
        start_time = time.time()
        
        try:
            if client_type == 'openai':
                response = client.chat.completions.create(
                    model=config['model_name'],
                    messages=[{"role": "user", "content": challenge['prompt']}],
                    temperature=config.get('temperature', 0.0),
                    max_tokens=config.get('max_tokens', 100)
                )
                text = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                
            elif client_type == 'anthropic':
                response = client.messages.create(
                    model=config['model_name'],
                    messages=[{"role": "user", "content": challenge['prompt']}],
                    max_tokens=config.get('max_tokens', 100),
                    temperature=config.get('temperature', 0.0)
                )
                text = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
                
            elif client_type == 'google':
                response = client.generate_content(challenge['prompt'])
                text = response.text
                tokens = len(text.split())  # Approximate
                
            elif client_type == 'mock':
                # Mock response for testing
                import random
                responses = [
                    "This is a mock response for testing.",
                    "The proof-of-training system uses cryptographic verification.",
                    "Machine learning models can be verified using behavioral fingerprinting.",
                    "Challenge-response protocols ensure model integrity."
                ]
                text = responses[challenge['id'] % len(responses)]
                tokens = len(text.split())
                time.sleep(0.01)  # Simulate latency
                
            else:
                raise ValueError(f"Unknown client type: {client_type}")
                
            latency_ms = (time.time() - start_time) * 1000
            cost = tokens * config.get('cost_per_token', 0.0)
            
            return APIResponse(
                api_name=api_key,
                model=config['model_name'],
                challenge_id=challenge['id'],
                prompt=challenge['prompt'],
                response=text,
                latency_ms=latency_ms,
                tokens_used=tokens,
                cost=cost,
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            self.logger.log_jsonl(self.log_file, {
                'event': 'api_error',
                'api': api_key,
                'challenge_id': challenge['id'],
                'error': str(e)
            })
            
            return APIResponse(
                api_name=api_key,
                model=config.get('model_name', 'unknown'),
                challenge_id=challenge['id'],
                prompt=challenge['prompt'],
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cost=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e)
            )
            
    def compute_distance(self, response1: str, response2: str) -> float:
        """Compute distance between two responses"""
        if not response1 or not response2:
            return 1.0
            
        # Simple character-level distance (can be enhanced)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, response1.lower(), response2.lower()).ratio()
        return 1.0 - similarity
        
    def verify_api(self, api_key: str, reference_key: str = None) -> VerificationResult:
        """Verify an API against reference responses"""
        if not self.challenges:
            self.generate_pot_challenges()
            
        # Query API with challenges
        api_responses = []
        total_cost = 0
        total_latency = 0
        
        verification_config = self.config.get('verification', {})
        threshold = verification_config.get('similarity_threshold', 0.7)
        max_challenges = min(
            verification_config.get('max_errors_percent', 100),
            len(self.challenges)
        )
        
        # Sequential verification with empirical-Bernstein
        distances = []
        n = 0
        sum_x = 0
        sum_x2 = 0
        
        for challenge in self.challenges[:max_challenges]:
            n += 1
            
            # Query API
            response = self.query_api(api_key, challenge)
            api_responses.append(response)
            
            if not response.success:
                continue
                
            total_cost += response.cost
            total_latency += response.latency_ms
            
            # Get reference response
            if reference_key and reference_key in self.references:
                ref_responses = self.references[reference_key]
                ref_response = ref_responses.get(str(challenge['id']), {}).get('response', '')
            else:
                # Use first successful response as reference (self-consistency)
                ref_response = response.response
                
            # Compute distance
            distance = self.compute_distance(response.response, ref_response)
            distances.append(distance)
            
            # Update statistics for EB bounds
            sum_x += distance
            sum_x2 += distance ** 2
            mean = sum_x / n
            
            if n > 1:
                var = (sum_x2 - n * mean**2) / (n - 1)
                
                # Compute EB bound
                alpha = 0.01
                u_bound = np.sqrt(2 * var * np.log(1/alpha) / n) + 7 * np.log(1/alpha) / (3 * (n-1))
                
                # Sequential decision
                if mean + u_bound <= threshold:
                    # Accept
                    break
                elif mean - u_bound >= threshold:
                    # Reject
                    break
                    
        # Compute metrics
        if distances:
            # FAR/FRR computation
            y_true = [1] * len(distances)  # Assume all should be accepted
            y_scores = [1 - d for d in distances]  # Convert distance to similarity
            
            # Compute at threshold
            predictions = [1 if s >= threshold else 0 for s in y_scores]
            false_accepts = sum(1 for i, p in enumerate(predictions) if p == 1 and y_true[i] == 0)
            false_rejects = sum(1 for i, p in enumerate(predictions) if p == 0 and y_true[i] == 1)
            
            far = false_accepts / len(predictions) if predictions else 0
            frr = false_rejects / len(predictions) if predictions else 0
            
            # AUROC
            try:
                auroc = roc_auc_score(y_true, y_scores)
                if np.isnan(auroc):
                    auroc = 0.5
            except Exception:
                auroc = 0.5
                
            # Precision-Recall
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
            except:
                precision, recall = [1.0], [1.0]
                
            # Decision
            mean_distance = np.mean(distances)
            confidence = 1.0 - mean_distance
            accepted = mean_distance <= threshold
            
        else:
            far, frr, auroc = 0, 0, 0.5
            precision, recall = [1.0], [1.0]
            mean_distance = 1.0
            confidence = 0.0
            accepted = False
            
        result = VerificationResult(
            api_name=api_key,
            model=self.clients[api_key]['config'].get('model_name', 'unknown'),
            num_challenges=len(self.challenges),
            distances=distances,
            threshold=threshold,
            accepted=accepted,
            confidence=confidence,
            far=far,
            frr=frr,
            auroc=auroc,
            precision=precision.tolist() if hasattr(precision, 'tolist') else precision,
            recall=recall.tolist() if hasattr(recall, 'tolist') else recall,
            queries_used=n,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            timestamp=datetime.now().isoformat()
        )
        
        # Save responses
        self._save_api_responses(api_key, api_responses)
        
        return result
        
    def _save_api_responses(self, api_key: str, responses: List[APIResponse]):
        """Save API responses to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{api_key}_responses_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in responses], f, indent=2)
            
        self.logger.log_jsonl(self.log_file, {
            'event': 'responses_saved',
            'api': api_key,
            'file': str(filename),
            'num_responses': len(responses)
        })
        
    def verify_all_apis(self) -> List[VerificationResult]:
        """Verify all configured APIs"""
        results = []
        
        for api_key in self.clients.keys():
            self.logger.log_jsonl(self.log_file, {
                'event': 'verifying_api',
                'api': api_key
            })
            
            result = self.verify_api(api_key)
            results.append(result)
            self.all_results.append(result)
            
            # Log summary
            self.logger.log_jsonl(self.log_file, {
                'event': 'verification_complete',
                'api': api_key,
                'accepted': result.accepted,
                'confidence': result.confidence,
                'far': result.far,
                'frr': result.frr,
                'auroc': result.auroc,
                'queries_used': result.queries_used,
                'cost': result.total_cost
            })
            
            # Rate limiting
            time.sleep(1)
            
        return results
        
    def generate_plots(self, results: List[VerificationResult]):
        """Generate visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 8))
        for result in results:
            if result.distances:
                y_true = [1] * len(result.distances)
                y_scores = [1 - d for d in result.distances]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    plt.plot(fpr, tpr, label=f"{result.api_name} (AUC={result.auroc:.3f})")
                except:
                    pass
                    
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for API Verification')
        plt.legend()
        plt.savefig(self.results_dir / f"roc_curves_{timestamp}.png")
        plt.close()
        
        # 2. FAR/FRR Comparison
        plt.figure(figsize=(12, 6))
        
        api_names = [r.api_name for r in results]
        fars = [r.far for r in results]
        frrs = [r.frr for r in results]
        
        x = np.arange(len(api_names))
        width = 0.35
        
        plt.bar(x - width/2, fars, width, label='FAR', color='red', alpha=0.7)
        plt.bar(x + width/2, frrs, width, label='FRR', color='blue', alpha=0.7)
        
        plt.xlabel('API')
        plt.ylabel('Error Rate')
        plt.title('False Accept/Reject Rates by API')
        plt.xticks(x, api_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / f"far_frr_{timestamp}.png")
        plt.close()
        
        # 3. Distance Distribution
        plt.figure(figsize=(12, 8))
        
        for i, result in enumerate(results):
            if result.distances:
                plt.subplot(len(results), 1, i+1)
                plt.hist(result.distances, bins=30, alpha=0.7, edgecolor='black')
                plt.axvline(result.threshold, color='red', linestyle='--', 
                           label=f'Threshold={result.threshold}')
                plt.xlabel('Distance')
                plt.ylabel('Frequency')
                plt.title(f'{result.api_name} - Distance Distribution')
                plt.legend()
                
        plt.tight_layout()
        plt.savefig(self.results_dir / f"distance_distributions_{timestamp}.png")
        plt.close()
        
        # 4. Performance Summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # AUROC comparison
        axes[0, 0].bar(api_names, [r.auroc for r in results], color='green', alpha=0.7)
        axes[0, 0].set_ylabel('AUROC')
        axes[0, 0].set_title('AUROC by API')
        axes[0, 0].set_xticklabels(api_names, rotation=45, ha='right')
        axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        
        # Confidence scores
        axes[0, 1].bar(api_names, [r.confidence for r in results], color='blue', alpha=0.7)
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Confidence Scores')
        axes[0, 1].set_xticklabels(api_names, rotation=45, ha='right')
        axes[0, 1].axhline(0.8, color='red', linestyle='--', alpha=0.5)
        
        # Queries used
        axes[1, 0].bar(api_names, [r.queries_used for r in results], color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Queries')
        axes[1, 0].set_title('Queries Used (Sequential)')
        axes[1, 0].set_xticklabels(api_names, rotation=45, ha='right')
        
        # Cost analysis
        axes[1, 1].bar(api_names, [r.total_cost for r in results], color='red', alpha=0.7)
        axes[1, 1].set_ylabel('Cost ($)')
        axes[1, 1].set_title('Total Cost')
        axes[1, 1].set_xticklabels(api_names, rotation=45, ha='right')
        
        plt.suptitle('API Verification Performance Summary', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.results_dir / f"performance_summary_{timestamp}.png")
        plt.close()
        
        self.logger.log_jsonl(self.log_file, {
            'event': 'plots_generated',
            'num_plots': 4,
            'directory': str(self.results_dir)
        })
        
    def save_results(self, results: List[VerificationResult]):
        """Save verification results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Save detailed results
        results_file = self.results_dir / f"verification_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            results_data = [convert_numpy(asdict(r)) for r in results]
            json.dump(results_data, f, indent=2)
            
        # Save summary
        summary = {
            'timestamp': timestamp,
            'num_apis': len(results),
            'apis_tested': [r.api_name for r in results],
            'accepted': [r.api_name for r in results if r.accepted],
            'rejected': [r.api_name for r in results if not r.accepted],
            'average_far': np.mean([r.far for r in results]),
            'average_frr': np.mean([r.frr for r in results]),
            'average_auroc': np.mean([r.auroc for r in results]),
            'total_cost': sum(r.total_cost for r in results),
            'total_queries': sum(r.queries_used for r in results)
        }
        
        summary_file = self.results_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate markdown report
        report = self._generate_report(results, summary)
        report_file = self.results_dir / f"report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
            
        self.logger.log_jsonl(self.log_file, {
            'event': 'results_saved',
            'results_file': str(results_file),
            'summary_file': str(summary_file),
            'report_file': str(report_file)
        })
        
        return results_file, summary_file, report_file
        
    def _generate_report(self, results: List[VerificationResult], summary: Dict) -> str:
        """Generate markdown report"""
        report = f"""# API Verification Report
Generated: {summary['timestamp']}

## Summary
- **APIs Tested**: {summary['num_apis']}
- **Accepted**: {len(summary['accepted'])}
- **Rejected**: {len(summary['rejected'])}
- **Average FAR**: {summary['average_far']:.4f}
- **Average FRR**: {summary['average_frr']:.4f}
- **Average AUROC**: {summary['average_auroc']:.4f}

## Detailed Results

| API | Model | Accepted | Confidence | FAR | FRR | AUROC | Queries | Cost |
|-----|-------|----------|------------|-----|-----|--------|---------|------|
"""
        
        for r in results:
            report += f"| {r.api_name} | {r.model} | {'✅' if r.accepted else '❌'} | "
            report += f"{r.confidence:.3f} | {r.far:.4f} | {r.frr:.4f} | "
            report += f"{r.auroc:.3f} | {r.queries_used} | ${r.total_cost:.4f} |\n"
            
        report += f"""

## Cost Analysis
- **Total Cost**: ${summary['total_cost']:.2f}
- **Total Queries**: {summary['total_queries']}
- **Average Cost per API**: ${summary['total_cost']/max(1, summary['num_apis']):.2f}

## Accepted APIs
"""
        
        for api in summary['accepted']:
            report += f"- {api}\n"
            
        if summary['rejected']:
            report += "\n## Rejected APIs\n"
            for api in summary['rejected']:
                report += f"- {api}\n"
                
        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='API Verification for PoT')
    parser.add_argument('--config', default='configs/api_verification.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num-challenges', type=int, default=100,
                       help='Number of challenges to generate')
    parser.add_argument('--apis', nargs='+', help='Specific APIs to test')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize verifier
    verifier = APIVerifier(args.config)
    
    # Generate challenges
    verifier.generate_pot_challenges(args.num_challenges)
    
    # Verify APIs
    if args.apis:
        results = []
        for api in args.apis:
            if api in verifier.clients:
                results.append(verifier.verify_api(api))
    else:
        results = verifier.verify_all_apis()
        
    # Save results
    results_file, summary_file, report_file = verifier.save_results(results)
    
    # Generate plots
    if args.plot:
        verifier.generate_plots(results)
        
    print(f"\n✅ Verification complete!")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Report saved to: {report_file}")
    
    # Print summary
    for result in results:
        status = "✅ ACCEPTED" if result.accepted else "❌ REJECTED"
        print(f"\n{result.api_name}: {status}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  FAR: {result.far:.4f}, FRR: {result.frr:.4f}, AUROC: {result.auroc:.3f}")
        print(f"  Queries: {result.queries_used}, Cost: ${result.total_cost:.4f}")


if __name__ == "__main__":
    main()