#!/usr/bin/env python3
"""
Enhanced attack simulation harness for Proof-of-Training.
Simulates wrapper, fine-tuning, and compression attacks with detailed metrics.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from pot.core.logging import StructuredLogger

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Using mock models.")

@dataclass
class AttackScenario:
    """Configuration for an attack scenario"""
    name: str
    attack_type: str  # 'wrapper', 'fine_tuning', 'compression', 'combined'
    difficulty: str  # 'easy', 'medium', 'hard'
    parameters: Dict[str, Any]
    expected_detection_rate: float

@dataclass
class AttackResult:
    """Results from an attack simulation"""
    scenario_name: str
    attack_type: str
    success_rate: float
    detection_rate: float
    false_positive_rate: float
    false_negative_rate: float
    average_similarity: float
    time_to_detection: Optional[float]
    evasion_duration: Optional[float]
    computational_cost: float
    timestamp: str

class ModelVerifier:
    """Simple model verifier for testing"""
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def verify(self, response1: np.ndarray, response2: np.ndarray):
        """Verify if two responses match"""
        # Calculate cosine similarity
        similarity = np.dot(response1.flatten(), response2.flatten()) / (
            np.linalg.norm(response1) * np.linalg.norm(response2) + 1e-8
        )
        passed = similarity > self.threshold
        return passed, similarity

class ChallengeGenerator:
    """Generate challenges for verification"""
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        np.random.seed(42)  # For reproducibility
    
    def generate_batch(self, batch_size: int = 1):
        """Generate batch of challenges"""
        return np.random.randn(batch_size, self.dimension).astype(np.float32)

class AttackSimulator:
    """Simulates realistic attacks against PoT"""
    
    def __init__(self, target_model, verifier: ModelVerifier):
        self.target_model = target_model
        self.verifier = verifier
        self.challenge_generator = ChallengeGenerator(dimension=512)
        self.results = []
    
    def simulate_wrapper_attack(self, 
                               scenario: AttackScenario,
                               n_rounds: int = 1000) -> AttackResult:
        """Simulate wrapper attack"""
        print(f"\nSimulating wrapper attack: {scenario.name}")
        
        # Create wrapper based on difficulty
        if scenario.difficulty == 'easy':
            wrapper = self._create_simple_wrapper()
        elif scenario.difficulty == 'medium':
            wrapper = self._create_adaptive_wrapper()
        else:  # hard
            wrapper = self._create_sophisticated_wrapper()
        
        successes = 0
        detections = 0
        similarities = []
        start_time = time.time()
        first_detection_time = None
        
        for round_idx in range(n_rounds):
            # Generate challenge
            challenge = self.challenge_generator.generate_batch(1)
            
            if HAS_TORCH:
                challenge_tensor = torch.FloatTensor(challenge)
                
                # Get responses
                with torch.no_grad():
                    target_response = self.target_model(challenge_tensor)
                    wrapper_response = wrapper(challenge_tensor)
                
                # Verify
                passed, similarity = self.verifier.verify(
                    wrapper_response.numpy(),
                    target_response.numpy()
                )
            else:
                # Mock verification
                similarity = 0.7 + np.random.random() * 0.3
                passed = similarity > 0.85
            
            similarities.append(similarity)
            
            if passed:
                successes += 1
            else:
                detections += 1
                if first_detection_time is None:
                    first_detection_time = time.time() - start_time
            
            # Adapt wrapper if it's adaptive
            if HAS_TORCH and hasattr(wrapper, 'adapt') and round_idx % 10 == 0:
                wrapper.adapt(challenge_tensor, target_response)
        
        computation_time = time.time() - start_time
        
        return AttackResult(
            scenario_name=scenario.name,
            attack_type='wrapper',
            success_rate=successes / n_rounds,
            detection_rate=detections / n_rounds,
            false_positive_rate=0.0,  # Will be calculated separately
            false_negative_rate=successes / n_rounds,
            average_similarity=np.mean(similarities),
            time_to_detection=first_detection_time,
            evasion_duration=computation_time if successes > 0 else 0,
            computational_cost=computation_time,
            timestamp=datetime.now().isoformat()
        )
    
    def simulate_fine_tuning_attack(self,
                                   scenario: AttackScenario,
                                   n_rounds: int = 1000) -> AttackResult:
        """Simulate fine-tuning attack"""
        print(f"\nSimulating fine-tuning attack: {scenario.name}")
        
        if HAS_TORCH:
            # Clone and fine-tune model
            attacker_model = self._clone_model(self.target_model)
            
            # Fine-tune based on difficulty
            if scenario.difficulty == 'easy':
                self._fine_tune_model(attacker_model, epochs=1, lr=0.001)
            elif scenario.difficulty == 'medium':
                self._fine_tune_model(attacker_model, epochs=5, lr=0.01)
            else:  # hard
                self._fine_tune_model(attacker_model, epochs=20, lr=0.1)
        else:
            attacker_model = None
        
        successes = 0
        detections = 0
        similarities = []
        start_time = time.time()
        
        for _ in range(n_rounds):
            challenge = self.challenge_generator.generate_batch(1)
            
            if HAS_TORCH and attacker_model:
                challenge_tensor = torch.FloatTensor(challenge)
                
                with torch.no_grad():
                    target_response = self.target_model(challenge_tensor)
                    attacker_response = attacker_model(challenge_tensor)
                
                passed, similarity = self.verifier.verify(
                    attacker_response.numpy(),
                    target_response.numpy()
                )
            else:
                # Mock results based on difficulty
                if scenario.difficulty == 'easy':
                    similarity = 0.8 + np.random.random() * 0.15
                elif scenario.difficulty == 'medium':
                    similarity = 0.7 + np.random.random() * 0.2
                else:
                    similarity = 0.5 + np.random.random() * 0.3
                passed = similarity > 0.85
            
            similarities.append(similarity)
            
            if passed:
                successes += 1
            else:
                detections += 1
        
        computation_time = time.time() - start_time
        
        return AttackResult(
            scenario_name=scenario.name,
            attack_type='fine_tuning',
            success_rate=successes / n_rounds,
            detection_rate=detections / n_rounds,
            false_positive_rate=0.0,
            false_negative_rate=successes / n_rounds,
            average_similarity=np.mean(similarities),
            time_to_detection=None,
            evasion_duration=computation_time if successes > 0 else 0,
            computational_cost=computation_time,
            timestamp=datetime.now().isoformat()
        )
    
    def simulate_compression_attack(self,
                                   scenario: AttackScenario,
                                   n_rounds: int = 1000) -> AttackResult:
        """Simulate model compression attack"""
        print(f"\nSimulating compression attack: {scenario.name}")
        
        if HAS_TORCH:
            # Compress model based on difficulty
            if scenario.difficulty == 'easy':
                compressed_model = self._compress_model(self.target_model, compression_ratio=0.9)
            elif scenario.difficulty == 'medium':
                compressed_model = self._compress_model(self.target_model, compression_ratio=0.5)
            else:  # hard
                compressed_model = self._compress_model(self.target_model, compression_ratio=0.1)
        else:
            compressed_model = None
        
        successes = 0
        detections = 0
        similarities = []
        start_time = time.time()
        
        for _ in range(n_rounds):
            challenge = self.challenge_generator.generate_batch(1)
            
            if HAS_TORCH and compressed_model:
                challenge_tensor = torch.FloatTensor(challenge)
                
                with torch.no_grad():
                    target_response = self.target_model(challenge_tensor)
                    compressed_response = compressed_model(challenge_tensor)
                
                passed, similarity = self.verifier.verify(
                    compressed_response.numpy(),
                    target_response.numpy()
                )
            else:
                # Mock results - compression degrades similarity
                if scenario.difficulty == 'easy':
                    similarity = 0.85 + np.random.random() * 0.1
                elif scenario.difficulty == 'medium':
                    similarity = 0.6 + np.random.random() * 0.2
                else:
                    similarity = 0.3 + np.random.random() * 0.2
                passed = similarity > 0.85
            
            similarities.append(similarity)
            
            if passed:
                successes += 1
            else:
                detections += 1
        
        computation_time = time.time() - start_time
        
        return AttackResult(
            scenario_name=scenario.name,
            attack_type='compression',
            success_rate=successes / n_rounds,
            detection_rate=detections / n_rounds,
            false_positive_rate=0.0,
            false_negative_rate=successes / n_rounds,
            average_similarity=np.mean(similarities),
            time_to_detection=None,
            evasion_duration=computation_time if successes > 0 else 0,
            computational_cost=computation_time,
            timestamp=datetime.now().isoformat()
        )
    
    def simulate_combined_attack(self,
                                scenario: AttackScenario,
                                n_rounds: int = 1000) -> AttackResult:
        """Simulate combined attack using multiple techniques"""
        print(f"\nSimulating combined attack: {scenario.name}")
        
        if HAS_TORCH:
            # Apply multiple attack techniques
            attacker_model = self._clone_model(self.target_model)
            
            # Fine-tune
            self._fine_tune_model(attacker_model, epochs=5, lr=0.01)
            
            # Compress
            attacker_model = self._compress_model(attacker_model, compression_ratio=0.7)
            
            # Add wrapper
            wrapper = self._create_adaptive_wrapper(base_model=attacker_model)
        else:
            wrapper = None
        
        successes = 0
        detections = 0
        similarities = []
        start_time = time.time()
        
        for round_idx in range(n_rounds):
            challenge = self.challenge_generator.generate_batch(1)
            
            if HAS_TORCH and wrapper:
                challenge_tensor = torch.FloatTensor(challenge)
                
                with torch.no_grad():
                    target_response = self.target_model(challenge_tensor)
                    attack_response = wrapper(challenge_tensor)
                
                passed, similarity = self.verifier.verify(
                    attack_response.numpy(),
                    target_response.numpy()
                )
                
                # Adapt wrapper periodically
                if hasattr(wrapper, 'adapt') and round_idx % 20 == 0:
                    wrapper.adapt(challenge_tensor, target_response)
            else:
                # Mock combined attack - generally less effective
                similarity = 0.4 + np.random.random() * 0.4
                passed = similarity > 0.85
            
            similarities.append(similarity)
            
            if passed:
                successes += 1
            else:
                detections += 1
        
        computation_time = time.time() - start_time
        
        return AttackResult(
            scenario_name=scenario.name,
            attack_type='combined',
            success_rate=successes / n_rounds,
            detection_rate=detections / n_rounds,
            false_positive_rate=0.0,
            false_negative_rate=successes / n_rounds,
            average_similarity=np.mean(similarities),
            time_to_detection=None,
            evasion_duration=computation_time if successes > 0 else 0,
            computational_cost=computation_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _create_simple_wrapper(self):
        """Create a simple wrapper with fixed transformations"""
        if not HAS_TORCH:
            return None
            
        class SimpleWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.transform = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 64)
                )
            
            def forward(self, x):
                return self.transform(x)
        
        return SimpleWrapper()
    
    def _create_adaptive_wrapper(self, base_model=None):
        """Create an adaptive wrapper that learns from feedback"""
        if not HAS_TORCH:
            return None
            
        class AdaptiveWrapper(nn.Module):
            def __init__(self, base=None):
                super().__init__()
                self.base_model = base if base else nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64)
                )
                self.adapter = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                self.optimizer = optim.Adam(self.parameters(), lr=0.01)
            
            def forward(self, x):
                if hasattr(self.base_model, 'forward'):
                    base_out = self.base_model(x)
                else:
                    base_out = x[:, :64]  # Fallback
                return self.adapter(base_out)
            
            def adapt(self, challenge, target_response):
                self.train()
                self.optimizer.zero_grad()
                output = self.forward(challenge)
                loss = nn.MSELoss()(output, target_response)
                loss.backward()
                self.optimizer.step()
                self.eval()
        
        return AdaptiveWrapper(base_model)
    
    def _create_sophisticated_wrapper(self):
        """Create a sophisticated wrapper with multiple evasion techniques"""
        if not HAS_TORCH:
            return None
            
        class SophisticatedWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                # Multiple pathways without BatchNorm (to avoid batch size issues)
                self.pathway1 = nn.Sequential(
                    nn.Linear(512, 384),
                    nn.LayerNorm(384),  # Use LayerNorm instead of BatchNorm
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(384, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64)
                )
                self.pathway2 = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                self.mixer = nn.Parameter(torch.tensor(0.5))
                self.noise_scale = 0.01
            
            def forward(self, x):
                # Add input noise
                x_noisy = x + torch.randn_like(x) * self.noise_scale
                
                # Process through both pathways
                out1 = self.pathway1(x_noisy)
                out2 = self.pathway2(x_noisy)
                
                # Mix outputs
                mixed = torch.sigmoid(self.mixer) * out1 + (1 - torch.sigmoid(self.mixer)) * out2
                
                # Add output noise
                return mixed + torch.randn_like(mixed) * self.noise_scale * 0.5
        
        return SophisticatedWrapper()
    
    def _clone_model(self, model):
        """Clone a model with same architecture"""
        if not HAS_TORCH:
            return None
            
        cloned = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Try to copy weights
        if model is not None:
            try:
                cloned.load_state_dict(model.state_dict())
            except:
                pass  # Different architecture, use random init
        
        return cloned
    
    def _fine_tune_model(self, model, epochs: int, lr: float):
        """Fine-tune a model on synthetic data"""
        if not HAS_TORCH or model is None:
            return
            
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for _ in range(epochs):
            # Generate synthetic training data
            synthetic_input = torch.randn(32, 512)
            synthetic_target = torch.randn(32, 64)
            
            optimizer.zero_grad()
            output = model(synthetic_input)
            loss = criterion(output, synthetic_target)
            loss.backward()
            optimizer.step()
    
    def _compress_model(self, model, compression_ratio: float):
        """Compress model using pruning"""
        if not HAS_TORCH:
            return None
            
        compressed = self._clone_model(model)
        
        if compressed is None:
            return None
            
        # Simple pruning: zero out small weights
        with torch.no_grad():
            for param in compressed.parameters():
                if len(param.shape) == 2:  # Weight matrix
                    threshold = torch.quantile(torch.abs(param), 1 - compression_ratio)
                    param[torch.abs(param) < threshold] = 0
        
        return compressed
    
    def evaluate_false_positives(self, n_samples: int = 100) -> float:
        """Evaluate false positive rate on legitimate models"""
        false_positives = 0
        
        for _ in range(n_samples):
            challenge = self.challenge_generator.generate_batch(1)
            
            if HAS_TORCH:
                challenge_tensor = torch.FloatTensor(challenge)
                
                with torch.no_grad():
                    response = self.target_model(challenge_tensor)
                
                # Add small legitimate variations (numerical precision, etc.)
                noise = torch.randn_like(response) * 0.001
                noisy_response = response + noise
                
                passed, _ = self.verifier.verify(
                    noisy_response.numpy(),
                    response.numpy()
                )
            else:
                # Mock false positive rate
                passed = np.random.random() > 0.02  # 2% false positive rate
            
            if not passed:
                false_positives += 1
        
        return false_positives / n_samples

class AttackHarness:
    """Main harness for running attack simulations"""
    
    def __init__(self, output_dir: str = "./attack_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize target model
        if HAS_TORCH:
            self.target_model = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        else:
            self.target_model = None
        
        # Initialize verifier
        self.verifier = ModelVerifier(threshold=0.85)
        
        # Initialize simulator
        self.simulator = AttackSimulator(self.target_model, self.verifier)
        
        # Define attack scenarios
        self.scenarios = self._define_scenarios()
    
    def _define_scenarios(self) -> List[AttackScenario]:
        """Define attack scenarios to test"""
        return [
            # Wrapper attacks
            AttackScenario(
                name="Simple Wrapper",
                attack_type="wrapper",
                difficulty="easy",
                parameters={"adaptation": False},
                expected_detection_rate=0.9
            ),
            AttackScenario(
                name="Adaptive Wrapper",
                attack_type="wrapper",
                difficulty="medium",
                parameters={"adaptation": True, "learning_rate": 0.01},
                expected_detection_rate=0.7
            ),
            AttackScenario(
                name="Sophisticated Wrapper",
                attack_type="wrapper",
                difficulty="hard",
                parameters={"evasion_techniques": ["noise", "multi_path", "adaptation"]},
                expected_detection_rate=0.5
            ),
            
            # Fine-tuning attacks
            AttackScenario(
                name="Minimal Fine-tuning",
                attack_type="fine_tuning",
                difficulty="easy",
                parameters={"epochs": 1},
                expected_detection_rate=0.8
            ),
            AttackScenario(
                name="Moderate Fine-tuning",
                attack_type="fine_tuning",
                difficulty="medium",
                parameters={"epochs": 5},
                expected_detection_rate=0.6
            ),
            AttackScenario(
                name="Aggressive Fine-tuning",
                attack_type="fine_tuning",
                difficulty="hard",
                parameters={"epochs": 20},
                expected_detection_rate=0.4
            ),
            
            # Compression attacks
            AttackScenario(
                name="Light Compression",
                attack_type="compression",
                difficulty="easy",
                parameters={"ratio": 0.9},
                expected_detection_rate=0.7
            ),
            AttackScenario(
                name="Medium Compression",
                attack_type="compression",
                difficulty="medium",
                parameters={"ratio": 0.5},
                expected_detection_rate=0.8
            ),
            AttackScenario(
                name="Heavy Compression",
                attack_type="compression",
                difficulty="hard",
                parameters={"ratio": 0.1},
                expected_detection_rate=0.95
            ),
            
            # Combined attacks
            AttackScenario(
                name="Multi-technique Attack",
                attack_type="combined",
                difficulty="hard",
                parameters={"techniques": ["fine_tuning", "compression", "wrapper"]},
                expected_detection_rate=0.3
            )
        ]
    
    def run_all_scenarios(self, n_rounds: int = 100):
        """Run all attack scenarios"""
        print("=" * 70)
        print("ATTACK SIMULATION HARNESS")
        print("=" * 70)
        
        results = []
        
        # Evaluate false positive rate first
        print("\nEvaluating false positive rate...")
        fp_rate = self.simulator.evaluate_false_positives(n_samples=100)
        print(f"False positive rate: {fp_rate:.2%}")
        
        # Run each scenario
        for scenario in self.scenarios:
            print(f"\n" + "-" * 60)
            print(f"Scenario: {scenario.name}")
            print(f"Type: {scenario.attack_type}")
            print(f"Difficulty: {scenario.difficulty}")
            
            # Run appropriate simulation
            if scenario.attack_type == "wrapper":
                result = self.simulator.simulate_wrapper_attack(scenario, n_rounds)
            elif scenario.attack_type == "fine_tuning":
                result = self.simulator.simulate_fine_tuning_attack(scenario, n_rounds)
            elif scenario.attack_type == "compression":
                result = self.simulator.simulate_compression_attack(scenario, n_rounds)
            elif scenario.attack_type == "combined":
                result = self.simulator.simulate_combined_attack(scenario, n_rounds)
            else:
                continue
            
            # Add false positive rate
            result.false_positive_rate = fp_rate
            
            results.append(result)
            
            # Print summary
            print(f"\nResults:")
            print(f"  Detection rate: {result.detection_rate:.2%}")
            print(f"  Success rate: {result.success_rate:.2%}")
            print(f"  Avg similarity: {result.average_similarity:.4f}")
            
            # Check against expected
            if abs(result.detection_rate - scenario.expected_detection_rate) > 0.2:
                print(f"  ⚠ Detection rate differs from expected ({scenario.expected_detection_rate:.2%})")
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _save_results(self, results: List[AttackResult]):
        """Save results to JSON"""
        output_file = self.output_dir / f"attack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    def _generate_visualizations(self, results: List[AttackResult]):
        """Generate visualization plots"""
        # Set style
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Detection rates by attack type
        ax = axes[0, 0]
        attack_types = list(set(r.attack_type for r in results))
        detection_rates = {t: [r.detection_rate for r in results if r.attack_type == t] 
                          for t in attack_types}
        
        positions = range(len(attack_types))
        for i, (attack_type, rates) in enumerate(detection_rates.items()):
            ax.bar(i, np.mean(rates), yerr=np.std(rates) if len(rates) > 1 else 0, 
                   capsize=5, color=f'C{i}')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(attack_types)
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rates by Attack Type')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Success rates by difficulty
        ax = axes[0, 1]
        difficulties = ['easy', 'medium', 'hard']
        colors = ['green', 'orange', 'red']
        
        difficulty_success = {d: [] for d in difficulties}
        for r in results:
            for s in self.scenarios:
                if s.name == r.scenario_name:
                    difficulty_success[s.difficulty].append(r.success_rate)
        
        for i, (difficulty, rates) in enumerate(difficulty_success.items()):
            if rates:
                ax.bar(i, np.mean(rates), color=colors[i], 
                       yerr=np.std(rates) if len(rates) > 1 else 0, capsize=5)
        
        ax.set_xticks(range(len(difficulties)))
        ax.set_xticklabels(difficulties)
        ax.set_ylabel('Success Rate')
        ax.set_title('Attack Success by Difficulty')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Similarity distribution
        ax = axes[1, 0]
        all_similarities = [r.average_similarity for r in results]
        ax.hist(all_similarities, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0.85, color='red', linestyle='--', label='Detection Threshold', linewidth=2)
        ax.set_xlabel('Average Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Similarity Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ROC-style plot
        ax = axes[1, 1]
        tpr = [1 - r.false_negative_rate for r in results]  # True positive rate
        fpr = [r.false_positive_rate for r in results]  # False positive rate
        
        scatter = ax.scatter(fpr, tpr, c=[r.detection_rate for r in results], 
                            cmap='RdYlGn', s=100, edgecolor='black', alpha=0.8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Detection Performance')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Detection Rate')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"attack_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved to {output_file}")
    
    def _generate_report(self, results: List[AttackResult]):
        """Generate comprehensive attack report"""
        report = []
        report.append("=" * 70)
        report.append("ATTACK SIMULATION REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"Scenarios tested: {len(results)}")
        
        # Overall statistics
        overall_detection = np.mean([r.detection_rate for r in results])
        overall_success = np.mean([r.success_rate for r in results])
        
        report.append(f"\n{'='*30} SUMMARY {'='*30}")
        report.append(f"Overall detection rate: {overall_detection:.2%}")
        report.append(f"Overall attack success rate: {overall_success:.2%}")
        report.append(f"False positive rate: {results[0].false_positive_rate:.2%}" if results else "N/A")
        
        # By attack type
        report.append(f"\n{'='*30} BY ATTACK TYPE {'='*30}")
        for attack_type in set(r.attack_type for r in results):
            type_results = [r for r in results if r.attack_type == attack_type]
            report.append(f"\n{attack_type.upper()}:")
            report.append(f"  Average detection rate: {np.mean([r.detection_rate for r in type_results]):.2%}")
            report.append(f"  Average success rate: {np.mean([r.success_rate for r in type_results]):.2%}")
            report.append(f"  Scenarios: {len(type_results)}")
        
        # Individual scenarios
        report.append(f"\n{'='*30} DETAILED RESULTS {'='*30}")
        for result in results:
            report.append(f"\n{result.scenario_name}:")
            report.append(f"  Type: {result.attack_type}")
            report.append(f"  Detection rate: {result.detection_rate:.2%}")
            report.append(f"  Success rate: {result.success_rate:.2%}")
            report.append(f"  Average similarity: {result.average_similarity:.4f}")
            report.append(f"  Computational cost: {result.computational_cost:.2f}s")
        
        # Security assessment
        report.append(f"\n{'='*30} SECURITY ASSESSMENT {'='*30}")
        
        critical_scenarios = [r for r in results if r.detection_rate < 0.5]
        if critical_scenarios:
            report.append("\n⚠ CRITICAL VULNERABILITIES:")
            for r in critical_scenarios:
                report.append(f"  - {r.scenario_name}: Only {r.detection_rate:.2%} detection rate")
        
        moderate_scenarios = [r for r in results if 0.5 <= r.detection_rate < 0.7]
        if moderate_scenarios:
            report.append("\n⚠ MODERATE VULNERABILITIES:")
            for r in moderate_scenarios:
                report.append(f"  - {r.scenario_name}: {r.detection_rate:.2%} detection rate")
        
        strong_scenarios = [r for r in results if r.detection_rate >= 0.7]
        if strong_scenarios:
            report.append("\n✓ STRONG DEFENSES:")
            for r in strong_scenarios:
                report.append(f"  - {r.scenario_name}: {r.detection_rate:.2%} detection rate")
        
        # Recommendations
        report.append(f"\n{'='*30} RECOMMENDATIONS {'='*30}")
        
        if overall_detection < 0.7:
            report.append("\n1. URGENT: Strengthen detection mechanisms")
            report.append("   - Increase challenge complexity")
            report.append("   - Add more verification rounds")
            report.append("   - Implement adaptive thresholds")
        
        if any(r.attack_type == 'wrapper' and r.detection_rate < 0.6 for r in results):
            report.append("\n2. Improve wrapper detection:")
            report.append("   - Add behavioral analysis")
            report.append("   - Implement timing analysis")
            report.append("   - Use ensemble verification")
        
        if any(r.attack_type == 'fine_tuning' and r.detection_rate < 0.6 for r in results):
            report.append("\n3. Enhance fine-tuning detection:")
            report.append("   - Increase parameter sensitivity")
            report.append("   - Add gradient analysis")
            report.append("   - Implement weight distribution checks")
        
        if results and results[0].false_positive_rate > 0.05:
            report.append("\n4. Reduce false positives:")
            report.append("   - Calibrate thresholds")
            report.append("   - Add tolerance for numerical precision")
            report.append("   - Implement confidence intervals")
        
        # Save report
        report_text = '\n'.join(report)
        output_file = self.output_dir / f"attack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {output_file}")
        
        # Also print to console
        print("\n" + report_text)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run attack simulations against PoT')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of rounds per scenario')
    parser.add_argument('--output', type=str, default='./attack_results',
                       help='Output directory for results')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       help='Specific scenarios to run (default: all)')
    
    args = parser.parse_args()
    
    # Initialize harness
    harness = AttackHarness(output_dir=args.output)
    
    # Filter scenarios if specified
    if args.scenarios:
        harness.scenarios = [s for s in harness.scenarios 
                            if s.name in args.scenarios]
    
    # Run simulations
    results = harness.run_all_scenarios(n_rounds=args.rounds)
    
    # Return success/failure for CI/CD
    if results:
        overall_detection = np.mean([r.detection_rate for r in results])
        
        if overall_detection < 0.5:
            print("\n❌ CRITICAL: Detection rate below 50%")
            sys.exit(1)
        elif overall_detection < 0.7:
            print("\n⚠ WARNING: Detection rate below 70%")
            sys.exit(0)
        else:
            print("\n✓ PASSED: Adequate detection rate")
            sys.exit(0)
    else:
        print("\n❌ ERROR: No results generated")
        sys.exit(1)

if __name__ == "__main__":
    main()