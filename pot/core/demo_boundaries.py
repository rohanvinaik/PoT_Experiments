"""Demo of anytime-valid confidence sequence boundaries."""

import numpy as np
from pot.core.boundaries import CSState, eb_radius, decide_one_sided, SequentialTest

def demo_anytime_validity():
    """Demonstrate anytime-valid property of confidence sequences."""
    print("Demonstrating Anytime-Valid Confidence Sequences")
    print("=" * 50)
    
    # Set up test
    true_mean = 0.6
    threshold = 0.5
    alpha = 0.05
    
    print(f"True mean: {true_mean}")
    print(f"Threshold: {threshold}")
    print(f"Alpha: {alpha}")
    print()
    
    # Run sequential test
    test = SequentialTest(threshold=threshold, alpha=alpha)
    
    print("Sample | Mean  | CI Lower | CI Upper | Decision")
    print("-" * 50)
    
    np.random.seed(42)
    for i in range(50):
        # Generate observation from true distribution
        z = np.random.beta(6, 4)  # Beta(6,4) has mean 0.6
        decision = test.update(z)
        
        if (i + 1) % 5 == 0 or decision != "continue":
            ci = test.get_confidence_interval()
            print(f"{i+1:6d} | {test.state.mean:.3f} | {ci[0]:8.3f} | {ci[1]:8.3f} | {decision}")
            
            if decision != "continue":
                print()
                print(f"Final decision after {i+1} samples: {decision}")
                break
    
    # Show that we can stop at any time
    print()
    print("Key property: The confidence interval is valid at ANY stopping time.")
    print("This allows for continuous monitoring without p-value adjustment.")

if __name__ == "__main__":
    demo_anytime_validity()