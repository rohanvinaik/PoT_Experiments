#!/usr/bin/env python3
"""
Custom Challenge Creation Example

This script demonstrates how to create and integrate custom
challenge types into the vision verification framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from pot.vision.verifier import EnhancedVisionVerifier


class CustomChallenger:
    """Custom challenge generator for specialized verification."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
    
    def generate_edge_detection_challenge(self, size=(224, 224), edge_strength=1.0):
        """Generate edge detection challenge using Sobel filters."""
        
        h, w = size
        
        # Create base pattern with geometric shapes
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Create multiple geometric shapes
        patterns = []
        
        # Circle
        circle = (xx**2 + yy**2 < 0.3).float()
        
        # Rectangle
        rectangle = ((torch.abs(xx) < 0.4) & (torch.abs(yy) < 0.2)).float()
        
        # Triangle (approximated)
        triangle = ((yy > -0.5) & (yy < xx + 0.5) & (yy < -xx + 0.5)).float()
        
        # Combine patterns
        pattern = circle + rectangle * 0.7 + triangle * 0.5
        pattern = torch.clamp(pattern, 0, 1)
        
        # Apply edge enhancement
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Apply Sobel filters
        pattern_expanded = pattern.unsqueeze(0).unsqueeze(0)
        
        edges_x = F.conv2d(pattern_expanded, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        edges_y = F.conv2d(pattern_expanded, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        # Combine edge responses
        edges = torch.sqrt(edges_x**2 + edges_y**2).squeeze()
        edges = edges * edge_strength
        
        # Normalize and convert to RGB
        edges = torch.clamp(edges, 0, 1)
        rgb_image = edges.unsqueeze(0).repeat(3, 1, 1)
        
        return rgb_image.to(self.device)
    
    def generate_color_constancy_challenge(self, size=(224, 224), illumination='daylight'):
        """Generate color constancy challenge with different illuminations."""
        
        h, w = size
        
        # Create a scene with multiple colored objects
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Create colored regions
        r_channel = torch.zeros(h, w)
        g_channel = torch.zeros(h, w)
        b_channel = torch.zeros(h, w)
        
        # Red object (top-left)
        red_mask = ((xx < 0.4) & (yy < 0.4))
        r_channel[red_mask] = 0.8
        
        # Green object (top-right)
        green_mask = ((xx > 0.6) & (yy < 0.4))
        g_channel[green_mask] = 0.8
        
        # Blue object (bottom-center)
        blue_mask = ((xx > 0.3) & (xx < 0.7) & (yy > 0.6))
        b_channel[blue_mask] = 0.8
        
        # White object (center)
        white_mask = ((xx > 0.4) & (xx < 0.6) & (yy > 0.4) & (yy < 0.6))
        r_channel[white_mask] = 0.9
        g_channel[white_mask] = 0.9
        b_channel[white_mask] = 0.9
        
        # Apply illumination
        illumination_colors = {
            'daylight': torch.tensor([1.0, 1.0, 1.0]),
            'tungsten': torch.tensor([1.2, 0.9, 0.6]),
            'fluorescent': torch.tensor([0.9, 1.1, 1.0]),
            'sunset': torch.tensor([1.3, 0.8, 0.5])
        }
        
        illum = illumination_colors.get(illumination, torch.tensor([1.0, 1.0, 1.0]))
        
        r_channel *= illum[0]
        g_channel *= illum[1]
        b_channel *= illum[2]
        
        # Stack channels and normalize
        rgb_image = torch.stack([r_channel, g_channel, b_channel])
        rgb_image = torch.clamp(rgb_image, 0, 1)
        
        return rgb_image.to(self.device)
    
    def generate_motion_blur_challenge(self, size=(224, 224), blur_length=15, angle=0):
        """Generate motion blur challenge."""
        
        h, w = size
        
        # Create a scene with multiple objects
        pattern = torch.zeros(h, w)
        
        # Add various shapes and patterns
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Checkerboard pattern
        checker_size = 20
        checker = torch.zeros(h, w)
        for i in range(0, h, checker_size):
            for j in range(0, w, checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    checker[i:i+checker_size, j:j+checker_size] = 1
        
        pattern += checker * 0.8
        
        # Add some vertical lines
        line_positions = [w//4, w//2, 3*w//4]
        for pos in line_positions:
            if 0 <= pos < w:
                pattern[:, max(0, pos-2):min(w, pos+2)] = 1.0
        
        # Create motion blur kernel
        kernel_size = blur_length
        kernel = torch.zeros(kernel_size, kernel_size)
        
        # Calculate line coordinates for motion blur
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        
        for i in range(kernel_size):
            offset = i - center
            x_offset = int(offset * np.cos(angle_rad))
            y_offset = int(offset * np.sin(angle_rad))
            
            x_pos = center + x_offset
            y_pos = center + y_offset
            
            if 0 <= x_pos < kernel_size and 0 <= y_pos < kernel_size:
                kernel[y_pos, x_pos] = 1.0
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Apply motion blur
        pattern_expanded = pattern.unsqueeze(0).unsqueeze(0)
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)
        
        blurred = F.conv2d(pattern_expanded, kernel_expanded, 
                          padding=kernel_size//2)
        blurred = blurred.squeeze()
        
        # Convert to RGB
        rgb_image = blurred.unsqueeze(0).repeat(3, 1, 1)
        rgb_image = torch.clamp(rgb_image, 0, 1)
        
        return rgb_image.to(self.device)
    
    def generate_depth_of_field_challenge(self, size=(224, 224), focal_plane=0.5, blur_strength=2.0):
        """Generate depth of field challenge with focus effects."""
        
        h, w = size
        
        # Create depth map (distance from camera)
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Create scene with objects at different depths
        scene = torch.zeros(h, w)
        depth_map = torch.ones(h, w) * 0.8  # Background depth
        
        # Foreground object (closer, depth = 0.2)
        fg_mask = ((xx - 0.3)**2 + (yy - 0.7)**2) < 0.05
        scene[fg_mask] = 0.9
        depth_map[fg_mask] = 0.2
        
        # Mid-ground object (medium depth = 0.5)
        mg_mask = ((xx - 0.7)**2 + (yy - 0.3)**2) < 0.08
        scene[mg_mask] = 0.7
        depth_map[mg_mask] = 0.5
        
        # Background objects (far, depth = 0.8-1.0)
        bg_mask = ((xx - 0.2)**2 + (yy - 0.2)**2) < 0.03
        scene[bg_mask] = 0.6
        depth_map[bg_mask] = 0.9
        
        # Add some texture to background
        noise = torch.rand(h, w) * 0.3
        scene += noise * (depth_map > 0.7).float()
        
        # Calculate blur amount based on distance from focal plane
        blur_amount = torch.abs(depth_map - focal_plane) * blur_strength
        
        # Apply variable blur (simplified approach)
        blurred_scene = scene.clone()
        
        # Create different blur kernels
        for blur_level in [1, 2, 3, 4]:
            mask = (blur_amount > blur_level - 0.5) & (blur_amount <= blur_level + 0.5)
            if mask.any():
                kernel_size = 2 * blur_level + 1
                kernel = torch.ones(kernel_size, kernel_size) / (kernel_size**2)
                
                # Apply blur to masked region
                scene_expanded = scene.unsqueeze(0).unsqueeze(0)
                kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)
                
                blur_result = F.conv2d(scene_expanded, kernel_expanded, 
                                     padding=kernel_size//2)
                blur_result = blur_result.squeeze()
                
                blurred_scene[mask] = blur_result[mask]
        
        # Convert to RGB with slight color variation
        r_channel = blurred_scene * 1.0
        g_channel = blurred_scene * 0.95
        b_channel = blurred_scene * 0.9
        
        rgb_image = torch.stack([r_channel, g_channel, b_channel])
        rgb_image = torch.clamp(rgb_image, 0, 1)
        
        return rgb_image.to(self.device)


def create_custom_verifier():
    """Create a verifier with custom challenge integration."""
    
    # Create a test model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    model.eval()
    
    # Create verifier configuration
    config = {
        'device': 'cpu',
        'verification_method': 'batch',
        'temperature': 1.0
    }
    
    verifier = EnhancedVisionVerifier(model, config)
    
    return verifier


def demonstrate_custom_challenges():
    """Demonstrate custom challenge generation and usage."""
    
    print("Custom Challenge Generation Demo")
    print("=" * 50)
    
    # Create custom challenger
    challenger = CustomChallenger(device='cpu')
    
    print("\\n1. Generating custom challenges...")
    
    # Generate different types of custom challenges
    challenges = {
        'Edge Detection': challenger.generate_edge_detection_challenge(
            size=(128, 128), edge_strength=1.5
        ),
        'Color Constancy (Daylight)': challenger.generate_color_constancy_challenge(
            size=(128, 128), illumination='daylight'
        ),
        'Color Constancy (Tungsten)': challenger.generate_color_constancy_challenge(
            size=(128, 128), illumination='tungsten'
        ),
        'Motion Blur (Horizontal)': challenger.generate_motion_blur_challenge(
            size=(128, 128), blur_length=15, angle=0
        ),
        'Motion Blur (Diagonal)': challenger.generate_motion_blur_challenge(
            size=(128, 128), blur_length=15, angle=45
        ),
        'Depth of Field (Near)': challenger.generate_depth_of_field_challenge(
            size=(128, 128), focal_plane=0.3, blur_strength=3.0
        ),
        'Depth of Field (Far)': challenger.generate_depth_of_field_challenge(
            size=(128, 128), focal_plane=0.8, blur_strength=2.0
        )
    }
    
    print(f"✓ Generated {len(challenges)} custom challenges")
    
    # Visualize challenges
    print("\\n2. Visualizing custom challenges...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Custom Vision Verification Challenges', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (name, challenge) in enumerate(challenges.items()):
        if i < len(axes):
            # Convert tensor to numpy for visualization
            if challenge.shape[0] == 3:  # RGB
                img = challenge.permute(1, 2, 0).cpu().numpy()
            else:
                img = challenge.squeeze().cpu().numpy()
            
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(name, fontsize=10, fontweight='bold')
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(challenges), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path('/tmp/custom_challenges_demo')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'custom_challenges.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualization saved to {output_dir / 'custom_challenges.png'}")
    
    return challenges


def test_custom_challenges_with_verifier(challenges):
    """Test custom challenges with the verifier."""
    
    print("\\n3. Testing custom challenges with verifier...")
    
    # Create verifier
    verifier = create_custom_verifier()
    
    print(f"✓ Verifier created with model: {type(verifier.model).__name__}")
    
    # Test each custom challenge
    results = {}
    
    for challenge_name, challenge in challenges.items():
        print(f"\\n   Testing {challenge_name}...")
        
        try:
            # Run model on challenge
            challenge_batch = challenge.unsqueeze(0)  # Add batch dimension
            output = verifier.run_model(challenge_batch)
            
            # Analyze output
            logits = output['logits']
            prediction = torch.argmax(logits, dim=1)
            confidence = torch.softmax(logits, dim=1).max()
            
            # Get embeddings if available
            embeddings = output.get('embeddings', {})
            embedding_info = f"{len(embeddings)} layers" if embeddings else "none"
            
            results[challenge_name] = {
                'prediction': prediction.item(),
                'confidence': confidence.item(),
                'inference_time': output.get('inference_time', 0),
                'embeddings': embedding_info,
                'logits_range': (logits.min().item(), logits.max().item()),
                'success': True
            }
            
            print(f"     ✓ Prediction: {prediction.item()}")
            print(f"     ✓ Confidence: {confidence.item():.3f}")
            print(f"     ✓ Inference time: {output.get('inference_time', 0):.4f}s")
            print(f"     ✓ Embeddings: {embedding_info}")
            
        except Exception as e:
            print(f"     ✗ Failed: {e}")
            results[challenge_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def analyze_custom_challenge_results(results):
    """Analyze results from custom challenge testing."""
    
    print("\\n4. Analyzing custom challenge results...")
    print("-" * 40)
    
    successful_tests = [name for name, result in results.items() if result.get('success', False)]
    failed_tests = [name for name, result in results.items() if not result.get('success', False)]
    
    print(f"✓ Successful tests: {len(successful_tests)}/{len(results)}")
    
    if failed_tests:
        print(f"✗ Failed tests: {len(failed_tests)}")
        for test in failed_tests:
            print(f"   - {test}: {results[test].get('error', 'Unknown error')}")
    
    if successful_tests:
        # Analyze successful results
        confidences = [results[test]['confidence'] for test in successful_tests]
        predictions = [results[test]['prediction'] for test in successful_tests]
        inference_times = [results[test]['inference_time'] for test in successful_tests]
        
        print(f"\\nResult Analysis:")
        print(f"  Average confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
        print(f"  Prediction range: {min(predictions)} - {max(predictions)}")
        print(f"  Average inference time: {np.mean(inference_times):.4f}s")
        
        # Find most and least confident predictions
        most_confident_idx = np.argmax(confidences)
        least_confident_idx = np.argmin(confidences)
        
        most_confident_test = successful_tests[most_confident_idx]
        least_confident_test = successful_tests[least_confident_idx]
        
        print(f"\\n  Most confident: {most_confident_test} ({confidences[most_confident_idx]:.3f})")
        print(f"  Least confident: {least_confident_test} ({confidences[least_confident_idx]:.3f})")
        
        # Analyze prediction distribution
        from collections import Counter
        pred_counts = Counter(predictions)
        print(f"\\n  Prediction distribution:")
        for pred, count in sorted(pred_counts.items()):
            print(f"    Class {pred}: {count} times")


def create_custom_challenge_integration_example():
    """Show how to integrate custom challenges into the verification framework."""
    
    print("\\n5. Custom Challenge Integration Example...")
    print("-" * 50)
    
    print(\"\"\"
To integrate custom challenges into the verification framework:

1. Create a custom challenger class:
   
   class MyCustomChallenger:
       def __init__(self, device='cpu'):
           self.device = device
       
       def generate_my_challenge(self, size=(224, 224), **kwargs):
           # Your custom challenge generation logic
           return challenge_tensor

2. Extend the verifier to support custom challenges:
   
   class ExtendedVisionVerifier(EnhancedVisionVerifier):
       def __init__(self, model, config):
           super().__init__(model, config)
           self.custom_challenger = MyCustomChallenger(device=config['device'])
       
       def generate_custom_challenges(self, num_challenges, challenge_type):
           challenges = []
           for i in range(num_challenges):
               if challenge_type == 'my_custom_type':
                   challenge = self.custom_challenger.generate_my_challenge()
                   challenges.append(challenge)
           return challenges

3. Use in verification:
   
   verifier = ExtendedVisionVerifier(model, config)
   result = verifier.verify_session(
       num_challenges=10,
       challenge_types=['frequency', 'texture', 'my_custom_type']
   )

Key considerations for custom challenges:
• Ensure challenges are deterministic for reproducibility
• Generate challenges that test specific model capabilities
• Consider computational efficiency for large-scale verification
• Validate challenges produce meaningful verification signals
• Document challenge parameters and expected behaviors
\"\"\"
    )


def main():
    """Run custom challenge demonstration."""
    
    print("Vision Verification Custom Challenges")
    print("=" * 60)
    
    # Demonstrate custom challenge generation
    challenges = demonstrate_custom_challenges()
    
    # Test challenges with verifier
    results = test_custom_challenges_with_verifier(challenges)
    
    # Analyze results
    analyze_custom_challenge_results(results)
    
    # Show integration example
    create_custom_challenge_integration_example()
    
    # Summary
    print("\\n" + "=" * 60)
    print("CUSTOM CHALLENGES DEMO SUMMARY")
    print("=" * 60)
    
    successful_count = sum(1 for r in results.values() if r.get('success', False))
    
    print(f"✓ Custom challenge types created: {len(challenges)}")
    print(f"✓ Challenges tested successfully: {successful_count}/{len(results)}")
    print(f"✓ Challenge categories demonstrated:")
    print(f"   - Edge detection and feature enhancement")
    print(f"   - Color constancy under different illuminations")
    print(f"   - Motion blur with directional effects")
    print(f"   - Depth of field and focus simulation")
    
    print(f"\\n📋 Key Takeaways:")
    print(f"   • Custom challenges enable specialized model testing")
    print(f"   • Challenge design should target specific model capabilities")
    print(f"   • Integration with existing framework is straightforward")
    print(f"   • Visualization helps validate challenge quality")
    
    print(f"\\n📁 Files generated:")
    output_dir = Path('/tmp/custom_challenges_demo')
    if output_dir.exists():
        files = list(output_dir.glob('*'))
        for file in files:
            print(f"   📄 {file.name}")
    
    print(f"\\n{'=' * 60}")
    print("Custom challenges demonstration completed!")
    
    return 0


if __name__ == "__main__":
    exit(main())