"""
Specialized Challenge Generators for Vision Models
Implements frequency, texture, and natural image challengers for vision model verification.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Union
import time

# Optional imports with fallbacks
try:
    from scipy import signal
    from scipy.spatial import Voronoi, distance_matrix
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class FrequencyChallenger:
    """Generate frequency-domain challenges for vision models."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def generate_fourier_pattern(self, 
                                size: Tuple[int, int],
                                frequency_range: Tuple[float, float],
                                num_components: int = 5) -> torch.Tensor:
        """
        Generate image using Fourier synthesis.
        Args:
            size: Image size (H, W)
            frequency_range: (min_freq, max_freq) in cycles per image
            num_components: Number of frequency components
        """
        h, w = size
        image = np.zeros((h, w))
        
        for _ in range(num_components):
            # Random frequency in specified range
            freq_x = np.random.uniform(*frequency_range)
            freq_y = np.random.uniform(*frequency_range)
            
            # Random phase
            phase = np.random.uniform(0, 2*np.pi)
            
            # Random amplitude
            amplitude = np.random.uniform(0.5, 1.0)
            
            # Generate component
            x = np.linspace(0, freq_x*2*np.pi, w)
            y = np.linspace(0, freq_y*2*np.pi, h)
            xx, yy = np.meshgrid(x, y)
            
            component = amplitude * np.sin(xx + yy + phase)
            image += component
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to RGB tensor
        tensor = torch.from_numpy(image).float()
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return tensor.to(self.device)
    
    def generate_gabor_filter_bank(self, 
                                   size: Tuple[int, int],
                                   orientations: int = 8,
                                   scales: int = 4) -> torch.Tensor:
        """
        Generate Gabor filter bank patterns.
        """
        h, w = size
        filters = []
        
        for scale in range(scales):
            frequency = 0.05 * (2 ** scale)
            
            for orientation in range(orientations):
                theta = orientation * np.pi / orientations
                
                # Generate Gabor kernel
                kernel = self._create_gabor_kernel(
                    size=(h, w),
                    frequency=frequency,
                    theta=theta
                )
                filters.append(kernel)
        
        # Combine filters
        combined = torch.stack(filters).mean(dim=0)
        
        # Convert to RGB
        combined = combined.unsqueeze(0).repeat(3, 1, 1)
        
        return combined.to(self.device)
    
    def _create_gabor_kernel(self, 
                            size: Tuple[int, int],
                            frequency: float,
                            theta: float,
                            sigma: float = None) -> torch.Tensor:
        """Create a Gabor kernel."""
        h, w = size
        
        if sigma is None:
            sigma = 0.56 * w / frequency
        
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        xx, yy = np.meshgrid(x, y)
        
        # Rotate coordinates
        x_theta = xx * np.cos(theta) + yy * np.sin(theta)
        y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
        
        # Gabor function
        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * frequency * x_theta)
        
        gabor = gaussian * sinusoid
        
        # Normalize
        gabor = (gabor - gabor.mean()) / gabor.std()
        
        return torch.from_numpy(gabor).float()
    
    def apply_frequency_mask(self, 
                            image: torch.Tensor,
                            mask_type: str = 'lowpass',
                            cutoff: float = 0.3) -> torch.Tensor:
        """
        Apply frequency domain mask to image.
        Args:
            mask_type: 'lowpass', 'highpass', 'bandpass', 'notch'
            cutoff: Normalized frequency cutoff
        """
        # Convert to numpy for FFT
        if image.dim() == 4:
            image = image.squeeze(0)
        
        result = []
        for channel in range(image.shape[0]):
            img_np = image[channel].cpu().numpy()
            
            # FFT
            f_transform = np.fft.fft2(img_np)
            f_shift = np.fft.fftshift(f_transform)
            
            # Create mask
            h, w = img_np.shape
            mask = self._create_frequency_mask(h, w, mask_type, cutoff)
            
            # Apply mask
            f_shift_masked = f_shift * mask
            
            # Inverse FFT
            f_ishift = np.fft.ifftshift(f_shift_masked)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            result.append(torch.from_numpy(img_back).float())
        
        return torch.stack(result).to(self.device)
    
    def _create_frequency_mask(self, h: int, w: int, 
                              mask_type: str, cutoff: float) -> np.ndarray:
        """Create frequency domain mask."""
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Normalize distance
        max_dist = np.sqrt(cx**2 + cy**2)
        dist_norm = dist / max_dist
        
        if mask_type == 'lowpass':
            mask = dist_norm <= cutoff
        elif mask_type == 'highpass':
            mask = dist_norm >= cutoff
        elif mask_type == 'bandpass':
            mask = (dist_norm >= cutoff*0.5) & (dist_norm <= cutoff*1.5)
        elif mask_type == 'notch':
            mask = ~((dist_norm >= cutoff*0.9) & (dist_norm <= cutoff*1.1))
        else:
            mask = np.ones((h, w))
            
        return mask.astype(float)
    
    def generate_sine_gratings(self,
                              size: Tuple[int, int],
                              frequency: float,
                              orientation: float = 0.0,
                              phase: float = 0.0,
                              contrast: float = 1.0) -> torch.Tensor:
        """Generate sine grating patterns."""
        h, w = size
        
        # Create coordinate system
        x = np.linspace(-np.pi, np.pi, w)
        y = np.linspace(-np.pi, np.pi, h)
        xx, yy = np.meshgrid(x, y)
        
        # Rotate coordinates
        x_rot = xx * np.cos(orientation) + yy * np.sin(orientation)
        
        # Generate sine grating
        grating = np.sin(frequency * x_rot + phase) * contrast
        
        # Normalize to [0, 1]
        grating = (grating + 1) / 2
        
        # Convert to RGB tensor
        tensor = torch.from_numpy(grating).float()
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return tensor.to(self.device)
    
    def generate_chirp_pattern(self,
                              size: Tuple[int, int],
                              f0: float = 0.1,
                              f1: float = 10.0) -> torch.Tensor:
        """Generate chirp pattern with varying frequency."""
        h, w = size
        
        # Create linear frequency sweep
        t = np.linspace(0, 1, w)
        freq = f0 + (f1 - f0) * t
        
        # Generate chirp signal
        chirp = np.zeros((h, w))
        for i, f in enumerate(freq):
            y_component = np.sin(2 * np.pi * f * np.linspace(0, 1, h))
            chirp[:, i] = y_component
        
        # Normalize
        chirp = (chirp - chirp.min()) / (chirp.max() - chirp.min())
        
        # Convert to RGB tensor
        tensor = torch.from_numpy(chirp).float()
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return tensor.to(self.device)


class TextureChallenger:
    """Generate texture-based challenges for vision models."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def generate_perlin_noise(self, 
                             size: Tuple[int, int],
                             scale: float = 100,
                             octaves: int = 4,
                             persistence: float = 0.5,
                             seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate Perlin noise texture.
        """
        if seed is not None:
            np.random.seed(seed)
            
        h, w = size
        
        # Generate base noise
        noise = np.zeros((h, w))
        
        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave
            
            # Generate random gradients
            grad_h = int(h / scale * freq) + 1
            grad_w = int(w / scale * freq) + 1
            
            gradients = np.random.randn(grad_h, grad_w, 2)
            
            # Interpolate
            octave_noise = self._interpolate_noise(gradients, (h, w))
            noise += octave_noise * amp
        
        # Normalize
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Convert to RGB tensor
        tensor = torch.from_numpy(noise).float()
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return tensor.to(self.device)
    
    def _interpolate_noise(self, gradients: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """Interpolate noise from gradients."""
        grad_h, grad_w, _ = gradients.shape
        h, w = output_size
        
        # Create output grid
        noise = np.zeros((h, w))
        
        # Scale factors
        scale_h = grad_h / h
        scale_w = grad_w / w
        
        for i in range(h):
            for j in range(w):
                # Grid coordinates
                gi = i * scale_h
                gj = j * scale_w
                
                # Grid indices
                gi0, gj0 = int(gi), int(gj)
                gi1, gj1 = min(gi0 + 1, grad_h - 1), min(gj0 + 1, grad_w - 1)
                
                # Fractional parts
                fi, fj = gi - gi0, gj - gj0
                
                # Distance vectors
                d00 = np.array([fj, fi])
                d10 = np.array([fj - 1, fi])
                d01 = np.array([fj, fi - 1])
                d11 = np.array([fj - 1, fi - 1])
                
                # Dot products
                n00 = np.dot(gradients[gi0, gj0], d00)
                n10 = np.dot(gradients[gi0, gj1], d10)
                n01 = np.dot(gradients[gi1, gj0], d01)
                n11 = np.dot(gradients[gi1, gj1], d11)
                
                # Interpolation weights (fade function)
                u = self._fade(fj)
                v = self._fade(fi)
                
                # Bilinear interpolation
                nx0 = n00 * (1 - u) + n10 * u
                nx1 = n01 * (1 - u) + n11 * u
                noise[i, j] = nx0 * (1 - v) + nx1 * v
        
        return noise
    
    def _fade(self, t: float) -> float:
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def generate_voronoi_texture(self, 
                                size: Tuple[int, int],
                                num_points: int = 50,
                                color_mode: str = 'random',
                                seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate Voronoi diagram texture.
        """
        if seed is not None:
            np.random.seed(seed)
            
        h, w = size
        
        # Random seed points
        points = np.random.rand(num_points, 2)
        points[:, 0] *= w
        points[:, 1] *= h
        
        # Create grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Find nearest seed point for each grid point
        if SCIPY_AVAILABLE:
            distances = distance_matrix(grid_points, points)
        else:
            # Fallback implementation
            distances = np.zeros((grid_points.shape[0], num_points))
            for i, point in enumerate(points):
                diff = grid_points - point
                distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))
        
        nearest = np.argmin(distances, axis=1)
        
        # Reshape to image
        voronoi_img = nearest.reshape(h, w)
        
        # Assign colors
        if color_mode == 'random':
            colors = np.random.rand(num_points, 3)
        elif color_mode == 'gradient':
            colors = np.array([[i/num_points, 0.5, 1-i/num_points] 
                              for i in range(num_points)])
        elif color_mode == 'distance':
            # Color based on distance to nearest point
            min_distances = np.min(distances, axis=1).reshape(h, w)
            colors = np.ones((num_points, 3))
            voronoi_img = min_distances
            voronoi_img = (voronoi_img - voronoi_img.min()) / (voronoi_img.max() - voronoi_img.min())
            
            # Convert to RGB
            rgb_image = np.stack([voronoi_img, voronoi_img, voronoi_img], axis=-1)
            tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1)
            return tensor.to(self.device)
        else:
            colors = np.random.rand(num_points, 3)
        
        # Create RGB image
        rgb_image = colors[voronoi_img]
        
        # Convert to tensor
        tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1)
        
        return tensor.to(self.device)
    
    def generate_fractal_texture(self, 
                                size: Tuple[int, int],
                                fractal_type: str = 'julia',
                                iterations: int = 256,
                                c: complex = -0.7 + 0.27015j) -> torch.Tensor:
        """
        Generate fractal texture (Julia or Mandelbrot set).
        """
        h, w = size
        
        if fractal_type == 'julia':
            image = self._julia_set(h, w, iterations, c)
        elif fractal_type == 'mandelbrot':
            image = self._mandelbrot_set(h, w, iterations)
        elif fractal_type == 'burning_ship':
            image = self._burning_ship_set(h, w, iterations)
        else:
            raise ValueError(f"Unknown fractal type: {fractal_type}")
        
        # Normalize and colorize
        image = np.log(image + 1)  # Log scale for better visualization
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create RGB with color gradient
        rgb = np.zeros((h, w, 3))
        rgb[:, :, 0] = image * 0.5
        rgb[:, :, 1] = image * 0.8
        rgb[:, :, 2] = image
        
        # Convert to tensor
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)
        
        return tensor.to(self.device)
    
    def _julia_set(self, h: int, w: int, iterations: int, 
                   c: complex = -0.7 + 0.27015j) -> np.ndarray:
        """Generate Julia set."""
        # Create complex plane
        x = np.linspace(-2, 2, w)
        y = np.linspace(-2, 2, h)
        xx, yy = np.meshgrid(x, y)
        z = xx + 1j * yy
        
        # Iterate
        output = np.zeros((h, w))
        for i in range(iterations):
            mask = np.abs(z) < 2
            z[mask] = z[mask]**2 + c
            output[mask] = i
        
        return output
    
    def _mandelbrot_set(self, h: int, w: int, iterations: int) -> np.ndarray:
        """Generate Mandelbrot set."""
        # Create complex plane
        x = np.linspace(-2.5, 1.5, w)
        y = np.linspace(-2, 2, h)
        xx, yy = np.meshgrid(x, y)
        c = xx + 1j * yy
        z = np.zeros_like(c)
        
        # Iterate
        output = np.zeros((h, w))
        for i in range(iterations):
            mask = np.abs(z) < 2
            z[mask] = z[mask]**2 + c[mask]
            output[mask] = i
        
        return output
    
    def _burning_ship_set(self, h: int, w: int, iterations: int) -> np.ndarray:
        """Generate Burning Ship fractal."""
        # Create complex plane
        x = np.linspace(-2.5, 1.5, w)
        y = np.linspace(-2, 2, h)
        xx, yy = np.meshgrid(x, y)
        c = xx + 1j * yy
        z = np.zeros_like(c)
        
        # Iterate
        output = np.zeros((h, w))
        for i in range(iterations):
            mask = np.abs(z) < 2
            z[mask] = (np.abs(z[mask].real) + 1j * np.abs(z[mask].imag))**2 + c[mask]
            output[mask] = i
        
        return output
    
    def generate_cellular_automata(self,
                                  size: Tuple[int, int],
                                  rule: int = 30,
                                  iterations: int = 100,
                                  seed: Optional[int] = None) -> torch.Tensor:
        """Generate cellular automata texture."""
        if seed is not None:
            np.random.seed(seed)
            
        h, w = size
        
        # Initialize with random state or single cell
        if rule == 30:  # Rule 30 - chaotic
            grid = np.random.randint(0, 2, w)
        else:
            grid = np.zeros(w, dtype=int)
            grid[w//2] = 1
        
        # Create output array
        automata = np.zeros((h, w))
        
        for i in range(min(h, iterations)):
            automata[i] = grid.copy()
            
            # Apply rule
            new_grid = np.zeros_like(grid)
            for j in range(w):
                left = grid[(j-1) % w]
                center = grid[j]
                right = grid[(j+1) % w]
                
                # Convert to binary representation
                neighborhood = left * 4 + center * 2 + right
                
                # Apply rule
                new_grid[j] = (rule >> neighborhood) & 1
            
            grid = new_grid
        
        # Fill remaining rows
        for i in range(iterations, h):
            automata[i] = grid
        
        # Convert to RGB tensor
        tensor = torch.from_numpy(automata).float()
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return tensor.to(self.device)
    
    def generate_reaction_diffusion(self,
                                   size: Tuple[int, int],
                                   steps: int = 1000,
                                   Da: float = 1.0,
                                   Db: float = 0.5,
                                   f: float = 0.055,
                                   k: float = 0.062) -> torch.Tensor:
        """Generate reaction-diffusion texture (Gray-Scott model)."""
        h, w = size
        
        # Initialize concentration arrays
        A = np.ones((h, w))
        B = np.zeros((h, w))
        
        # Add initial perturbation
        center_h, center_w = h // 2, w // 2
        r = min(h, w) // 10
        for i in range(center_h - r, center_h + r):
            for j in range(center_w - r, center_w + r):
                if 0 <= i < h and 0 <= j < w:
                    if (i - center_h)**2 + (j - center_w)**2 <= r**2:
                        A[i, j] = 0.5
                        B[i, j] = 0.25
        
        # Time evolution
        dt = 1.0
        for _ in range(steps):
            # Laplacian (diffusion)
            lA = self._laplacian(A)
            lB = self._laplacian(B)
            
            # Reaction-diffusion equations
            dA = Da * lA - A * B * B + f * (1 - A)
            dB = Db * lB + A * B * B - (k + f) * B
            
            A += dA * dt
            B += dB * dt
            
            # Clamp values
            A = np.clip(A, 0, 1)
            B = np.clip(B, 0, 1)
        
        # Create RGB visualization
        rgb = np.zeros((h, w, 3))
        rgb[:, :, 0] = A
        rgb[:, :, 1] = B
        rgb[:, :, 2] = (A + B) / 2
        
        # Convert to tensor
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)
        
        return tensor.to(self.device)
    
    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute Laplacian using finite differences."""
        lapl = np.zeros_like(field)
        
        # Interior points
        lapl[1:-1, 1:-1] = (field[2:, 1:-1] + field[:-2, 1:-1] + 
                           field[1:-1, 2:] + field[1:-1, :-2] - 
                           4 * field[1:-1, 1:-1])
        
        # Boundary conditions (Neumann)
        lapl[0, :] = lapl[1, :]
        lapl[-1, :] = lapl[-2, :]
        lapl[:, 0] = lapl[:, 1]
        lapl[:, -1] = lapl[:, -2]
        
        return lapl


class NaturalImageChallenger:
    """Generate natural-looking synthetic images."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def generate_synthetic_natural(self, 
                                  size: Tuple[int, int],
                                  scene_type: str = 'landscape',
                                  seed: Optional[int] = None) -> torch.Tensor:
        """Generate synthetic natural-looking images."""
        if seed is not None:
            np.random.seed(seed)
        
        if scene_type == 'landscape':
            return self._generate_landscape(size)
        elif scene_type == 'clouds':
            return self._generate_clouds(size)
        elif scene_type == 'abstract':
            return self._generate_abstract(size)
        elif scene_type == 'water':
            return self._generate_water(size)
        elif scene_type == 'forest':
            return self._generate_forest(size)
        else:
            return self._generate_landscape(size)
            
    def _generate_landscape(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate simple landscape."""
        h, w = size
        image = np.zeros((h, w, 3))
        
        # Sky gradient (blue to light blue)
        for i in range(h//2):
            progress = i / (h//2)
            sky_color = [
                0.5 + 0.3 * progress,  # Red
                0.7 + 0.2 * progress,  # Green  
                0.9 + 0.1 * progress   # Blue
            ]
            image[i, :] = sky_color
        
        # Generate terrain height
        terrain_height = self._generate_terrain(w, h//4)
        
        # Ground (green with variation)
        for x in range(w):
            ground_start = h//2 + terrain_height[x]
            if ground_start < h:
                # Vary green intensity based on height
                height_factor = 1.0 - terrain_height[x] / (h//4)
                ground_color = [
                    0.2 + 0.1 * height_factor,  # Red
                    0.4 + 0.3 * height_factor,  # Green
                    0.1 + 0.1 * height_factor   # Blue
                ]
                image[int(ground_start):, x] = ground_color
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        return tensor.to(self.device)
    
    def _generate_terrain(self, width: int, max_height: int) -> np.ndarray:
        """Generate terrain height profile using midpoint displacement."""
        # Start with endpoints
        heights = np.zeros(width)
        heights[0] = np.random.randint(0, max_height//2)
        heights[-1] = np.random.randint(0, max_height//2)
        
        # Recursive subdivision
        segment_size = width - 1
        roughness = 0.5
        
        while segment_size > 1:
            # Process each segment
            for i in range(0, width - 1, segment_size):
                if i + segment_size < width:
                    # Midpoint
                    mid = i + segment_size // 2
                    if segment_size // 2 > 0:
                        # Average of endpoints plus random displacement
                        avg = (heights[i] + heights[i + segment_size]) / 2
                        displacement = np.random.uniform(-roughness * max_height, 
                                                       roughness * max_height)
                        heights[mid] = avg + displacement
            
            segment_size //= 2
            roughness *= 0.7  # Reduce roughness at each level
        
        # Smooth and ensure non-negative
        if SCIPY_AVAILABLE:
            heights = gaussian_filter(heights, sigma=1.0)
        heights = np.maximum(heights, 0)
        heights = np.minimum(heights, max_height)
        
        return heights.astype(int)
    
    def _generate_clouds(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate cloud-like patterns."""
        h, w = size
        
        # Generate multiple layers of Perlin noise
        cloud_density = np.zeros((h, w))
        
        # Large-scale cloud structure
        for octave in range(3):
            freq = 2 ** octave
            amp = 1.0 / (2 ** octave)
            
            # Simple noise generation
            noise = np.random.randn(h//freq + 1, w//freq + 1)
            
            # Interpolate to full size
            if SCIPY_AVAILABLE:
                from scipy.ndimage import zoom
                upsampled = zoom(noise, (freq, freq), order=1)[:h, :w]
            else:
                # Simple nearest neighbor upsampling
                upsampled = np.repeat(np.repeat(noise, freq, axis=0), freq, axis=1)[:h, :w]
            
            cloud_density += upsampled * amp
        
        # Normalize and apply cloud threshold
        cloud_density = (cloud_density - cloud_density.min()) / (cloud_density.max() - cloud_density.min())
        clouds = np.where(cloud_density > 0.4, 1.0, 0.0)
        
        # Create sky background
        image = np.zeros((h, w, 3))
        
        # Blue sky gradient
        for i in range(h):
            progress = i / h
            sky_color = [0.5, 0.7 + 0.3 * progress, 1.0]
            image[i, :] = sky_color
        
        # Add white clouds
        cloud_color = [1.0, 1.0, 1.0]
        for i in range(h):
            for j in range(w):
                if clouds[i, j] > 0.5:
                    # Blend cloud with sky
                    alpha = clouds[i, j] * 0.8
                    image[i, j] = alpha * np.array(cloud_color) + (1 - alpha) * image[i, j]
        
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        return tensor.to(self.device)
    
    def _generate_abstract(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate abstract patterns."""
        h, w = size
        
        # Create multiple overlapping shapes
        image = np.zeros((h, w, 3))
        
        # Random circles and ellipses
        num_shapes = np.random.randint(5, 15)
        
        for _ in range(num_shapes):
            # Random center
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            
            # Random size
            rx = np.random.randint(10, min(w, h) // 4)
            ry = np.random.randint(10, min(w, h) // 4)
            
            # Random color
            color = np.random.rand(3)
            
            # Draw shape
            for i in range(h):
                for j in range(w):
                    # Ellipse equation
                    dist = ((j - cx) / rx)**2 + ((i - cy) / ry)**2
                    if dist <= 1:
                        # Blend colors
                        alpha = 0.3
                        image[i, j] = alpha * color + (1 - alpha) * image[i, j]
        
        # Add some noise
        noise = np.random.normal(0, 0.1, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        return tensor.to(self.device)
    
    def _generate_water(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate water-like patterns with waves."""
        h, w = size
        
        # Create base water color (blue-green)
        base_color = np.array([0.1, 0.4, 0.7])
        image = np.tile(base_color, (h, w, 1))
        
        # Generate wave patterns
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        
        # Multiple wave components
        waves = (np.sin(xx) * 0.1 + 
                np.sin(2 * yy) * 0.05 + 
                np.sin(xx + yy) * 0.07)
        
        # Apply waves to color intensity
        for i in range(3):
            image[:, :, i] = np.clip(image[:, :, i] + waves, 0, 1)
        
        # Add reflections (lighter streaks)
        reflection_mask = np.sin(3 * xx) > 0.7
        image[reflection_mask] += 0.2
        image = np.clip(image, 0, 1)
        
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        return tensor.to(self.device)
    
    def _generate_forest(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate forest-like patterns."""
        h, w = size
        
        # Create gradient from dark green (bottom) to lighter green (top)
        image = np.zeros((h, w, 3))
        
        for i in range(h):
            progress = i / h
            forest_color = [
                0.1 + 0.1 * progress,  # Red
                0.2 + 0.3 * progress,  # Green
                0.1 + 0.1 * progress   # Blue
            ]
            image[i, :] = forest_color
        
        # Add vertical tree-like structures
        num_trees = w // 20
        for _ in range(num_trees):
            tree_x = np.random.randint(0, w)
            tree_height = np.random.randint(h//3, h)
            tree_width = np.random.randint(2, 8)
            
            # Draw tree trunk/foliage
            for i in range(h - tree_height, h):
                for j in range(max(0, tree_x - tree_width//2), 
                              min(w, tree_x + tree_width//2)):
                    # Darker green for trees
                    image[i, j] *= 0.7
        
        # Add some texture
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        return tensor.to(self.device)


# Factory function for easy access
def create_challenger(challenger_type: str, device: str = 'cuda') -> Union[FrequencyChallenger, TextureChallenger, NaturalImageChallenger]:
    """Factory function to create challengers."""
    if challenger_type.lower() == 'frequency':
        return FrequencyChallenger(device)
    elif challenger_type.lower() == 'texture':
        return TextureChallenger(device)
    elif challenger_type.lower() == 'natural':
        return NaturalImageChallenger(device)
    else:
        raise ValueError(f"Unknown challenger type: {challenger_type}")


# Utility functions
def combine_challenges(challenges: List[torch.Tensor], 
                      weights: Optional[List[float]] = None) -> torch.Tensor:
    """Combine multiple challenges with optional weights."""
    if weights is None:
        weights = [1.0 / len(challenges)] * len(challenges)
    
    if len(challenges) != len(weights):
        raise ValueError("Number of challenges and weights must match")
    
    combined = torch.zeros_like(challenges[0])
    for challenge, weight in zip(challenges, weights):
        combined += weight * challenge
    
    # Normalize to [0, 1]
    combined = torch.clamp(combined, 0, 1)
    
    return combined


def apply_transformations(challenge: torch.Tensor,
                         transformations: List[str]) -> List[torch.Tensor]:
    """Apply various transformations to a challenge."""
    results = [challenge]
    
    for transform in transformations:
        if transform == 'rotate90':
            results.append(torch.rot90(challenge, k=1, dims=[1, 2]))
        elif transform == 'flip_horizontal':
            results.append(torch.flip(challenge, dims=[2]))
        elif transform == 'flip_vertical':
            results.append(torch.flip(challenge, dims=[1]))
        elif transform == 'transpose':
            results.append(torch.transpose(challenge, 1, 2))
        elif transform == 'invert':
            results.append(1.0 - challenge)
        elif transform == 'gamma':
            # Apply gamma correction
            gamma = np.random.uniform(0.5, 2.0)
            results.append(torch.pow(challenge, gamma))
    
    return results