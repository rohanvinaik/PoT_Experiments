"""
Performance benchmarks for topographical learning methods.
Compares methods on various dataset sizes and measures quality vs. speed tradeoffs.
"""

import numpy as np
import torch
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pot.semantic import TopographicalProjector
from pot.semantic.topography_optimized import (
    IncrementalUMAP,
    OnlineSOM,
    project_latents_batched,
    CachedProjector,
    ApproximateProjector,
    optimize_projection_pipeline,
    RAPIDS_AVAILABLE
)
from pot.semantic.topography_utils import (
    prepare_latents_for_projection,
    compute_trustworthiness,
    compute_continuity,
    compute_stress_metrics
)


class BenchmarkSuite:
    """Comprehensive benchmarking suite for topographical methods."""
    
    def __init__(self, save_results: bool = True, results_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.
        
        Args:
            save_results: Whether to save results to disk
            results_dir: Directory to save results
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        if save_results:
            self.results_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.dataset_cache = {}
    
    def generate_dataset(self, n_samples: int, n_features: int, 
                        dataset_type: str = 'clustered') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic datasets for benchmarking.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            dataset_type: Type of dataset structure
            
        Returns:
            Tuple of (data, labels)
        """
        cache_key = (n_samples, n_features, dataset_type)
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
        
        np.random.seed(42)
        
        if dataset_type == 'clustered':
            # Gaussian clusters
            n_clusters = min(5, n_samples // 100)
            samples_per_cluster = n_samples // n_clusters
            
            data_parts = []
            labels = []
            
            for i in range(n_clusters):
                center = np.random.randn(n_features) * 3
                cluster = np.random.randn(samples_per_cluster, n_features) + center
                data_parts.append(cluster)
                labels.extend([i] * samples_per_cluster)
            
            # Add remaining samples to last cluster
            remaining = n_samples - len(labels)
            if remaining > 0:
                cluster = np.random.randn(remaining, n_features) + center
                data_parts.append(cluster)
                labels.extend([n_clusters-1] * remaining)
            
            data = np.vstack(data_parts)
            labels = np.array(labels)
        
        elif dataset_type == 'manifold':
            # Swiss roll manifold
            t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
            height = 20 * np.random.rand(n_samples)
            
            # 3D swiss roll
            swiss_3d = np.zeros((n_samples, 3))
            swiss_3d[:, 0] = t * np.cos(t)
            swiss_3d[:, 1] = height
            swiss_3d[:, 2] = t * np.sin(t)
            
            # Embed in higher dimensions
            if n_features > 3:
                # Add noise dimensions
                noise = np.random.randn(n_samples, n_features - 3) * 0.1
                data = np.hstack([swiss_3d, noise])
            else:
                data = swiss_3d[:, :n_features]
            
            labels = (t - t.min()) / (t.max() - t.min())  # Continuous labels
            labels = (labels * 4).astype(int)  # Discretize
        
        elif dataset_type == 'uniform':
            # Uniform random data
            data = np.random.randn(n_samples, n_features)
            labels = np.random.randint(0, 2, n_samples)
        
        elif dataset_type == 'sparse':
            # Sparse data (many zeros)
            data = np.random.randn(n_samples, n_features)
            mask = np.random.rand(n_samples, n_features) > 0.2  # 80% zeros
            data[mask] = 0
            labels = np.random.randint(0, 3, n_samples)
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        data = data[indices]
        labels = labels[indices]
        
        self.dataset_cache[cache_key] = (data, labels)
        return data, labels
    
    def benchmark_method(self, method_name: str, data: np.ndarray,
                        labels: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a single method.
        
        Args:
            method_name: Name of the method
            data: Input data
            labels: True labels for quality assessment
            **kwargs: Method parameters
            
        Returns:
            Dictionary of benchmark results
        """
        n_samples, n_features = data.shape
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the projection
        start_time = time.time()
        
        try:
            if method_name == 'pca':
                projector = TopographicalProjector('pca')
                projection = projector.project_latents(torch.tensor(data, dtype=torch.float32))
            
            elif method_name == 'umap':
                projector = TopographicalProjector('umap')
                projection = projector.project_latents(torch.tensor(data, dtype=torch.float32))
            
            elif method_name == 'tsne':
                projector = TopographicalProjector('tsne')
                projection = projector.project_latents(torch.tensor(data[:min(1000, len(data))], dtype=torch.float32))
                # Pad if needed
                if len(projection) < len(data):
                    # Use subset for t-SNE
                    data = data[:len(projection)]
                    labels = labels[:len(projection)]
            
            elif method_name == 'incremental_umap':
                projector = IncrementalUMAP(**kwargs)
                projection = projector.fit(data)
            
            elif method_name == 'online_som':
                projector = OnlineSOM(**kwargs)
                projector.partial_fit(data)
                projection = projector.transform(data)
            
            elif method_name == 'batched_umap':
                projection = project_latents_batched(
                    torch.tensor(data, dtype=torch.float32),
                    method='umap',
                    batch_size=kwargs.get('batch_size', 1000)
                )
            
            elif method_name == 'cached_umap':
                base_projector = TopographicalProjector('umap')
                projector = CachedProjector(base_projector)
                projection = projector.project_latents(torch.tensor(data, dtype=torch.float32))
            
            elif method_name == 'approximate_umap':
                projector = ApproximateProjector('pca_umap', **kwargs)
                projection = projector.fit_transform(data)
            
            elif method_name == 'optimized_auto':
                projection = optimize_projection_pipeline(
                    torch.tensor(data, dtype=torch.float32),
                    method='auto'
                )
            
            elif method_name == 'gpu_umap' and RAPIDS_AVAILABLE:
                projection = project_latents_batched(
                    torch.tensor(data, dtype=torch.float32),
                    method='umap',
                    use_gpu=True,
                    batch_size=kwargs.get('batch_size', 2000)
                )
            
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            projection_time = time.time() - start_time
            
            # Monitor peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - initial_memory
            
            # Compute quality metrics
            try:
                trust = compute_trustworthiness(data, projection, n_neighbors=min(10, len(data)-1))
                cont = compute_continuity(data, projection, n_neighbors=min(10, len(data)-1))
                stress_metrics = compute_stress_metrics(data, projection)
                
                quality_metrics = {
                    'trustworthiness': trust,
                    'continuity': cont,
                    'kruskal_stress': stress_metrics['kruskal_stress_1'],
                    'shepard_correlation': stress_metrics['shepard_correlation']
                }
            except Exception as e:
                print(f"Failed to compute quality metrics: {e}")
                quality_metrics = {
                    'trustworthiness': 0.0,
                    'continuity': 0.0,
                    'kruskal_stress': 1.0,
                    'shepard_correlation': 0.0
                }
            
            success = True
            error_msg = None
            
        except Exception as e:
            projection_time = time.time() - start_time
            memory_usage = 0
            quality_metrics = {
                'trustworthiness': 0.0,
                'continuity': 0.0,
                'kruskal_stress': 1.0,
                'shepard_correlation': 0.0
            }
            success = False
            error_msg = str(e)
            projection = None
        
        # Clean up memory
        gc.collect()
        
        result = {
            'method': method_name,
            'n_samples': n_samples,
            'n_features': n_features,
            'projection_time': projection_time,
            'memory_usage_mb': memory_usage,
            'success': success,
            'error': error_msg,
            **quality_metrics
        }
        
        return result
    
    def run_scalability_benchmark(self, methods: List[str],
                                 sample_sizes: List[int],
                                 feature_sizes: List[int],
                                 dataset_type: str = 'clustered') -> pd.DataFrame:
        """
        Run scalability benchmark across different data sizes.
        
        Args:
            methods: List of method names to benchmark
            sample_sizes: List of sample sizes to test
            feature_sizes: List of feature sizes to test
            dataset_type: Type of dataset to generate
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        total_tests = len(methods) * len(sample_sizes) * len(feature_sizes)
        test_count = 0
        
        print(f"Running scalability benchmark: {total_tests} tests")
        print(f"Methods: {methods}")
        print(f"Sample sizes: {sample_sizes}")
        print(f"Feature sizes: {feature_sizes}")
        print("-" * 50)
        
        for method in methods:
            for n_samples in sample_sizes:
                for n_features in feature_sizes:
                    test_count += 1
                    print(f"Test {test_count}/{total_tests}: {method} - "
                          f"{n_samples} samples, {n_features} features")
                    
                    # Generate data
                    data, labels = self.generate_dataset(n_samples, n_features, dataset_type)
                    
                    # Run benchmark
                    result = self.benchmark_method(method, data, labels)
                    result['dataset_type'] = dataset_type
                    results.append(result)
                    
                    print(f"  Time: {result['projection_time']:.2f}s, "
                          f"Memory: {result['memory_usage_mb']:.1f}MB, "
                          f"Success: {result['success']}")
        
        df = pd.DataFrame(results)
        
        if self.save_results:
            output_file = self.results_dir / f"scalability_{dataset_type}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")
        
        return df
    
    def run_quality_vs_speed_benchmark(self, methods: List[str],
                                      n_samples: int = 2000,
                                      n_features: int = 100) -> pd.DataFrame:
        """
        Compare quality vs speed tradeoffs.
        
        Args:
            methods: Methods to compare
            n_samples: Fixed number of samples
            n_features: Fixed number of features
            
        Returns:
            DataFrame with results
        """
        results = []
        
        print(f"Running quality vs speed benchmark")
        print(f"Data size: {n_samples} x {n_features}")
        print("-" * 50)
        
        # Test on different dataset types
        dataset_types = ['clustered', 'manifold', 'uniform']
        
        for dataset_type in dataset_types:
            print(f"\nDataset type: {dataset_type}")
            
            data, labels = self.generate_dataset(n_samples, n_features, dataset_type)
            
            for method in methods:
                print(f"  Testing {method}...")
                
                result = self.benchmark_method(method, data, labels)
                result['dataset_type'] = dataset_type
                results.append(result)
        
        df = pd.DataFrame(results)
        
        if self.save_results:
            output_file = self.results_dir / "quality_vs_speed.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved results to {output_file}")
        
        return df
    
    def run_incremental_learning_benchmark(self, n_total: int = 5000,
                                         batch_sizes: List[int] = None) -> pd.DataFrame:
        """
        Benchmark incremental learning methods.
        
        Args:
            n_total: Total number of samples
            batch_sizes: List of batch sizes to test
            
        Returns:
            DataFrame with results
        """
        if batch_sizes is None:
            batch_sizes = [100, 500, 1000]
        
        results = []
        
        print(f"Running incremental learning benchmark")
        print(f"Total samples: {n_total}, Batch sizes: {batch_sizes}")
        print("-" * 50)
        
        # Generate full dataset
        data, labels = self.generate_dataset(n_total, 50, 'clustered')
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Test IncrementalUMAP
            print("  Testing IncrementalUMAP...")
            start_time = time.time()
            
            inc_umap = IncrementalUMAP()
            
            # Process in batches
            n_batches = n_total // batch_size
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_total)
                batch_data = data[start_idx:end_idx]
                
                if i == 0:
                    inc_umap.fit(batch_data)
                else:
                    inc_umap.partial_fit(batch_data)
            
            total_time = time.time() - start_time
            
            # Get final projection
            final_projection = inc_umap.embeddings_history[-1]
            
            # Compute quality
            subset_data = data[:len(final_projection)]
            trust = compute_trustworthiness(subset_data, final_projection, n_neighbors=10)
            cont = compute_continuity(subset_data, final_projection, n_neighbors=10)
            
            results.append({
                'method': 'IncrementalUMAP',
                'batch_size': batch_size,
                'n_batches': n_batches,
                'total_time': total_time,
                'time_per_batch': total_time / n_batches,
                'trustworthiness': trust,
                'continuity': cont,
                'final_samples': inc_umap.total_samples
            })\n            \n            # Test OnlineSOM\n            print(\"  Testing OnlineSOM...\")\n            start_time = time.time()\n            \n            online_som = OnlineSOM(grid_size=(15, 15))\n            \n            # Process in batches\n            for i in range(n_batches):\n                start_idx = i * batch_size\n                end_idx = min(start_idx + batch_size, n_total)\n                batch_data = data[start_idx:end_idx]\n                \n                online_som.partial_fit(batch_data, epochs=1)\n            \n            total_time = time.time() - start_time\n            \n            # Get projection\n            final_projection = online_som.transform(data)\n            \n            # Compute quality\n            trust = compute_trustworthiness(data, final_projection, n_neighbors=10)\n            cont = compute_continuity(data, final_projection, n_neighbors=10)\n            \n            results.append({\n                'method': 'OnlineSOM',\n                'batch_size': batch_size,\n                'n_batches': n_batches,\n                'total_time': total_time,\n                'time_per_batch': total_time / n_batches,\n                'trustworthiness': trust,\n                'continuity': cont,\n                'final_samples': len(data)\n            })\n        \n        df = pd.DataFrame(results)\n        \n        if self.save_results:\n            output_file = self.results_dir / \"incremental_learning.csv\"\n            df.to_csv(output_file, index=False)\n            print(f\"Saved results to {output_file}\")\n        \n        return df\n    \n    def run_caching_benchmark(self, cache_sizes: List[int] = None) -> pd.DataFrame:\n        \"\"\"\n        Benchmark caching effectiveness.\n        \n        Args:\n            cache_sizes: List of cache sizes to test\n            \n        Returns:\n            DataFrame with caching results\n        \"\"\"\n        if cache_sizes is None:\n            cache_sizes = [10, 50, 100, 500]\n        \n        results = []\n        \n        print(f\"Running caching benchmark\")\n        print(f\"Cache sizes: {cache_sizes}\")\n        print(\"-\" * 50)\n        \n        # Generate test datasets\n        datasets = []\n        for i in range(20):\n            data, _ = self.generate_dataset(500, 50, 'clustered')\n            datasets.append(data)\n        \n        for cache_size in cache_sizes:\n            print(f\"\\nCache size: {cache_size}\")\n            \n            # Create cached projector\n            base_projector = TopographicalProjector('pca')  # Fast method for testing\n            cached_projector = CachedProjector(base_projector, cache_size=cache_size)\n            \n            # Time projections with cache\n            times = []\n            for i, data in enumerate(datasets):\n                start_time = time.time()\n                projection = cached_projector.project_latents(\n                    torch.tensor(data, dtype=torch.float32)\n                )\n                times.append(time.time() - start_time)\n                \n                if i % 5 == 0:\n                    print(f\"  Iteration {i}: {times[-1]:.3f}s, \"\n                          f\"Hit rate: {cached_projector.get_hit_rate():.2%}\")\n            \n            # Test cache hits by repeating some datasets\n            print(\"  Testing cache hits...\")\n            repeat_times = []\n            for i in range(5):\n                data = datasets[i % 3]  # Repeat first 3 datasets\n                start_time = time.time()\n                projection = cached_projector.project_latents(\n                    torch.tensor(data, dtype=torch.float32)\n                )\n                repeat_times.append(time.time() - start_time)\n            \n            results.append({\n                'cache_size': cache_size,\n                'mean_time_first_run': np.mean(times),\n                'mean_time_repeat': np.mean(repeat_times),\n                'final_hit_rate': cached_projector.get_hit_rate(),\n                'speedup': np.mean(times) / np.mean(repeat_times)\n            })\n        \n        df = pd.DataFrame(results)\n        \n        if self.save_results:\n            output_file = self.results_dir / \"caching.csv\"\n            df.to_csv(output_file, index=False)\n            print(f\"Saved results to {output_file}\")\n        \n        return df\n    \n    def create_visualizations(self, scalability_df: pd.DataFrame,\n                            quality_speed_df: pd.DataFrame,\n                            incremental_df: pd.DataFrame,\n                            caching_df: pd.DataFrame):\n        \"\"\"\n        Create benchmark visualization plots.\n        \n        Args:\n            scalability_df: Scalability benchmark results\n            quality_speed_df: Quality vs speed results\n            incremental_df: Incremental learning results\n            caching_df: Caching benchmark results\n        \"\"\"\n        # Set up plotting\n        plt.style.use('seaborn-v0_8-whitegrid')\n        \n        # 1. Scalability plots\n        fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n        \n        # Time vs samples\n        successful_results = scalability_df[scalability_df['success'] == True]\n        \n        ax = axes[0, 0]\n        for method in successful_results['method'].unique():\n            method_data = successful_results[successful_results['method'] == method]\n            ax.loglog(method_data['n_samples'], method_data['projection_time'], \n                     'o-', label=method, markersize=4)\n        ax.set_xlabel('Number of Samples')\n        ax.set_ylabel('Projection Time (s)')\n        ax.set_title('Scalability: Time vs Sample Size')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        \n        # Memory vs samples\n        ax = axes[0, 1]\n        for method in successful_results['method'].unique():\n            method_data = successful_results[successful_results['method'] == method]\n            ax.loglog(method_data['n_samples'], method_data['memory_usage_mb'], \n                     's-', label=method, markersize=4)\n        ax.set_xlabel('Number of Samples')\n        ax.set_ylabel('Memory Usage (MB)')\n        ax.set_title('Scalability: Memory vs Sample Size')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        \n        # Quality vs speed\n        ax = axes[1, 0]\n        quality_successful = quality_speed_df[quality_speed_df['success'] == True]\n        \n        for dataset_type in quality_successful['dataset_type'].unique():\n            subset = quality_successful[quality_successful['dataset_type'] == dataset_type]\n            scatter = ax.scatter(subset['projection_time'], subset['trustworthiness'],\n                               c=subset['method'].astype('category').cat.codes,\n                               label=dataset_type, alpha=0.7, s=50)\n        \n        ax.set_xlabel('Projection Time (s)')\n        ax.set_ylabel('Trustworthiness')\n        ax.set_title('Quality vs Speed Tradeoff')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        \n        # Incremental learning\n        ax = axes[1, 1]\n        for method in incremental_df['method'].unique():\n            method_data = incremental_df[incremental_df['method'] == method]\n            ax.plot(method_data['batch_size'], method_data['time_per_batch'],\n                   'o-', label=method, markersize=6)\n        ax.set_xlabel('Batch Size')\n        ax.set_ylabel('Time per Batch (s)')\n        ax.set_title('Incremental Learning Performance')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        \n        if self.save_results:\n            plt.savefig(self.results_dir / 'benchmark_overview.png', \n                       dpi=150, bbox_inches='tight')\n        \n        plt.show()\n        \n        # 2. Detailed caching plot\n        if len(caching_df) > 0:\n            fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n            \n            # Cache hit rate\n            ax = axes[0]\n            ax.plot(caching_df['cache_size'], caching_df['final_hit_rate'], \n                   'bo-', markersize=8)\n            ax.set_xlabel('Cache Size')\n            ax.set_ylabel('Hit Rate')\n            ax.set_title('Cache Hit Rate vs Cache Size')\n            ax.grid(True, alpha=0.3)\n            \n            # Speedup\n            ax = axes[1]\n            ax.plot(caching_df['cache_size'], caching_df['speedup'], \n                   'ro-', markersize=8)\n            ax.set_xlabel('Cache Size')\n            ax.set_ylabel('Speedup Factor')\n            ax.set_title('Caching Speedup vs Cache Size')\n            ax.grid(True, alpha=0.3)\n            \n            plt.tight_layout()\n            \n            if self.save_results:\n                plt.savefig(self.results_dir / 'caching_analysis.png', \n                           dpi=150, bbox_inches='tight')\n            \n            plt.show()\n    \n    def print_summary(self, scalability_df: pd.DataFrame,\n                     quality_speed_df: pd.DataFrame):\n        \"\"\"\n        Print benchmark summary.\n        \n        Args:\n            scalability_df: Scalability results\n            quality_speed_df: Quality vs speed results\n        \"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"BENCHMARK SUMMARY\")\n        print(\"=\"*60)\n        \n        # Scalability summary\n        successful = scalability_df[scalability_df['success'] == True]\n        if len(successful) > 0:\n            print(\"\\nScalability (per method):\")\n            print(\"-\" * 40)\n            \n            summary = successful.groupby('method').agg({\n                'projection_time': ['mean', 'std'],\n                'memory_usage_mb': ['mean', 'std'],\n                'trustworthiness': ['mean', 'std'],\n                'n_samples': ['min', 'max']\n            }).round(3)\n            \n            for method in summary.index:\n                print(f\"\\n{method}:\")\n                print(f\"  Time: {summary.loc[method, ('projection_time', 'mean')]:.3f} ± \"\n                      f\"{summary.loc[method, ('projection_time', 'std')]:.3f} s\")\n                print(f\"  Memory: {summary.loc[method, ('memory_usage_mb', 'mean')]:.1f} ± \"\n                      f\"{summary.loc[method, ('memory_usage_mb', 'std')]:.1f} MB\")\n                print(f\"  Quality: {summary.loc[method, ('trustworthiness', 'mean')]:.3f} ± \"\n                      f\"{summary.loc[method, ('trustworthiness', 'std')]:.3f}\")\n                print(f\"  Sample range: {int(summary.loc[method, ('n_samples', 'min')])} - \"\n                      f\"{int(summary.loc[method, ('n_samples', 'max')])}\")\n        \n        # Quality vs speed summary\n        quality_successful = quality_speed_df[quality_speed_df['success'] == True]\n        if len(quality_successful) > 0:\n            print(\"\\n\\nQuality vs Speed (best performers):\")\n            print(\"-\" * 40)\n            \n            # Find best methods for different criteria\n            fastest = quality_successful.loc[quality_successful['projection_time'].idxmin()]\n            highest_quality = quality_successful.loc[quality_successful['trustworthiness'].idxmax()]\n            best_ratio = quality_successful.copy()\n            best_ratio['quality_speed_ratio'] = best_ratio['trustworthiness'] / best_ratio['projection_time']\n            best_balanced = best_ratio.loc[best_ratio['quality_speed_ratio'].idxmax()]\n            \n            print(f\"\\nFastest: {fastest['method']} ({fastest['projection_time']:.3f}s)\")\n            print(f\"Highest Quality: {highest_quality['method']} (trust: {highest_quality['trustworthiness']:.3f})\")\n            print(f\"Best Balanced: {best_balanced['method']} (ratio: {best_balanced['quality_speed_ratio']:.2f})\")\n        \n        # Recommendations\n        print(\"\\n\\nRecommendations:\")\n        print(\"-\" * 40)\n        print(\"• For real-time applications: Use PCA or approximate methods\")\n        print(\"• For highest quality: Use UMAP or t-SNE with full parameters\")\n        print(\"• For large datasets: Use batched processing or GPU acceleration\")\n        print(\"• For incremental data: Use IncrementalUMAP or OnlineSOM\")\n        print(\"• For repeated projections: Use CachedProjector\")\n\n\ndef main():\n    \"\"\"\n    Run complete benchmark suite.\n    \"\"\"\n    print(\"Starting Topographical Learning Benchmarks\")\n    print(\"=\"*50)\n    \n    # Initialize benchmark suite\n    benchmark = BenchmarkSuite(save_results=True)\n    \n    # Define methods to test\n    base_methods = ['pca', 'umap', 'approximate_umap', 'optimized_auto']\n    \n    # Add GPU methods if available\n    methods = base_methods.copy()\n    if RAPIDS_AVAILABLE:\n        methods.append('gpu_umap')\n        print(\"GPU acceleration available - including GPU methods\")\n    else:\n        print(\"GPU acceleration not available - CPU methods only\")\n    \n    # Add caching and batching\n    methods.extend(['batched_umap', 'cached_umap'])\n    \n    # 1. Scalability benchmark\n    print(\"\\n1. Running scalability benchmark...\")\n    sample_sizes = [100, 500, 1000, 2000]\n    feature_sizes = [10, 50, 100]\n    \n    scalability_df = benchmark.run_scalability_benchmark(\n        methods=['pca', 'umap', 'approximate_umap', 'optimized_auto'],\n        sample_sizes=sample_sizes,\n        feature_sizes=feature_sizes,\n        dataset_type='clustered'\n    )\n    \n    # 2. Quality vs speed benchmark\n    print(\"\\n2. Running quality vs speed benchmark...\")\n    quality_speed_df = benchmark.run_quality_vs_speed_benchmark(\n        methods=['pca', 'umap', 'approximate_umap', 'optimized_auto'],\n        n_samples=1000,\n        n_features=50\n    )\n    \n    # 3. Incremental learning benchmark\n    print(\"\\n3. Running incremental learning benchmark...\")\n    incremental_df = benchmark.run_incremental_learning_benchmark(\n        n_total=2000,\n        batch_sizes=[100, 500, 1000]\n    )\n    \n    # 4. Caching benchmark\n    print(\"\\n4. Running caching benchmark...\")\n    caching_df = benchmark.run_caching_benchmark(\n        cache_sizes=[10, 50, 100]\n    )\n    \n    # Create visualizations\n    print(\"\\n5. Creating visualizations...\")\n    benchmark.create_visualizations(\n        scalability_df, quality_speed_df, incremental_df, caching_df\n    )\n    \n    # Print summary\n    benchmark.print_summary(scalability_df, quality_speed_df)\n    \n    print(\"\\n\" + \"=\"*50)\n    print(\"Benchmarks completed!\")\n    print(f\"Results saved to: {benchmark.results_dir}\")\n    print(\"=\"*50)\n\n\nif __name__ == \"__main__\":\n    main()