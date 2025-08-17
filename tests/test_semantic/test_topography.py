"""
Comprehensive tests for topographical learning module.
Tests projection methods, evolution tracking, and visualization capabilities.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
import warnings

# Add parent directory to path for pot imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

from pot.semantic import (
    ConceptLibrary,
    SemanticMatcher,
    TopographicalProjector,
    TopographicalEvolutionTracker,
    create_topographical_semantic_system,
    visualize_semantic_landscape,
    analyze_semantic_evolution
)
from pot.semantic.topography import (
    TopographicalMethod,
    TopographicalConfig,
    SOMProjector,
    ConceptSpaceNavigator
)
from pot.semantic.topography_utils import (
    prepare_latents_for_projection,
    compute_trustworthiness,
    compute_continuity,
    compute_shepard_correlation,
    identify_clusters_in_projection,
    track_cluster_evolution,
    select_optimal_parameters
)


class TestTopographicalProjector(unittest.TestCase):
    """Test TopographicalProjector with different methods."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic data
        self.n_samples = 100
        self.n_features = 50
        self.data = torch.randn(self.n_samples, self.n_features)
        
        # Create clusters for testing
        cluster1 = torch.randn(30, self.n_features) + torch.tensor([2.0] * self.n_features)
        cluster2 = torch.randn(30, self.n_features) - torch.tensor([2.0] * self.n_features)
        cluster3 = torch.randn(40, self.n_features)
        self.clustered_data = torch.cat([cluster1, cluster2, cluster3])
    
    def test_umap_projection(self):
        """Test UMAP projection with various parameters."""
        projector = TopographicalProjector(method='umap')
        
        # Test basic projection
        projected = projector.project_latents(self.data)
        self.assertEqual(projected.shape, (self.n_samples, 2))
        
        # Test with custom parameters
        projector_custom = TopographicalProjector(
            method='umap',
            n_neighbors=10,
            min_dist=0.05,
            n_components=3
        )
        projected_3d = projector_custom.project_latents(self.data)
        self.assertEqual(projected_3d.shape, (self.n_samples, 3))
        
        # Test that projections are deterministic
        projected2 = projector.project_latents(self.data)
        np.testing.assert_array_almost_equal(projected, projected2, decimal=5)
    
    def test_tsne_projection(self):
        """Test t-SNE projection."""
        projector = TopographicalProjector(method='tsne')
        
        # Test basic projection
        projected = projector.project_latents(self.data[:50])  # Use fewer samples for speed
        self.assertEqual(projected.shape, (50, 2))
        
        # Test with custom perplexity
        projector_custom = TopographicalProjector(
            method='tsne',
            perplexity=15,
            n_iter=500
        )
        projected_custom = projector_custom.project_latents(self.data[:50])
        self.assertEqual(projected_custom.shape, (50, 2))
        
        # Test that different perplexities give different results
        self.assertFalse(np.allclose(projected, projected_custom))
    
    def test_pca_projection(self):
        """Test PCA projection."""
        projector = TopographicalProjector(method='pca')
        
        # Test basic projection
        projected = projector.project_latents(self.data)
        self.assertEqual(projected.shape, (self.n_samples, 2))
        
        # Test variance preservation
        projector_3d = TopographicalProjector(method='pca', n_components=3)
        projected_3d = projector_3d.project_latents(self.data)
        self.assertEqual(projected_3d.shape, (self.n_samples, 3))
        
        # Verify PCA properties (orthogonality)
        cov_matrix = np.cov(projected.T)
        self.assertAlmostEqual(cov_matrix[0, 1], 0, places=5)
    
    def test_som_training(self):
        """Test SOM training and projection."""
        projector = SOMProjector(grid_size=(10, 10))
        
        # Train SOM
        projector.train(self.data.numpy()[:50])  # Use subset for speed
        self.assertTrue(projector.is_trained)
        
        # Project data
        projected = projector.project(self.data.numpy()[:50])
        self.assertEqual(projected.shape, (50, 2))
        
        # Test quantization error
        qe = projector.quantization_error(self.data.numpy()[:50])
        self.assertIsInstance(qe, float)
        self.assertGreater(qe, 0)
        
        # Test topographic error
        te = projector.topographic_error(self.data.numpy()[:50])
        self.assertIsInstance(te, float)
        self.assertGreaterEqual(te, 0)
        self.assertLessEqual(te, 1)
    
    def test_fallback_mechanisms(self):
        """Test fallback to PCA when other methods fail."""
        # Create data that might cause issues
        tiny_data = torch.randn(2, 100)  # Very few samples
        
        projector = TopographicalProjector(method='umap')
        projected = projector.project_latents(tiny_data)
        
        # Should still produce valid output (may fall back to PCA)
        self.assertEqual(len(projected), 2)
        self.assertGreaterEqual(projected.shape[1], 1)


class TestEvolutionTracking(unittest.TestCase):
    """Test topographical evolution tracking."""
    
    def setUp(self):
        """Set up evolution tracking test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create evolving data (simulating drift)
        self.snapshots = []
        base_data = np.random.randn(50, 20)
        
        for t in range(5):
            # Add progressive drift
            drift = np.random.randn(50, 20) * (t * 0.1)
            snapshot = base_data + drift
            self.snapshots.append(snapshot)
    
    def test_evolution_tracking(self):
        """Test topographical evolution tracking."""
        tracker = TopographicalEvolutionTracker()
        
        # Add snapshots
        for i, snapshot in enumerate(self.snapshots):
            tracker.add_snapshot(
                snapshot,
                timestamp=float(i),
                compute_metrics=True
            )
        
        # Check snapshots were added
        self.assertEqual(len(tracker.snapshots), 5)
        
        # Compute drift metrics
        drift_metrics = tracker.compute_drift_metrics()
        self.assertIn('centroid_shift', drift_metrics)
        self.assertIn('spread_change', drift_metrics)
        self.assertIn('density_shift', drift_metrics)
        self.assertIn('cumulative_drift', drift_metrics)
        
        # Drift should increase over time
        self.assertGreater(drift_metrics['cumulative_drift'], 0)
    
    def test_regime_change_detection(self):
        """Test regime change detection."""
        tracker = TopographicalEvolutionTracker()
        
        # Create data with sudden change
        normal_data = [np.random.randn(50, 20) for _ in range(3)]
        changed_data = [np.random.randn(50, 20) + 5 for _ in range(2)]  # Sudden shift
        
        for i, snapshot in enumerate(normal_data + changed_data):
            tracker.add_snapshot(snapshot, timestamp=float(i))
        
        # Detect regime changes
        changes_gradient = tracker.detect_regime_changes(method='gradient')
        changes_variance = tracker.detect_regime_changes(method='variance')
        
        # Should detect change around index 3
        self.assertTrue(len(changes_gradient) > 0 or len(changes_variance) > 0)
    
    def test_cluster_evolution(self):
        """Test cluster evolution tracking."""
        # Create evolving clusters
        snapshots = []
        for t in range(4):
            # Clusters that merge over time
            cluster1 = np.random.randn(25, 2) + [t, 0]
            cluster2 = np.random.randn(25, 2) + [5-t, 0]
            snapshot = np.vstack([cluster1, cluster2])
            snapshots.append(snapshot)
        
        # Track cluster evolution
        evolution = track_cluster_evolution(snapshots, method='kmeans', n_clusters=2)
        
        self.assertEqual(evolution['n_snapshots'], 4)
        self.assertEqual(len(evolution['n_clusters']), 4)
        self.assertIn('mean_stability', evolution)
        
        # Check transition tracking
        self.assertEqual(len(evolution['transitions']), 3)  # n_snapshots - 1


class TestProjectionMetrics(unittest.TestCase):
    """Test projection quality metrics."""
    
    def setUp(self):
        """Set up test data for metrics."""
        np.random.seed(42)
        
        # Create high-dimensional data
        self.high_dim = np.random.randn(100, 50)
        
        # Create low-dimensional projection (using PCA for consistency)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        self.low_dim = pca.fit_transform(self.high_dim)
    
    def test_trustworthiness(self):
        """Test trustworthiness metric."""
        trust = compute_trustworthiness(
            self.high_dim,
            self.low_dim,
            n_neighbors=10
        )
        
        self.assertIsInstance(trust, float)
        self.assertGreaterEqual(trust, 0)
        self.assertLessEqual(trust, 1)
    
    def test_continuity(self):
        """Test continuity metric."""
        cont = compute_continuity(
            self.high_dim,
            self.low_dim,
            n_neighbors=10
        )
        
        self.assertIsInstance(cont, float)
        self.assertGreaterEqual(cont, 0)
        self.assertLessEqual(cont, 1)
    
    def test_shepard_correlation(self):
        """Test Shepard correlation."""
        from scipy.spatial.distance import pdist
        
        high_distances = pdist(self.high_dim)
        low_distances = pdist(self.low_dim)
        
        corr = compute_shepard_correlation(high_distances, low_distances)
        
        self.assertIsInstance(corr, float)
        self.assertGreaterEqual(corr, -1)
        self.assertLessEqual(corr, 1)
        
        # PCA should have reasonable correlation
        self.assertGreater(corr, 0.5)
    
    def test_projection_metrics(self):
        """Test all projection metrics together."""
        from pot.semantic.topography_utils import (
            compute_neighborhood_preservation,
            compute_stress_metrics
        )
        
        # Neighborhood preservation
        metrics = compute_neighborhood_preservation(
            self.high_dim,
            self.low_dim,
            k=10
        )
        
        self.assertIn('trustworthiness', metrics)
        self.assertIn('continuity', metrics)
        self.assertIn('mean_relative_rank_error', metrics)
        self.assertIn('lcmc', metrics)
        
        # Stress metrics
        stress = compute_stress_metrics(
            self.high_dim,
            self.low_dim,
            normalized=True
        )
        
        self.assertIn('kruskal_stress_1', stress)
        self.assertIn('sammon_stress', stress)
        self.assertIn('shepard_correlation', stress)


class TestSemanticIntegration(unittest.TestCase):
    """Test integration with semantic verification."""
    
    def setUp(self):
        """Set up semantic integration tests."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.dim = 128
        self.n_concepts = 5
    
    def test_unified_system_creation(self):
        """Test unified topographical semantic system."""
        library, matcher, projector = create_topographical_semantic_system(
            dim=self.dim,
            projection_method='pca'
        )
        
        self.assertIsInstance(library, ConceptLibrary)
        self.assertIsInstance(matcher, SemanticMatcher)
        self.assertIsInstance(projector, TopographicalProjector)
        
        self.assertEqual(library.dim, self.dim)
    
    def test_concept_projection(self):
        """Test projecting concepts in library."""
        library, matcher, projector = create_topographical_semantic_system(
            dim=self.dim,
            projection_method='pca'
        )
        
        # Add concepts
        for i in range(self.n_concepts):
            embeddings = torch.randn(10, self.dim)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Get positions
        positions = library.get_concept_positions(method='pca')
        
        self.assertEqual(len(positions), self.n_concepts)
        for name, pos in positions.items():
            self.assertEqual(pos.shape, (2,))
    
    def test_semantic_trajectory(self):
        """Test semantic trajectory tracking."""
        library, matcher, projector = create_topographical_semantic_system(
            dim=self.dim,
            projection_method='pca'
        )
        
        # Add concepts
        for i in range(3):
            embeddings = torch.randn(10, self.dim)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Create trajectory
        trajectory_embeddings = [torch.randn(self.dim) for _ in range(10)]
        result = matcher.track_semantic_trajectory(
            trajectory_embeddings,
            projection_method='pca',
            smooth=True
        )
        
        self.assertEqual(result['trajectory_length'], 10)
        self.assertGreater(result['total_distance'], 0)
        self.assertIn('velocities', result)
        self.assertIn('accelerations', result)
    
    def test_semantic_evolution_analysis(self):
        """Test analyzing semantic evolution."""
        # Create library
        library = ConceptLibrary(dim=self.dim, method='gaussian')
        
        # Add reference concepts
        for i in range(3):
            embeddings = torch.randn(10, self.dim)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Create evolving embeddings
        history = []
        for t in range(3):
            embeddings = [torch.randn(self.dim) + t * 0.1 for _ in range(20)]
            history.append(embeddings)
        
        # Analyze evolution
        result = analyze_semantic_evolution(
            history,
            library,
            timestamps=[0.0, 1.0, 2.0],
            method='pca'
        )
        
        self.assertIn('drift_metrics', result)
        self.assertIn('cluster_evolution', result)
        self.assertIn('regime_changes', result)
        self.assertEqual(result['n_snapshots'], 3)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_prepare_latents(self):
        """Test latent preparation."""
        # Test with torch tensor
        latents = torch.randn(50, 100)
        prepared = prepare_latents_for_projection(latents, normalize=True)
        
        self.assertIsInstance(prepared, np.ndarray)
        self.assertEqual(prepared.shape, (50, 100))
        
        # Check normalization
        norms = np.linalg.norm(prepared, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(50), decimal=5)
        
        # Test dimensionality reduction
        prepared_reduced = prepare_latents_for_projection(
            latents,
            normalize=True,
            reduce_dims=10
        )
        self.assertEqual(prepared_reduced.shape, (50, 10))
    
    def test_optimal_parameters(self):
        """Test automatic parameter selection."""
        data = np.random.randn(100, 50)
        
        # UMAP parameters
        umap_params = select_optimal_parameters(data, 'umap')
        self.assertIn('n_neighbors', umap_params)
        self.assertIn('min_dist', umap_params)
        
        # t-SNE parameters
        tsne_params = select_optimal_parameters(data, 'tsne')
        self.assertIn('perplexity', tsne_params)
        self.assertIn('learning_rate', tsne_params)
        
        # SOM parameters
        som_params = select_optimal_parameters(data, 'som')
        self.assertIn('x_size', som_params)
        self.assertIn('y_size', som_params)
    
    def test_cluster_identification(self):
        """Test cluster identification in projections."""
        # Create clear clusters
        cluster1 = np.random.randn(30, 2) + [0, 0]
        cluster2 = np.random.randn(30, 2) + [5, 5]
        data = np.vstack([cluster1, cluster2])
        
        # Identify clusters
        labels = identify_clusters_in_projection(data, method='kmeans', n_clusters=2)
        
        self.assertEqual(len(labels), 60)
        self.assertEqual(len(np.unique(labels)), 2)
        
        # Test DBSCAN
        labels_dbscan = identify_clusters_in_projection(data, method='dbscan')
        self.assertEqual(len(labels_dbscan), 60)


class TestVisualization(unittest.TestCase):
    """Test visualization capabilities."""
    
    def setUp(self):
        """Set up visualization tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_concept_space_visualization(self):
        """Test concept space visualization."""
        # Create library with concepts
        library = ConceptLibrary(dim=50, method='gaussian')
        
        for i in range(3):
            embeddings = torch.randn(10, 50)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Visualize
        save_path = os.path.join(self.temp_dir, 'concept_space.png')
        fig = library.visualize_concept_space(
            method='pca',
            save_path=save_path
        )
        
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.exists(save_path))
    
    def test_semantic_landscape(self):
        """Test semantic landscape visualization."""
        library = ConceptLibrary(dim=50, method='gaussian')
        
        for i in range(5):
            embeddings = torch.randn(10, 50)
            library.add_concept(f'concept_{i}', embeddings)
        
        # Static visualization
        fig = visualize_semantic_landscape(library, method='pca')
        self.assertIsNotNone(fig)
        
        # Interactive visualization would require plotly
        # Skipping interactive test in unit tests


class TestConceptNavigation(unittest.TestCase):
    """Test concept space navigation."""
    
    def test_navigator_creation(self):
        """Test ConceptSpaceNavigator creation."""
        # Create some 2D positions
        positions = {
            'concept_a': np.array([0, 0]),
            'concept_b': np.array([1, 0]),
            'concept_c': np.array([0, 1])
        }
        
        navigator = ConceptSpaceNavigator(positions)
        
        # Find nearest
        nearest = navigator.find_nearest_concepts(np.array([0.5, 0.5]), k=2)
        self.assertEqual(len(nearest), 2)
        
        # Find path
        path = navigator.find_path('concept_a', 'concept_c')
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], 'concept_a')
        self.assertEqual(path[-1], 'concept_c')
        
        # Compute centrality
        centrality = navigator.compute_concept_centrality()
        self.assertEqual(len(centrality), 3)


if __name__ == '__main__':
    unittest.main()