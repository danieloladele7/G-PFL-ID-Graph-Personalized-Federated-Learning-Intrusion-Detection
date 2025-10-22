# src/non_iid_metrics.py
import torch
import numpy as np
import networkx as nx
from scipy.stats import entropy
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from src.utils import load_client_data
from typing import Dict, List


class NonIIDMetrics:
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions"""
        p = p + 1e-10  # Avoid division by zero
        q = q + 1e-10
        return entropy(p, q)
    
    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions"""
        p = p + 1e-10
        q = q + 1e-10
        return jensenshannon(p, q) ** 2  # scipy returns sqrt(JS)
    
    @staticmethod
    def earth_movers_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Earth Mover's Distance between two distributions"""
        return wasserstein_distance(p, q)
    
    @staticmethod
    def compute_label_distribution(labels: List[int], num_classes: int) -> np.ndarray:
        """Compute normalized label distribution"""
        dist = np.zeros(num_classes)
        for label in labels:
            if label < num_classes:
                dist[label] += 1
        return dist / dist.sum() if dist.sum() > 0 else dist
    
    @staticmethod
    def compute_graph_homophily(edge_index, labels):
        """Compute homophily ratio for a graph with bounds checking."""
        if len(labels) == 0:
            return 0.0
        
        # Convert to tensor if needed
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        src, dst = edge_index
        num_nodes = labels.size(0)
        
        # Create mask for valid edges (within bounds)
        valid_mask = (src < num_nodes) & (dst < num_nodes)
        
        if not valid_mask.all():
            # Log warning about invalid edges
            invalid_count = (~valid_mask).sum().item()
            total_edges = src.size(0)
            print(f"WARNING: {invalid_count}/{total_edges} edges have out-of-bounds node indices. These will be filtered out.")
            
            # Filter to only valid edges
            src = src[valid_mask]
            dst = dst[valid_mask]
        
        # Calculate homophily on valid edges only
        if src.size(0) == 0:
            return 0.0  # No valid edges
        
        same_label = (labels[src] == labels[dst]).sum().item()
        total_valid_edges = src.size(0)
        
        return same_label / total_valid_edges if total_valid_edges > 0 else 0.0
    
    def analyze_non_iid(self, client_data: Dict[str, Dict], global_dist: np.ndarray) -> Dict:
        """Comprehensive Non-IID analysis across clients"""
        results = {
            'kl_divergences': {},
            'js_divergences': {},
            'em_distances': {},
            'homophily_scores': {}
        }
        
        for client_id, data in client_data.items():
            # Get client label distribution
            client_labels = data['labels']
            client_dist = self.compute_label_distribution(client_labels, len(global_dist))
            
            # Debug: print distributions
            print(f"Client {client_id}:")
            print(f"  Global dist: {global_dist}")
            print(f"  Client dist: {client_dist}")
        
            # Compute divergences
            results['kl_divergences'][client_id] = self.kl_divergence(client_dist, global_dist)
            results['js_divergences'][client_id] = self.js_divergence(client_dist, global_dist)
            results['em_distances'][client_id] = self.earth_movers_distance(client_dist, global_dist)
            
            # Debug: print computed values
            print(f"  KL: {results['kl_divergences'][client_id]}")
        
            # Compute graph homophily if graph data is available
            if 'graph' in data:
                graph = data['graph']
                if hasattr(graph, 'y') and graph.y is not None:
                    results['homophily_scores'][client_id] = self.compute_graph_homophily(
                        graph.edge_index, graph.y
                    )
        
        return results
    
    # src/non_iid_metrics.py
    def analyze_feature_distributions(self, client_data: Dict[str, Dict]) -> Dict:
        """Analyze feature distributions for one-class scenario"""
        results = {
            'feature_means': {},
            'feature_variances': {},
            'js_divergences': {},
            'global_feature_dist': None
        }
        
        # Compute global feature distribution
        all_features = []
        for data in client_data.values():
            if 'features' in data:
                all_features.extend(data['features'])
        
        if all_features:
            # Flatten all features and remove NaN values
            flat_features = np.concatenate([f.flatten() for f in all_features])
            flat_features = flat_features[~np.isnan(flat_features)]  # Remove NaN values
            
            if len(flat_features) > 0:  # Check if we have valid features
                results['global_feature_dist'] = {
                    'mean': float(np.mean(flat_features)),
                    'variance': float(np.var(flat_features))
                }
        
        # Analyze each client's feature distribution
        for client_id, data in client_data.items():
            if 'features' in data and data['features']:
                # Flatten client features and remove NaN values
                client_flat = np.concatenate([f.flatten() for f in data['features']])
                client_flat = client_flat[~np.isnan(client_flat)]  # Remove NaN values
                
                if len(client_flat) > 0:  # Only process if we have valid features
                    # Compute mean and variance
                    results['feature_means'][client_id] = float(np.mean(client_flat))
                    results['feature_variances'][client_id] = float(np.var(client_flat))
                    
                    # Compute JS divergence from global distribution
                    if results['global_feature_dist']:
                        # Create histograms for comparison
                        global_hist, _ = np.histogram(flat_features, bins=50, density=True)
                        client_hist, _ = np.histogram(client_flat, bins=50, density=True)
                        
                        # Add small value to avoid division by zero
                        global_hist = global_hist + 1e-10
                        client_hist = client_hist + 1e-10
                        
                        # Normalize
                        global_hist = global_hist / np.sum(global_hist)
                        client_hist = client_hist / np.sum(client_hist)
                        
                        # Compute JS divergence
                        results['js_divergences'][client_id] = self.js_divergence(global_hist, client_hist)
        
        return results

    @staticmethod
    def compute_feature_distribution(features: List[np.ndarray]) -> Dict[str, float]:
        """Compute statistics about feature distributions"""
        if not features:
            return {}
        
        features = np.array(features)
        return {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'skewness': float(float((features - np.mean(features)) ** 3).mean() / np.std(features) ** 3),
            'kurtosis': float(float((features - np.mean(features)) ** 4).mean() / np.std(features) ** 4 - 3)
        }
    
    def analyze_graph_structures(self, client_data: Dict[str, Dict]) -> Dict:
        """Analyze graph structure metrics for non-IID analysis"""
        results = {
            'node_counts': {},
            'edge_counts': {},
            'avg_degrees': {},
            'graph_densities': {}
        }
        
        for client_id, data in client_data.items():
            if 'graph' in data and data['graph'] is not None:
                graph = data['graph']
                results['node_counts'][client_id] = graph.num_nodes
                results['edge_counts'][client_id] = graph.num_edges
                
                # Compute average degree
                if graph.num_nodes > 0 and hasattr(graph, 'edge_index'):
                    # For undirected graphs, degree is number of edges * 2 / num_nodes
                    results['avg_degrees'][client_id] = (2 * graph.num_edges) / graph.num_nodes
                else:
                    results['avg_degrees'][client_id] = 0
                    
                # Compute graph density
                if graph.num_nodes > 1:
                    max_edges = graph.num_nodes * (graph.num_nodes - 1) / 2  # For undirected graphs
                    results['graph_densities'][client_id] = graph.num_edges / max_edges
                else:
                    results['graph_densities'][client_id] = 0
        
        return results

    def analyze_non_iid_comprehensive(self, data_dir: str, num_clients: int, mode: str = "binary") -> Dict:
        """Comprehensive Non-IID analysis including feature distributions and graph properties"""
        results = {
            'label_distributions': {},
            'feature_statistics': {},
            'graph_properties': {},
            'divergence_metrics': {}
        }
        
        # Load client data
        client_data = {}
        for cid in range(num_clients):
            data = load_client_data(data_dir, str(cid))
            if data:
                if mode == "oneclass":
                    client_data[str(cid)] = {
                        'features': [d.x.numpy() for d in data if hasattr(d, 'x')],
                        'graph': data[0] if data else None
                    }
                else:
                    if data and hasattr(data[0], 'y'):
                        labels = [d.y.item() for d in data if hasattr(d, 'y')]
                        client_data[str(cid)] = {
                            'labels': labels,
                            'graph': data[0]
                        }
        
        if mode == "oneclass":
            # Analyze feature distributions
            feature_results = self.analyze_feature_distributions(client_data)
            results['feature_statistics'] = feature_results
            
            # Analyze graph structures
            graph_results = self.analyze_graph_structures(client_data)
            results['graph_properties'] = graph_results
        else:
            # Get global label distribution
            all_labels = []
            for data in client_data.values():
                if 'labels' in data:
                    all_labels.extend(data['labels'])
            
            if all_labels:
                global_label_dist = self.compute_label_distribution(all_labels, num_classes=2)
                
                # Analyze each client
                for cid, data in client_data.items():
                    if 'labels' in data:
                        # Label distribution
                        client_labels = data['labels']
                        client_label_dist = self.compute_label_distribution(client_labels, len(global_label_dist))
                        results['label_distributions'][cid] = client_label_dist
                        
                        # Divergence metrics
                        results['divergence_metrics'][cid] = {
                            'kl_divergence': self.kl_divergence(client_label_dist, global_label_dist),
                            'js_divergence': self.js_divergence(client_label_dist, global_label_dist),
                            'em_distance': self.earth_movers_distance(client_label_dist, global_label_dist)
                        }
                    
                    # Graph properties
                    if 'graph' in data and data['graph'] is not None:
                        graph = data['graph']
                        G = nx.Graph()
                        if hasattr(graph, 'edge_index'):
                            G.add_edges_from(graph.edge_index.t().numpy())
                            
                        results['graph_properties'][cid] = {
                            'num_nodes': G.number_of_nodes(),
                            'num_edges': G.number_of_edges(),
                            'density': nx.density(G),
                            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
                        }
        
        return results

# Helper function for the script
def analyze_non_iid_characteristics(data_dir: str, num_clients: int) -> Dict:
    analyzer = NonIIDMetrics()
    return analyzer.analyze_non_iid_comprehensive(data_dir, num_clients)
