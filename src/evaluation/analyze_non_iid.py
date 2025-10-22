# src/analyze_non_iid.py (updated)
import argparse
from non_iid_metrics import NonIIDMetrics
from src.utils import load_client_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

def extract_features_from_graph(graph):
    """Extract features from graph object, handling both unified and individual feature formats"""
    if hasattr(graph, 'x') and graph.x is not None:
        # Unified feature matrix exists
        return graph.x.numpy()
    else:
        # Extract features from individual attributes
        feature_arrays = []
        reserved_attrs = ['edge_index', 'y', 'num_nodes']
        
        # Get all attributes that are likely features
        for attr_name in dir(graph):
            if not attr_name.startswith('_') and attr_name not in reserved_attrs:
                attr = getattr(graph, attr_name)
                if isinstance(attr, torch.Tensor) and attr.dim() == 1:
                    feature_arrays.append(attr.numpy())
        
        if feature_arrays:
            # Stack individual features into a matrix
            return np.column_stack(feature_arrays)
        else:
            return np.array([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="oneclass", choices=["oneclass", "binary"])
    args = parser.parse_args()
    
    analyzer = NonIIDMetrics()
    
    # Load data from all clients
    client_data = {}
    for cid in range(args.num_clients):
        data = load_client_data(args.data_dir, str(cid))
        if data and isinstance(data, dict) and 'graph' in data:
            graph = data['graph']
            
            # Extract features from graph
            features = extract_features_from_graph(graph)
            
            if args.mode == "oneclass":
                client_data[str(cid)] = {
                    'features': [features],
                    'graph': graph
                }
            else:
                # For binary mode, use labels
                if hasattr(graph, 'y') and graph.y is not None:
                    labels = graph.y.cpu().numpy().tolist()
                    client_data[str(cid)] = {
                        'labels': labels,
                        'graph': graph
                    }
                else:
                    print(f"Client {cid} graph has no labels")
        else:
            print(f"No valid data for client {cid}")
    
    if args.mode == "oneclass":
        # Analyze feature distributions for one-class scenario
        results = analyzer.analyze_feature_distributions(client_data)
        
        # Create visualizations for feature distributions
        plt.figure(figsize=(12, 8))
        
        # Plot feature means
        plt.subplot(2, 2, 1)
        means = [results['feature_means'].get(str(i), 0) for i in range(args.num_clients)]
        plt.bar(range(len(means)), means)
        plt.title("Feature Means by Client")
        plt.xlabel("Client ID")
        plt.ylabel("Mean Feature Value")
        
        # Plot feature variances
        plt.subplot(2, 2, 2)
        variances = [results['feature_variances'].get(str(i), 0) for i in range(args.num_clients)]
        plt.bar(range(len(variances)), variances)
        plt.title("Feature Variances by Client")
        plt.xlabel("Client ID")
        plt.ylabel("Feature Variance")
        
        # Plot JS divergences of feature distributions
        plt.subplot(2, 2, 3)
        js_divergences = [results['js_divergences'].get(str(i), 0) for i in range(args.num_clients)]
        plt.bar(range(len(js_divergences)), js_divergences)
        plt.title("JS Divergence of Feature Distributions")
        plt.xlabel("Client ID")
        plt.ylabel("JS Divergence")
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/feature_distributions.png")
        
    else:
        # Compute global distribution for binary mode
        all_labels = []
        for data in client_data.values():
            if 'labels' in data:
                all_labels.extend(data['labels'])
        
        if all_labels:
            global_dist = analyzer.compute_label_distribution(all_labels, num_classes=2)
            
            # Analyze Non-IID characteristics
            results = analyzer.analyze_non_iid(client_data, global_dist)
            
            # Create visualizations for all metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot KL divergences
            axes[0, 0].bar(range(len(results['kl_divergences'])), list(results['kl_divergences'].values()))
            axes[0, 0].set_title("KL Divergence by Client")
            axes[0, 0].set_xlabel("Client ID")
            axes[0, 0].set_ylabel("KL Divergence")
            
            # Plot JS divergences
            axes[0, 1].bar(range(len(results['js_divergences'])), list(results['js_divergences'].values()))
            axes[0, 1].set_title("JS Divergence by Client")
            axes[0, 1].set_xlabel("Client ID")
            axes[0, 1].set_ylabel("JS Divergence")
            
            # Plot EM distances
            axes[1, 0].bar(range(len(results['em_distances'])), list(results['em_distances'].values()))
            axes[1, 0].set_title("Earth Mover's Distance by Client")
            axes[1, 0].set_xlabel("Client ID")
            axes[1, 0].set_ylabel("EM Distance")
            
            # Plot homophily scores if available
            if results['homophily_scores']:
                axes[1, 1].bar(range(len(results['homophily_scores'])), list(results['homophily_scores'].values()))
                axes[1, 1].set_title("Homophily Scores by Client")
                axes[1, 1].set_xlabel("Client ID")
                axes[1, 1].set_ylabel("Homophily Score")
            
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/non_iid_metrics.png")
    
    # Save results
    if results:
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(f"{args.output_dir}/non_iid_metrics.csv")
    
    print("Non-IID analysis completed")

if __name__ == "__main__":
    main()