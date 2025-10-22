# src/utils/graph_builder.py
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import logging

logger = logging.getLogger(__name__)

def safe_statistics(series, default=0.0):
    """Compute statistics safely, handling NaN/Inf values"""
    if series.empty:
        return [default] * 5
    
    # Replace inf with nan, then replace nan with default
    clean_series = series.replace([np.inf, -np.inf], np.nan).fillna(default)
    
    # Compute statistics
    return [
        clean_series.mean(),
        clean_series.std() if len(clean_series) > 1 else default,
        clean_series.max(),
        clean_series.min(),
        clean_series.sum()
    ]

def build_per_flow_graph_optimized(client_data: pd.DataFrame) -> Data:
    """Build a graph where each flow is a node - optimized version with NaN handling"""
    G = nx.Graph()
    
    # Clean the data first - replace NaN/Inf with 0
    numeric_cols = client_data.select_dtypes(include=[np.number]).columns
    client_data_clean = client_data.copy()
    for col in numeric_cols:
        client_data_clean[col] = client_data_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Add nodes (flows)
    logger.info("Adding nodes...")
    for idx, row in tqdm(client_data_clean.iterrows(), total=len(client_data_clean), desc="Adding nodes"):
        # Use flow features as node attributes
        features = {col: row[col] for col in client_data_clean.columns 
                   if col not in ['id.orig_h', 'id.resp_h', 'label'] 
                   and not col.startswith('proto_') 
                   and not col.startswith('service_') 
                   and not col.startswith('conn_')}
        G.add_node(idx, **features)
    
    # Create IP to flow index mapping
    logger.info("Creating IP to flow mapping...")
    ip_to_flows = defaultdict(set)
    for idx, row in tqdm(client_data_clean.iterrows(), total=len(client_data_clean), desc="Mapping IPs"):
        src_ip = row['id.orig_h']
        dst_ip = row['id.resp_h']
        ip_to_flows[src_ip].add(idx)
        ip_to_flows[dst_ip].add(idx)
    
    # Add edges based on shared IP addresses
    logger.info("Adding edges...")
    edges_added = set()
    
    for ip, flow_indices in tqdm(ip_to_flows.items(), desc="Processing IPs"):
        flow_list = list(flow_indices)
        # Connect all flows that share this IP
        for i in range(len(flow_list)):
            for j in range(i+1, len(flow_list)):
                edge = tuple(sorted([flow_list[i], flow_list[j]]))
                if edge not in edges_added:
                    G.add_edge(flow_list[i], flow_list[j])
                    edges_added.add(edge)
    
    # Convert to PyG data
    logger.info("Converting to PyG format...")
    pyg_data = from_networkx(G)
    
    # Add labels
    if 'label' in client_data_clean.columns:
        labels = client_data_clean['label'].apply(lambda x: 0 if 'benign' in str(x).lower() else 1).values
        pyg_data.y = torch.tensor(labels, dtype=torch.long)
    
    return pyg_data

def build_per_flow_graph_memory_efficient(client_data: pd.DataFrame) -> Data:
    """Build a per-flow graph with memory efficiency in mind"""
    # Create a sparse representation of the graph
    from scipy.sparse import lil_matrix
    import numpy as np
    
    n_nodes = len(client_data)
    
    # Create IP to node index mapping
    ip_to_nodes = defaultdict(list)
    for idx, row in client_data.iterrows():
        src_ip = row['id.orig_h']
        dst_ip = row['id.resp_h']
        ip_to_nodes[src_ip].append(idx)
        ip_to_nodes[dst_ip].append(idx)
    
    # Create adjacency matrix in sparse format
    adj_matrix = lil_matrix((n_nodes, n_nodes), dtype=np.int8)
    
    # Add edges based on shared IPs
    for ip, nodes in tqdm(ip_to_nodes.items(), desc="Building adjacency matrix"):
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                adj_matrix[nodes[i], nodes[j]] = 1
                adj_matrix[nodes[j], nodes[i]] = 1  # Undirected graph
    
    # Convert to edge index format for PyG
    edge_index = adj_matrix.nonzero()
    edge_index = torch.tensor([edge_index[0], edge_index[1]], dtype=torch.long)
    
    # Create node features
    feature_cols = [col for col in client_data.columns 
                   if col not in ['id.orig_h', 'id.resp_h', 'label'] 
                   and not col.startswith('proto_') 
                   and not col.startswith('service_') 
                   and not col.startswith('conn_')]
    
    x = torch.tensor(client_data[feature_cols].values, dtype=torch.float)
    
    # Create labels
    if 'label' in client_data.columns:
        labels = client_data['label'].apply(lambda x: 0 if 'benign' in str(x) else 1).values
        y = torch.tensor(labels, dtype=torch.long)
    else:
        y = None
    
    # Create PyG data object
    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    
    return pyg_data

def build_per_flow_graph_chunked(client_data: pd.DataFrame, chunk_size: int = 10000) -> Data:
    """Build per-flow graph by processing in chunks to save memory"""
    # This is a more complex implementation that processes the data in chunks
    # Graph is created incrementally
    
    G = nx.Graph()
    
    # Add nodes in chunks
    logger.info("Adding nodes...")
    for start in range(0, len(client_data), chunk_size):
        end = min(start + chunk_size, len(client_data))
        chunk = client_data.iloc[start:end]
        
        for idx, row in chunk.iterrows():
            features = {col: row[col] for col in client_data.columns 
                       if col not in ['id.orig_h', 'id.resp_h', 'label'] 
                       and not col.startswith('proto_') 
                       and not col.startswith('service_') 
                       and not col.startswith('conn_')}
            G.add_node(idx, **features)
    
    # Create IP to node mapping
    logger.info("Creating IP to node mapping...")
    ip_to_nodes = defaultdict(list)
    for idx, row in client_data.iterrows():
        src_ip = row['id.orig_h']
        dst_ip = row['id.resp_h']
        ip_to_nodes[src_ip].append(idx)
        ip_to_nodes[dst_ip].append(idx)
    
    # Add edges in chunks
    logger.info("Adding edges...")
    for ip, nodes in tqdm(ip_to_nodes.items(), desc="Processing IPs"):
        # Process this IP's nodes in chunks if there are too many
        if len(nodes) > chunk_size:
            for i in range(0, len(nodes), chunk_size):
                chunk_nodes = nodes[i:i+chunk_size]
                for j in range(i, min(i+chunk_size, len(nodes))):
                    for k in range(j+1, min(i+chunk_size, len(nodes))):
                        G.add_edge(nodes[j], nodes[k])
        else:
            # Connect all nodes that share this IP
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    G.add_edge(nodes[i], nodes[j])
    
    # Convert to PyG data
    logger.info("Converting to PyG format...")
    pyg_data = from_networkx(G)
    
    # Add labels
    if 'label' in client_data.columns:
        labels = client_data['label'].apply(lambda x: 0 if 'benign' in str(x) else 1).values
        pyg_data.y = torch.tensor(labels, dtype=torch.long)
    
    return pyg_data

def build_bipartite_graph(client_data: pd.DataFrame) -> Data:
    """Build a bipartite graph between IPs and flows"""
    G = nx.Graph()
    
    # Add flow nodes
    for idx, row in client_data.iterrows():
        features = {col: row[col] for col in client_data.columns 
                   if col not in ['id.orig_h', 'id.resp_h', 'label'] 
                   and not col.startswith('proto_') 
                   and not col.startswith('service_') 
                   and not col.startswith('conn_')}
        G.add_node(f"flow_{idx}", bipartite=0, **features)
    
    # Add IP nodes and connect to flows
    all_ips = set(client_data['id.orig_h'].unique()) | set(client_data['id.resp_h'].unique())
    for ip in all_ips:
        G.add_node(f"ip_{ip}", bipartite=1)
    
    # Connect flows to IPs
    for idx, row in client_data.iterrows():
        src_ip = row['id.orig_h']
        dst_ip = row['id.resp_h']
        G.add_edge(f"flow_{idx}", f"ip_{src_ip}")
        G.add_edge(f"flow_{idx}", f"ip_{dst_ip}")
    
    # Convert to PyG data
    pyg_data = from_networkx(G)
    
    # Add labels (only for flow nodes)
    if 'label' in client_data.columns:
        labels = []
        for node in G.nodes():
            if node.startswith("flow_"):
                idx = int(node.split("_")[1])
                label = 0 if 'benign' in str(client_data.iloc[idx]['label']) else 1
                labels.append(label)
        pyg_data.y = torch.tensor(labels, dtype=torch.long)
    
    return pyg_data

def build_per_host_graph_optimized(client_data: pd.DataFrame) -> Data:
    """Build a graph where each host is a node - optimized version"""
    G = nx.Graph()
    
    # Collect all unique IPs
    all_ips = set(client_data['id.orig_h'].unique()) | set(client_data['id.resp_h'].unique())
    
    # Aggregate features per host
    logger.info("Aggregating host features...")
    host_features = {}
    numeric_cols = client_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['id.orig_h', 'id.resp_h', 'label']]
    
    for ip in tqdm(all_ips, desc="Processing hosts"):
        # Flows where this IP is the source
        src_flows = client_data[client_data['id.orig_h'] == ip]
        # Flows where this IP is the destination
        dst_flows = client_data[client_data['id.resp_h'] == ip]
        
        if not src_flows.empty:
            src_features = src_flows[numeric_cols].mean().to_dict()
        else:
            src_features = {col: 0 for col in numeric_cols}
            
        if not dst_flows.empty:
            dst_features = dst_flows[numeric_cols].mean().to_dict()
        else:
            dst_features = {col: 0 for col in numeric_cols}
        
        features = {}
        for col in numeric_cols:
            features[f'src_{col}'] = src_features.get(col, 0)
            features[f'dst_{col}'] = dst_features.get(col, 0)
        
        host_features[ip] = features
        G.add_node(ip, **features)
    
    # Add edges based on flows between hosts
    logger.info("Adding edges between hosts...")
    # Use groupby to get unique connections
    connections = client_data[['id.orig_h', 'id.resp_h']].drop_duplicates()
    
    for _, row in tqdm(connections.iterrows(), total=len(connections), desc="Adding edges"):
        src_ip = row['id.orig_h']
        dst_ip = row['id.resp_h']
        if src_ip != dst_ip:  # Avoid self-loops
            G.add_edge(src_ip, dst_ip)
    
    # Convert to PyG data
    logger.info("Converting to PyG format...")
    pyg_data = from_networkx(G)
    
    # Add labels (mark host as malicious if it appears in any malicious flow)
    if 'label' in client_data.columns:
        malicious_ips = set()
        malicious_flows = client_data[client_data['label'].str.contains('malicious', case=False, na=False)]
        
        malicious_ips.update(malicious_flows['id.orig_h'].unique())
        malicious_ips.update(malicious_flows['id.resp_h'].unique())
        
        labels = [1 if ip in malicious_ips else 0 for ip in all_ips]
        pyg_data.y = torch.tensor(labels, dtype=torch.long)
    
    return pyg_data

def build_per_host_graph_enhanced(client_data: pd.DataFrame) -> Data:
    """Build a per-host graph with enhanced feature aggregation and unified feature matrix"""
    G = nx.Graph()
    
    # Clean the data first - replace NaN/Inf with 0
    numeric_cols = client_data.select_dtypes(include=[np.number]).columns
    client_data_clean = client_data.copy()
    for col in numeric_cols:
        client_data_clean[col] = client_data_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Collect all unique IPs
    all_ips = set(client_data_clean['id.orig_h'].unique()) | set(client_data_clean['id.resp_h'].unique())
    
    # Identify numeric columns for aggregation (excluding IP and label columns)
    numeric_cols = [col for col in numeric_cols 
                   if col not in ['id.orig_h', 'id.resp_h', 'label'] 
                   and not col.startswith('proto_') 
                   and not col.startswith('service_') 
                   and not col.startswith('conn_')]
    
    # For each host, compute multiple statistics
    all_features = []
    ip_to_idx = {}
    
    for idx, ip in enumerate(tqdm(all_ips, desc="Processing hosts")):
        ip_to_idx[ip] = idx
        
        # Flows where this IP is the source
        src_flows = client_data_clean[client_data_clean['id.orig_h'] == ip]
        # Flows where this IP is the destination
        dst_flows = client_data_clean[client_data_clean['id.resp_h'] == ip]
        
        features = []
        
        # Source-side statistics
        for col in numeric_cols:
            features.extend(safe_statistics(src_flows[col] if not src_flows.empty else pd.Series([])))
        
        # Destination-side statistics
        for col in numeric_cols:
            features.extend(safe_statistics(dst_flows[col] if not dst_flows.empty else pd.Series([])))
        
        # Count-based features
        features.append(len(src_flows))
        features.append(len(dst_flows))
        features.append(len(src_flows) + len(dst_flows))
        
        all_features.append(features)
        G.add_node(ip)
    
    # Add edges based on flows between hosts
    connections = client_data_clean[['id.orig_h', 'id.resp_h']].drop_duplicates()
    
    for _, row in tqdm(connections.iterrows(), total=len(connections), desc="Adding edges"):
        src_ip = row['id.orig_h']
        dst_ip = row['id.resp_h']
        if src_ip != dst_ip and src_ip in ip_to_idx and dst_ip in ip_to_idx:  # Avoid self-loops
            G.add_edge(src_ip, dst_ip)
    
    # Convert to PyG data
    pyg_data = from_networkx(G)
    
    # Add unified feature matrix
    if all_features:
        pyg_data.x = torch.tensor(all_features, dtype=torch.float)
    else:
        # Create default features if no features were computed
        n_features = len(numeric_cols) * 10 + 3  # 5 stats per col for src + dst, plus 3 count features
        pyg_data.x = torch.zeros((len(all_ips), n_features), dtype=torch.float)
    
    # Add labels (mark host as malicious if it appears in any malicious flow)
    if 'label' in client_data_clean.columns:
        malicious_ips = set()
        malicious_flows = client_data_clean[client_data_clean['label'].str.contains('malicious', case=False, na=False)]
        
        malicious_ips.update(malicious_flows['id.orig_h'].unique())
        malicious_ips.update(malicious_flows['id.resp_h'].unique())
        
        labels = [1 if ip in malicious_ips else 0 for ip in all_ips]
        pyg_data.y = torch.tensor(labels, dtype=torch.long)
    else:
        # Default to all benign if no labels
        pyg_data.y = torch.zeros(len(all_ips), dtype=torch.long)
    
    return pyg_data

def process_client_graph(args):
    """Process a single client graph (for parallel processing)"""
    i, processed_dir, graph_dir, graph_type = args
    try:
        # Load client data
        client_data = torch.load(processed_dir / f"client_{i}.pt", weights_only=False)
        
        # Build graph based on the selected type
        if graph_type == "per_flow":
            graph_data = build_per_flow_graph_optimized(client_data['data'])
        else:  # per_host
            graph_data = build_per_host_graph_enhanced(client_data['data'])
        
        # Save graph
        torch.save({
            'graph': graph_data,
            'client_id': i,
            'graph_type': graph_type,
            'labeling_strategy': client_data['labeling_strategy']
        }, graph_dir / f"client_graph_{i}.pt")
        
        return i, True
    except Exception as e:
        logger.error(f"Error processing client {i}: {e}")
        return i, False

def build_client_graphs_parallel(processed_dir: Path, graph_dir: Path, graph_type: str = "per_host"):
    """Build graphs for all clients using parallel processing"""
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset info
    dataset_info = torch.load(processed_dir / "dataset_info.pt", weights_only=False)
    n_clients = dataset_info['n_clients']
    
    # Prepare arguments for parallel processing
    args_list = [(i, processed_dir, graph_dir, graph_type) for i in range(n_clients)]
    
    # Use multiprocessing to process clients in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_client_graph, args_list),
            total=n_clients,
            desc="Processing clients"
        ))
    
    # Log results
    success_count = sum(1 for _, success in results if success)
    logger.info(f"Successfully processed {success_count}/{n_clients} clients")

def build_client_graphs(processed_dir: Path, graph_dir: Path, graph_type: str = "per_host", parallel: bool = True):
    """Build graphs for all clients"""
    if parallel:
        build_client_graphs_parallel(processed_dir, graph_dir, graph_type)
    else:
        # Load dataset info
        dataset_info = torch.load(processed_dir / "dataset_info.pt", weights_only=False)
        n_clients = dataset_info['n_clients']
        
        for i in tqdm(range(n_clients), desc="Clients"):
            # Load client data
            client_data = torch.load(processed_dir / f"client_{i}.pt", weights_only=False)
            
            # Build graph based on the selected type
            if graph_type == "per_flow":
                graph_data = build_per_flow_graph_optimized(client_data['data'])
            else:  # per_host
                graph_data = build_per_host_graph_optimized(client_data['data'])
            
            # Save graph
            torch.save({
                'graph': graph_data,
                'client_id': i,
                'graph_type': graph_type,
                'labeling_strategy': client_data['labeling_strategy']
            }, graph_dir / f"client_graph_{i}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graphs from processed data")
    parser.add_argument("--processed_dir", type=str, required=True, help="Processed data directory")
    parser.add_argument("--graph_dir", type=str, required=True, help="Output graph directory")
    parser.add_argument("--graph_type", type=str, default="per_host", 
                       choices=["per_flow", "per_host"], help="Type of graph to build")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    args = parser.parse_args()
    
    build_client_graphs(Path(args.processed_dir), Path(args.graph_dir), args.graph_type, args.parallel)
