# Containing src functions on Adjacency creation (k-NN, spatial joins, etc.) and PyG Data object assembly from features, edge_index, targets

# def build_node_feature_tensor(df: pd.DataFrame, feature_cols: List[str]) -> torch.Tensor:
#     """Extracts node features and converts to float32 tensor."""
# def build_target_tensor(df: pd.DataFrame, target_col: str) -> torch.Tensor:
#     """Extracts target GWL values for observed nodes."""
# def build_graph_data_object(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
#                             y: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> Data:
#     """Assembles PyTorch Geometric Data object with optional node masking."""