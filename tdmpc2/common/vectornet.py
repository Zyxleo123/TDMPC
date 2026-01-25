"""
VectorNet Encoder for TD-MPC2.
Adapted from VTN (VectorNet Transformer) architecture for processing
vectorized scene representations in autonomous driving.
"""
import torch
import torch.nn as nn


class MLPLayer(nn.Module):
    """MLP layer matching VTN structure for weight compatibility."""
    def __init__(self, in_channel, out_channel, hidden=64):
        super().__init__()
        self.linear1 = nn.Linear(in_channel, hidden)
        self.linear2 = nn.Linear(hidden, out_channel)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(out_channel)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.act(out)
        return out


class VectorNetSubGraph(nn.Module):
    """
    SubGraph module for VectorNet that processes polylines (sequences of vectors).
    Structure matches VTN from vlm-clean for pre-trained weight compatibility.
    """
    def __init__(self, in_channel, num_layers=3, hidden_dim=64, num_steps=9):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # GLP (Graph Layer Propagation) layers - use ModuleList for VTN compatibility
        self.glp_layers = nn.ModuleList()
        self.agg_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.glp_layers.append(MLPLayer(in_channel, hidden_dim, hidden_dim))
            self.agg_layers.append(MLPLayer(num_steps, 1, hidden_dim))
            in_channel = hidden_dim * 2
        
        # Final aggregation
        self.final_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_agg = nn.Linear(num_steps, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch * n_objects, n_steps, n_features)
        Returns:
            (batch * n_objects, hidden_dim)
        """
        # GLP iterations
        for glp_layer, agg_layer in zip(self.glp_layers, self.agg_layers):
            glp_out = glp_layer(x)  # (B*O, S, H)
            # Aggregate across time
            agg_out = agg_layer(glp_out.transpose(-2, -1))  # (B*O, H, 1)
            agg_out = agg_out.expand(-1, -1, self.num_steps).transpose(-2, -1)  # (B*O, S, H)
            x = torch.cat([glp_out, agg_out], dim=-1)  # (B*O, S, 2H)
        
        # Final aggregation
        x = self.final_linear(x)  # (B*O, S, H)
        x = self.final_agg(x.transpose(-2, -1))  # (B*O, H, 1)
        x = x.squeeze(-1)  # (B*O, H)
        return torch.nn.functional.normalize(x, p=2.0, dim=-1)


class VectorNetGlobalGraph(nn.Module):
    """
    Global Graph module using self-attention to aggregate object-level features.
    """
    def __init__(self, in_channel, hidden_dim, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SelfAttentionLayer(in_channel, hidden_dim))
            in_channel = hidden_dim
    
    def forward(self, x, valid_lens):
        """
        Args:
            x: (batch, n_objects, features)
            valid_lens: (batch,) number of valid objects per batch
        Returns:
            (batch, n_objects, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x, valid_lens)
        return x


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for global graph aggregation."""
    def __init__(self, in_channel, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_lin = nn.Linear(in_channel, hidden_dim)
        self.k_lin = nn.Linear(in_channel, hidden_dim)
        self.v_lin = nn.Linear(in_channel, hidden_dim)
    
    def forward(self, x, valid_lens):
        """
        Args:
            x: (batch, n_objects, features)
            valid_lens: (batch,) or None
        """
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        
        # Attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        
        # Masked softmax
        if valid_lens is not None:
            mask = self._create_mask(scores, valid_lens)
            scores = scores.masked_fill(mask, -1e12)
        
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        if valid_lens is not None:
            attn_weights = attn_weights * (~mask).float()
        
        return torch.bmm(attn_weights, value)
    
    def _create_mask(self, scores, valid_lens):
        """Create attention mask based on valid lengths."""
        batch_size, n_objects, _ = scores.shape
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for batch_id, valid_len in enumerate(valid_lens):
            valid_len = int(valid_len.item())
            mask[batch_id, :, valid_len:] = True
            mask[batch_id, valid_len:, :] = True
        return mask


class VectorNetEncoder(nn.Module):
    """
    VectorNet encoder for vectorized scene representation.
    Processes structured observations of agents, lanes, traffic lights, and waypoints.
    
    Args:
        n_objects: Total number of objects (e.g., 151 = 50 agents + 50 lanes + 50 lights + 1 waypoint)
        n_steps: Number of temporal steps (e.g., 9)
        n_features: Number of features per object-step (e.g., 10)
        latent_dim: Output latent dimension
        task_dim: Task embedding dimension (for multi-task)
        hidden_dim: Hidden dimension for VectorNet layers
    """
    def __init__(self, n_objects, n_steps, n_features, latent_dim, task_dim=0, hidden_dim=64):
        super().__init__()
        self.n_objects = n_objects
        self.n_steps = n_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        
        # SubGraph processes each object's temporal sequence
        self.subgraph = VectorNetSubGraph(
            in_channel=n_features,
            num_layers=3,
            hidden_dim=hidden_dim,
            num_steps=n_steps
        )
        
        # Global graph aggregates across all objects
        # Input: hidden_dim + 2 (relative position x,y)
        self.global_graph = VectorNetGlobalGraph(
            in_channel=hidden_dim + 2,
            hidden_dim=hidden_dim,
            num_layers=1
        )
        
        # Final projection to latent space
        # We extract waypoint info (last object's x,y) and concatenate
        # hidden_dim (from first object/ego) + 2 (waypoint x,y) + task_dim
        final_input_dim = hidden_dim + 2 + task_dim
        self.final_projection = nn.Sequential(
            nn.Linear(final_input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, obs):
        """
        Args:
            obs: Can be one of:
                - (batch, n_objects, n_steps, n_features) - standard
                - (horizon, batch, n_objects, n_steps, n_features) - trajectory from replay buffer
                - (batch, n_objects * n_steps * n_features) - flattened
                - (n_objects, n_steps, n_features) - single sample
        Returns:
            (batch, latent_dim) or (horizon, batch, latent_dim)
        """
        original_shape = obs.shape
        has_horizon = False
        
        # Handle different input shapes
        if obs.ndim == 5:
            # Trajectory: (horizon, batch, n_objects, n_steps, n_features)
            has_horizon = True
            horizon, batch_size = obs.shape[0], obs.shape[1]
            # Merge horizon and batch: (horizon*batch, n_objects, n_steps, n_features)
            obs = obs.reshape(horizon * batch_size, self.n_objects, self.n_steps, self.n_features)
        elif obs.ndim == 4:
            # Already structured: (batch, n_objects, n_steps, n_features)
            batch_size = obs.shape[0]
        elif obs.ndim == 3:
            # Missing batch dimension: (n_objects, n_steps, n_features)
            obs = obs.unsqueeze(0)
            batch_size = 1
        elif obs.ndim == 2:
            # Flattened: (batch, n_objects * n_steps * n_features)
            batch_size = obs.shape[0]
            total_size = self.n_objects * self.n_steps * self.n_features
            if obs.shape[1] != total_size:
                raise ValueError(
                    f"Flattened observation size {obs.shape[1]} doesn't match "
                    f"expected {total_size} ({self.n_objects}*{self.n_steps}*{self.n_features})"
                )
            obs = obs.reshape(batch_size, self.n_objects, self.n_steps, self.n_features)
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}, ndim={obs.ndim}")
        
        # Now obs is always (batch_size, n_objects, n_steps, n_features)
        # where batch_size might be horizon*batch if has_horizon=True
        current_batch = obs.shape[0]
        
        # Separate waypoint (last object) from other objects
        waypoint = obs[:, -1, -1, :2]  # (current_batch, 2) - x, y position
        objects = obs[:, :-1, :, :]  # (current_batch, n_objects-1, n_steps, n_features)
        n_valid_objects = objects.shape[1]
        
        # Process each object's polyline through SubGraph
        # Reshape: (current_batch, n_objects-1, n_steps, n_features) -> (current_batch*(n_objects-1), n_steps, n_features)
        objects_flat = objects.reshape(-1, self.n_steps, self.n_features)
        subgraph_out = self.subgraph(objects_flat)  # (current_batch*(n_objects-1), hidden_dim)
        subgraph_out = subgraph_out.reshape(current_batch, n_valid_objects, self.hidden_dim)
        
        # Add relative position information (x, y from last timestep)
        relative_pos = objects[:, :, -1, :2]  # (current_batch, n_objects-1, 2)
        subgraph_out = torch.cat([subgraph_out, relative_pos], dim=-1)  # (current_batch, n_objects-1, hidden_dim+2)
        
        # Global aggregation with self-attention
        valid_lens = torch.ones(current_batch, device=obs.device) * n_valid_objects
        global_out = self.global_graph(subgraph_out, valid_lens)  # (current_batch, n_objects-1, hidden_dim)
        
        # Extract ego vehicle feature (first object) and concatenate with waypoint
        ego_feature = global_out[:, 0, :]  # (current_batch, hidden_dim)
        combined = torch.cat([ego_feature, waypoint], dim=-1)  # (current_batch, hidden_dim+2)
        
        # Add task embedding if provided (will be concatenated in world_model.encode)
        # For now, just project to latent space
        if self.task_dim > 0:
            # Add zero task embedding placeholder (will be replaced by world_model.task_emb)
            task_placeholder = torch.zeros(current_batch, self.task_dim, device=obs.device)
            combined = torch.cat([combined, task_placeholder], dim=-1)
        
        latent = self.final_projection(combined)  # (current_batch, latent_dim)
        
        # Restore horizon dimension if it was present
        if has_horizon:
            latent = latent.reshape(horizon, batch_size, self.latent_dim)
        
        return latent

