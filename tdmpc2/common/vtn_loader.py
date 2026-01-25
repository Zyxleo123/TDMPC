"""
Load pre-trained VTN encoder weights from SAC checkpoint into TD-MPC2 VectorNet encoder.
"""
import os
import sys
import torch
import zipfile
import tempfile
from pathlib import Path


def load_pretrained_vtn_weights(vectornet_encoder, checkpoint_path, device='cuda', verbose=True):
    """
    Load pre-trained VTN encoder weights from SAC checkpoint into VectorNet encoder.
    
    The SAC checkpoint contains weights under structure:
    - actor.features_extractor.subgraph.glp_layer.mlp_X.*
    - actor.features_extractor.subgraph.agg_layer.mlp_X.*
    - actor.features_extractor.global_graph.layers.glp_X.*
    
    We map these to VectorNetEncoder structure:
    - subgraph.glp_layers.X.*
    - subgraph.agg_layers.X.*
    - global_graph.layers.X.*
    
    Args:
        vectornet_encoder: VectorNetEncoder instance
        checkpoint_path: Path to the SAC checkpoint (.zip file)
        device: Device to load weights to
        verbose: Print detailed loading info
        
    Returns:
        vectornet_encoder with loaded weights
    """
    if verbose:
        print(f"Loading pre-trained VTN weights from {checkpoint_path}")
    
    # Load the SAC checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract zip
        with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Load the policy (contains the VTN encoder)
        policy_path = os.path.join(tmpdir, 'policy.pth')
        checkpoint = torch.load(policy_path, map_location=device)
    
    # Extract and map VTN encoder weights
    vtn_weights = {}
    
    for key, value in checkpoint.items():
        if not key.startswith('actor.features_extractor'):
            continue
        
        # Remove prefix
        new_key = key.replace('actor.features_extractor.', '')
        
        # Map SubGraph weights
        # SAC: subgraph.glp_layer.mlp_X -> VectorNet: subgraph.glp_layers.X
        # SAC: subgraph.agg_layer.mlp_X -> VectorNet: subgraph.agg_layers.X
        new_key = new_key.replace('subgraph.glp_layer.mlp_', 'subgraph.glp_layers.')
        new_key = new_key.replace('subgraph.agg_layer.mlp_', 'subgraph.agg_layers.')
        
        # Map final mlp_dims and mlp_step
        new_key = new_key.replace('subgraph.mlp_dims', 'subgraph.final_linear')
        new_key = new_key.replace('subgraph.mlp_step', 'subgraph.final_agg')
        
        # Map GlobalGraph weights
        # SAC: global_graph.layers.glp_X -> VectorNet: global_graph.layers.X
        new_key = new_key.replace('global_graph.layers.glp_', 'global_graph.layers.')
        
        vtn_weights[new_key] = value
    
    if verbose:
        print(f"Extracted {len(vtn_weights)} VTN weights from checkpoint")
    
    # Load weights into VectorNet encoder
    model_dict = vectornet_encoder.state_dict()
    
    # Filter and load matching keys
    matched_weights = {}
    mismatched_shapes = []
    missing_keys = []
    
    for key, value in vtn_weights.items():
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                matched_weights[key] = value
                if verbose:
                    print(f"  ✓ {key}: {tuple(value.shape)}")
            else:
                mismatched_shapes.append((key, value.shape, model_dict[key].shape))
                if verbose:
                    print(f"  ✗ Shape mismatch {key}: checkpoint={tuple(value.shape)}, model={tuple(model_dict[key].shape)}")
        else:
            missing_keys.append(key)
            if verbose:
                print(f"  ✗ Key not in model: {key}")
    
    # Update model with matched weights
    model_dict.update(matched_weights)
    vectornet_encoder.load_state_dict(model_dict, strict=False)
    
    print(f"\n{'='*60}")
    print(f"Pre-trained VTN weights loaded successfully!")
    print(f"  ✓ Loaded: {len(matched_weights)} weights")
    if mismatched_shapes:
        print(f"  ✗ Shape mismatches: {len(mismatched_shapes)}")
    if missing_keys:
        print(f"  ✗ Missing keys: {len(missing_keys)}")
    print(f"{'='*60}\n")
    
    return vectornet_encoder


def load_full_sac_checkpoint(checkpoint_path, device='cuda'):
    """
    Load and inspect the full SAC checkpoint structure.
    
    Args:
        checkpoint_path: Path to the SAC checkpoint (.zip file)
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint contents
    """
    print(f"Inspecting SAC checkpoint: {checkpoint_path}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract zip
        with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # List all files
        files = []
        for root, dirs, filenames in os.walk(tmpdir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, tmpdir)
                files.append(rel_path)
                print(f"  - {rel_path}")
        
        # Try to load main checkpoint files
        checkpoint_data = {}
        for file in files:
            if file.endswith('.pth') or file.endswith('.pt') or 'policy' in file:
                try:
                    full_path = os.path.join(tmpdir, file)
                    data = torch.load(full_path, map_location=device)
                    checkpoint_data[file] = data
                    
                    # Print structure
                    if isinstance(data, dict):
                        print(f"\n{file} keys:")
                        for key in list(data.keys())[:20]:  # First 20 keys
                            value = data[key]
                            if torch.is_tensor(value):
                                print(f"    {key}: {value.shape}")
                            else:
                                print(f"    {key}: {type(value)}")
                except Exception as e:
                    print(f"  Could not load {file}: {e}")
        
        return checkpoint_data


if __name__ == "__main__":
    # Test loading
    checkpoint_path = "/zfsauton2/home/vlin3/projects/humanoid-bench/dependency/vlm-clean/model-call-080000-steps-0000800000.zip"
    
    if os.path.exists(checkpoint_path):
        print("=" * 60)
        print("Inspecting SAC checkpoint structure")
        print("=" * 60)
        load_full_sac_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

