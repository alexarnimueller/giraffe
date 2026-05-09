#!/usr/bin/env python
"""Simple tests for AttentiveFP encoder improvements."""

import torch
from giraffe.model import (
    AttentiveFP, 
    AttentiveFP2,
    MeanAggregation,
    SumAggregation,
    AttentionPooling,
    GRUUpdate,
    ResidualUpdate,
)


def create_dummy_batch(num_atoms=10, batch_size=2):
    """Create dummy graph data for testing."""
    # Create a simple molecule graph
    num_edges = num_atoms  # Simple ring-like structure
    
    # Edge index: [2, num_edges]
    edge_index = torch.randint(0, num_atoms, (2, num_edges))
    # Ensure no self-loops
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.shape[1] == 0:
        edge_index = torch.tensor([[0, 1], [1, 0]])
    
    # Node features [num_atoms, in_channels]
    in_channels = 133  # Default for AttentiveFP
    x = torch.randn(num_atoms, in_channels)
    
    # Edge features [num_edges, edge_dim]
    edge_dim = 14  # Default bond feature dim
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    
    # Batch assignment [num_atoms]
    batch = torch.randint(0, batch_size, (num_atoms,))
    
    return x, edge_index, edge_attr, batch


def test_basic_forward():
    """Test basic forward pass works."""
    print("Testing basic forward pass...")
    model = AttentiveFP(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
    )
    
    x, edge_index, edge_attr, batch = create_dummy_batch()
    out = model(x, edge_index, edge_attr, batch)
    
    assert out.shape == (2, 128), f"Expected shape (2, 128), got {out.shape}"
    print("  ✓ Basic forward pass OK")


def test_undirected():
    """Test undirected message passing."""
    print("Testing undirected message passing...")
    model = AttentiveFP(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        undirected=True,
    )
    
    x, edge_index, edge_attr, batch = create_dummy_batch()
    out = model(x, edge_index, edge_attr, batch)
    
    assert out.shape == (2, 128), f"Expected shape (2, 128), got {out.shape}"
    print("  ✓ Undirected forward pass OK")


def test_aggregators():
    """Test different aggregator options."""
    print("Testing aggregator options...")
    
    for agg_name, agg_cls in [
        ("add", SumAggregation),
        ("mean", MeanAggregation),
        ("attention", AttentionPooling),
    ]:
        model = AttentiveFP(
            in_channels=133,
            hidden_channels=256,
            out_channels=128,
            edge_dim=14,
            num_layers=2,
            num_timesteps=2,
            dropout=0.1,
            aggregator=agg_name,
        )
        
        x, edge_index, edge_attr, batch = create_dummy_batch()
        out = model(x, edge_index, edge_attr, batch)
        
        assert out.shape == (2, 128), f"Aggregator {agg_name}: expected shape (2, 128), got {out.shape}"
        print(f"  ✓ Aggregator '{agg_name}' OK")


def test_return_vertex_embeddings():
    """Test returning vertex embeddings."""
    print("Testing return_vertex_embeddings...")
    model = AttentiveFP(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        return_vertex_embeddings=True,
    )
    
    x, edge_index, edge_attr, batch = create_dummy_batch()
    mol_emb, atom_emb = model(x, edge_index, edge_attr, batch)
    
    assert mol_emb.shape == (2, 128), f"Expected mol shape (2, 128), got {mol_emb.shape}"
    assert atom_emb.shape[0] == x.shape[0], f"Expected atom dim {x.shape[0]}, got {atom_emb.shape[0]}"
    assert atom_emb.shape[1] == 256, f"Expected atom hidden 256, got {atom_emb.shape[1]}"
    print("  ✓ Return vertex embeddings OK")


def test_use_gru_flag():
    """Test use_gru flag for lighter model."""
    print("Testing use_gru flag...")
    
    # With GRU (default, more expressive)
    model_gru = AttentiveFP(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        use_gru=True,
    )
    
    # With residual (lighter)
    model_residual = AttentiveFP(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        use_gru=False,
    )
    
    x, edge_index, edge_attr, batch = create_dummy_batch()
    
    out_gru = model_gru(x, edge_index, edge_attr, batch)
    out_residual = model_residual(x, edge_index, edge_attr, batch)
    
    assert out_gru.shape == out_residual.shape == (2, 128)
    print("  ✓ use_gru flag OK")


def test_gradients():
    """Test that gradients flow correctly."""
    print("Testing gradient flow...")
    
    model = AttentiveFP(
        in_channels=133,
        hidden_channels=128,
        out_channels=64,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.0,  # No dropout for deterministic gradients
    )
    
    x, edge_index, edge_attr, batch = create_dummy_batch()
    x.requires_grad = True
    
    out = model(x, edge_index, edge_attr, batch)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients not flowing to input"
    assert model.lin1.weight.grad is not None, "Gradients not flowing to lin1"
    print("  ✓ Gradient flow OK")


def test_vae_variant():
    """Test AttentiveFP2 VAE variant."""
    print("Testing AttentiveFP2 VAE variant...")
    
    model = AttentiveFP2(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
    )
    
    x, edge_index, edge_attr, batch = create_dummy_batch()
    mu, logvar = model(x, edge_index, edge_attr, batch)
    
    assert mu.shape == logvar.shape == (2, 128)
    print("  ✓ AttentiveFP2 forward OK")
    
    # Test with return_vertex_embeddings
    model_return = AttentiveFP2(
        in_channels=133,
        hidden_channels=256,
        out_channels=128,
        edge_dim=14,
        num_layers=2,
        num_timesteps=2,
        dropout=0.1,
        return_vertex_embeddings=True,
    )
    
    mu, logvar, atom_emb = model_return(x, edge_index, edge_attr, batch)
    assert atom_emb.shape[0] == x.shape[0]
    print("  ✓ AttentiveFP2 with vertex embeddings OK")


def test_update_modules():
    """Test GRUUpdate and ResidualUpdate modules."""
    print("Testing update modules...")
    
    hidden_channels = 128
    batch_size = 10
    
    gru_update = GRUUpdate(hidden_channels)
    residual_update = ResidualUpdate(hidden_channels)
    
    message = torch.randn(batch_size, hidden_channels)
    hidden = torch.randn(batch_size, hidden_channels)
    
    out_gru = gru_update(message, hidden)
    out_residual = residual_update(message, hidden)
    
    assert out_gru.shape == out_residual.shape == (batch_size, hidden_channels)
    print("  ✓ Update modules OK")


def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("Running AttentiveFP encoder tests")
    print("="*50 + "\n")
    
    test_basic_forward()
    test_undirected()
    test_aggregators()
    test_return_vertex_embeddings()
    test_use_gru_flag()
    test_gradients()
    test_vae_variant()
    test_update_modules()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()