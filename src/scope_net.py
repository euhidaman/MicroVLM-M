"""
ScopeNet Module
Lightweight classifier to determine whether memory injection should be applied
"""

import torch
import torch.nn as nn
import json
import os

class ScopeNet(nn.Module):
    """
    Lightweight MLP classifier for memory scope detection
    
    Takes fused context embedding and outputs binary decision:
        - 1: Apply memory injection
        - 0: Skip memory injection
    
    Architecture:
        - Multi-layer MLP with ReLU activations
        - Dropout for regularization
        - Sigmoid output for binary classification
    """
    
    def __init__(self, config_path=None, config_dict=None):
        super().__init__()
        
        # Load configuration
        if config_dict is not None:
            config = config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                master_config = json.load(f)
            config = master_config['scope']
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'relu')
        
        # Build MLP layers
        layers = []
        current_dim = self.input_dim
        
        for i in range(self.num_layers):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            
            # Activation
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            # Dropout
            layers.append(nn.Dropout(self.dropout))
            current_dim = self.hidden_dim
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, context_embedding, threshold=0.5):
        """
        Forward pass
        
        Args:
            context_embedding: (batch, input_dim) fused context representation
            threshold: decision threshold for binary classification
        
        Returns:
            scope_decision: (batch,) binary decision (0 or 1)
            scope_prob: (batch,) probability of applying memory
        """
        # Get probability
        scope_prob = self.mlp(context_embedding).squeeze(-1)  # (batch,)
        
        # Binary decision
        scope_decision = (scope_prob > threshold).float()
        
        return scope_decision, scope_prob
    
    def get_loss(self, context_embedding, targets):
        """
        Compute binary cross-entropy loss for training
        
        Args:
            context_embedding: (batch, input_dim)
            targets: (batch,) binary ground truth
        
        Returns:
            loss: scalar BCE loss
        """
        _, scope_prob = self.forward(context_embedding, threshold=0.5)
        loss = F.binary_cross_entropy(scope_prob, targets)
        return loss


if __name__ == "__main__":
    # Test ScopeNet
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.json")
    
    scope_net = ScopeNet(config_path=config_path)
    
    # Test forward pass
    batch_size = 4
    input_dim = 2560  # BitNet hidden size
    
    dummy_context = torch.randn(batch_size, input_dim)
    decisions, probs = scope_net(dummy_context)
    
    print(f"Input shape: {dummy_context.shape}")
    print(f"Decisions shape: {decisions.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Decisions: {decisions}")
    print(f"Probabilities: {probs}")
    
    # Test loss computation
    targets = torch.randint(0, 2, (batch_size,)).float()
    loss = scope_net.get_loss(dummy_context, targets)
    print(f"Loss: {loss.item():.4f}")
    
    print("ScopeNet test passed!")
