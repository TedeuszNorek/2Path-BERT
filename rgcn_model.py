import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np

class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Network layer.
    This layer implements the RGCN propagation rule as described in the paper
    "Modeling Relational Data with Graph Convolutional Networks"
    """
    
    def __init__(self, in_features: int, out_features: int, num_relations: int, 
                 num_bases: int = None, bias: bool = True, activation: nn.Module = None):
        """
        Initialize the RGCN layer.
        
        Args:
            in_features: Size of input node features
            out_features: Size of output node features
            num_relations: Number of relation types
            num_bases: Number of bases to use for weight matrices decomposition
            bias: Whether to include bias
            activation: Activation function to use
        """
        super(RGCNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        
        # Define weights
        if num_bases is None or num_bases > num_relations or num_bases <= 0:
            self.num_bases = num_relations
            
        # Weight bases and coefficients
        if self.num_bases < self.num_relations:
            # Weight decomposition using bases
            self.weight_bases = nn.Parameter(torch.Tensor(self.num_bases, self.in_features, self.out_features))
            self.weight_coefficients = nn.Parameter(torch.Tensor(self.num_relations, self.num_bases))
        else:
            # No decomposition
            self.weight_bases = nn.Parameter(torch.Tensor(self.num_relations, self.in_features, self.out_features))
            
        # Bias
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and biases."""
        nn.init.xavier_uniform_(self.weight_bases)
        if self.num_bases < self.num_relations:
            nn.init.xavier_uniform_(self.weight_coefficients)
        if self.bias:
            nn.init.zeros_(self.bias_param)
            
    def forward(self, features: torch.Tensor, adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the RGCN layer.
        
        Args:
            features: Node features tensor of shape (num_nodes, in_features)
            adjacency_matrices: List of adjacency matrices for each relation type
                               Each matrix has shape (num_nodes, num_nodes)
        
        Returns:
            Updated node features tensor of shape (num_nodes, out_features)
        """
        if self.num_bases < self.num_relations:
            # Generate relation-specific weights from bases and coefficients
            weights = torch.einsum('rb,bio->rio', self.weight_coefficients, self.weight_bases)
        else:
            weights = self.weight_bases
            
        # Initialize output tensor
        output = torch.zeros(features.size(0), self.out_features, device=features.device)
        
        # Aggregate messages for each relation type
        for r in range(self.num_relations):
            # Get relation-specific weight
            weight = weights[r]
            
            # Message passing: transform features using relation-specific weight
            transformed_features = torch.matmul(features, weight)
            
            # Propagate: multiply with adjacency matrix (this gives neighbor aggregation)
            propagated = torch.matmul(adjacency_matrices[r], transformed_features)
            
            # Sum for all relations
            output += propagated
            
        # Add bias
        if self.bias:
            output += self.bias_param
            
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network model for semantic relationship modeling.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_relations: int, num_layers: int = 2, num_bases: int = None,
                 dropout: float = 0.0):
        """
        Initialize the RGCN model.
        
        Args:
            input_dim: Size of input node features
            hidden_dim: Size of hidden layers
            output_dim: Size of output node features 
            num_relations: Number of relation types
            num_layers: Number of RGCN layers
            num_bases: Number of bases for weight decomposition
            dropout: Dropout probability
        """
        super(RGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.dropout = dropout
        
        # Build RGCN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            RGCNLayer(input_dim, hidden_dim, num_relations, num_bases, 
                     activation=F.relu)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases,
                         activation=F.relu)
            )
            
        # Output layer
        self.layers.append(
            RGCNLayer(hidden_dim, output_dim, num_relations, num_bases,
                     activation=None)
        )
            
    def forward(self, features: torch.Tensor, 
                adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the RGCN model.
        
        Args:
            features: Node features tensor of shape (num_nodes, input_dim)
            adjacency_matrices: List of adjacency matrices for each relation type
        
        Returns:
            Updated node embeddings of shape (num_nodes, output_dim)
        """
        x = features
        
        # Pass through RGCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, adjacency_matrices)
            
            # Apply dropout except for the last layer
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x

class RGCNProcessor:
    """
    Processor class for using RGCN to model and analyze semantic relationships.
    """
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64, 
                 output_dim: int = 32, num_layers: int = 2):
        """
        Initialize the RGCN processor.
        
        Args:
            embedding_dim: Size of initial node embeddings
            hidden_dim: Size of hidden layers
            output_dim: Size of output embeddings
            num_layers: Number of RGCN layers
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Dictionary to map entity and relation strings to indices
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self.idx_to_entity = {}
        self.idx_to_relation = {}
        
        # Entity and relation counts
        self.num_entities = 0
        self.num_relations = 0
        
        # Model is initialized later when we know num_relations
        self.model = None
        self.entity_embeddings = None
        
    def process_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a list of relationships to build and run the RGCN model.
        
        Args:
            relationships: List of relationship dictionaries containing
                          subject, predicate, object
                        
        Returns:
            Dictionary with entities, relations, and embeddings
        """
        # Extract entities and relations
        entities = set()
        relations = set()
        triples = []
        
        for rel in relationships:
            subject = rel.get("subject", "").strip()
            obj = rel.get("object", "").strip() 
            predicate = rel.get("predicate", "").strip()
            
            # Skip invalid relationships
            if not subject or not obj or not predicate:
                continue
                
            entities.add(subject)
            entities.add(obj)
            relations.add(predicate)
            
            triples.append((subject, predicate, obj))
            
        # Create mapping from entity/relation strings to indices
        self._build_mappings(entities, relations)
        
        # Create adjacency matrices for each relation
        adjacency_matrices = self._create_adjacency_matrices(triples)
        
        # Initialize or update model if needed
        if self.model is None or self.num_relations != len(relations):
            self._initialize_model()
            
        # Initialize entity embeddings
        if self.entity_embeddings is None or self.entity_embeddings.size(0) != self.num_entities:
            self.entity_embeddings = torch.randn(self.num_entities, self.embedding_dim)
            
        # Run RGCN model
        with torch.no_grad():
            output_embeddings = self.model(self.entity_embeddings, adjacency_matrices)
            
        # Format results
        return {
            "entities": {entity: self.entity_to_idx[entity] for entity in entities},
            "relations": {relation: self.relation_to_idx[relation] for relation in relations},
            "embeddings": output_embeddings.numpy(),
            "triples": triples,
            "entity_to_idx": self.entity_to_idx,
            "relation_to_idx": self.relation_to_idx,
            "idx_to_entity": self.idx_to_entity,
            "idx_to_relation": self.idx_to_relation
        }
        
    def _build_mappings(self, entities: Set[str], relations: Set[str]) -> None:
        """
        Build mappings between entity/relation strings and indices.
        
        Args:
            entities: Set of entity strings
            relations: Set of relation strings
        """
        # Add new entities
        for entity in entities:
            if entity not in self.entity_to_idx:
                self.entity_to_idx[entity] = self.num_entities
                self.idx_to_entity[self.num_entities] = entity
                self.num_entities += 1
                
        # Add new relations
        for relation in relations:
            if relation not in self.relation_to_idx:
                self.relation_to_idx[relation] = self.num_relations
                self.idx_to_relation[self.num_relations] = relation
                self.num_relations += 1
                
    def _initialize_model(self) -> None:
        """Initialize the RGCN model with current parameters."""
        self.model = RGCN(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_relations=self.num_relations,
            num_layers=self.num_layers,
            num_bases=min(self.num_relations, 5),  # Use 5 bases or num_relations, whichever is smaller
            dropout=0.1
        )
        
    def _create_adjacency_matrices(self, triples: List[Tuple[str, str, str]]) -> List[torch.Tensor]:
        """
        Create adjacency matrices for each relation type.
        
        Args:
            triples: List of (subject, predicate, object) triples
            
        Returns:
            List of adjacency matrices, one per relation type
        """
        # Initialize adjacency matrices
        adjacency_matrices = [torch.zeros(self.num_entities, self.num_entities) 
                             for _ in range(self.num_relations)]
        
        # Fill adjacency matrices
        for subject, predicate, obj in triples:
            if subject in self.entity_to_idx and predicate in self.relation_to_idx and obj in self.entity_to_idx:
                subj_idx = self.entity_to_idx[subject]
                obj_idx = self.entity_to_idx[obj]
                rel_idx = self.relation_to_idx[predicate]
                
                # Set entry to 1
                adjacency_matrices[rel_idx][subj_idx, obj_idx] = 1.0
                
        # Normalize adjacency matrices
        for i, adj in enumerate(adjacency_matrices):
            # Add self-loops
            adj = adj + torch.eye(self.num_entities)
            
            # Normalize using degree matrix
            rowsum = adj.sum(dim=1)
            rowsum[rowsum == 0] = 1  # Avoid division by zero
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt = torch.diag(d_inv_sqrt)
            adjacency_matrices[i] = torch.mm(torch.mm(d_inv_sqrt, adj), d_inv_sqrt)
            
        return adjacency_matrices
