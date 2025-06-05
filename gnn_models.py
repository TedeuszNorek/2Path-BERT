import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import time
import logging

class RGCNLayer(nn.Module):
    """Relational Graph Convolutional Network layer"""
    
    def __init__(self, in_features: int, out_features: int, num_relations: int, 
                 num_bases: int = None, bias: bool = True, activation: nn.Module = None):
        super(RGCNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases is not None else num_relations
        
        # Weight matrices for each relation using basis decomposition
        self.basis_weights = nn.Parameter(torch.FloatTensor(self.num_bases, in_features, out_features))
        self.combination_weights = nn.Parameter(torch.FloatTensor(num_relations, self.num_bases))
        
        # Self-loop weight
        self.self_weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
            
        self.activation = activation
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis_weights)
        nn.init.xavier_uniform_(self.combination_weights)
        nn.init.xavier_uniform_(self.self_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, features: torch.Tensor, adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        output = torch.mm(features, self.self_weight)
        
        for relation_idx, adj_matrix in enumerate(adjacency_matrices):
            if relation_idx < self.num_relations:
                # Compute relation-specific weight matrix
                relation_weight = torch.sum(
                    self.combination_weights[relation_idx].unsqueeze(-1).unsqueeze(-1) * self.basis_weights,
                    dim=0
                )
                
                # Apply message passing
                messages = torch.mm(adj_matrix, torch.mm(features, relation_weight))
                output = output + messages
        
        if self.bias is not None:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class CompGCNLayer(nn.Module):
    """Composition-based Graph Convolutional Network layer"""
    
    def __init__(self, in_features: int, out_features: int, num_relations: int, 
                 composition_func: str = 'mult', bias: bool = True, activation: nn.Module = None):
        super(CompGCNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.composition_func = composition_func
        
        # Entity transformation
        self.entity_transform = nn.Linear(in_features, out_features, bias=False)
        
        # Relation embeddings
        self.relation_embeddings = nn.Parameter(torch.FloatTensor(num_relations, in_features))
        self.relation_transform = nn.Linear(in_features, out_features, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
            
        self.activation = activation
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_embeddings)
        nn.init.xavier_uniform_(self.entity_transform.weight)
        nn.init.xavier_uniform_(self.relation_transform.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def compose(self, entity_emb: torch.Tensor, relation_emb: torch.Tensor) -> torch.Tensor:
        """Composition function for entity and relation embeddings"""
        if self.composition_func == 'mult':
            return entity_emb * relation_emb
        elif self.composition_func == 'add':
            return entity_emb + relation_emb
        elif self.composition_func == 'concat':
            return torch.cat([entity_emb, relation_emb], dim=-1)
        else:
            return entity_emb * relation_emb  # Default to multiplication
    
    def forward(self, features: torch.Tensor, adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        num_nodes = features.size(0)
        output = torch.zeros(num_nodes, self.out_features)
        
        for relation_idx, adj_matrix in enumerate(adjacency_matrices):
            if relation_idx < self.num_relations:
                relation_emb = self.relation_embeddings[relation_idx]
                
                # Compose entity and relation embeddings
                composed = self.compose(features, relation_emb.unsqueeze(0).expand_as(features))
                
                # Apply adjacency matrix
                messages = torch.mm(adj_matrix, composed)
                output = output + self.entity_transform(messages)
        
        if self.bias is not None:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class RGATLayer(nn.Module):
    """Relational Graph Attention Network layer"""
    
    def __init__(self, in_features: int, out_features: int, num_relations: int, 
                 num_heads: int = 1, dropout: float = 0.0, bias: bool = True, activation: nn.Module = None):
        super(RGATLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert out_features % num_heads == 0
        self.head_dim = out_features // num_heads
        
        # Linear transformations for each relation
        self.relation_transforms = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) 
            for _ in range(num_relations)
        ])
        
        # Attention mechanisms for each relation
        self.attention_weights = nn.ModuleList([
            nn.Linear(2 * self.head_dim, 1, bias=False)
            for _ in range(num_relations)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
            
        self.activation = activation
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        for transform in self.relation_transforms:
            nn.init.xavier_uniform_(transform.weight)
        for attention in self.attention_weights:
            nn.init.xavier_uniform_(attention.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, features: torch.Tensor, adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        num_nodes = features.size(0)
        outputs = []
        
        for relation_idx, adj_matrix in enumerate(adjacency_matrices):
            if relation_idx < self.num_relations:
                # Transform features
                transformed = self.relation_transforms[relation_idx](features)
                transformed = transformed.view(num_nodes, self.num_heads, self.head_dim)
                
                # Compute attention scores
                attention_scores = torch.zeros(num_nodes, num_nodes, self.num_heads)
                
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if adj_matrix[i, j] != 0:
                            for h in range(self.num_heads):
                                concat_features = torch.cat([transformed[i, h], transformed[j, h]], dim=0)
                                attention_scores[i, j, h] = self.attention_weights[relation_idx](concat_features)
                
                # Apply softmax to attention scores
                attention_weights = F.softmax(attention_scores, dim=1)
                attention_weights = self.dropout_layer(attention_weights)
                
                # Apply attention
                output = torch.zeros(num_nodes, self.num_heads, self.head_dim)
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if adj_matrix[i, j] != 0:
                            for h in range(self.num_heads):
                                output[i, h] += attention_weights[i, j, h] * transformed[j, h]
                
                output = output.view(num_nodes, self.out_features)
                outputs.append(output)
        
        # Combine outputs from all relations
        if outputs:
            final_output = torch.stack(outputs).mean(dim=0)
        else:
            final_output = torch.zeros(num_nodes, self.out_features)
        
        if self.bias is not None:
            final_output = final_output + self.bias
            
        if self.activation is not None:
            final_output = self.activation(final_output)
            
        return final_output

class GNNProcessor:
    """Unified processor for different GNN architectures"""
    
    def __init__(self, model_type: str = 'rgcn', embedding_dim: int = 128, 
                 hidden_dim: int = 64, output_dim: int = 32, num_layers: int = 2,
                 temperature: float = 1.0):
        self.model_type = model_type.lower()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.temperature = temperature
        
        self.entity_to_idx = {}
        self.idx_to_entity = {}
        self.relation_to_idx = {}
        self.idx_to_relation = {}
        self.model = None
        
        # Performance metrics
        self.processing_time = 0.0
        self.memory_usage = 0.0
        
        # Clean logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def clear_cache(self):
        """Clear all cached data for clean measurements"""
        self.entity_to_idx.clear()
        self.idx_to_entity.clear()
        self.relation_to_idx.clear()
        self.idx_to_relation.clear()
        self.model = None
        self.processing_time = 0.0
        self.memory_usage = 0.0
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process relationships using the selected GNN architecture"""
        start_time = time.time()
        
        # Clear previous state
        self.clear_cache()
        
        if not relationships:
            return {
                "embeddings": np.array([]),
                "entity_to_idx": {},
                "relation_to_idx": {},
                "processing_time": 0.0,
                "model_type": self.model_type
            }
        
        # Extract entities and relations
        entities = set()
        relations = set()
        
        for rel in relationships:
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            
            if subject:
                entities.add(subject)
            if obj:
                entities.add(obj)
            if predicate:
                relations.add(predicate)
        
        entities = list(entities)
        relations = list(relations)
        
        if not entities or not relations:
            self.logger.warning("No valid entities or relations found")
            return {
                "embeddings": np.array([]),
                "entity_to_idx": {},
                "relation_to_idx": {},
                "processing_time": 0.0,
                "model_type": self.model_type
            }
        
        # Build mappings
        self._build_mappings(entities, relations)
        
        # Initialize model
        self._initialize_model()
        
        # Create adjacency matrices
        adjacency_matrices = self._create_adjacency_matrices(relationships)
        
        # Initialize node features
        num_entities = len(entities)
        node_features = torch.randn(num_entities, self.embedding_dim) / self.temperature
        
        # Forward pass through the model
        with torch.no_grad():
            if self.model is not None:
                embeddings = self._forward_pass(node_features, adjacency_matrices)
            else:
                embeddings = torch.zeros(num_entities, self.output_dim)
        
        self.processing_time = time.time() - start_time
        
        self.logger.info(f"{self.model_type.upper()} processing completed in {self.processing_time:.4f}s")
        
        return {
            "embeddings": embeddings.numpy(),
            "entity_to_idx": self.entity_to_idx.copy(),
            "relation_to_idx": self.relation_to_idx.copy(),
            "processing_time": self.processing_time,
            "model_type": self.model_type,
            "num_entities": len(entities),
            "num_relations": len(relations)
        }
    
    def _build_mappings(self, entities: List[str], relations: List[str]) -> None:
        """Build entity and relation mappings"""
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
        self.idx_to_entity = {idx: entity for entity, idx in self.entity_to_idx.items()}
        
        self.relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
        self.idx_to_relation = {idx: relation for relation, idx in self.relation_to_idx.items()}
    
    def _initialize_model(self) -> None:
        """Initialize the selected GNN model"""
        num_relations = len(self.relation_to_idx)
        
        if self.model_type == 'rgcn':
            self.model = nn.Sequential()
            
            # First layer
            self.model.add_module('rgcn1', RGCNLayer(
                self.embedding_dim, self.hidden_dim, num_relations, 
                activation=nn.ReLU()
            ))
            
            # Additional layers
            for i in range(1, self.num_layers - 1):
                self.model.add_module(f'rgcn{i+1}', RGCNLayer(
                    self.hidden_dim, self.hidden_dim, num_relations,
                    activation=nn.ReLU()
                ))
            
            # Output layer
            if self.num_layers > 1:
                self.model.add_module(f'rgcn{self.num_layers}', RGCNLayer(
                    self.hidden_dim, self.output_dim, num_relations
                ))
            else:
                self.model.add_module('rgcn_out', RGCNLayer(
                    self.embedding_dim, self.output_dim, num_relations
                ))
        
        elif self.model_type == 'compgcn':
            self.model = nn.Sequential()
            
            # First layer
            self.model.add_module('compgcn1', CompGCNLayer(
                self.embedding_dim, self.hidden_dim, num_relations,
                activation=nn.ReLU()
            ))
            
            # Additional layers
            for i in range(1, self.num_layers - 1):
                self.model.add_module(f'compgcn{i+1}', CompGCNLayer(
                    self.hidden_dim, self.hidden_dim, num_relations,
                    activation=nn.ReLU()
                ))
            
            # Output layer
            if self.num_layers > 1:
                self.model.add_module(f'compgcn{self.num_layers}', CompGCNLayer(
                    self.hidden_dim, self.output_dim, num_relations
                ))
            else:
                self.model.add_module('compgcn_out', CompGCNLayer(
                    self.embedding_dim, self.output_dim, num_relations
                ))
        
        elif self.model_type == 'rgat':
            self.model = nn.Sequential()
            
            # First layer
            self.model.add_module('rgat1', RGATLayer(
                self.embedding_dim, self.hidden_dim, num_relations,
                num_heads=4, activation=nn.ReLU()
            ))
            
            # Additional layers
            for i in range(1, self.num_layers - 1):
                self.model.add_module(f'rgat{i+1}', RGATLayer(
                    self.hidden_dim, self.hidden_dim, num_relations,
                    num_heads=4, activation=nn.ReLU()
                ))
            
            # Output layer
            if self.num_layers > 1:
                self.model.add_module(f'rgat{self.num_layers}', RGATLayer(
                    self.hidden_dim, self.output_dim, num_relations,
                    num_heads=1
                ))
            else:
                self.model.add_module('rgat_out', RGATLayer(
                    self.embedding_dim, self.output_dim, num_relations,
                    num_heads=1
                ))
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_adjacency_matrices(self, relationships: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """Create adjacency matrices for each relation type"""
        num_entities = len(self.entity_to_idx)
        num_relations = len(self.relation_to_idx)
        
        # Initialize adjacency matrices
        adjacency_matrices = [torch.zeros(num_entities, num_entities) for _ in range(num_relations)]
        
        # Fill adjacency matrices
        for rel in relationships:
            subject = rel.get("subject", "").strip()
            predicate = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            confidence = rel.get("confidence", 1.0)
            
            if subject in self.entity_to_idx and obj in self.entity_to_idx and predicate in self.relation_to_idx:
                subj_idx = self.entity_to_idx[subject]
                obj_idx = self.entity_to_idx[obj]
                rel_idx = self.relation_to_idx[predicate]
                
                # Apply temperature scaling to confidence
                scaled_confidence = confidence / self.temperature
                adjacency_matrices[rel_idx][subj_idx, obj_idx] = scaled_confidence
        
        return adjacency_matrices
    
    def _forward_pass(self, node_features: torch.Tensor, adjacency_matrices: List[torch.Tensor]) -> torch.Tensor:
        """Perform forward pass through the GNN layers"""
        x = node_features
        
        if self.model_type == 'rgcn':
            for layer in self.model:
                x = layer(x, adjacency_matrices)
        elif self.model_type == 'compgcn':
            for layer in self.model:
                x = layer(x, adjacency_matrices)
        elif self.model_type == 'rgat':
            for layer in self.model:
                x = layer(x, adjacency_matrices)
        
        return x
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for comparison"""
        return {
            "model_type": self.model_type,
            "processing_time": self.processing_time,
            "memory_usage": self.memory_usage,
            "temperature": self.temperature,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers
        }