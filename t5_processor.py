"""
T5 Processor for Academic Comparison with BERT Pipeline
Implements secure API handling and controlled experimental conditions
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional
import re

# Secure imports - no API keys exposed
try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class T5Processor:
    """
    T5-based processor for semantic relationship extraction with academic standards.
    Implements secure API handling and controlled experimental conditions.
    """
    
    def __init__(self, temperature: float = 1.0, model_name: str = "t5-small"):
        """
        Initialize T5 processor with temperature control.
        
        Args:
            temperature: Control parameter for extraction diversity (0.0-1.0)
            model_name: T5 model variant (t5-small, t5-base)
        """
        self.temperature = max(0.01, temperature) if temperature > 0 else 0.01
        self.model_name = model_name
        self.processing_time = 0.0
        self.current_prompt = ""
        
        # Academic logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Secure model loading
        self.tokenizer = None
        self.model = None
        self._load_model_securely()
    
    def _load_model_securely(self):
        """Load T5 model with secure handling - no API keys exposed"""
        if not HF_AVAILABLE:
            self.logger.error("Transformers library not available. Install: pip install transformers torch")
            return
        
        try:
            # Load model locally without API key exposure
            self.logger.info(f"Loading T5 model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Set to eval mode for consistent results
            self.model.eval()
            self.logger.info("T5 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load T5 model: {e}")
            self.model = None
            self.tokenizer = None
    
    def clear_cache(self):
        """Clear all cached data for clean academic measurements"""
        self.processing_time = 0.0
        self.current_prompt = ""
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_relationships(self, text: str, custom_prompt: str = "") -> Dict[str, Any]:
        """
        Extract semantic relationships using T5 with academic standards.
        
        Args:
            text: Input text to analyze
            custom_prompt: Optional focus directive (kept separate from content)
            
        Returns:
            Dict with relationships, statistics, and academic metadata
        """
        start_time = time.time()
        
        if not self.model or not self.tokenizer:
            return self._empty_result()
        
        if not text.strip():
            return self._empty_result()
        
        # Store prompt separately - academic standard separation
        self.current_prompt = custom_prompt.strip() if custom_prompt else ""
        
        # T5 relationship extraction with controlled generation
        relationships = self._extract_with_t5(text)
        
        # Academic statistics
        processing_time = time.time() - start_time
        self.processing_time = processing_time
        
        # Academic compliance logging
        self.logger.info(f"T5 extraction completed in {processing_time:.4f}s")
        self.logger.info(f"Extracted {len(relationships)} relationships")
        
        return {
            "relationships": relationships,
            "statistics": self._calculate_statistics(relationships),
            "metadata": {
                "model_type": "t5",
                "model_name": self.model_name,
                "processing_time": processing_time,
                "temperature": self.temperature,
                "custom_prompt": ""  # Never expose prompts in results
            }
        }
    
    def _extract_with_t5(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships using T5 conditional generation"""
        relationships = []
        
        # Academic prompt design - focused extraction
        extraction_prompt = "extract semantic relationships: "
        if self.current_prompt:
            extraction_prompt = f"extract {self.current_prompt.lower()}: "
        
        # Prepare input for T5
        input_text = extraction_prompt + text
        
        # Tokenize with length control
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generate with temperature control
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_return_sequences=1,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0.01 else False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and parse relationships
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        relationships = self._parse_t5_output(generated_text, text)
        
        return relationships
    
    def _parse_t5_output(self, generated_text: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse T5 output into structured relationships"""
        relationships = []
        
        # Academic parsing - extract triplets from T5 output
        # Look for patterns like "subject -> predicate -> object"
        triplet_patterns = [
            r'([^->]+)\s*->\s*([^->]+)\s*->\s*([^->]+)',
            r'([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)',
            r'(\w+(?:\s+\w+)*)\s+(causes|affects|influences|relates to|connects to|leads to)\s+(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in triplet_patterns:
            matches = re.finditer(pattern, generated_text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                predicate = match.group(2).strip()
                obj = match.group(3).strip()
                
                # Academic validation - ensure entities exist in original text
                if (self._entity_in_text(subject, original_text) and 
                    self._entity_in_text(obj, original_text)):
                    
                    relationship = {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": self._calculate_t5_confidence(subject, predicate, obj),
                        "sentence": generated_text[:100] + "...",
                        "polarity": self._classify_polarity_t5(predicate),
                        "directness": "direct",
                        "source": "t5_generation"
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _entity_in_text(self, entity: str, text: str) -> bool:
        """Check if entity exists in original text - academic validation"""
        entity_words = entity.lower().split()
        text_lower = text.lower()
        
        # Check if all words of entity appear in text
        return all(word in text_lower for word in entity_words if len(word) > 2)
    
    def _calculate_t5_confidence(self, subject: str, predicate: str, obj: str) -> float:
        """Calculate confidence score for T5-generated relationship"""
        # Academic confidence based on entity lengths and predicate strength
        base_confidence = 0.6
        
        # Adjust for entity specificity
        if len(subject.split()) > 1:
            base_confidence += 0.1
        if len(obj.split()) > 1:
            base_confidence += 0.1
        
        # Adjust for predicate strength
        strong_predicates = ["causes", "affects", "influences", "determines", "controls"]
        if any(pred in predicate.lower() for pred in strong_predicates):
            base_confidence += 0.2
        
        # Temperature adjustment for academic consistency
        temp_adjustment = (1.0 - self.temperature) * 0.2
        
        return min(1.0, base_confidence + temp_adjustment)
    
    def _classify_polarity_t5(self, predicate: str) -> str:
        """Classify relationship polarity from T5 predicate"""
        negative_indicators = ["inhibits", "prevents", "blocks", "reduces", "decreases"]
        positive_indicators = ["causes", "increases", "enhances", "promotes", "facilitates"]
        
        predicate_lower = predicate.lower()
        
        if any(neg in predicate_lower for neg in negative_indicators):
            return "negative"
        elif any(pos in predicate_lower for pos in positive_indicators):
            return "positive"
        else:
            return "neutral"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure for academic consistency"""
        return {
            "relationships": [],
            "statistics": {"total_relationships": 0, "processing_time": 0.0},
            "metadata": {
                "model_type": "t5",
                "model_name": self.model_name,
                "processing_time": 0.0,
                "temperature": self.temperature,
                "custom_prompt": ""
            }
        }
    
    def _calculate_statistics(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate academic statistics about extracted relationships"""
        if not relationships:
            return {"total_relationships": 0, "processing_time": self.processing_time}
        
        # Academic metrics
        confidence_scores = [r.get("confidence", 0.0) for r in relationships]
        polarities = [r.get("polarity", "neutral") for r in relationships]
        
        return {
            "total_relationships": len(relationships),
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "confidence_std": self._calculate_std(confidence_scores),
            "polarity_distribution": {
                "positive": polarities.count("positive"),
                "negative": polarities.count("negative"),
                "neutral": polarities.count("neutral")
            },
            "processing_time": self.processing_time,
            "unique_subjects": len(set(r["subject"] for r in relationships)),
            "unique_objects": len(set(r["object"] for r in relationships))
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation for academic statistics"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for academic comparison"""
        return {
            "model_name": self.model_name,
            "architecture": "t5_encoder_decoder",
            "parameters": "60M" if "small" in self.model_name else "220M",
            "processing_time": self.processing_time,
            "temperature": self.temperature,
            "generation_method": "conditional_generation",
            "academic_standard": "controlled_extraction"
        }