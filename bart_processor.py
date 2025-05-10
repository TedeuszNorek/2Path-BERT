import re
import spacy
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

class BARTProcessor:
    """
    BART-inspired processor for extracting semantic relationships from text.
    This simulates how BART would process text for extraction tasks.
    """
    
    def __init__(self):
        """Initialize the BART-inspired processor."""
        # Load spaCy model for basic NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model is not installed, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Log initialization
        self._log_operation("initialize", {"model": "BART-inspired", "timestamp": datetime.now().isoformat()})
    
    def extract_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic relationships from text including polarity and directness.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with relationships, statistics, and metadata
        """
        start_time = time.time()
        
        # Preprocess and split into sentences
        sentences = self._preprocess_text(text)
        
        # Extract relationships from each sentence
        all_relationships = []
        for sentence in sentences:
            relationships = self._extract_from_sentence(sentence)
            all_relationships.extend(relationships)
        
        # Classify polarity and directness
        for relationship in all_relationships:
            relationship["polarity"] = self._classify_polarity(relationship["sentence"], 
                                                           relationship["subject"], 
                                                           relationship["predicate"], 
                                                           relationship["object"])
            relationship["directness"] = self._classify_directness(relationship["sentence"], 
                                                               relationship["subject"], 
                                                               relationship["predicate"], 
                                                               relationship["object"])
            relationship["confidence"] = self._calculate_confidence(relationship)
        
        # Calculate statistics
        statistics = self._calculate_statistics(all_relationships)
        
        # Prepare RDF triples
        rdf_triples = self.convert_to_rdf(all_relationships)
        
        # Log extraction
        self._log_operation("extract_relationships", {
            "input_length": len(text),
            "num_sentences": len(sentences),
            "num_relationships": len(all_relationships),
            "processing_time": time.time() - start_time
        })
        
        return {
            "relationships": all_relationships,
            "statistics": statistics,
            "rdf_triples": rdf_triples,
            "processing_time": time.time() - start_time
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Split text into sentences and clean."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences
    
    def _extract_from_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract relationships from a single sentence."""
        doc = self.nlp(sentence)
        relationships = []
        
        # Find main verb and its arguments
        for token in doc:
            # Look for verbs as the predicate
            if token.pos_ == "VERB":
                # Find subject
                subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                
                # Find objects
                objects = [child for child in token.children 
                          if child.dep_ in ["dobj", "pobj", "attr", "iobj"]]
                
                # If we have both subject and object, create a relationship
                for subj in subjects:
                    for obj in objects:
                        # Expand subject and object to noun phrases
                        subject_span = self._expand_span(doc, subj)
                        object_span = self._expand_span(doc, obj)
                        
                        # Skip if subject and object are the same
                        if subject_span.text.lower() == object_span.text.lower():
                            continue
                        
                        # Create relationship
                        relationship = {
                            "subject": subject_span.text,
                            "predicate": token.lemma_,
                            "object": object_span.text,
                            "sentence": sentence
                        }
                        relationships.append(relationship)
        
        # If no relationships found, try a rule-based approach
        if not relationships:
            relationships = self._rule_based_extraction(sentence)
            
        return relationships
    
    def _expand_span(self, doc, token):
        """Expand token to a noun phrase if possible."""
        # Start with the token itself
        start, end = token.i, token.i + 1
        
        # If token is part of a compound, expand to include the full compound
        if token.dep_ == "compound":
            head = token.head
            while head.dep_ == "compound":
                head = head.head
            if head.pos_ in ["NOUN", "PROPN"]:
                start = min(token.i, head.i)
                end = max(token.i + 1, head.i + 1)
        
        # Include modifiers
        for child in token.children:
            if child.dep_ in ["amod", "compound", "det", "nummod", "poss"]:
                start = min(start, child.i)
                end = max(end, child.i + 1)
        
        return doc[start:end]
    
    def _rule_based_extraction(self, sentence: str) -> List[Dict[str, Any]]:
        """Apply rule-based extraction for sentences without clear SVO structure."""
        relationships = []
        doc = self.nlp(sentence)
        
        # Check for possession patterns
        has_pattern = re.search(r'(\w+)\s+has\s+(\w+)', sentence, re.IGNORECASE)
        if has_pattern:
            relationship = {
                "subject": has_pattern.group(1),
                "predicate": "has",
                "object": has_pattern.group(2),
                "sentence": sentence
            }
            relationships.append(relationship)
        
        # Check for negation patterns
        not_pattern = re.search(r'(\w+)\s+(?:isn\'t|is not|aren\'t|are not)\s+(\w+)', sentence, re.IGNORECASE)
        if not_pattern:
            relationship = {
                "subject": not_pattern.group(1),
                "predicate": "is not",
                "object": not_pattern.group(2),
                "sentence": sentence
            }
            relationships.append(relationship)
        
        # Check for action patterns
        action_pattern = re.search(r'(\w+)\s+(\w+)\s+(\w+)', sentence, re.IGNORECASE)
        if action_pattern and not (has_pattern or not_pattern):
            subject = action_pattern.group(1)
            verb = action_pattern.group(2)
            obj = action_pattern.group(3)
            
            # Check if the middle word is a verb
            if self.nlp(verb)[0].pos_ == "VERB":
                relationship = {
                    "subject": subject,
                    "predicate": verb,
                    "object": obj,
                    "sentence": sentence
                }
                relationships.append(relationship)
        
        return relationships
    
    def _classify_polarity(self, sentence: str, subject: str, predicate: str, obj: str) -> str:
        """Classify the polarity of a relationship."""
        # Check for negation words
        negation_words = ["not", "n't", "no", "never", "neither", "nor", "none"]
        
        # Default polarity is neutral
        polarity = "neutral"
        
        # Check for explicit negation
        for word in negation_words:
            if word in sentence.lower():
                polarity = "negative"
                break
        
        # Check for positive sentiment words
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "positive", 
                         "beneficial", "helpful", "improve", "benefit", "advantage"]
        
        for word in positive_words:
            if word in sentence.lower():
                polarity = "positive"
                break
        
        # Check for negative sentiment words
        negative_words = ["bad", "terrible", "horrible", "awful", "negative", "harmful", 
                         "detrimental", "worsen", "damage", "disadvantage"]
        
        for word in negative_words:
            if word in sentence.lower():
                polarity = "negative"
                break
        
        # Special cases for common predicates
        if predicate.lower() in ["cause", "lead to", "result in"]:
            # Check if the object has negative connotations
            if any(word in obj.lower() for word in negative_words):
                polarity = "negative"
        
        return polarity
    
    def _classify_directness(self, sentence: str, subject: str, predicate: str, obj: str) -> str:
        """Classify the directness of a relationship."""
        # Default is direct
        directness = "direct"
        
        # Check for modal verbs indicating indirectness
        modal_verbs = ["may", "might", "could", "can", "would", "should", "possibly", 
                      "perhaps", "potentially", "likely", "unlikely"]
        
        if any(modal in sentence.lower() for modal in modal_verbs):
            directness = "indirect"
        
        # Check for uncertainty indicators
        uncertainty_phrases = ["not sure", "uncertain", "unclear", "may be", "might be", 
                              "is possible", "possibility", "hypothesis", "theory", 
                              "correlation", "association", "potential"]
        
        if any(phrase in sentence.lower() for phrase in uncertainty_phrases):
            directness = "indirect"
        
        return directness
    
    def _calculate_confidence(self, relationship: Dict[str, Any]) -> float:
        """Calculate confidence score for the relationship."""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on presence of subject, predicate, and object
        if all(relationship.get(k) for k in ["subject", "predicate", "object"]):
            confidence += 0.2
        
        # Penalize very short components
        for component in ["subject", "predicate", "object"]:
            if component in relationship and len(relationship[component]) < 2:
                confidence -= 0.1
        
        # Adjust based on directness
        if relationship.get("directness") == "indirect":
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def convert_to_rdf(self, relationships: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """Convert relationships to RDF-like triples."""
        rdf_triples = []
        
        for rel in relationships:
            # Basic triple
            triple = (rel["subject"], rel["predicate"], rel["object"])
            rdf_triples.append(triple)
            
            # Add polarity as a triple if not neutral
            if rel.get("polarity", "neutral") != "neutral":
                polarity_triple = (
                    f"{rel['subject']}_{rel['predicate']}_{rel['object']}", 
                    "has_polarity", 
                    rel["polarity"]
                )
                rdf_triples.append(polarity_triple)
            
            # Add directness as a triple if indirect
            if rel.get("directness", "direct") != "direct":
                directness_triple = (
                    f"{rel['subject']}_{rel['predicate']}_{rel['object']}", 
                    "has_directness", 
                    rel["directness"]
                )
                rdf_triples.append(directness_triple)
            
            # Add confidence as a triple
            if "confidence" in rel:
                confidence_triple = (
                    f"{rel['subject']}_{rel['predicate']}_{rel['object']}", 
                    "has_confidence", 
                    str(rel["confidence"])
                )
                rdf_triples.append(confidence_triple)
        
        return rdf_triples
    
    def _calculate_statistics(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the extracted relationships."""
        if not relationships:
            return {
                "num_relationships": 0,
                "avg_confidence": 0,
                "polarity_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "directness_distribution": {"direct": 0, "indirect": 0}
            }
            
        # Count polarities
        polarity_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for rel in relationships:
            polarity = rel.get("polarity", "neutral")
            polarity_counts[polarity] += 1
            
        # Count directness
        directness_counts = {"direct": 0, "indirect": 0}
        for rel in relationships:
            directness = rel.get("directness", "direct")
            directness_counts[directness] += 1
            
        # Calculate average confidence
        total_confidence = sum(rel.get("confidence", 0) for rel in relationships)
        avg_confidence = total_confidence / len(relationships) if relationships else 0
            
        return {
            "num_relationships": len(relationships),
            "avg_confidence": avg_confidence,
            "polarity_distribution": polarity_counts,
            "directness_distribution": directness_counts
        }
        
    def _log_operation(self, operation: str, data: Dict[str, Any]) -> None:
        """Log operations for auditing and debugging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "data": data
        }
        
        # Log to a file
        log_file = os.path.join("logs", f"bart_processor_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
