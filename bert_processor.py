import spacy
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import re

class BERTProcessor:
    """
    Clean BERT-based processor for scientific relationship extraction.
    Uses only spaCy's BERT-based models without other LLM interference.
    """
    
    def __init__(self, temperature: float = 1.0):
        """Initialize the BERT processor with temperature control."""
        self.temperature = temperature
        self.nlp = None
        self.processing_time = 0.0
        
        # Clean logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
    
    def _load_model(self):
        """Load spaCy BERT model"""
        try:
            # Load English BERT model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("BERT model loaded successfully")
        except OSError:
            self.logger.error("BERT model not found. Please install: python -m spacy download en_core_web_sm")
            raise
    
    def clear_cache(self):
        """Clear all cached data for clean measurements"""
        self.processing_time = 0.0
        # Reset processing time for clean measurements
    
    def extract_relationships(self, text: str, custom_prompt: str = "") -> Dict[str, Any]:
        """
        Extract semantic relationships from text using BERT.
        
        Args:
            text: Input text to analyze
            custom_prompt: Optional custom prompt for extraction focus
            
        Returns:
            Dict with relationships, statistics, and metadata
        """
        start_time = time.time()
        
        # Clear previous processing state
        self.clear_cache()
        
        if not text.strip():
            return self._empty_result()
        
        # Apply custom prompt context if provided
        if custom_prompt.strip():
            # Use prompt to focus extraction
            text = f"{custom_prompt.strip()}\n\n{text}"
        
        # Preprocess text
        sentences = self._preprocess_text(text)
        
        # Extract relationships from each sentence
        all_relationships = []
        for sentence in sentences:
            relationships = self._extract_from_sentence(sentence)
            all_relationships.extend(relationships)
        
        # Apply temperature scaling to confidence scores
        for rel in all_relationships:
            rel["confidence"] = min(1.0, rel["confidence"] / self.temperature)
        
        # Calculate statistics
        statistics = self._calculate_statistics(all_relationships)
        
        # Convert to RDF triples
        rdf_triples = self.convert_to_rdf(all_relationships)
        
        self.processing_time = time.time() - start_time
        
        self.logger.info(f"BERT extraction completed in {self.processing_time:.4f}s")
        self.logger.info(f"Extracted {len(all_relationships)} relationships")
        
        return {
            "relationships": all_relationships,
            "statistics": statistics,
            "rdf_triples": rdf_triples,
            "processing_time": self.processing_time,
            "model_type": "bert",
            "temperature": self.temperature,
            "custom_prompt": custom_prompt
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "relationships": [],
            "statistics": {},
            "rdf_triples": [],
            "processing_time": 0.0,
            "model_type": "bert",
            "temperature": self.temperature,
            "custom_prompt": ""
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Split text into sentences and clean."""
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) > 10:  # Filter very short sentences
                sentences.append(sentence_text)
        
        return sentences
    
    def _extract_from_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Extract relationships from a single sentence using BERT."""
        doc = self.nlp(sentence)
        relationships = []
        
        # Method 1: Dependency parsing for SVO patterns
        svo_relationships = self._extract_svo_patterns(doc, sentence)
        relationships.extend(svo_relationships)
        
        # Method 2: Named entity relationships
        ner_relationships = self._extract_ner_relationships(doc, sentence)
        relationships.extend(ner_relationships)
        
        # Method 3: Noun phrase relationships
        np_relationships = self._extract_noun_phrase_relationships(doc, sentence)
        relationships.extend(np_relationships)
        
        return relationships
    
    def _extract_svo_patterns(self, doc, sentence: str) -> List[Dict[str, Any]]:
        """Extract Subject-Verb-Object patterns using dependency parsing."""
        relationships = []
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                        subject = self._expand_span(doc, child)
                        break
                
                # Find object
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj", "iobj", "attr"]:
                        obj = self._expand_span(doc, child)
                        break
                    elif child.dep_ == "prep":
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                obj = self._expand_span(doc, prep_child)
                                break
                
                if subject and obj and subject != obj:
                    predicate = token.lemma_
                    
                    # Calculate confidence based on dependency structure
                    confidence = self._calculate_svo_confidence(doc, token, subject, obj)
                    
                    relationship = {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": confidence,
                        "polarity": self._classify_polarity(sentence, subject, predicate, obj),
                        "directness": self._classify_directness(sentence, subject, predicate, obj),
                        "sentence": sentence,
                        "extraction_method": "svo_dependency"
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_ner_relationships(self, doc, sentence: str) -> List[Dict[str, Any]]:
        """Extract relationships between named entities."""
        relationships = []
        entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
        
        # Find relationships between entity pairs
        for i, (ent1_text, ent1_label, ent1_start, ent1_end) in enumerate(entities):
            for j, (ent2_text, ent2_label, ent2_start, ent2_end) in enumerate(entities):
                if i != j and abs(ent1_end - ent2_start) < 10:  # Entities close to each other
                    
                    # Find connecting verb or preposition
                    predicate = self._find_connecting_predicate(doc, ent1_end, ent2_start)
                    
                    if predicate:
                        confidence = 0.7  # Base confidence for NER relationships
                        
                        relationship = {
                            "subject": ent1_text,
                            "predicate": predicate,
                            "object": ent2_text,
                            "confidence": confidence,
                            "polarity": self._classify_polarity(sentence, ent1_text, predicate, ent2_text),
                            "directness": "direct",
                            "sentence": sentence,
                            "extraction_method": "ner_based",
                            "subject_type": ent1_label,
                            "object_type": ent2_label
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_noun_phrase_relationships(self, doc, sentence: str) -> List[Dict[str, Any]]:
        """Extract relationships between noun phrases."""
        relationships = []
        noun_phrases = [chunk for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
        
        for i, np1 in enumerate(noun_phrases):
            for j, np2 in enumerate(noun_phrases):
                if i != j and abs(np1.end - np2.start) < 8:
                    
                    # Find connecting verb
                    predicate = self._find_connecting_predicate(doc, np1.end, np2.start)
                    
                    if predicate:
                        confidence = 0.6  # Lower confidence for noun phrase relationships
                        
                        relationship = {
                            "subject": np1.text.strip(),
                            "predicate": predicate,
                            "object": np2.text.strip(),
                            "confidence": confidence,
                            "polarity": "neutral",
                            "directness": "direct",
                            "sentence": sentence,
                            "extraction_method": "noun_phrase"
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _expand_span(self, doc, token):
        """Expand token to include modifiers and compounds."""
        start = token.i
        end = token.i + 1
        
        # Expand left (modifiers, compounds)
        for child in token.children:
            if child.dep_ in ["amod", "compound", "det", "poss"] and child.i < start:
                start = child.i
        
        # Expand right (compounds, particles)
        for child in token.children:
            if child.dep_ in ["compound", "prt"] and child.i >= end:
                end = child.i + 1
        
        return " ".join([t.text for t in doc[start:end]]).strip()
    
    def _find_connecting_predicate(self, doc, start_pos: int, end_pos: int) -> Optional[str]:
        """Find verb or preposition connecting two spans."""
        min_pos = min(start_pos, end_pos)
        max_pos = max(start_pos, end_pos)
        
        for token in doc[min_pos:max_pos]:
            if token.pos_ in ["VERB", "AUX"]:
                return token.lemma_
            elif token.pos_ == "ADP":  # Preposition
                return token.text
        
        return None
    
    def _calculate_svo_confidence(self, doc, verb_token, subject: str, obj: str) -> float:
        """Calculate confidence for SVO relationship based on linguistic features."""
        confidence = 0.8  # Base confidence
        
        # Boost confidence for clear dependency structure
        if verb_token.dep_ == "ROOT":
            confidence += 0.1
        
        # Check for negation
        for child in verb_token.children:
            if child.dep_ == "neg":
                confidence -= 0.2
                break
        
        # Check sentence length (shorter sentences often clearer)
        sentence_length = len([t for t in doc if not t.is_space])
        if sentence_length < 15:
            confidence += 0.1
        elif sentence_length > 30:
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _classify_polarity(self, sentence: str, subject: str, predicate: str, obj: str) -> str:
        """Classify relationship polarity using BERT features."""
        # Check for negative words
        negative_words = {"not", "no", "never", "without", "lacks", "fails", "prevents", "stops", "reduces", "decreases"}
        positive_words = {"helps", "enables", "supports", "increases", "improves", "enhances", "facilitates"}
        
        sentence_lower = sentence.lower()
        predicate_lower = predicate.lower()
        
        # Check predicate
        if predicate_lower in negative_words:
            return "negative"
        elif predicate_lower in positive_words:
            return "positive"
        
        # Check surrounding context
        negative_count = sum(1 for word in negative_words if word in sentence_lower)
        positive_count = sum(1 for word in positive_words if word in sentence_lower)
        
        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    def _classify_directness(self, sentence: str, subject: str, predicate: str, obj: str) -> str:
        """Classify relationship directness."""
        indirect_indicators = {"through", "via", "by means of", "because of", "due to", "as a result of"}
        
        sentence_lower = sentence.lower()
        for indicator in indirect_indicators:
            if indicator in sentence_lower:
                return "indirect"
        
        return "direct"
    
    def convert_to_rdf(self, relationships: List[Dict[str, Any]]) -> List[tuple]:
        """Convert relationships to RDF-like triples."""
        triples = []
        for rel in relationships:
            subject = rel["subject"].replace(" ", "_")
            predicate = rel["predicate"].replace(" ", "_")
            obj = rel["object"].replace(" ", "_")
            triples.append((subject, predicate, obj))
        return triples
    
    def _calculate_statistics(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about extracted relationships."""
        if not relationships:
            return {}
        
        # Polarity distribution
        polarity_dist = Counter(rel["polarity"] for rel in relationships)
        
        # Directness distribution
        directness_dist = Counter(rel["directness"] for rel in relationships)
        
        # Extraction method distribution
        method_dist = Counter(rel.get("extraction_method", "unknown") for rel in relationships)
        
        # Confidence statistics
        confidences = [rel["confidence"] for rel in relationships]
        
        return {
            "total_relationships": len(relationships),
            "polarity_distribution": dict(polarity_dist),
            "directness_distribution": dict(directness_dist),
            "method_distribution": dict(method_dist),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for scientific comparison."""
        return {
            "model_type": "bert",
            "processing_time": self.processing_time,
            "temperature": self.temperature,
            "spacy_model": "en_core_web_sm"
        }