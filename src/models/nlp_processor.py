"""
Natural Language Processing module for gesture-to-text mapping
Handles context understanding, phrase generation, and multi-language support
"""
import nltk
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import logging
from collections import deque, defaultdict
import re

from config.config import config

class GestureToTextMapper:
    """Maps recognized gestures to meaningful text with context awareness"""
    
    def __init__(self):
        # Download required NLTK data
        self._download_nltk_requirements()
        
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy English model not found. Some features may be limited.")
            self.nlp = None
        
        # Initialize gesture mappings
        self.gesture_mappings = self._load_gesture_mappings()
        
        # Context buffer for maintaining conversation flow
        self.context_buffer = deque(maxlen=config.nlp.context_window)
        
        # Language detection and translation
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul") if config.nlp.use_offline_translation else None
        
        # Gesture confidence tracking
        self.gesture_confidence_buffer = deque(maxlen=5)
    
    def _download_nltk_requirements(self):
        """Download required NLTK data"""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'vader_lexicon'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    logging.warning(f"Could not download NLTK data: {data}")
    
    def _load_gesture_mappings(self) -> Dict[str, Dict]:
        """Load gesture-to-word/phrase mappings"""
        # Basic ASL gesture mappings
        mappings = {
            # Basic greetings and politeness
            'hello': {
                'word': 'hello',
                'phrases': ['Hello!', 'Hi there!', 'Good to see you!'],
                'category': 'greeting',
                'context_boost': ['morning', 'meeting', 'introduction']
            },
            'thank_you': {
                'word': 'thank you',
                'phrases': ['Thank you!', 'Thanks!', 'I appreciate it!'],
                'category': 'politeness',
                'context_boost': ['help', 'gift', 'assistance']
            },
            'please': {
                'word': 'please',
                'phrases': ['Please', 'If you could please', 'Would you please'],
                'category': 'politeness',
                'context_boost': ['request', 'ask', 'need']
            },
            'yes': {
                'word': 'yes',
                'phrases': ['Yes', 'Absolutely', 'That\'s right', 'Correct'],
                'category': 'response',
                'context_boost': ['question', 'confirm', 'agree']
            },
            'no': {
                'word': 'no',
                'phrases': ['No', 'Not really', 'I don\'t think so', 'Nope'],
                'category': 'response',
                'context_boost': ['question', 'disagree', 'refuse']
            },
            
            # Emotions and states
            'happy': {
                'word': 'happy',
                'phrases': ['I\'m happy', 'That makes me happy', 'I feel good'],
                'category': 'emotion',
                'context_boost': ['feeling', 'mood', 'emotion']
            },
            'sad': {
                'word': 'sad',
                'phrases': ['I\'m sad', 'That makes me sad', 'I feel down'],
                'category': 'emotion',
                'context_boost': ['feeling', 'mood', 'emotion']
            },
            
            # Basic needs
            'water': {
                'word': 'water',
                'phrases': ['I need water', 'Water please', 'Can I have water?'],
                'category': 'need',
                'context_boost': ['thirsty', 'drink', 'need']
            },
            'food': {
                'word': 'food',
                'phrases': ['I need food', 'I\'m hungry', 'Time to eat'],
                'category': 'need',
                'context_boost': ['hungry', 'eat', 'meal']
            },
            'help': {
                'word': 'help',
                'phrases': ['I need help', 'Can you help me?', 'Please help'],
                'category': 'request',
                'context_boost': ['problem', 'assistance', 'support']
            },
            
            # Actions
            'more': {
                'word': 'more',
                'phrases': ['More please', 'I want more', 'Give me more'],
                'category': 'quantity',
                'context_boost': ['want', 'need', 'additional']
            },
            'finish': {
                'word': 'finished',
                'phrases': ['I\'m finished', 'All done', 'That\'s enough'],
                'category': 'status',
                'context_boost': ['complete', 'done', 'end']
            },
            'sorry': {
                'word': 'sorry',
                'phrases': ['I\'m sorry', 'Sorry about that', 'My apologies'],
                'category': 'politeness',
                'context_boost': ['mistake', 'apologize', 'regret']
            },
            
            # Relationships
            'family': {
                'word': 'family',
                'phrases': ['My family', 'Family is important', 'I love my family'],
                'category': 'relationship',
                'context_boost': ['home', 'relatives', 'love']
            },
            'friend': {
                'word': 'friend',
                'phrases': ['My friend', 'Good friend', 'Close friend'],
                'category': 'relationship',
                'context_boost': ['friendship', 'buddy', 'companion']
            },
            
            # Places
            'home': {
                'word': 'home',
                'phrases': ['I\'m going home', 'At home', 'My home'],
                'category': 'place',
                'context_boost': ['house', 'family', 'live']
            },
            'work': {
                'word': 'work',
                'phrases': ['I work', 'At work', 'Going to work'],
                'category': 'place',
                'context_boost': ['job', 'office', 'career']
            },
            'school': {
                'word': 'school',
                'phrases': ['At school', 'Going to school', 'I study'],
                'category': 'place',
                'context_boost': ['learn', 'education', 'student']
            }
        }
        
        # Add alphabet mappings
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for letter in alphabet:
            mappings[letter] = {
                'word': letter,
                'phrases': [f'The letter {letter}', f'{letter}'],
                'category': 'alphabet',
                'context_boost': ['spell', 'letter', 'alphabet']
            }
        
        return mappings
    
    def map_gesture_to_text(
        self,
        gesture: str,
        confidence: float,
        use_context: bool = True
    ) -> Dict[str, any]:
        """
        Map a recognized gesture to appropriate text
        
        Args:
            gesture: Recognized gesture label
            confidence: Recognition confidence score
            use_context: Whether to use context for text generation
            
        Returns:
            Dictionary containing mapped text and metadata
        """
        # Store gesture confidence
        self.gesture_confidence_buffer.append((gesture, confidence))
        
        # Get base mapping
        if gesture not in self.gesture_mappings:
            return {
                'text': gesture,
                'phrase': f"[{gesture}]",
                'confidence': confidence,
                'category': 'unknown',
                'context_used': False
            }
        
        mapping = self.gesture_mappings[gesture]
        
        # Select appropriate phrase based on context
        if use_context and len(self.context_buffer) > 0:
            phrase = self._select_contextual_phrase(mapping)
            context_used = True
        else:
            phrase = mapping['phrases'][0]  # Default phrase
            context_used = False
        
        # Add to context buffer
        self.context_buffer.append({
            'gesture': gesture,
            'text': mapping['word'],
            'category': mapping['category'],
            'confidence': confidence
        })
        
        return {
            'text': mapping['word'],
            'phrase': phrase,
            'confidence': confidence,
            'category': mapping['category'],
            'context_used': context_used
        }
    
    def _select_contextual_phrase(self, mapping: Dict) -> str:
        """Select the most appropriate phrase based on context"""
        # Analyze recent context
        recent_categories = [item['category'] for item in list(self.context_buffer)[-3:]]
        recent_words = [item['text'] for item in list(self.context_buffer)[-3:]]
        
        # Check for context boost keywords
        context_score = 0
        for boost_word in mapping.get('context_boost', []):
            if boost_word in recent_words:
                context_score += 1
        
        # Select phrase based on context and category patterns
        phrases = mapping['phrases']
        
        if context_score > 0 and len(phrases) > 1:
            # Use more contextual phrase
            return phrases[min(context_score, len(phrases) - 1)]
        
        # Check for conversation patterns
        if len(recent_categories) >= 2:
            if 'question' in recent_categories and mapping['category'] == 'response':
                # This is likely an answer to a question
                return phrases[0]
            elif 'greeting' in recent_categories and mapping['category'] == 'politeness':
                # Polite response to greeting
                return phrases[-1] if len(phrases) > 1 else phrases[0]
        
        return phrases[0]  # Default
    
    def generate_sentence(self, gesture_sequence: List[Tuple[str, float]]) -> str:
        """
        Generate a coherent sentence from a sequence of gestures
        
        Args:
            gesture_sequence: List of (gesture, confidence) tuples
            
        Returns:
            Generated sentence
        """
        if not gesture_sequence:
            return ""
        
        # Filter low-confidence gestures
        filtered_sequence = [
            (gesture, conf) for gesture, conf in gesture_sequence
            if conf >= config.ui.prediction_threshold
        ]
        
        if not filtered_sequence:
            return ""
        
        # Map each gesture to text
        mapped_texts = []
        for gesture, confidence in filtered_sequence:
            result = self.map_gesture_to_text(gesture, confidence, use_context=True)
            mapped_texts.append(result['text'])
        
        # Generate coherent sentence
        sentence = self._construct_sentence(mapped_texts)
        
        return sentence
    
    def _construct_sentence(self, words: List[str]) -> str:
        """Construct a grammatically correct sentence from words"""
        if not words:
            return ""
        
        if len(words) == 1:
            return words[0].capitalize() + "."
        
        # Simple sentence construction rules
        sentence_patterns = {
            # Subject + Verb patterns
            ('I', 'need'): "I need {}.",
            ('I', 'want'): "I want {}.",
            ('I', 'love'): "I love {}.",
            ('I', 'am'): "I am {}.",
            
            # Politeness patterns
            ('please', 'help'): "Please help me.",
            ('thank', 'you'): "Thank you!",
            
            # Question patterns
            ('where', 'is'): "Where is {}?",
            ('what', 'is'): "What is {}?",
        }
        
        # Try to match patterns
        if len(words) >= 2:
            key = (words[0].lower(), words[1].lower())
            if key in sentence_patterns:
                pattern = sentence_patterns[key]
                if len(words) > 2:
                    return pattern.format(" ".join(words[2:]))
                else:
                    return pattern
        
        # Fallback: simple concatenation with basic grammar
        sentence = " ".join(words)
        sentence = sentence.capitalize()
        
        # Add punctuation
        if not sentence.endswith(('.', '!', '?')):
            # Determine punctuation based on content
            if any(word in sentence.lower() for word in ['where', 'what', 'when', 'how', 'why']):
                sentence += "?"
            elif any(word in sentence.lower() for word in ['hello', 'thank', 'please', 'help']):
                sentence += "!"
            else:
                sentence += "."
        
        return sentence
    
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text to target language
        
        Args:
            text: Input text to translate
            target_language: Target language code
            
        Returns:
            Translated text
        """
        if target_language == 'en' or not self.translator:
            return text
        
        try:
            # Use offline translation if available
            if self.translator:
                result = self.translator(text, target_language=target_language)
                return result[0]['translation_text']
        except Exception as e:
            logging.warning(f"Translation failed: {e}")
        
        return text  # Fallback to original text
    
    def get_gesture_suggestions(self, partial_sentence: str) -> List[str]:
        """
        Get gesture suggestions based on partial sentence context
        
        Args:
            partial_sentence: Partially constructed sentence
            
        Returns:
            List of suggested gestures
        """
        suggestions = []
        
        # Analyze partial sentence for context
        words = partial_sentence.lower().split()
        
        # Simple rule-based suggestions
        if not words:
            # Start with greetings
            suggestions = ['hello', 'good', 'thank_you']
        elif words[-1] in ['i', 'me']:
            suggestions = ['need', 'want', 'am', 'love']
        elif 'need' in words:
            suggestions = ['help', 'water', 'food', 'more']
        elif 'thank' in words:
            suggestions = ['you']
        else:
            # General suggestions based on frequency
            suggestions = ['please', 'yes', 'no', 'help', 'more']
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def clear_context(self):
        """Clear the context buffer"""
        self.context_buffer.clear()
        self.gesture_confidence_buffer.clear()
    
    def get_context_summary(self) -> Dict:
        """Get summary of current context"""
        if not self.context_buffer:
            return {'empty': True}
        
        categories = [item['category'] for item in self.context_buffer]
        words = [item['text'] for item in self.context_buffer]
        avg_confidence = np.mean([item['confidence'] for item in self.context_buffer])
        
        return {
            'recent_words': words[-3:],
            'dominant_category': max(set(categories), key=categories.count),
            'average_confidence': avg_confidence,
            'context_length': len(self.context_buffer)
        }

class LanguageDetector:
    """Simple language detection for multi-language support"""
    
    def __init__(self):
        self.language_patterns = {
            'en': ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has'],
            'es': ['el', 'la', 'y', 'es', 'son', 'fue', 'fueron', 'tiene', 'hay'],
            'fr': ['le', 'la', 'et', 'est', 'sont', 'était', 'étaient', 'avoir', 'a'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'sind', 'war', 'waren', 'haben'],
            'it': ['il', 'la', 'e', 'è', 'sono', 'era', 'erano', 'avere', 'ha']
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        words = text.lower().split()
        
        if not words:
            return config.nlp.default_language
        
        language_scores = defaultdict(int)
        
        for word in words:
            for lang, patterns in self.language_patterns.items():
                if word in patterns:
                    language_scores[lang] += 1
        
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return config.nlp.default_language