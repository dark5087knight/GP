"""
Enhanced multi-language support for sign language recognition system
Provides translation services and localization features
"""
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class LanguageConfig:
    """Configuration for language support"""
    code: str
    name: str
    native_name: str
    rtl: bool = False  # Right-to-left text
    gesture_mappings: Optional[Dict] = None

class MultiLanguageSupport:
    """Enhanced multi-language support system"""
    
    def __init__(self):
        self.supported_languages = self._initialize_languages()
        self.gesture_translations = self._load_gesture_translations()
        self.ui_translations = self._load_ui_translations()
        self.current_language = 'en'
        
        # Simple offline translation dictionaries
        self.translation_dict = self._load_translation_dictionaries()
        
        logging.info(f"Multi-language support initialized with {len(self.supported_languages)} languages")
    
    def _initialize_languages(self) -> Dict[str, LanguageConfig]:
        """Initialize supported languages with configurations"""
        languages = {
            'en': LanguageConfig('en', 'English', 'English'),
            'es': LanguageConfig('es', 'Spanish', 'Español'),
            'fr': LanguageConfig('fr', 'French', 'Français'),
            'de': LanguageConfig('de', 'German', 'Deutsch'),
            'it': LanguageConfig('it', 'Italian', 'Italiano'),
            'pt': LanguageConfig('pt', 'Portuguese', 'Português'),
            'zh': LanguageConfig('zh', 'Chinese', '中文'),
            'ja': LanguageConfig('ja', 'Japanese', '日本語'),
            'ko': LanguageConfig('ko', 'Korean', '한국어'),
            'ar': LanguageConfig('ar', 'Arabic', 'العربية', rtl=True)
        }
        return languages
    
    def _load_gesture_translations(self) -> Dict[str, Dict[str, str]]:
        """Load gesture-specific translations"""
        # In a real implementation, these would be loaded from files
        translations = {
            'hello': {
                'en': 'Hello',
                'es': 'Hola',
                'fr': 'Bonjour',
                'de': 'Hallo',
                'it': 'Ciao',
                'pt': 'Olá',
                'zh': '你好',
                'ja': 'こんにちは',
                'ko': '안녕하세요',
                'ar': 'مرحبا'
            },
            'thank_you': {
                'en': 'Thank you',
                'es': 'Gracias',
                'fr': 'Merci',
                'de': 'Danke',
                'it': 'Grazie',
                'pt': 'Obrigado',
                'zh': '谢谢',
                'ja': 'ありがとう',
                'ko': '감사합니다',
                'ar': 'شكرا'
            },
            'please': {
                'en': 'Please',
                'es': 'Por favor',
                'fr': 'S\'il vous plaît',
                'de': 'Bitte',
                'it': 'Per favore',
                'pt': 'Por favor',
                'zh': '请',
                'ja': 'お願いします',
                'ko': '제발',
                'ar': 'من فضلك'
            },
            'yes': {
                'en': 'Yes',
                'es': 'Sí',
                'fr': 'Oui',
                'de': 'Ja',
                'it': 'Sì',
                'pt': 'Sim',
                'zh': '是',
                'ja': 'はい',
                'ko': '네',
                'ar': 'نعم'
            },
            'no': {
                'en': 'No',
                'es': 'No',
                'fr': 'Non',
                'de': 'Nein',
                'it': 'No',
                'pt': 'Não',
                'zh': '不',
                'ja': 'いいえ',
                'ko': '아니요',
                'ar': 'لا'
            },
            'help': {
                'en': 'Help',
                'es': 'Ayuda',
                'fr': 'Aide',
                'de': 'Hilfe',
                'it': 'Aiuto',
                'pt': 'Ajuda',
                'zh': '帮助',
                'ja': '助けて',
                'ko': '도움',
                'ar': 'مساعدة'
            },
            'water': {
                'en': 'Water',
                'es': 'Agua',
                'fr': 'Eau',
                'de': 'Wasser',
                'it': 'Acqua',
                'pt': 'Água',
                'zh': '水',
                'ja': '水',
                'ko': '물',
                'ar': 'ماء'
            },
            'food': {
                'en': 'Food',
                'es': 'Comida',
                'fr': 'Nourriture',
                'de': 'Essen',
                'it': 'Cibo',
                'pt': 'Comida',
                'zh': '食物',
                'ja': '食べ物',
                'ko': '음식',
                'ar': 'طعام'
            },
            'family': {
                'en': 'Family',
                'es': 'Familia',
                'fr': 'Famille',
                'de': 'Familie',
                'it': 'Famiglia',
                'pt': 'Família',
                'zh': '家庭',
                'ja': '家族',
                'ko': '가족',
                'ar': 'عائلة'
            },
            'home': {
                'en': 'Home',
                'es': 'Casa',
                'fr': 'Maison',
                'de': 'Zuhause',
                'it': 'Casa',
                'pt': 'Casa',
                'zh': '家',
                'ja': '家',
                'ko': '집',
                'ar': 'منزل'
            }
        }
        return translations
    
    def _load_ui_translations(self) -> Dict[str, Dict[str, str]]:
        """Load UI element translations"""
        ui_translations = {
            'app_title': {
                'en': 'Sign Language to Text Translator',
                'es': 'Traductor de Lenguaje de Señas a Texto',
                'fr': 'Traducteur de Langue des Signes vers Texte',
                'de': 'Gebärdensprache zu Text Übersetzer',
                'it': 'Traduttore da Lingua dei Segni a Testo',
                'pt': 'Tradutor de Linguagem de Sinais para Texto',
                'zh': '手语转文本翻译器',
                'ja': '手話からテキストへの翻訳機',
                'ko': '수화에서 텍스트로 번역기',
                'ar': 'مترجم لغة الإشارة إلى نص'
            },
            'start_camera': {
                'en': 'Start Camera',
                'es': 'Iniciar Cámara',
                'fr': 'Démarrer la Caméra',
                'de': 'Kamera Starten',
                'it': 'Avvia Fotocamera',
                'pt': 'Iniciar Câmera',
                'zh': '启动摄像头',
                'ja': 'カメラを開始',
                'ko': '카메라 시작',
                'ar': 'تشغيل الكاميرا'
            },
            'stop_camera': {
                'en': 'Stop Camera',
                'es': 'Detener Cámara',
                'fr': 'Arrêter la Caméra',
                'de': 'Kamera Stoppen',
                'it': 'Ferma Fotocamera',
                'pt': 'Parar Câmera',
                'zh': '停止摄像头',
                'ja': 'カメラを停止',
                'ko': '카메라 정지',
                'ar': 'إيقاف الكاميرا'
            },
            'clear_text': {
                'en': 'Clear Text',
                'es': 'Limpiar Texto',
                'fr': 'Effacer le Texte',
                'de': 'Text Löschen',
                'it': 'Cancella Testo',
                'pt': 'Limpar Texto',
                'zh': '清除文本',
                'ja': 'テキストをクリア',
                'ko': '텍스트 지우기',
                'ar': 'مسح النص'
            },
            'confidence': {
                'en': 'Confidence',
                'es': 'Confianza',
                'fr': 'Confiance',
                'de': 'Vertrauen',
                'it': 'Confidenza',
                'pt': 'Confiança',
                'zh': '置信度',
                'ja': '信頼度',
                'ko': '신뢰도',
                'ar': 'الثقة'
            },
            'gesture_detected': {
                'en': 'Gesture Detected',
                'es': 'Gesto Detectado',
                'fr': 'Geste Détecté',
                'de': 'Geste Erkannt',
                'it': 'Gesto Rilevato',
                'pt': 'Gesto Detectado',
                'zh': '检测到手势',
                'ja': 'ジェスチャーを検出',
                'ko': '제스처 감지됨',
                'ar': 'تم اكتشاف الإيماءة'
            }
        }
        return ui_translations
    
    def _load_translation_dictionaries(self) -> Dict[str, Dict[str, str]]:
        """Load comprehensive translation dictionaries"""
        # Basic translation dictionaries for common phrases
        translations = {
            'en_to_es': {
                'I need help': 'Necesito ayuda',
                'Thank you very much': 'Muchas gracias',
                'Please help me': 'Por favor ayúdame',
                'I am happy': 'Estoy feliz',
                'Good morning': 'Buenos días',
                'How are you': 'Cómo estás',
                'Nice to meet you': 'Mucho gusto'
            },
            'en_to_fr': {
                'I need help': 'J\\'ai besoin d\\'aide',
                'Thank you very much': 'Merci beaucoup',
                'Please help me': 'Aidez-moi s\\'il vous plaît',
                'I am happy': 'Je suis heureux',
                'Good morning': 'Bonjour',
                'How are you': 'Comment allez-vous',
                'Nice to meet you': 'Enchanté de vous rencontrer'
            },
            'en_to_de': {
                'I need help': 'Ich brauche Hilfe',
                'Thank you very much': 'Vielen Dank',
                'Please help me': 'Bitte hilf mir',
                'I am happy': 'Ich bin glücklich',
                'Good morning': 'Guten Morgen',
                'How are you': 'Wie geht es dir',
                'Nice to meet you': 'Schön dich kennenzulernen'
            }
        }
        return translations
    
    def translate_gesture(self, gesture: str, target_language: str) -> str:
        """Translate a single gesture to target language"""
        if gesture in self.gesture_translations:
            return self.gesture_translations[gesture].get(target_language, gesture)
        return gesture
    
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language"""
        if target_language == 'en':
            return text
        
        # Try exact phrase match first
        translation_key = f'en_to_{target_language}'
        if translation_key in self.translation_dict:
            if text in self.translation_dict[translation_key]:
                return self.translation_dict[translation_key][text]
        
        # Try word-by-word translation for gestures
        words = text.split()
        translated_words = []
        
        for word in words:
            # Remove punctuation for lookup
            clean_word = word.strip('.,!?').lower()
            translated_word = self.translate_gesture(clean_word, target_language)
            
            # Preserve original case and punctuation
            if translated_word != clean_word:
                if word[0].isupper():
                    translated_word = translated_word.capitalize()
                if word.endswith(('.', '!', '?', ',')):
                    translated_word += word[-1]
            else:
                translated_word = word  # Keep original if no translation found
            
            translated_words.append(translated_word)
        
        return ' '.join(translated_words)
    
    def get_ui_text(self, key: str, language: str = None) -> str:
        """Get UI text in specified language"""
        if language is None:
            language = self.current_language
        
        if key in self.ui_translations:
            return self.ui_translations[key].get(language, 
                                               self.ui_translations[key].get('en', key))
        return key
    
    def set_language(self, language_code: str) -> bool:
        """Set current system language"""
        if language_code in self.supported_languages:
            self.current_language = language_code
            logging.info(f"Language set to {self.supported_languages[language_code].name}")
            return True
        return False
    
    def get_supported_languages(self) -> List[Tuple[str, str]]:
        """Get list of supported languages as (code, name) tuples"""
        return [(code, config.name) for code, config in self.supported_languages.items()]
    
    def get_language_info(self, language_code: str) -> Optional[LanguageConfig]:
        """Get detailed information about a language"""
        return self.supported_languages.get(language_code)
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (enhanced version)"""
        # Language-specific character patterns
        patterns = {
            'zh': lambda t: any(ord(char) > 0x4E00 and ord(char) < 0x9FFF for char in t),
            'ja': lambda t: any(ord(char) > 0x3040 and ord(char) < 0x309F for char in t) or 
                           any(ord(char) > 0x30A0 and ord(char) < 0x30FF for char in t),
            'ko': lambda t: any(ord(char) > 0xAC00 and ord(char) < 0xD7AF for char in t),
            'ar': lambda t: any(ord(char) > 0x0600 and ord(char) < 0x06FF for char in t)
        }
        
        # Check character-based languages first
        for lang, pattern in patterns.items():
            if pattern(text):
                return lang
        
        # For Latin-based languages, use word patterns
        text_lower = text.lower()
        
        # Simple keyword-based detection
        language_keywords = {
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf'],
            'it': ['il', 'di', 'che', 'e', 'la', 'per', 'in', 'un', 'è', 'si', 'da', 'non'],
            'pt': ['de', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os']
        }
        
        word_scores = {lang: 0 for lang in language_keywords}
        words = text_lower.split()
        
        for word in words:
            for lang, keywords in language_keywords.items():
                if word in keywords:
                    word_scores[lang] += 1
        
        # Return language with highest score, default to English
        if max(word_scores.values()) > 0:
            return max(word_scores, key=word_scores.get)
        
        return 'en'  # Default to English
    
    def get_gesture_suggestions_by_language(self, language: str) -> List[str]:
        """Get common gesture suggestions for a specific language/culture"""
        # Cultural gesture preferences
        cultural_gestures = {
            'en': ['hello', 'thank_you', 'please', 'yes', 'no', 'help'],
            'es': ['hello', 'gracias', 'por_favor', 'sí', 'no', 'ayuda'],
            'fr': ['bonjour', 'merci', 'sil_vous_plait', 'oui', 'non', 'aide'],
            'de': ['hallo', 'danke', 'bitte', 'ja', 'nein', 'hilfe'],
            'it': ['ciao', 'grazie', 'per_favore', 'sì', 'no', 'aiuto'],
            'zh': ['hello', 'thank_you', 'please', 'yes', 'no', 'help'],  # Using English gestures
            'ja': ['hello', 'thank_you', 'please', 'yes', 'no', 'help'],
            'ko': ['hello', 'thank_you', 'please', 'yes', 'no', 'help'],
            'ar': ['hello', 'thank_you', 'please', 'yes', 'no', 'help']
        }
        
        return cultural_gestures.get(language, cultural_gestures['en'])
    
    def format_text_for_language(self, text: str, language: str) -> str:
        """Format text according to language conventions"""
        language_config = self.get_language_info(language)
        
        if language_config and language_config.rtl:
            # For RTL languages like Arabic, add RTL marker
            return f"\\u202B{text}\\u202C"
        
        return text
    
    def get_localized_gesture_mappings(self, language: str) -> Dict[str, str]:
        """Get gesture mappings localized for specific language"""
        localized_mappings = {}
        
        for gesture, translations in self.gesture_translations.items():
            localized_mappings[gesture] = translations.get(language, 
                                                         translations.get('en', gesture))
        
        return localized_mappings

# Global instance
multi_language_support = MultiLanguageSupport()