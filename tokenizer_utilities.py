import json
import os
from typing import Dict, List, Optional
import numpy as np


class NOOVERLAPTokenizer:
    def __init__(self, tokenizers_dict: Dict[str, object], 
                 language_tags: Dict[str, str]):
        self.tokenizers = tokenizers_dict
        self.language_tags = language_tags
        self.languages = list(tokenizers_dict.keys())
        
        self.vocab = {}
        self.id_to_token = {}
        self.token_to_id = {}
        self._build_combined_vocab()
    
    def _build_combined_vocab(self):
        current_id = 0
        
        # Add language tags first
        for lang, tag in self.language_tags.items():
            self.vocab[tag] = current_id
            self.token_to_id[tag] = current_id
            self.id_to_token[current_id] = tag
            current_id += 1
        
        # Add special tokens
        special_tokens = ["<UNK>", "<PAD>", "<CLS>", "<SEP>", "<MASK>"]
        for token in special_tokens:
            self.vocab[token] = current_id
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
        
        # Add each language's vocabulary
        for lang in self.languages:
            tokenizer = self.tokenizers[lang]
            tag = self.language_tags[lang]
            vocab = tokenizer.get_vocab()
            
            for token, original_id in vocab.items():
                if token in special_tokens or token in self.language_tags.values():
                    continue
                
                tagged_token = f"{tag}_{token}"
                self.vocab[tagged_token] = current_id
                self.token_to_id[tagged_token] = current_id
                self.id_to_token[current_id] = tagged_token
                current_id += 1
    
    def _detect_language_from_tag(self, text: str) -> str:
        """Detect language from language tag at start of text"""
        for lang, tag in self.language_tags.items():
            if text.startswith(f"{tag} "):
                return lang
        # Default to first language if no tag found
        return self.languages[0]
    
    def _remove_language_tag(self, text: str) -> str:
        """Remove language tag from text"""
        for tag in self.language_tags.values():
            if text.startswith(f"{tag} "):
                return text[len(tag) + 1:]  # +1 for the space
        return text
    
    def encode(self, text: str, language: Optional[str] = None) -> List[int]:
        """
        Encode a single text.
        
        Args:
            text: Text to encode (can include language tag prefix like "<EN> text")
            language: Optional language code. If not provided, will be detected from tag.
        
        Returns:
            List of token IDs
        """
        # Auto-detect language from tag if not provided
        if language is None:
            language = self._detect_language_from_tag(text)
        
        if language not in self.tokenizers:
            raise ValueError(f"Language {language} not supported")
        
        # Remove language tag if present
        text = self._remove_language_tag(text)
        
        tokenizer = self.tokenizers[language]
        tag = self.language_tags[language]
        
        tokens = tokenizer.encode(text).ids
        
        tag_id = self.vocab[tag]
        return [tag_id] + tokens
    
    def encode_batch(self, texts: List[str]) -> List['EncodedText']:
        """
        Encode a batch of texts efficiently.
        
        Args:
            texts: List of texts to encode (can include language tag prefixes)
        
        Returns:
            List of encoded text objects with .ids and .attention_mask attributes
        """
        encoded_batch = []
        
        for text in texts:
            ids = self.encode(text)
            
            # Create a simple object with .ids attribute (compatible with training)
            encoded = _EncodedText(ids=ids)
            encoded_batch.append(encoded)
        
        return encoded_batch
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(id_, "<UNK>") for id_ in ids]
        text_parts = []
        for token in tokens:
            if token.startswith("<") and token.endswith(">"):
                continue
            text_parts.append(token)
        return "".join(text_parts)
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab_size_per_language(self) -> Dict[str, int]:
        sizes = {}
        for lang in self.languages:
            vocab = self.tokenizers[lang].get_vocab()
            sizes[lang] = len(vocab)
        return sizes
    
    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "vocab.json"), "w") as f:
            json.dump(self.vocab, f, indent=2)
        
        metadata = {
            "strategy": "NOOVERLAP",
            "languages": self.languages,
            "language_tags": self.language_tags,
            "vocab_size": self.get_vocab_size(),
            "vocab_per_language": self.get_vocab_size_per_language(),
        }
        
        with open(os.path.join(output_dir, "nooverlap_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


class _EncodedText:
    """Simple wrapper class for encoded text output (compatible with training code)"""
    def __init__(self, ids: List[int]):
        self.ids = ids
        # Create attention mask (all 1s since all tokens are valid)
        self.attention_mask = [1] * len(ids)
    
    def __repr__(self):
        return f"EncodedText(ids={len(self.ids)} tokens)"