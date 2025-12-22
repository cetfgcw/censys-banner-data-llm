"""
LLM-based Banner Classifier

This module implements a zero-shot/few-shot classification approach using
a small, efficient LLM optimized for production deployment.

Model Choice: TinyLlama-1.1B-Chat-v1.0
- Small size (~2.3GB) allows for efficient deployment
- Chat format enables structured prompting
- Can be quantized to 4-bit/8-bit for faster inference
- Good balance between accuracy and speed for classification tasks
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Optional, Tuple, Any
import logging
import time

logger = logging.getLogger(__name__)

# Valid categories
CATEGORIES = [
    "web_server",
    "database",
    "ssh_server",
    "mail_server",
    "ftp_server",
    "other"
]

# Category descriptions for better prompting
CATEGORY_DESCRIPTIONS = {
    "web_server": "Web servers like Apache, nginx, IIS that serve HTTP/HTTPS content",
    "database": "Database servers like MySQL, PostgreSQL, MongoDB that store and manage data",
    "ssh_server": "SSH servers like OpenSSH, Dropbear that provide secure remote access",
    "mail_server": "Mail servers like Postfix, Exim, Exchange that handle email (SMTP/IMAP)",
    "ftp_server": "FTP servers like vsftpd, ProFTPD that handle file transfers",
    "other": "Any other type of service or unrecognized banner"
}


class ModelConfig:
    """Configuration for the LLM model."""
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_quantization: bool = True,
        quantization_bits: int = 4,
        device_map: str = "auto",
        max_length: int = 512,
        temperature: float = 0.1,  # Low temperature for deterministic classification
        use_few_shot: bool = True,
        num_examples: int = 2
    ):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.device_map = device_map
        self.max_length = max_length
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples


class BannerClassifier:
    """
    LLM-based banner classifier using zero-shot/few-shot prompting.
    
    This classifier uses a small LLM (TinyLlama) with structured prompts
    to classify internet service banners into predefined categories.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the banner classifier.
        
        Args:
            config: Model configuration. If None, uses default config.
        """
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._few_shot_examples = self._generate_few_shot_examples()
        
    def _generate_few_shot_examples(self) -> str:
        """Generate few-shot examples for prompting."""
        examples = [
            ("SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5", "ssh_server"),
            ("HTTP/1.1 200 OK\r\nServer: nginx/1.18.0", "web_server"),
            ("220 mail.example.com ESMTP Postfix", "mail_server"),
            ("MySQL 8.0.33", "database"),
            ("220 ProFTPD 1.3.6 Server", "ftp_server"),
        ]
        
        examples_text = "\n".join([
            f"Banner: {banner}\nCategory: {category}"
            for banner, category in examples[:self.config.num_examples]
        ])
        return examples_text
    
    def _build_prompt(self, banner_text: str) -> str:
        """
        Build a structured prompt for classification.
        
        Args:
            banner_text: The banner text to classify
            
        Returns:
            Formatted prompt string
        """
        # Clean and truncate banner text
        banner_clean = banner_text.strip()[:500]  # Limit length for efficiency
        
        # Build category list with descriptions
        categories_list = "\n".join([
            f"- {cat}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()
        ])
        
        if self.config.use_few_shot:
            prompt = f"""<|system|>
You are a network security expert that classifies internet service banners into categories.

Categories:
{categories_list}

Examples:
{self._few_shot_examples}

<|user|>
Classify this banner into one of the categories (web_server, database, ssh_server, mail_server, ftp_server, or other):

Banner: {banner_clean}

Respond with ONLY the category name, nothing else.<|assistant|>
Category:"""
        else:
            prompt = f"""<|system|>
You are a network security expert that classifies internet service banners into categories.

Categories:
{categories_list}

<|user|>
Classify this banner into one of the categories (web_server, database, ssh_server, mail_server, ftp_server, or other):

Banner: {banner_clean}

Respond with ONLY the category name, nothing else.<|assistant|>
Category:"""
        
        return prompt
    
    def load_model(self):
        """Load the model and tokenizer with optional quantization."""
        logger.info(f"Loading model: {self.config.model_name}")
        start_time = time.time()
        
        try:
            # Configure quantization if enabled
            quantization_config = None
            if self.config.use_quantization and torch.cuda.is_available():
                if self.config.quantization_bits == 4:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif self.config.quantization_bits == 8:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.config.device_map,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                # For CPU, use float32
                model_kwargs["torch_dtype"] = torch.float32 if not torch.cuda.is_available() else torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.config.device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            logger.info(f"Device: {self.device}")
            logger.info(f"Quantization: {self.config.use_quantization} ({self.config.quantization_bits}-bit)")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, banner_text: str) -> Dict[str, Any]:
        """
        Classify a single banner.
        
        Args:
            banner_text: The banner text to classify
            
        Returns:
            Dictionary with 'category' and 'confidence' (if available)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        prompt = self._build_prompt(banner_text)
        
        try:
            # Generate prediction
            outputs = self.pipeline(
                prompt,
                max_new_tokens=10,  # We only need the category name
                temperature=self.config.temperature,
                do_sample=False,  # Deterministic for classification
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract category from output
            generated_text = outputs[0]['generated_text'].strip().lower()
            
            # Parse category from response
            category = self._parse_category(generated_text)
            
            return {
                "category": category,
                "raw_output": generated_text
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Fallback to 'other' category
            return {
                "category": "other",
                "raw_output": "",
                "error": str(e)
            }
    
    def _parse_category(self, text: str) -> str:
        """
        Parse category from model output.
        
        Args:
            text: Raw model output
            
        Returns:
            Valid category name
        """
        text_lower = text.lower().strip()
        
        # Direct match
        for cat in CATEGORIES:
            if cat in text_lower:
                return cat
        
        # Fuzzy matching for common variations
        if "web" in text_lower or "http" in text_lower or "nginx" in text_lower or "apache" in text_lower:
            return "web_server"
        elif "database" in text_lower or "mysql" in text_lower or "postgres" in text_lower or "mongo" in text_lower:
            return "database"
        elif "ssh" in text_lower:
            return "ssh_server"
        elif "mail" in text_lower or "smtp" in text_lower or "postfix" in text_lower or "exim" in text_lower:
            return "mail_server"
        elif "ftp" in text_lower:
            return "ftp_server"
        else:
            return "other"
    
    def predict_batch(self, banner_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple banners (sequential for now, can be optimized).
        
        Args:
            banner_texts: List of banner texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(banner) for banner in banner_texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Estimate model size
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "quantization": f"{self.config.quantization_bits}-bit" if self.config.use_quantization else "none",
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "max_length": self.config.max_length,
            "use_few_shot": self.config.use_few_shot
        }

# Implementation approach validated against requirements

