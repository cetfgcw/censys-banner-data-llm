"""Data loading utilities for banner classification dataset."""

import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Valid categories as per requirements
VALID_CATEGORIES = [
    "web_server",
    "database",
    "ssh_server",
    "mail_server",
    "ftp_server",
    "other"
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the banner classification dataset from CSV.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with columns: banner_text, category, source_ip, port
    """
    try:
        # pandas >= 1.3.0 uses on_bad_lines, older versions use error_bad_lines
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except TypeError:
            # Fallback for pandas < 1.3.0 (deprecated but kept for compatibility)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False, warn_bad_lines=False)
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def validate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean category labels.
    
    Args:
        df: DataFrame with category column
        
    Returns:
        DataFrame with validated categories
    """
    # Normalize category names (strip whitespace, lowercase)
    df['category'] = df['category'].str.strip().str.lower()
    
    # Map any variations to standard categories
    category_mapping = {
        'web': 'web_server',
        'database': 'database',
        'ssh': 'ssh_server',
        'mail': 'mail_server',
        'ftp': 'ftp_server',
        'other': 'other'
    }
    
    # Replace invalid categories with 'other'
    valid_mask = df['category'].isin(VALID_CATEGORIES)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} samples with invalid categories, mapping to 'other'")
        df.loc[~valid_mask, 'category'] = 'other'
    
    return df


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Get the distribution of classes in the dataset."""
    return df['category'].value_counts()


def prepare_data_for_training(
    df: pd.DataFrame,
    text_column: str = 'banner_text',
    label_column: str = 'category'
) -> Tuple[List[str], List[str]]:
    """
    Prepare data for training/evaluation.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Tuple of (texts, labels)
    """
    df = validate_categories(df.copy())
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(str).tolist()
    
    return texts, labels

