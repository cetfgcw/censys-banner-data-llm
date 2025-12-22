"""Tests for data loading utilities."""

import pytest
import pandas as pd
from src.data_loader import (
    load_dataset,
    validate_categories,
    get_class_distribution,
    prepare_data_for_training,
    VALID_CATEGORIES
)


def test_validate_categories():
    """Test category validation."""
    df = pd.DataFrame({
        "category": ["web_server", "ssh_server", "invalid", "  WEB_SERVER  "],
        "banner_text": ["test"] * 4
    })
    
    validated = validate_categories(df)
    
    # Should normalize
    assert validated["category"].iloc[3] == "web_server"
    # Invalid should become "other"
    assert validated["category"].iloc[2] == "other"


def test_get_class_distribution():
    """Test class distribution calculation."""
    df = pd.DataFrame({
        "category": ["web_server", "ssh_server", "web_server"],
        "banner_text": ["test"] * 3
    })
    
    dist = get_class_distribution(df)
    assert dist["web_server"] == 2
    assert dist["ssh_server"] == 1


def test_prepare_data_for_training():
    """Test data preparation."""
    df = pd.DataFrame({
        "category": ["web_server", "ssh_server"],
        "banner_text": ["banner1", "banner2"]
    })
    
    texts, labels = prepare_data_for_training(df)
    
    assert len(texts) == 2
    assert len(labels) == 2
    assert texts[0] == "banner1"
    assert labels[0] == "web_server"

