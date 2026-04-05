#!/usr/bin/env python3
"""
Test script to verify embedder imports work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.embedder import Embedder
    print("SUCCESS: Embedder imported correctly")
    
    # Test instantiation
    embedder = Embedder()
    print("SUCCESS: Embedder instantiated correctly")
    
    # Test encoding
    test_embedding = embedder.encode_single("Hello world")
    print(f"SUCCESS: Encoding works, shape: {test_embedding.shape}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()