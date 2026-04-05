#!/usr/bin/env python3
"""
Test script to verify retriever imports work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.retriever import Retriever, create_sample_data
    print("SUCCESS: Retriever imported correctly")
    
    # Test instantiation
    retriever = Retriever()
    print("SUCCESS: Retriever instantiated correctly")
    
    # Test with sample data
    embeddings, ids = create_sample_data(num_vectors=10, dim=384)
    print(f"SUCCESS: Sample data created, shape: {embeddings.shape}")
    
    # Test adding embeddings
    added_ids = retriever.add_embeddings(embeddings, ids)
    print(f"SUCCESS: Added embeddings, count: {len(added_ids)}")
    
    # Test search
    query_emb, _ = create_sample_data(num_vectors=2, dim=384)
    distances, indices, mapped_ids, search_time = retriever.search(query_emb, k=3)
    print(f"SUCCESS: Search completed, results shape: {distances.shape}, time: {search_time:.4f}s")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()