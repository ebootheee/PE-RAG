#!/usr/bin/env python3
"""
Test Pinecone Retrieval - Debug Script
This script helps you understand what's happening when you query Pinecone
and how the text content should be retrieved.
"""

import os
from pinecone import Pinecone
from openai import OpenAI
import json

def test_pinecone_retrieval():
    """Test what happens when we query Pinecone and how to get text content back"""
    
    # Initialize clients
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Connect to index
    index = pc.Index("tvp-legal")
    
    # Create a test query
    test_query = "What is the management fee structure?"
    
    print(f"🔍 Testing query: '{test_query}'")
    print("=" * 80)
    
    # Get embedding for the query
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=[test_query],
        dimensions=3072
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        namespace="tvp-2"
    )
    
    print(f"Found {len(results.matches)} results:")
    print()
    
    for i, match in enumerate(results.matches):
        print(f"Result {i+1}:")
        print(f"  ID: {match.id}")
        print(f"  Score: {match.score:.4f}")
        print(f"  Metadata keys: {list(match.metadata.keys()) if match.metadata else 'None'}")
        
        # Check for text content in different possible locations
        text_content = ""
        if match.metadata:
            text_content = (
                match.metadata.get('page_content', '') or
                match.metadata.get('text', '') or
                match.metadata.get('content', '')
            )
        
        print(f"  Text content found: {'Yes' if text_content else 'No'}")
        if text_content:
            print(f"  Text preview: {text_content[:200]}...")
        else:
            print("  ❌ No text content found in metadata!")
            
        print(f"  Source file: {match.metadata.get('source_file', 'Unknown') if match.metadata else 'Unknown'}")
        print()
    
    # Show what the raw response looks like
    print("Raw Pinecone Response Structure:")
    print("=" * 40)
    if results.matches:
        sample_match = results.matches[0]
        sample_dict = {
            'id': sample_match.id,
            'score': sample_match.score,
            'metadata': dict(sample_match.metadata) if sample_match.metadata else {}
        }
        print(json.dumps(sample_dict, indent=2, default=str))
    
    # Show what format n8n might expect
    print("\nFormat that n8n expects:")
    print("=" * 40)
    if results.matches and results.matches[0].metadata:
        expected_format = {
            'pageContent': results.matches[0].metadata.get('page_content', ''),
            'metadata': {k: v for k, v in results.matches[0].metadata.items() 
                        if k not in ['page_content', 'text', 'content']},
            'id': results.matches[0].id
        }
        print(json.dumps(expected_format, indent=2, default=str))

if __name__ == "__main__":
    test_pinecone_retrieval()