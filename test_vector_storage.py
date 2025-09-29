#!/usr/bin/env python3
"""
Test Vector Storage - Verify Text Field
This script tests if the text content is now stored properly in the 'text' field
and will be mapped to pageContent by LangChain/n8n integrations.
"""

import os
from pinecone import Pinecone
from openai import OpenAI
import json

def test_vector_storage():
    """Test what happens when we query vectors with the new 'text' field"""
    
    # Initialize clients
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Connect to index
    index = pc.Index("tvp-legal")
    
    # First, let's check what fields exist in current vectors
    print("🔍 Checking existing vector structure...")
    print("=" * 60)
    
    # Get a few random vectors to inspect
    results = index.query(
        vector=[0.0] * 3072,  # Zero vector to get random results
        top_k=3,
        include_metadata=True,
        namespace="tvp-2"
    )
    
    if results.matches:
        sample_vector = results.matches[0]
        print(f"Sample Vector ID: {sample_vector.id}")
        print(f"Metadata keys: {list(sample_vector.metadata.keys()) if sample_vector.metadata else 'None'}")
        
        if sample_vector.metadata:
            # Check which text fields exist
            text_fields = []
            for key in ['text', 'page_content', 'content']:
                if key in sample_vector.metadata:
                    content = sample_vector.metadata[key]
                    text_fields.append(f"{key}: {len(content)} chars")
            
            print(f"Text fields found: {text_fields}")
            
            # Show the structure expected by n8n/LangChain
            print("\nWhat n8n should receive with 'text' field:")
            print("-" * 40)
            mock_langchain_result = {
                'pageContent': sample_vector.metadata.get('text', ''),
                'metadata': {k: v for k, v in sample_vector.metadata.items() if k != 'text'},
                'id': sample_vector.id
            }
            print(json.dumps(mock_langchain_result, indent=2, default=str)[:500] + "...")
    
    print("\n" + "=" * 60)
    print("Now testing a real query...")
    
    # Create a test query
    test_query = "What is the management fee structure?"
    
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
    
    print(f"Query: '{test_query}'")
    print(f"Found {len(results.matches)} results:")
    print()
    
    for i, match in enumerate(results.matches):
        print(f"Result {i+1}:")
        print(f"  ID: {match.id}")
        print(f"  Score: {match.score:.4f}")
        
        if match.metadata:
            # Check for text content in the expected field
            text_content = match.metadata.get('text', '')
            page_content = match.metadata.get('page_content', '')
            
            print(f"  'text' field: {'✅ Found' if text_content else '❌ Empty'} ({len(text_content)} chars)")
            print(f"  'page_content' field: {'✅ Found' if page_content else '❌ Empty'} ({len(page_content)} chars)")
            
            if text_content:
                print(f"  Text preview: {text_content[:150]}...")
            elif page_content:
                print(f"  Page content preview: {page_content[:150]}...")
            else:
                print("  ❌ No text content found!")
                
        print()

if __name__ == "__main__":
    test_vector_storage()