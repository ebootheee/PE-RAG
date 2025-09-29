#!/usr/bin/env python3
"""
Verify Test Vectors
Check if the test vectors exist and their structure
"""

import os
from pinecone import Pinecone
import json

def verify_test_vectors():
    """Check the test vectors we just uploaded"""
    
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("tvp-legal")
    namespace = "tvp-2-test"
    
    print("🔍 Checking test namespace...")
    
    # Get index stats
    stats = index.describe_index_stats()
    test_namespace_stats = stats.namespaces.get(namespace, {})
    
    print(f"Namespace '{namespace}' stats:")
    print(f"  Vector count: {test_namespace_stats.get('vector_count', 0)}")
    
    if test_namespace_stats.get('vector_count', 0) > 0:
        print("\n📋 Fetching test vectors by ID...")
        
        # Try to fetch the specific vectors we uploaded
        try:
            fetch_response = index.fetch(
                ids=['test_vector_1', 'test_vector_2'],
                namespace=namespace
            )
            
            for vector_id, vector_data in fetch_response.vectors.items():
                print(f"\nVector ID: {vector_id}")
                print(f"Metadata keys: {list(vector_data.metadata.keys()) if vector_data.metadata else 'None'}")
                
                if vector_data.metadata:
                    text_content = vector_data.metadata.get('text', '')
                    print(f"'text' field: {'✅ Found' if text_content else '❌ Empty'} ({len(text_content)} chars)")
                    if text_content:
                        print(f"Content: {text_content}")
                    
                    # Show what n8n would see
                    print("\nWhat n8n should receive:")
                    langchain_format = {
                        'pageContent': text_content,
                        'metadata': {k: v for k, v in vector_data.metadata.items() if k != 'text'},
                        'id': vector_id
                    }
                    print(json.dumps(langchain_format, indent=2))
        
        except Exception as e:
            print(f"Error fetching vectors: {e}")
            
        # Try a simple query with zero vector
        print("\n🔍 Testing with zero vector query...")
        try:
            results = index.query(
                vector=[0.0] * 3072,
                top_k=5,
                include_metadata=True,
                namespace=namespace
            )
            
            print(f"Found {len(results.matches)} vectors in test namespace")
            for match in results.matches:
                if match.metadata:
                    text_content = match.metadata.get('text', '')
                    print(f"  {match.id}: text field has {len(text_content)} chars")
        
        except Exception as e:
            print(f"Error querying: {e}")
    
    else:
        print("❌ No vectors found in test namespace")

if __name__ == "__main__":
    verify_test_vectors()