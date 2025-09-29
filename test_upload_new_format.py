#!/usr/bin/env python3
"""
Test Upload with New Format
Upload a few test vectors using the 'text' field to verify n8n integration works
"""

import os
from pinecone import Pinecone
from openai import OpenAI
import json

def create_test_vectors():
    """Create a few test vectors with text content in 'text' field"""
    
    # Initialize clients
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Connect to index
    index = pc.Index("tvp-legal")
    namespace = "tvp-2-test"  # Use a test namespace
    
    # Test content
    test_documents = [
        {
            'content': 'This is a test document about management fees. The management fee is 2% annually.',
            'metadata': {
                'source_file': 'test_doc_1.pdf',
                'document_type': 'test_document',
                'chunk_index': 1,
                'total_chunks': 1
            }
        },
        {
            'content': 'This is another test document discussing partnership agreements and investment terms.',
            'metadata': {
                'source_file': 'test_doc_2.pdf', 
                'document_type': 'test_document',
                'chunk_index': 1,
                'total_chunks': 1
            }
        }
    ]
    
    print("🚀 Creating test vectors with 'text' field...")
    
    # Create embeddings and vectors
    vectors_to_upload = []
    
    for i, doc in enumerate(test_documents):
        # Get embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=[doc['content']],
            dimensions=3072
        )
        
        # Create vector in new format
        vector = {
            'id': f'test_vector_{i+1}',
            'values': response.data[0].embedding,
            'metadata': {
                'text': doc['content'],  # Using 'text' field instead of 'page_content'
                'source_file': doc['metadata']['source_file'],
                'document_type': doc['metadata']['document_type'],
                'chunk_index': doc['metadata']['chunk_index'],
                'total_chunks': doc['metadata']['total_chunks'],
                'priority': 'test',
                'fund_name': 'Test Fund',
                'is_draft': False,
                'is_executed': True,
                'file_size': 1024,
                'processing_date': '2025-09-29T12:00:00.000000'
            }
        }
        
        vectors_to_upload.append(vector)
    
    # Upload to Pinecone
    index.upsert(
        vectors=vectors_to_upload,
        namespace=namespace
    )
    
    print(f"✅ Uploaded {len(vectors_to_upload)} test vectors to namespace '{namespace}'")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    
    test_query = "management fee"
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=[test_query],
        dimensions=3072
    )
    
    results = index.query(
        vector=response.data[0].embedding,
        top_k=2,
        include_metadata=True,
        namespace=namespace
    )
    
    print(f"Query: '{test_query}'")
    print(f"Found {len(results.matches)} results:")
    
    for i, match in enumerate(results.matches):
        print(f"\nResult {i+1}:")
        print(f"  ID: {match.id}")
        print(f"  Score: {match.score:.4f}")
        
        if match.metadata:
            text_content = match.metadata.get('text', '')
            print(f"  'text' field: {'✅ Found' if text_content else '❌ Empty'} ({len(text_content)} chars)")
            if text_content:
                print(f"  Content: {text_content}")
    
    # Show what n8n should receive
    print(f"\n📋 What n8n should receive (simulated LangChain format):")
    if results.matches:
        match = results.matches[0]
        if match.metadata and 'text' in match.metadata:
            langchain_format = {
                'pageContent': match.metadata['text'],
                'metadata': {k: v for k, v in match.metadata.items() if k != 'text'},
                'id': match.id,
                'score': float(match.score)
            }
            print(json.dumps(langchain_format, indent=2))
    
    print(f"\n🎯 Instructions for testing in n8n:")
    print(f"1. Use namespace '{namespace}' in your Pinecone query")
    print(f"2. Query for 'management fee' or 'partnership agreement'")
    print(f"3. The pageContent field should now contain the actual text content")
    print(f"4. If it works, you can re-upload all vectors with the new format")

if __name__ == "__main__":
    create_test_vectors()