#!/usr/bin/env python3
"""
Pinecone Retrieval Wrapper for n8n
This script provides a properly formatted response for n8n workflows
"""

import os
from pinecone import Pinecone
from openai import OpenAI
import json
from typing import List, Dict, Any

def query_pinecone_for_n8n(query_text: str, top_k: int = 5, namespace: str = "legal-docs") -> List[Dict[str, Any]]:
    """
    Query Pinecone and return results in the format expected by n8n/LangChain
    """
    # Initialize clients
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Connect to index
    index = pc.Index("legal-documents")
    
    # Get embedding for the query
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=[query_text],
        dimensions=3072
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    
    # Transform results to n8n/LangChain format
    formatted_results = []
    for match in results.matches:
        if not match.metadata:
            continue
            
        # Extract text content from metadata
        page_content = (
            match.metadata.get('page_content', '') or
            match.metadata.get('text', '') or
            match.metadata.get('content', '') or
            ''
        )
        
        # Create clean metadata without text content
        clean_metadata = {k: v for k, v in match.metadata.items() 
                         if k not in ['page_content', 'text', 'content']}
        
        # Format result as expected by n8n
        formatted_result = {
            'pageContent': page_content,
            'metadata': clean_metadata,
            'id': match.id,
            'score': float(match.score)
        }
        
        formatted_results.append(formatted_result)
    
    return formatted_results

if __name__ == "__main__":
    # Test the function directly
    print("Testing Pinecone query wrapper...")
    results = query_pinecone_for_n8n("What is the management fee structure?", 3)
    
    print(f"Found {len(results)} results")
    print("\nFirst result:")
    print(json.dumps(results[0] if results else {}, indent=2))
    
    print(f"\nComparison:")
    print("✅ pageContent field has content:", bool(results[0]['pageContent']) if results else False)
    print("✅ metadata.page_content removed:", 'page_content' not in results[0]['metadata'] if results else False)