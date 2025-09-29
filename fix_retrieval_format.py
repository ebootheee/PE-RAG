#!/usr/bin/env python3
"""
Fix Retrieval Format Script
This script helps convert Pinecone retrieval results to the format expected by n8n workflows.

The issue: Pinecone returns vectors with metadata, but n8n expects a pageContent field at the top level.
This script shows how to transform the data properly.
"""

import json
from typing import List, Dict, Any

def transform_pinecone_results_to_langchain_format(pinecone_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform Pinecone retrieval results to LangChain Document format expected by n8n
    
    Input format (what Pinecone returns):
    {
        "id": "chunk_id",
        "score": 0.123,
        "metadata": {
            "page_content": "actual text content...",
            "source_file": "document.pdf",
            ...other metadata...
        }
    }
    
    Output format (what n8n expects):
    {
        "pageContent": "actual text content...",
        "metadata": {
            "source_file": "document.pdf",
            ...other metadata WITHOUT page_content...
        },
        "id": "chunk_id"
    }
    """
    transformed_results = []
    
    for result in pinecone_results:
        # Extract the text content from metadata
        metadata = result.get('metadata', {})
        
        # Get text content from various possible field names
        page_content = (
            metadata.get('page_content', '') or
            metadata.get('text', '') or
            metadata.get('content', '') or
            ''
        )
        
        # Create new metadata without the text content
        clean_metadata = {k: v for k, v in metadata.items() 
                         if k not in ['page_content', 'text', 'content']}
        
        # Create the transformed result
        transformed_result = {
            'pageContent': page_content,
            'metadata': clean_metadata,
            'id': result.get('id', '')
        }
        
        # Include score if available
        if 'score' in result:
            transformed_result['score'] = result['score']
            
        transformed_results.append(transformed_result)
    
    return transformed_results

def example_usage():
    """Example of how to use the transformation function"""
    
    # Example input (what you're currently getting from Pinecone)
    pinecone_input = [
        {
            "pageContent": "",
            "metadata": {
                "chunk_index": 21,
                "document_type": "legal_document",
                "file_size": 417423,
                "fund_name": "GreenPoint TVP",
                "is_draft": False,
                "is_executed": False,
                "page_content": "Management Fee:\nDuring the Commitment Period, the TVP Partnership will pay the Manager\nan annual management fee...",
                "priority": "med",
                "processing_date": "2025-09-29T10:17:24.564782",
                "source_file": "2_MED_DOC_137873384-v5 and GreenPoint TVP Partnership,.pdf",
                "total_chunks": 50
            },
            "id": "53195cba_0021"
        }
    ]
    
    # Transform the data
    transformed = transform_pinecone_results_to_langchain_format(pinecone_input)
    
    print("Original format:")
    print(json.dumps(pinecone_input[0], indent=2))
    print("\nTransformed format:")
    print(json.dumps(transformed[0], indent=2))
    
    return transformed

if __name__ == "__main__":
    example_usage()