#!/usr/bin/env python3

import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add credential path
os.environ['PINECONE_CREDENTIAL_PATH'] = os.path.expanduser('~/.pinecone')

def check_vector_content():
    """Check what's actually stored in Pinecone vectors"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('tvp-legal')
    
    print("Checking vector content in tvp-2 namespace...")
    
    # Get some sample vectors
    results = index.query(
        vector=[0.1] * 3072,  # Dummy vector for text-embedding-3-large
        top_k=5,
        namespace='tvp-2',
        include_metadata=True
    )
    
    print(f"Found {len(results['matches'])} vectors")
    print("=" * 80)
    
    for i, match in enumerate(results['matches'], 1):
        print(f"Vector {i}:")
        print(f"  ID: {match['id']}")
        print(f"  Score: {match['score']:.4f}")
        print(f"  Metadata keys: {list(match['metadata'].keys())}")
        
        # Check if text content is stored in metadata
        text_content = match['metadata'].get('text', '')
        page_content = match['metadata'].get('page_content', '')
        content = match['metadata'].get('content', '')
        
        print(f"  Text field length: {len(text_content)}")
        print(f"  Page_content field length: {len(page_content)}")
        print(f"  Content field length: {len(content)}")
        
        # Show first 200 chars if any content exists
        if text_content:
            print(f"  Text preview: {text_content[:200]}...")
        elif page_content:
            print(f"  Page content preview: {page_content[:200]}...")
        elif content:
            print(f"  Content preview: {content[:200]}...")
        else:
            print("  ⚠️  NO TEXT CONTENT FOUND!")
            
        print(f"  Source file: {match['metadata'].get('source_file', 'unknown')}")
        print(f"  Chunk index: {match['metadata'].get('chunk_index', 'unknown')}")
        print("-" * 40)
    
    # Also check namespace stats
    try:
        stats = index.describe_index_stats()
        print(f"\nNamespace stats:")
        for namespace, info in stats['namespaces'].items():
            print(f"  {namespace}: {info['vector_count']} vectors")
    except Exception as e:
        print(f"Error getting stats: {e}")

if __name__ == "__main__":
    check_vector_content()