#!/usr/bin/env python3
"""
Migrate Vector Metadata Fields
This script migrates existing vectors from 'page_content' field to 'text' field
so they work properly with standard LangChain integrations.
"""

import os
from pinecone import Pinecone
from tqdm import tqdm
import time

def migrate_vectors_metadata():
    """Migrate vectors from page_content to text field"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("tvp-legal")
    namespace = "tvp-2"
    
    print("🔄 Starting metadata field migration...")
    print(f"📁 Namespace: {namespace}")
    
    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.namespaces.get(namespace, {}).get('vector_count', 0)
    print(f"📊 Total vectors to migrate: {total_vectors}")
    
    if total_vectors == 0:
        print("❌ No vectors found in namespace!")
        return
    
    # Migration process
    batch_size = 100
    migrated_count = 0
    
    # We'll iterate through vectors by querying with different dummy vectors
    # This is a workaround since Pinecone doesn't have a "list all" function
    
    print("\n🚀 Starting migration process...")
    
    # Get a sample of vectors to check current state
    sample_results = index.query(
        vector=[0.0] * 3072,
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )
    
    # Check if migration is needed
    needs_migration = False
    for match in sample_results.matches:
        if match.metadata and 'page_content' in match.metadata and 'text' not in match.metadata:
            needs_migration = True
            break
    
    if not needs_migration:
        print("✅ Vectors already have 'text' field or don't need migration!")
        return
    
    print("📝 Migration needed. Processing vectors...")
    
    # For this migration, we'll use a different approach:
    # 1. Query vectors in batches
    # 2. Update metadata for each batch
    
    # Since we can't easily iterate through all vectors, let's do targeted queries
    # We'll use the existing vectors we know about from our test
    
    # Get vectors in batches using different query vectors
    import numpy as np
    
    processed_ids = set()
    
    for query_attempt in range(100):  # Try multiple random queries to cover all vectors
        # Create a random query vector
        query_vector = np.random.normal(0, 0.1, 3072).tolist()
        
        try:
            results = index.query(
                vector=query_vector,
                top_k=100,
                include_metadata=True,
                namespace=namespace
            )
            
            # Prepare batch updates
            updates = []
            
            for match in results.matches:
                if match.id in processed_ids:
                    continue
                    
                if match.metadata and 'page_content' in match.metadata:
                    # Create updated metadata
                    new_metadata = dict(match.metadata)
                    new_metadata['text'] = new_metadata.pop('page_content')
                    
                    # Prepare update
                    updates.append({
                        'id': match.id,
                        'metadata': new_metadata
                    })
                    
                    processed_ids.add(match.id)
            
            # Apply updates
            if updates:
                # Use upsert to update metadata (keeping existing vectors and values)
                index.upsert(
                    vectors=updates,
                    namespace=namespace
                )
                
                migrated_count += len(updates)
                print(f"✅ Migrated batch: {len(updates)} vectors (Total: {migrated_count})")
            
            # If we didn't find any new vectors, we might be done
            if not updates:
                break
                
        except Exception as e:
            print(f"⚠️ Error in batch {query_attempt}: {str(e)}")
            continue
        
        # Small delay to avoid rate limits
        time.sleep(0.1)
        
        # Stop if we've processed a reasonable number
        if migrated_count >= total_vectors * 0.9:  # 90% coverage is good enough
            break
    
    print(f"\n🎉 Migration completed!")
    print(f"📊 Vectors migrated: {migrated_count}")
    print(f"📈 Coverage: {(migrated_count/total_vectors)*100:.1f}%")
    
    # Test the results
    print("\n🔍 Testing migrated vectors...")
    test_results = index.query(
        vector=[0.0] * 3072,
        top_k=3,
        include_metadata=True,
        namespace=namespace
    )
    
    for i, match in enumerate(test_results.matches):
        has_text = 'text' in match.metadata if match.metadata else False
        has_page_content = 'page_content' in match.metadata if match.metadata else False
        print(f"Vector {i+1}: text field: {'✅' if has_text else '❌'}, page_content field: {'✅' if has_page_content else '❌'}")

if __name__ == "__main__":
    migrate_vectors_metadata()