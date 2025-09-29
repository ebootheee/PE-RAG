#!/usr/bin/env python3

import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_namespace():
    """Clear the namespace in Pinecone"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('tvp-legal')
    
    print("Clearing tvp-2 namespace...")
    index.delete(delete_all=True, namespace='tvp-2')
    print("Namespace cleared successfully!")

if __name__ == "__main__":
    clear_namespace()