from pinecone import Pinecone
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class VectorDatabase:
    def __init__(self, index_name):
        api_key = os.getenv('PINECONE_API_KEY')
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)

    def upsert(self, vectors, namespace):
        return self.index.upsert(vectors=vectors, namespace=namespace)

    def query(self, vector, top_k, include_metadata, namespace):
        logging.info("Querying Pinecone with vector: %s", vector)
        response = self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata, namespace=namespace)
        logging.info("Received response: %s", response)
        return response

