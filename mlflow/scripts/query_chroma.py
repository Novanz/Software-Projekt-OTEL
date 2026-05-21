import chromadb
from pprint import pprint

CHROMA_HOST = "192.168.56.103"   # change if needed
CHROMA_PORT = 8008
COLLECTION_NAME = "rag_docs"

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = client.get_collection(name=COLLECTION_NAME)

results = collection.query(
    query_texts=["how does mlflow tracing work"],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

pprint(results)
