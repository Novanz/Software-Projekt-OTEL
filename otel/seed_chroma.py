import chromadb

CHROMA_HOST = "192.168.56.103"
CHROMA_PORT = 8008
COLLECTION_NAME = "rag_docs"

DOCS = [
    {
        "id": "doc-1",
        "title": "MLflow tracing overview",
        "text": "MLflow tracing can capture inputs, outputs, latency, and nested execution steps."
    },
    {
        "id": "doc-2",
        "title": "Ollama local inference",
        "text": "Ollama serves local LLM inference over an HTTP API and works well for prototype setups."
    },
    {
        "id": "doc-3",
        "title": "RAG basics",
        "text": "Retrieval-augmented generation combines retrieved context with a model prompt to improve answers."
    },
    {
        "id": "doc-4",
        "title": "ChromaDB note",
        "text": "A vector database like ChromaDB can later replace mock retrieval with embedding-based search."
    },
]

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

collection.add(
    ids=[d["id"] for d in DOCS],
    documents=[f"{d['title']}\n{d['text']}" for d in DOCS],
    metadatas=[{"title": d["title"], "source": "rag_moc_v0"} for d in DOCS],
)

print(collection.count())
