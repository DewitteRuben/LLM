from chromadb import Client
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client with in-memory persistence (adjust as needed for file-based persistence)
client = Client()

# Create or get a collection
collection = client.get_or_create_collection(name="cat_collection")

# Load text data
with open('data/cat-wiki.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Split on periods and clean sentences
sentences = text.split('.')
sentences = [s.strip() for s in sentences if s.strip()]
sentences = list(set(sentences))  # Remove duplicates

# Initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Create sentence embeddings
sentence_embeddings = model.encode(sentences)

# Add sentences and embeddings to the collection
collection.add(
    embeddings=sentence_embeddings.tolist(),
    documents=sentences,
    metadatas=[{"index": i} for i in range(len(sentences))],
    ids=[f"sentence_{i}" for i in range(len(sentences))]
)

# Query the collection
query = "fur and teeth"
query_embedding = model.encode([query]).tolist()

# Perform the search
results = collection.query(
    query_embeddings=query_embedding,
    n_results=4
)

# Print results
for idx, document in enumerate(results["documents"][0]):
    print(f"Result {idx + 1}: {document}")
