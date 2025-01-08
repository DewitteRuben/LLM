import faiss
from sentence_transformers import SentenceTransformer

with open('data/cat-wiki.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Split on periods.
sentences = text.split('.')

# Strip whitespace and remove empty entries.
sentences = [s.strip() for s in sentences if s.strip()]

sentences = [word for word in list(set(sentences)) if type(word) is str]

# initialize sentence transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
# create sentence embeddings
sentence_embeddings = model.encode(sentences)

# get dimension of vector databse
d = sentence_embeddings.shape[1]

nlist = 10  # how many cells
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

index.train(sentence_embeddings)

# add embeddings to index
index.add(sentence_embeddings)

# We can change the number of nearby cells to search too with `nprobe`.
index.nprobe = 5

# encode query with the same embeddings
k = 4
xq = model.encode(["fur and teeth"])

D, I = index.search(xq, k)  # search
print(I)

# use the found indices to return the data
results = [f'{i}: {sentences[i]}' for i in I[0]]

print(results)

