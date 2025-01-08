
## FAISS with LangChain and Hugging Face: Building an AI Cat Expert

We’ll combine FAISS for vector search, LangChain for managing AI workflows, and Hugging Face for natural language processing. By the end, you’ll understand how these tools work together to retrieve and generate intelligent responses based on a Wikipedia article.

## Getting started

```
pip install -U langchain_community langchain_core langchain_huggingface faiss-cpu
```

## Code Example

In this example, we’ll:

- Extract sentences from a Wikipedia article on cats.
- Use FAISS to index these sentences as vectors.
- Build a pipeline that retrieves relevant sentences and answers questions using a Hugging Face model.

### 1. Read and Prepare the Data
We begin by loading and preprocessing a text file:

```python
with open('FAISS/data/cat-wiki.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Split the text into sentences and clean up
sentences = text.split('.')
sentences = [s.strip() for s in sentences if s.strip()]
sentences = [word for word in list(set(sentences)) if type(word) is str]
```

This ensures our data is clean, unique, and ready for indexing.

### 2. Create FAISS Index

Next, we create a FAISS index and add the preprocessed sentences:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Generate embeddings for the sentences
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create and populate the FAISS index
vector_store = FAISS.from_texts(sentences, embeddings)
```

### 3. Define the Assistant’s Behavior

```python
from langchain import PromptTemplate

prompt_template = """
<|system|>
You are an expert assistant that answers questions about cats.
...
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
```

### 4. Add a HuggingFace Model

We use a Hugging Face language model to generate answers:

```python
from langchain_huggingface.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs=dict(max_new_tokens=512)
)
```

### 5. Build the Question-Answering Pipeline
Finally, we create a pipeline that connects FAISS, the prompt template, and the language model:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

question = "Tell me about fur"
qa_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = qa_chain.invoke(question)
print(answer)
```
