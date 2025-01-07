import os
import faiss
from typing import List

# Langchain and Hugging Face imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from uuid import uuid4
# Transformers imports
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Enable MPS fallback for Mac compatibility
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def main():
    with open('FAISS/data/cat-wiki.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = text.split('.')

    sentences = [s.strip() for s in sentences if s.strip()]

    sentences = [word for word in list(set(sentences)) if type(word) is str]

    documents: List[dict] = [
        Document(
            page_content=sentence,
        )
        for sentence in sentences
    ]
    
    prompt_template = """
    <|system|>
    You are an expert assistant that answers questions about cats.
    You are given some extracted parts from a cat wikipedia article along with a question.
    If you don't know the answer, just say "I don't know." Don't try to make up an answer.
    It is very important that you ALWAYS answer the question in the same language the question is in.
    Use only the following pieces of context to answer the question at the end.
    <|end|>

    <|user|>
    Context: {context}
    Question is below. Remember to answer in the same language:
    Question: {question}
    <|end|>
    <|assistant|>
    """

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Extract embedding dimension from sample query
    # sample_embedding = embeddings.embed_query("sample text")
    # embedding_dimension = len(sample_embedding)

    # index = faiss.IndexFlatL2(embedding_dimension)
    
    vector_store = FAISS.from_texts(sentences, embeddings)
    
    # vector_store = FAISS(
    #     embedding_function=embeddings,
    #     index=index,
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )

    
    # uuids = [str(uuid4()) for _ in range(len(documents))]
    
    # print(len(documents))

    vector_store.add_documents(documents=documents)
  
    llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512
        ),
    )

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

if __name__ == "__main__":
    main()