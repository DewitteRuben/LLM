import os
import bs4
from typing import List

# Langchain and Hugging Face imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFacePipeline
from langchain import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# Enable MPS fallback for Mac compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    

def main():
    # 1. Initialize Embeddings
    # Use a pre-trained sentence transformer model for creating embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # 2. Create Vector Store
    # In-memory vector store to store document embeddings
    vector_store = InMemoryVectorStore(embeddings)
        
    system_prompt = """You are an expert assistant that answers questions about machine learning and Large Language Models (LLMs). 
    You are given some extracted parts from machine learning papers along with a question. 
    If you don't know the answer, just say "I don't know." Don't try to make up an answer. 
    It is very important that you ALWAYS answer the question in the same language the question is in. 
    Use only the following pieces of context to answer the question at the end.
    """
    
    user_prompt = """
    Context: {context}
    Question is below. Remember to answer in the same language"
    Question: {question}"""
    
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", user_prompt),
    ])
    
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # 5. Split Documents into Chunks
    # Use RecursiveCharacterTextSplitter to create manageable document chunks
    max_seq_length_embeddings = embeddings._client.max_seq_length
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_seq_length_embeddings, 
        chunk_overlap=int(max_seq_length_embeddings/10)
    )

    all_splits = text_splitter.split_documents(docs)

    # 6. Add Document Chunks to Vector Store
    vector_store.add_documents(documents=all_splits)
    
    llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
        ),
    )
    
    llm_engine_hf = ChatHuggingFace(llm=llm)

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm_engine_hf.invoke(messages)
        return {"answer": response.content}
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is Task Decomposition?"})
    print(response["answer"])

if __name__ == "__main__":
    main()