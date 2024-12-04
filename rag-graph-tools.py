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
from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

# Enable MPS fallback for Mac compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
    
    graph_builder = StateGraph(MessagesState)

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information related to a query."""
        retrieved_docs = vector_store.similarity_search(query)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm_engine_hf.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    tools = ToolNode([retrieve])

    def generate(state: MessagesState):
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm_engine_hf.invoke(prompt)
        return {"messages": [response]}
    
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()
    
    input_message = "What is Task Decomposition?"

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()