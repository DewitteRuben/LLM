# Introduction

Large Language Models (LLMs) have revolutionized natural language processing (NLP), enabling applications ranging from text generation to complex problem-solving. While there are many LLMs to choose from, they share common underlying principles.

An LLM on its own is essentially a text generator with. It generates text by predicting the next sequence of words based on the input (prompt) it receives. Its output is determined by patterns and relationships learned from the datasets during training.

This raises some questions, such as:

- How can I integrate an LLM into a system that performs tasks beyond text generation?
- How do I choose the right model for my use case?
- How can I enable the AI to process large amounts of documents and files effectively?
- ...

## AI Models

For the sake of scope, we will focus exclusively on LLMs, which specialize in NLP (Natural Language Processing). Within AI, other domains like Computer Vision and Audio exist, but these fall outside the scope of this discussion.

### Popular General-Purpose Models

Selecting the right model is one of the most critical decisions. Each LLM offers unique advantages and trade-offs. Here are some of the most popular vendors and their general-purpose models:

- **OpenAI**: GPT-4, GPT-3.5, DALL-E 3
- **Anthropic**: Claude 3, Claude 2, Claude Instant
- **Azure**: Phi-1.5, Mistral-7B-V01
- **Google**: Gemini, Bard
- **AWS**: Amazon Titan, Amazon Bedrock
- **Cohere**: Command R+, Command
- **NVIDIA**: Nemotron
- **Groq**: Language Processing Unit (LPU)

Many publicly available general-purpose LLM APIs charge based on token usage. For example, OpenAI’s GPT-4 costs $2.50 per 1 million input tokens. These costs can quickly add up, especially when processing large datasets or multiple documents through the model.

### Task specific models

The LLMs above are general-purpose models and pretty good at a lot of things. But for certain tasks, it might make sense to use a model that's specialized for that job. For example, Salesforce's CodeGen is a text-generation model that's trained specifically to be great at generating code.

### Fine-tuning a pretrained model

While pre-trained models such as the ones mentioned above are often sufficient, fine-tuning allows developers to adapt these models for specific tasks, think of things like summarizing legal documents or analyzing scientific literature.

Fine-tuning combines the advantages of pre-trained general-purpose models with custom capabilities, and it is far more resource-efficient than training a model from scratch.

### Training Architectures: MLM vs. CLM

There are 2 types of training architectures that are employed to train a language model. Each model training method having advantages for specific use cases.

#### Masked language modeling (MLM)

An MLM can predict a token (e.g., a word) that is masked (e.g., missing) in a sequence (e.g., a sentence). To do so, an MLM has access to tokens bidirectionally, meaning it is aware of the tokens on both the left and right sides of the sequence. MLMs excel at tasks requiring strong contextual understanding, such as text classification, sentiment analysis, and named entity recognition. While they are proficient at understanding text, they are not well-suited for generating new text.

In short: MLMs optimize for understanding (contextualizing data without necessarily generating new content).

#### Causal language modeling (CLM)

A CLM can predict a token (e.g., a word) in a sequence of tokens. To achieve this, the model can only access tokens on the left side of the token being predicted, meaning it lacks knowledge of future tokens. A good example of a CLM in practice is an intelligent coding assistant, such as Copilot.

Before the introduction of [BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) in 2018, CLMs were the gold standard for language modeling. While CLMs remain essential for generative tasks, MLMs are increasingly used for tasks requiring in-depth text understanding.

In short: CLMs optimize for generation (producing coherent outputs from given inputs).

### Open source and self-hosting models

For many individuals and companies, sending their data to a remote LLM hosted elsewhere can pose privacy and security concerns, especially for sensitive organizations like banks.

Fortunately, numerous open-source and publicly available models can be run directly on personal machines or virtual private servers.

Some popular open-source LLM models are [Llama from Meta](https://www.llama.com/) and [Qwen](https://huggingface.co/Qwen) from Alibaba cloud

#### Hugging Face

Hugging Face is a platform where people can share various types of (open-source) AI models, datasets as well as fine-tuned models based on existing ones. It is the largest hub for sharing AI models and related resources.

Major players in the industry, such as OpenAI, have also uploaded some of their older GPT models to Hugging Face under the MIT license. Hugging Face is free to use unless you want to use their cloud to host your AI model.

While not all models are open source and each typically comes with its own license, they are all available for download and can be used on your own machine.

Hugging Face also provides its own (as well as community-maintained) [transformers package](https://huggingface.co/docs/hub/transformers) in Python, enabling you to download various models—from object detection to text generation—directly into your application.

## Development using LangChain and Hugging Face

In essence, developing applications using LLMs involves providing a pre-trained model with a prompt in its required format, receiving the result, and then parsing the text output for use in a user-facing application.

An LLM on its own is limited to text generation based on the input it receives. Features loading data from websites, memory, and other advanced features such as those found in tools like ChatGPT—are not inherent to the LLM itself but must be built on top of it.

Managing these tasks manually can be cumbersome, especially since each LLM application may have its own specific requirements. Fortunately, LangChain simplifies this process.

LangChain provides tools and abstractions that allow developers to focus on the application's needs rather than dealing with API peculiarities, boilerplate code, or reinventing the wheel.

### Programming Languages

LangChain is available in Python and JavaScript. Python is more popular and has better support from other libraries like Hugging Face. For example, the Hugging Face pipeline library isn’t available in JavaScript, so you’re stuck using their Inference API, which means interacting with a model hosted remotely instead of running it on your own machine. Overall, JavaScript seems like the better option if all you need is to interface with a model that is running in the cloud.

### Model interfacing

LangChain can be used in two primary ways: either by interfacing directly with a local model or through an API provided by major vendors. LangChain provides abstractions for all the major vendors. For this we will make use of the `langchain-huggingface` package, which is a package that is maintained in collaboration with Hugging Face themselves.

Locally running a model with LangChain and invoking it is straightforward. For this demo, we will use the `microsoft/Phi-3-mini-4k-instruct` model, a lightweight model that runs efficiently on most desktop computers while still delivering impressive performance.

```python
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,  # Limits the number of tokens generated in response to input
        "top_k": 50,  # Reduces vocabulary size
        "temperature": 0.1,  # Controls randomness: higher values increase diversity
    },
)
llm.invoke("Hugging Face is")
```

LangChain also provides wrappers for the Hugging Face API, allowing you to run prompts on a dedicated endpoint hosted by Hugging Face. This can be achieved using the `HuggingFaceEndpoint` class. However, to keep this demo cost-free, we will focus on local usage.

### Prompts

It's very important to note that every model has it's own 'syntax' for prompting to the model. LLM models are VERY sensitive to you providing the correct prompt format. If you don't the model will perform significantly worse.

In a nutshell language models operate with three distinct roles for prompts:

1. **System**: Represents the core language model itself. In roleplaying terms, this corresponds to "Out of Character."
2. **Assistant**: The role or persona applied to the System, shaping its personality and behavior. In roleplaying, this is "In Character."
3. **User**: The human interacting with the model.

Prompt formats distinguish these roles, helping the System identify who is communicating. The initial message in any sequence is the System prompt, which sets the model's behavior. After that, interactions are with the Assistant role created by the System prompt.

You can either format your own prompt by looking up the tokens required for that prompt, or use the `Chat` abstractions provided by LangChain.

For Hugging Face you can wrap the the LLM with the `ChatHuggingFace` class, which when invoked, will automatically use the appropriate `ChatTemplate` for the defined `model_id`.

```python
from langchain_huggingface import ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        ),
    )

llm_engine_hf = ChatHuggingFace(llm=llm)
llm_engine_hf.invoke("Hugging Face is")  # Corresponds to: <|user|>Hugging Face is<|end|><|assistant|>
```

### Context window

Something to keep in mind while developing using LLMs, is LLMs have a limited context window size. Usually the bigger models do support a large context window, however, this might still not be enough depending on your application use case.

If you use case does require you to load loads of data within a given context, e.g. when loading a codebase of files into a project.

To combat this it is practical for your application to basically look up the appropriate context on the fly depending on the prompt that was given. This process is called RAG (Retrieval Augmented Generation)

Here's the corrected version of your text with errors and inaccuracies addressed, without rewriting unnecessarily:

### RAG

In short, RAG essentially means that we will look up the context we want to provide, together with the prompt to the LLM, using some sort of prepared storage (usually a vector store).

**Loading the data**

First, we need to load the data we require. In LangChain, this can be done using document loaders. There are many kinds of document loaders, such as CSV loaders, Markdown loaders, and more. The one we will use for this example is a website loader.

```python
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=["post-content", "post-title", "post-header"]
        )
    ),
)
docs = loader.load()
```

**Splitting the data**

After we've loaded the required data, we will split it into chunks. This is because they are easier to search over, and larger documents won't fit in the context window. For this, we will use the `RecursiveCharacterTextSplitter` class provided by LangChain.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)
```

This will split the data in equal chunks of the provided size, with also providing some overlap between chunks, so that no context is lost that might be stored in the previous or next chunk.

**Storing/embedding the data**

After splitting the data, we need to store it in a way that allows it to be easily searched later. For this, we will use a vector store.

An embeddings model is trained to understand context and capture semantic relationships and similarities between words (think of an MLM). This model will generate a vector representation of the input as an array, e.g., `[-0.1, 0.5, 0.8]`. Each embedding essentially represents a set of coordinates. These vectors are then stored in a vector store.

If we use the same embeddings model on our prompt or question, it will generate a set of vectors. We can then compare these vectors from the prompt with the vectors stored in the vector store using simple math. Essentially, we calculate the "distance" between these "coordinates" to determine the similarity or relationship of the words.

That way, we can easily find the chunk of context that matches the prompt that was given.

```python
# Use a pre-trained sentence transformer model for creating embeddings
from langchain.vectorstores import InMemoryVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embedding_model=embeddings)

# Add the chunks to the vector store
vector_store.add_documents(chunks)

# We can then run a similarity search using the prompt that was provided
retrieved_docs = vector_store.similarity_search("What is Task Decomposition?")
```

**Retrieval chain using LCEL**

After preparing our vector store with embedded documents, we can create a complete RAG pipeline using LangChain Expression Language (LCEL). LCEL provides a simple, intuitive way to chain together different components of our retrieval and generation workflow.

```python
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

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

qa_chain = qa_chain.invoke("What is Task Decomposition?")
```