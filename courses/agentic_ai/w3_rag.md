# Week 3: RAG (Retrieval-Augmented Generation)
[back](README.md)
## Student Reading Material

### Introduction

Welcome to Week 3 of the Agentic AI program. This week introduces Retrieval-Augmented Generation (RAG), a powerful technique that grounds AI responses in real data. RAG bridges the gap between large language models and factual accuracy, enabling agents to answer questions based on specific documents and knowledge bases rather than relying solely on training data.

### What is RAG and Why Do We Need It?

Large Language Models (LLMs) are powerful, but they have a critical limitation: they can "hallucinate" or generate plausible-sounding but incorrect information. This happens because LLMs generate responses based on patterns learned during training, not by consulting actual sources.

**The Problem:**
- LLMs don't have access to your private documents
- They can't cite specific sources
- Their knowledge has a cutoff date
- They may confidently state incorrect facts

**The Solution: RAG**

Retrieval-Augmented Generation combines two approaches:
1. **Retrieval**: Search for relevant information from a knowledge base
2. **Generation**: Use an LLM to formulate an answer based on the retrieved information

This grounds the AI's responses in actual documents, dramatically reducing hallucinations and enabling agents to work with domain-specific knowledge.

### Learning Outcomes

By the end of this week, you will be able to:
- Understand the RAG architecture and its components
- Load and process documents (PDFs, text files)
- Create text embeddings using transformer models
- Store and retrieve embeddings using vector databases
- Build a complete RAG question-answering system
- Evaluate RAG system accuracy and identify failure modes
- Create agent tools that leverage RAG for knowledge retrieval

---

## RAG Architecture

The RAG pipeline consists of six key stages:

### 1. Load Documents

Import documents from various sources (PDFs, text files, web pages, databases).

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("textbook.pdf")
documents = loader.load()
```

### 2. Chunk

Break documents into smaller pieces. LLMs have token limits, and smaller chunks improve retrieval precision.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

**Why chunking matters:**
- Enables precise retrieval of relevant sections
- Fits within LLM context windows
- Improves response quality by focusing on specific information

### 3. Embed

Convert text chunks into numerical vectors (embeddings) that capture semantic meaning.

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**What are embeddings?**

Embeddings transform text into high-dimensional vectors where semantically similar text has similar vector representations. This allows mathematical comparison of meaning.

For example:
- "dog" and "puppy" have similar embeddings
- "dog" and "car" have very different embeddings

### 4. Store

Save embeddings in a vector database for efficient similarity search.

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_vectorstore"
)
```

**Vector Store Options:**
- **Chroma**: Lightweight, easy to use, great for development
- **FAISS**: Fast, efficient, good for production
- **Pinecone**: Cloud-based, scalable
- **Weaviate**: Feature-rich, production-ready

### 5. Retrieve

Search the vector store for chunks most relevant to a user's question.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.get_relevant_documents("What is machine learning?")
```

This performs semantic search, finding documents that match the meaning of the query, not just keyword matches.

### 6. Generate

Pass the retrieved documents to an LLM along with the user's question to generate a grounded answer.

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "What is machine learning?"})
print(result["result"])
```

---

## Understanding Embeddings

Embeddings are the foundation of RAG systems. They enable semantic search by representing text as vectors in a high-dimensional space.

### How Embeddings Work

Traditional keyword search matches exact words. Embeddings match meaning:

**Keyword Search:**
- Query: "automobile"
- Matches: Documents containing "automobile"
- Misses: Documents about "car" or "vehicle"

**Embedding Search:**
- Query: "automobile"
- Matches: Documents about cars, vehicles, transportation
- Understands semantic relationships

### Sentence Transformers

The `sentence-transformers` library provides pre-trained models optimized for creating embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("This is a sample sentence")
print(embedding.shape)  # Output: (384,)
```

This converts the sentence into a 384-dimensional vector that captures its meaning.

---

## Vector Stores Explained

Vector stores are specialized databases optimized for similarity search on high-dimensional vectors.

### Chroma

Chroma is a lightweight vector database perfect for development and small to medium-scale applications.

**Key Features:**
- Easy setup with no external dependencies
- Persistent storage
- Built-in embedding support
- Simple API

**Example:**

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_texts(
    texts=["Machine learning is a subset of AI", "Python is a programming language"],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Query
results = vectorstore.similarity_search("What is ML?", k=2)
```

### FAISS

FAISS (Facebook AI Similarity Search) is optimized for speed and scale.

**Key Features:**
- Extremely fast similarity search
- Handles millions of vectors
- GPU acceleration support
- Lower memory footprint

**Example:**

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    texts=["Machine learning is a subset of AI", "Python is a programming language"],
    embedding=embeddings
)

# Save and load
vectorstore.save_local("faiss_index")
loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## Building a Complete RAG System

Here's how all the pieces fit together:

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load documents
loader = PyPDFLoader("textbook.pdf")
documents = loader.load()

# 2. Chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_vectorstore"
)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    return_source_documents=True
)

# Query the system
def answer_query(question):
    result = qa_chain({"query": question})
    return result["result"]

# Use it
answer = answer_query("What is machine learning?")
print(answer)
```

---

## Creating Agent Tools with RAG

For AI agents, RAG becomes a powerful tool for knowledge retrieval:

```python
def query_rag(question):
    """
    Answer questions based on document knowledge base.
    
    Args:
        question: User's question as a string
        
    Returns:
        Answer grounded in retrieved documents
    """
    # Load existing vectorstore
    vectorstore = Chroma(
        persist_directory="./rag_vectorstore",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=retriever
    )
    
    result = qa_chain({"query": question})
    return result["result"]
```

This function can be called by an agent whenever it needs to consult the knowledge base.

---

## Hands-On Practice

### Saturday Workshop Activities

During the hands-on session, you'll:

1. **Build a PDF Loader**
   - Load a textbook chapter or technical document
   - Extract and process text content
   - Handle multi-page PDFs

2. **Create Embeddings**
   - Use sentence-transformers to generate embeddings
   - Understand embedding dimensions and properties
   - Compare different embedding models

3. **Store in Chroma**
   - Set up a Chroma vector database
   - Index your document chunks
   - Persist the database for reuse

4. **Create a RAG QA System**
   - Build the complete retrieval pipeline
   - Implement the `answer_query(question)` function
   - Test with various questions

---

## Assignments

### Assignment 1: Build a PDF RAG Bot

**Objective**: Create a complete RAG system for a PDF document

**Requirements:**
- Input: Any textbook chapter or technical document (PDF format)
- Process the PDF through the full RAG pipeline
- Create a persistent vector store in `rag_vectorstore/` directory
- Implement `rag_bot.py` with a `query_rag(question)` function
- The function should return accurate answers based on the document

**Deliverables:**
- `rag_vectorstore/` directory with indexed documents
- `rag_bot.py` with working `query_rag()` function
- Sample queries demonstrating the system works

### Assignment 2: Evaluate Accuracy

**Objective**: Understand when RAG works well and when it fails

**Requirements:**
- Prepare 20 questions about your document
- Include a mix of:
  - Direct factual questions
  - Questions requiring inference
  - Questions about topics not in the document
- Run each question through your RAG system
- Document which questions were answered correctly
- Identify patterns in failures

**Analysis Questions:**
- When does RAG provide accurate answers?
- What types of questions cause problems?
- How does chunk size affect retrieval quality?
- What happens when information isn't in the document?

---

## RAG Best Practices

### Chunking Strategy

- **Chunk size**: 500-1500 characters typically works well
- **Overlap**: 10-20% overlap helps maintain context
- **Semantic boundaries**: Split on paragraphs or sentences, not mid-sentence

### Retrieval Tuning

- **k parameter**: Retrieve 3-5 chunks for most questions
- **Similarity threshold**: Filter out low-relevance results
- **Reranking**: Consider reranking retrieved chunks for better quality

### Evaluation

- Test with diverse question types
- Check for hallucinations (answers not supported by documents)
- Verify source attribution
- Measure response time and cost

---

## Common Challenges and Solutions

### Challenge 1: Poor Retrieval Quality

**Symptoms**: System retrieves irrelevant chunks

**Solutions:**
- Adjust chunk size and overlap
- Try different embedding models
- Increase k (number of retrieved chunks)
- Improve document preprocessing

### Challenge 2: Incomplete Answers

**Symptoms**: Answers miss important information

**Solutions:**
- Retrieve more chunks (increase k)
- Use larger chunk sizes
- Implement multi-hop retrieval
- Improve question formulation

### Challenge 3: Hallucinations

**Symptoms**: System adds information not in documents

**Solutions:**
- Use stricter prompts ("Only answer based on provided context")
- Return source documents for verification
- Implement confidence scoring
- Use models fine-tuned for factual accuracy

---

## Connecting to Agents

RAG transforms agents from general assistants into domain experts:

- **Knowledge grounding**: Agents can cite specific sources
- **Up-to-date information**: Add new documents without retraining
- **Domain expertise**: Agents become experts in your specific content
- **Reduced hallucinations**: Answers are grounded in real data
- **Transparency**: Source documents provide audit trails

In upcoming weeks, you'll integrate RAG with agent frameworks, creating agents that can reason over your documents, answer complex questions, and take actions based on retrieved knowledge.

---

## Preparation for Class

Before attending class, please:
1. Install required libraries:
   ```bash
   pip install langchain
   pip install chromadb
   pip install sentence-transformers
   pip install pypdf
   pip install openai
   ```
2. Obtain an OpenAI API key (or prepare to use an alternative LLM)
3. Find a PDF document to use (textbook chapter, technical paper, manual)
4. Review Week 2 concepts on data processing
5. Read through this material and prepare questions

### Recommended Pre-Reading

Familiarize yourself with:
- What are vector embeddings?
- How does semantic search differ from keyword search?
- What are the limitations of LLMs?

---

## Looking Ahead

Week 3 equips your agents with knowledge retrieval capabilities. In Week 4, you'll explore agent frameworks like LangChain and CrewAI, learning how to orchestrate multiple tools (including ML models and RAG systems) into sophisticated agent workflows that can plan, reason, and execute complex tasks autonomously.
