# Tax Q&A Chatbot with RAG Pipeline

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Korean income tax law. Built with LangChain, Ollama, Pinecone, and Streamlit.

## Project Overview

This project builds a RAG pipeline step by step, from basic LLM testing to a fully functional chatbot with chat history support. Each notebook represents a progressive stage of development.

## Data Source

This chatbot generates answers based on the official Korean Income Tax Act published by the government:
- [소득세법 - 국가법령정보센터](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%86%8C%EB%93%9D%EC%84%B8%EB%B2%95)

## Tech Stack

- **LLM**: Ollama (exaone3.5:7.8b) - local LLM, replacing OpenAI API from the original course
- **Embedding**: nomic-embed-text (via Ollama)
- **Vector DB**: Pinecone (migrated from ChromaDB)
- **Framework**: LangChain 0.3.x
- **Frontend**: Streamlit

## Project Structure

```
.
├── 1. langchain_llm_test.ipynb              # Initial LangChain + Ollama setup test
├── 2. rag_with_chroma.ipynb                 # RAG pipeline with ChromaDB
├── 3. rag_without_langchain_with_chroma.ipynb # RAG without LangChain (python-docx + tiktoken)
├── 4. rag_with_pinecone.ipynb               # Migration from ChromaDB to Pinecone
├── 4.1. rag_with_pinecone_tax_with_table.ipynb    # Test with table-formatted docx
├── 4.2. rag_with_pinecone_tax_with_markdown.ipynb  # Test with markdown-formatted docx
├── 4.3 rag_with_query_rewriting.ipynb       # Query rewriting with keyword dictionary
├── 5. for_streamlit.ipynb                   # Streamlit app prototype
├── 5.1. for_streamlit_update.ipynb          # Updated Streamlit prototype
├── streamlit/
│   ├── chat.py              # Streamlit UI with streaming response
│   ├── llm.py               # RAG chain with chat history and few-shot examples
│   ├── config.py            # Few-shot answer examples for prompt engineering
│   └── requirements.txt     # Python dependencies
├── tax.docx                 # Original tax law document
├── tax_with_table.docx      # Table-formatted version
└── tax_with_markdown.docx   # Markdown-formatted version
```

## RAG Pipeline Flow

1. **Document Loading** - Load tax law documents (.docx) using LangChain's `Docx2txtLoader` or `python-docx`
2. **Text Splitting** - Split documents into chunks using `RecursiveCharacterTextSplitter`
3. **Embedding** - Vectorize chunks using `nomic-embed-text` via Ollama
4. **Vector Storage** - Store embeddings in Pinecone cloud database
5. **Retrieval** - On user query, find top-k similar documents from Pinecone
6. **Generation** - Pass retrieved documents + user question to LLM for answer generation

## Key Features

- **Chat History Support** - Uses `RunnableWithMessageHistory` and `create_history_aware_retriever` for multi-turn conversations
- **Query Rewriting** - Dictionary-based keyword replacement (e.g., "사람" -> "거주자") for better retrieval
- **Few-Shot Prompting** - Provides answer format examples via `FewShotChatMessagePromptTemplate`
- **Streaming Response** - Real-time response output using `st.write_stream`

## Setup

### Prerequisites

- Python 3.10
- [Ollama](https://ollama.ai) installed and running locally
- Pinecone account and API key

### Installation

```bash
# Set up Python virtual environment (using pyenv)
pyenv virtualenv 3.10.14 llm-application-rag
pyenv shell llm-application-rag

# Install dependencies
pip install -r streamlit/requirements.txt

# Pull Ollama models
ollama pull exaone3.5:7.8b
ollama pull nomic-embed-text
```

### Environment Variables

Create a `.env` file in the `streamlit/` directory:

```
PINECONE_API_KEY=your_pinecone_api_key
```

### Run

```bash
cd streamlit
streamlit run chat.py
```

## Why LangChain 0.3.x Instead of 1.x?

As of 2026, `pip install langchain` installs version 1.x by default. However, LangChain 1.x removed the `langchain.chains` module entirely, migrating chat history support to **LangGraph**.

Since this project follows a course built on LangChain 0.x, the entire LangChain ecosystem was downgraded to 0.3.x for compatibility:

| Package | 1.x (default) | 0.3.x (used) |
|---------|---------------|---------------|
| langchain | 1.2.15 | 0.3.25 |
| langchain-core | 1.3.0 | 0.3.84 |
| langchain-ollama | 1.1.0 | 0.3.10 |
| langchain-community | 0.4.1 | 0.3.21 |
| langchain-chroma | 1.1.0 | 0.2.2 |
| langchain-pinecone | 0.2.13 | 0.2.0 |

## Troubleshooting

### Model Instruction Following Issues

The `exaone3.5:7.8b` model does not reliably follow system prompt instructions:

- **Answer length** - Ignores "answer in 2-3 sentences" instruction, produces long responses
- **Source citation** - Does not cite specific tax law articles as instructed
- **Chat history context** - Fails to reformulate follow-up questions using conversation history (e.g., "How about 100 million?" loses context of the previous income tax question)

**Attempted fix**: Changed prompts from English to Korean - no improvement.

**Root cause**: This is an instruction following limitation of the 7.8B parameter model, not a code or prompt language issue. Larger models (32B+) are expected to handle these instructions better.

### Deployment

Deployment to Streamlit Cloud is not planned for this project, as it uses Ollama (local LLM) instead of a cloud-based LLM API (e.g., OpenAI). Streamlit Cloud does not have access to a local Ollama server, so the app would not function without switching to a paid API.
