
# 🧠 Agentic RAG with LangGraph + PDF

This project implements a **dynamic Agentic RAG system** that intelligently decides whether to use document retrieval or answer directly using an LLM.

Unlike traditional RAG pipelines (Retrieve → Generate), this system uses an **AI Agent (LangGraph)** to control the flow dynamically.

---

## 🚀 Features

* ✅ Agent-based decision making (retrieve vs generate)
* ✅ Dynamic PDF ingestion
* ✅ FAISS vector store for fast similarity search
* ✅ OpenRouter integration for LLM + embeddings
* ✅ Gradio UI for interactive usage
* ✅ Optimized performance (no repeated vector building)
* ✅ Decision transparency (shows retrieval mode)

---

## 🏗️ Architecture

```
User Question
      ↓
   Decide (Agent)
   ↙        ↘
Retrieve     Generate
   ↓            ↓
   →→→ Final Answer ←←←
```

---

## ⚙️ Tech Stack

* Python
* LangGraph
* LangChain
* FAISS
* OpenRouter
* Gradio

---

## 📂 How It Works

### 1. Upload PDF

* Document is loaded and split into chunks
* Embeddings are generated
* Stored in FAISS vector database

### 2. Ask Question

* Agent decides:

  * 🔍 Use retrieval
  * 🧠 Answer directly

### 3. Generate Answer

* Uses context (if retrieved)
* Otherwise uses LLM knowledge

---

## 🧪 Run Locally

```bash
git clone https://github.com/Masouma96/Agentic-RAG.git
cd agentic-rag

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

python agent.py
```

---

## 🔑 Environment Variables

Create `.env` file:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## 🖥️ UI

* Upload PDF
* Ask questions
* See:

  * Answer
  * Decision mode (retrieve / generate)

---

## ⚡ Performance Optimization

* FAISS built only once per PDF
* Retriever reused across queries
* Removed LLM decision bottleneck
* Reduced chunk size for faster retrieval

---

## 📌 Example Use Case

* Ask: *"What is step 2?"* → Retrieval
* Ask: *"What is AI?"* → Direct LLM answer

---

## 🔮 Future Improvements

* Add caching layer (Redis / memory)
* Multi-document support
* Streaming responses
* Better retrieval classifier (lightweight model)
* Docker deployment

---

## 👩‍💻 Author

Built with ❤️ for learning advanced AI system design.
