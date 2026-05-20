import os
from typing import TypedDict
from langchain_core import documents
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph , END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import gradio as gr

# Suppress annoying logging and user warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

model ="openai/gpt-oss-20b:free"

def get_llm():
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url=BASE_URL, 
        api_key=OPENROUTER_API_KEY,
        
    )
# Configured perfectly via OpenRouter's native embedding endpoint
embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small", 
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY
        )  

class AgentState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    needs_retrieval: bool
    decision: str

def build_vector_store(file):
    loader = PyPDFLoader(file)
    docs = loader.load()

    spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100, 
        length_function=len,
        )
    chunks = spliter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store



def decide_retrieval(state: AgentState):
    topic_keywords = [
        "rag",
        "langchain",
        "vector store",
        "vectorstore",
        "split",
        "splitter",
        "document",
        "pdf",
        "retrieval",
        "embeddings",
    ]

    if current_pdf_path is not None and state["question"].strip():
        needs_retrieval = True
    else:
        needs_retrieval = any(
            keyword in state["question"].lower() for keyword in topic_keywords
        )

    decision = "retrieve" if needs_retrieval else "generate"
    return {**state, "needs_retrieval": needs_retrieval, "decision": decision}

def retrieve_documents(state:AgentState, retriever) -> AgentState:
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}

def generate_answer(state:AgentState) ->AgentState:
    question = state["question"]
    documents = state["documents"]

    llm = get_llm()

    if documents:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""
        Use the following retrieved context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        response = llm.invoke(prompt)
        return {**state, "answer": response.content}
    
    #direct response without retrieval
    else:
        prompt = f"Answer the following question to the best of your ability:\n\nQuestion: {question}\nAnswer:"

        response = llm.invoke(prompt)
        answer = response.content
    return {**state, "answer": answer}

def should_retrieve(state:AgentState) ->str:
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"
    
# build langgraph workflow
def build_graph(retriever):

    workflow = StateGraph(AgentState)

    workflow.add_node("decide", decide_retrieval)

    workflow.add_node(
        "retrieve",
        lambda state: retrieve_documents(state, retriever)
    )

    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("decide")

    workflow.add_conditional_edges(
        "decide",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

vector_store = None
retriever = None
app = None
current_pdf_path = None

def build_pdf_app(file_path: str):
    global vector_store, retriever, app, current_pdf_path

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = spliter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    app = build_graph(retriever)
    current_pdf_path = file_path


def rag_pipeline(file, question):
    global app, current_pdf_path

    if file is None:
        return "Please upload a PDF file first."

    file_path = file.name if hasattr(file, "name") else file
    if app is None or file_path != current_pdf_path:
        try:
            build_pdf_app(file_path)
        except Exception as e:
            return f"Error loading PDF: {str(e)}"

    try:
        result = app.invoke({
            "question": question,
            "documents": [],
            "answer": "",
            "needs_retrieval": False,
            "decision": "",
        })
        return result["answer"], f"Mode: {result['decision']}"
    except Exception as e:
        return f"Error answering question: {str(e)}", "Error"


interface = gr.Interface(
    fn=rag_pipeline,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"),
        gr.Textbox(label="Ask a question"),
    ],
    outputs=[gr.Textbox(label="Answer", lines=8), gr.Textbox(label="Decision", interactive=False, lines=1)],
    title="Agentic RAG with PDF",
    description="Upload a PDF and ask a question. If the question is about RAG/langchain/vector store/splitting, the answer will use document retrieval; otherwise it will use the LLM directly.",
)

if __name__ == "__main__":
    interface.launch()










