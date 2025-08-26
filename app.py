import streamlit as st
import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import requests
import subprocess
import time
import torch
from sentence_transformers import SentenceTransformer

# Force CPU usage and disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OLLAMA_NUM_GPU"] = "0"  # Force Ollama to use CPU
torch.set_default_device('cpu')

# Set page configuration
st.set_page_config(
    page_title="GIKI GPT",
    page_icon="logo.ico",
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = "unknown"

class CPUHuggingFaceEmbeddings:
    """Custom CPU-compatible HuggingFace embeddings"""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Load model with CPU explicitly
        self.model = SentenceTransformer(
            f'sentence-transformers/{model_name}',
            device='cpu'
        )
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text):
        """Embed a single query"""
        return self.model.encode(text, convert_to_tensor=False).tolist()

class GIKIGPT:
    def __init__(self):
        # Use custom CPU-compatible embeddings
        self.embeddings = CPUHuggingFaceEmbeddings("all-MiniLM-L6-v2")
        self.vector_store = None
    
    def check_ollama_status(self):
        """Check if Ollama is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                return "running"
            return "not_responding"
        except requests.ConnectionError:
            return "not_running"
        except Exception:
            return "error"
    
    def start_ollama_service(self):
        """Attempt to start Ollama service"""
        try:
            # For Windows
            if os.name == 'nt':
                result = subprocess.run(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                result = subprocess.run(
                    ["ollama", "serve"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            time.sleep(3)  # Wait for service to start
            return self.check_ollama_status() == "running"
        except Exception as e:
            st.error(f"Failed to start Ollama: {str(e)}")
            return False
        
    def load_documents(self, files: List) -> List[Document]:
        """Load documents from various file formats"""
        documents = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                if file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
                elif file.name.lower().endswith(('.doc', '.docx')):
                    loader = Docx2txtLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
                elif file.name.lower().endswith('.txt'):
                    loader = TextLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
                elif file.name.lower().endswith(('.ppt', '.pptx')):
                    loader = UnstructuredPowerPointLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
        
        return documents
    
    def process_documents(self, documents: List[Document]):
        """Process and chunk documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return self.vector_store
    
    def get_llm(self):
        """Get LLM instance with proper fallback handling"""
        ollama_status = self.check_ollama_status()
        st.session_state.ollama_status = ollama_status
        
        if ollama_status == "running":
            try:
                return Ollama(
                    model="phi", 
                    temperature=0.1, 
                    timeout=120,  # Longer timeout for CPU
                    num_thread=4  # Use multiple CPU threads
                )
            except Exception as e:
                st.warning(f"Ollama connection failed: {str(e)}")
        
        # Fallback to HuggingFace endpoint
        hf_api_key = st.session_state.get('hf_api_key', '') or st.secrets.get("HF_API_KEY", "")
        if hf_api_key:
            try:
                return HuggingFaceEndpoint(
                    repo_id="microsoft/phi-2",
                    huggingfacehub_api_token=hf_api_key,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.1,
                        "max_length": 512,
                        "max_new_tokens": 256
                    }
                )
            except Exception as e:
                st.error(f"HuggingFace Error: {str(e)}")
        
        st.error("No LLM available. Please check Ollama or configure API keys.")
        return None
    
    def create_qa_chain(self):
        """Create QA chain with RAG"""
        if not self.vector_store:
            return None
        
        prompt_template = """Use the context below to answer the question. If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        llm = self.get_llm()
        if not llm:
            return None
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False,
            verbose=False
        )
        
        return qa_chain

def main():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .status-running {
            color: green;
            font-weight: bold;
        }
        .status-not-running {
            color: red;
            font-weight: bold;
        }
        .cpu-mode {
            background-color: #262730; /* dark background */
            color: white;              /* white text */
            padding: 10px 18px;
            border-radius: 8px;
            margin: 10px auto;
            text-align: center;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
            width: fit-content;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App Title
    st.markdown('<h1 class="main-header">🤖 GIKI GPT</h1>', unsafe_allow_html=True)
    
    # Initialize GIKI GPT
    giki_gpt = GIKIGPT()
    
    # Check Ollama status
    if st.session_state.ollama_status == "unknown":
        st.session_state.ollama_status = giki_gpt.check_ollama_status()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        status_display = {
            "running": "✅ Ollama is running",
            "not_running": "❌ Ollama not running",
            "not_responding": "❗️ Ollama not responding",
            "error": "💢 Ollama status unknown"
        }
        
        status_text = status_display.get(st.session_state.ollama_status, "❓ Unknown status")
        st.write(f"**Ollama:** {status_text}")
        
        if st.session_state.ollama_status != "running":
            if st.button("🔄 Try to Start Ollama"):
                with st.spinner("Starting Ollama..."):
                    if giki_gpt.start_ollama_service():
                        st.success("Ollama service started!")
                        st.session_state.ollama_status = giki_gpt.check_ollama_status()
                        st.rerun()
                    else:
                        st.error("Could not start Ollama service")
        
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, DOC, TXT, PPT)",
            type=['pdf', 'doc', 'docx', 'txt', 'ppt', 'pptx'],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) > 5:
            st.warning("Maximum 5 files allowed. Using first 5.")
            uploaded_files = uploaded_files[:5]
        
        if st.button("⚙️ Process Documents", type="primary") and uploaded_files:
            with st.spinner("Processing documents (CPU mode may be slow)..."):
                try:
                    documents = giki_gpt.load_documents(uploaded_files)
                    if documents:
                        giki_gpt.process_documents(documents)
                        st.session_state.qa_chain = giki_gpt.create_qa_chain()
                        st.session_state.documents_processed = True
                        st.success(f"✅ Processed {len(documents)} documents!")
                    else:
                        st.error("❌ No valid content found.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.header("🔑 API Fallback")
        st.text_input("HuggingFace API Key", key="hf_api_key", type="password")

    # Main chat interface
    if st.session_state.documents_processed:
        st.header("💬 Chat with GIKI GPT")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.info(f"**You:** {message['content']}")
            else:
                st.success(f"**GIKI GPT:** {message['content']}")
        
        # Chat input
        question = st.text_input("Ask a question:", key="question_input")
        
        if st.button("Send") and question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": question})
                    answer = result["result"]
                    st.session_state.chat_history.append({"role": "bot", "content": answer})
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.chat_history.append({"role": "bot", "content": error_msg})
                    st.error(error_msg)
                    st.rerun()
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload and process documents to start chatting")

if __name__ == "__main__":
    main()