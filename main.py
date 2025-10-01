import os
import time
import psutil  # New import for resource monitoring
from dotenv import load_dotenv

# --- NEW: Using DirectoryLoader to load all files ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# ---------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()
# google_api_key = os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
#     exit()

print("--- RAG CLI Chatbot with Resource Monitoring ---")

# --- 1. DOCUMENT LOADING (Modified for all files in a directory) ---
docs_folder_path = "docs/"
# Using DirectoryLoader to load all supported files.
source_file_count = len(
    [
        f
        for f in os.listdir(docs_folder_path)
        if os.path.isfile(os.path.join(docs_folder_path, f))
    ]
)

# It automatically uses PyPDFLoader for .pdf, TextLoader for .txt, etc.
loader = DirectoryLoader(docs_folder_path, glob="**/*", loader_cls=PyPDFLoader)
documents = loader.load()
print(
    f"Loaded {len(documents)} page(s) from {source_file_count} file(s) in '{docs_folder_path}'."
)
# -----------------------------------------------------------------

# --- 2. TEXT SPLITTING (No changes here) ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"Split document into {len(texts)} text chunks.")

# --- 3. VECTOR STORE CREATION (with timing) ---
print("\n--- Creating Embeddings and Vector Store ---")
# --- NEW: Start the timer ---
start_time = time.time()
# ---------------------------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = None
batch_size = 20
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    print(f"Processing batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}...")

    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch_texts, embeddings)
    else:
        vectorstore.add_documents(batch_texts)
    # Note: No sleep needed for local embeddings, this is fast

# --- NEW: Stop the timer ---
end_time = time.time()
# -------------------------

# --- NEW: Save the index to disk to measure its size ---
index_path = "faiss_index"
vectorstore.save_local(index_path)
# ----------------------------------------------------
print("Vector store created and saved successfully.")

# --- NEW: Resource Usage Report ---
print("\n--- Resource Usage Report ---")

# 1. Time Taken
duration = end_time - start_time
print(f"Time to create embeddings and index: {duration:.2f} seconds")

# 2. RAM Usage
process = psutil.Process(os.getpid())
ram_usage_mb = process.memory_info().rss / (1024 * 1024)  # in MB
print(f"RAM used by this script: {ram_usage_mb:.2f} MB")


# 3. Storage and Location
# Helper function to get directory size


def get_dir_stats(path="."):
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1
    return total_size, file_count  # Return both values


# MODIFIED calling code to receive both values
storage_usage_bytes, total_files = get_dir_stats(index_path)
storage_usage_mb = storage_usage_bytes / (1024 * 1024)

storage_location = os.path.abspath(index_path)
print(f"Vector store size on disk: {storage_usage_mb:.2f} MB")
# NEW line to print the file count
# print(f"Total files in vector store directory: {total_files}")
print(f"Vector store location: {storage_location}")
# -----------------------------------

# --- 4. RAG CHAIN CREATION (No changes) ---
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-pro", temperature=0.7, google_api_key=google_api_key
# )

llm = ChatGroq(model_name="qwen/qwen3-32b")
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
# )

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a precise assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
Your answer MUST be based only on the provided context.

For every fact you state, you MUST cite the page number from which you got the information. The page number is in the 'page' field of the document's metadata.
Format your citation at the end of the relevant sentence like this: (Source: Page X).

Context:
{context}

Question: {question}
Answer:"""
)

# # The new chain takes the LLM and retriever separately
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm, retriever=vectorstore.as_retriever()
# )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,  # Your new Groq or Hugging Face LLM
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
)

print("\n--- RAG chain created. The chatbot is ready. ---")
print("-" * 50)

# --- 5. USER INTERACTION LOOP ---
chat_history = []
while True:
    user_question = input("\nAsk a question about your PDF (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    print("Thinking...")
    # response = qa_chain.invoke({"query": user_question})

    # Pass the question and chat history to the chain
    result = qa_chain.invoke({"question": user_question, "chat_history": chat_history})
    answer = result["answer"]

    # Update the chat history
    chat_history.append((user_question, answer))
    response = {"result": answer}

    print("\nAnswer:")
    print(response["result"])
    print("-" * 50)
