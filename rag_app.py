import os
import shutil
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ------------------------
# 1. Load API Key
# ------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

print("API KEY FOUND:", bool(api_key))

# ------------------------
# 2. Load PDFs
# ------------------------
pdf_folder = "PDFs"
documents = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, file)

        print(f"Loading {pdf_path}...")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Add filename metadata
        for doc in docs:
            doc.metadata["filename"] = file

        documents.extend(docs)

# ------------------------
# 3. Split Documents
# ------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = text_splitter.split_documents(documents)

# ------------------------
# 4. Filter Small Chunks
# ------------------------
filtered_chunks = []
for chunk in chunks:

    text = chunk.page_content.strip()

    if len(set(text.split())) < 20:
        continue

    if "filename" not in chunk.metadata:
        chunk.metadata["filename"] = chunk.metadata.get(
            "source",
            "unknown"
        ).split("\\")[-1].split("/")[-1]

    filtered_chunks.append(chunk)

chunks = filtered_chunks
print("Sample Metadata:")
print(chunks[0].metadata)

# ------------------------
# 5. Metrics
# ------------------------
total_pdfs = len(
    [f for f in os.listdir(pdf_folder)
     if f.endswith(".pdf")]
)

print(f"Total PDFs: {total_pdfs}")
print(f"Total Pages: {len(documents)}")
print(f"Total Chunks: {len(chunks)}")

# ------------------------
# 6. Embeddings
# ------------------------
print("Loading Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embeddings Loaded")

# ------------------------
# 7. Create Vector DB
# ------------------------
db_path = "./chroma_db"

if not os.path.exists(db_path):

    print("Creating Vector Database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

else:

    print("Loading existing Vector Database...")

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
# ------------------------
# 8. Retriever
# ------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        
    }
)
# ------------------------
# 9. LLM
# ------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant"
    # Optional:
    # model_name="llama-3.3-70b-versatile"
)

print("Groq LLM Ready")

# ------------------------
# 10. Prompt
# ------------------------
prompt = ChatPromptTemplate.from_template("""
You are an expert science explainer.

Your job is to synthesize information from multiple PDF sources into a clear, simple explanation.

Rules:
- Combine all relevant information into one coherent answer
- Avoid repeating the same idea in different words
- Do NOT sound like a research paper
- Keep it natural and easy to read
- Use context only
- If the context is sufficient, always answer

Context:
{context}

Question:
{question}

Answer:
""")

# ------------------------
# 11. Format Retrieved Docs
# ------------------------
def format_docs(docs):

    return "\n\n".join([
        doc.page_content.strip().replace("\n", " ")
        for doc in docs
    ])
# ------------------------
# 13. Ask Question
# ------------------------
def get_context(query):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    return context, docs

def ask_question(query):
    context, docs = get_context(query)

    response = llm.invoke(
        prompt.format(
            context=context,
            question=query
        )
    )

    return response.content, docs