from langchain_huggingface import HuggingFaceEmbeddings

print("Starting")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loaded")