import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# === Load environment variables ===
load_dotenv()

# === Configuration ===
TXT_FOLDER = "dataset"  # <- replace with your folder containing .txt files
CHROMA_DB_DIR = "chroma_txt_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === Text splitter ===
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

# === Load and process each .txt file ===
def load_and_chunk_txt_files(folder_path):
    all_documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            category = os.path.splitext(filename)[0]  # use filename (without .txt) as category
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            chunks = splitter.split_text(text)

            documents = [
                Document(page_content=chunk, metadata={"category": category})
                for chunk in chunks
            ]
            all_documents.extend(documents)
            print(f"✅ Processed '{filename}' into {len(chunks)} chunks")
    
    return all_documents

# === Embed and store in Chroma ===
def embed_and_store(documents, db_dir):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir
    )

    vectordb.persist()
    print(f"📦 Stored {len(documents)} total chunks in Chroma DB at '{db_dir}'")

def main():
    if not os.path.exists(TXT_FOLDER):
        print(f"❌ Folder not found: {TXT_FOLDER}")
        return

    print("📄 Loading and chunking text files...")
    documents = load_and_chunk_txt_files(TXT_FOLDER)

    print("🔢 Embedding and saving to Chroma...")
    embed_and_store(documents, CHROMA_DB_DIR)

    print("🎉 All done!")

if __name__ == "__main__":
    main()
