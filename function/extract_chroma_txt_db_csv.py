import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# === Load environment variables ===
load_dotenv()

# === Configuration ===
TXT_FOLDER = "dataset"            # Folder containing .txt files
CHROMA_DB_DIR = "chroma_txt_db_csv"      # Chroma DB storage folder
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CSV_OUTPUT_PATH = "chunk_info1.csv"

# === Text splitter ===
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

# === Load and chunk each .txt file ===
def load_and_chunk_txt_files(folder_path):
    all_documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            category = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            chunks = splitter.split_text(text)
            documents = [
                Document(page_content=chunk, metadata={"category": category})
                for chunk in chunks
            ]
            all_documents.extend(documents)

            # Print chunk preview info
            print(f"\n📄 File: {filename}")
            print(f"🧩 Chunks created: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i + 1} ---")
                print(chunk)


    return all_documents

# === Embed and store in ChromaDB ===
def embed_and_store(documents, db_dir):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir
    )
    vectordb.persist()

    # Print category-wise chunk summary
    print(f"\n📦 Stored {len(documents)} total chunks in Chroma DB at '{db_dir}'")
    from collections import Counter
    metadata_list = [doc.metadata.get("category", "unknown") for doc in documents]
    print("\n📊 Stored categories and chunk counts:")
    for cat, count in Counter(metadata_list).items():
        print(f" - {cat}: {count} chunks")

# === Save chunking info to CSV ===
def save_chunk_info_to_csv(documents, csv_file_path):
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Chunk_Index", "Category", "Chunk_Content", "Timestamp"])

        for i, doc in enumerate(documents):
            content = doc.page_content.replace('\n', ' ')
            category = doc.metadata.get("category", "unknown")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([i + 1, category, content, timestamp])

    print(f"📝 Chunk info saved to CSV at: {csv_file_path}")


# === Main ===
def main():
    if not os.path.exists(TXT_FOLDER):
        print(f"❌ Folder not found: {TXT_FOLDER}")
        return

    print("📄 Loading and chunking text files...")
    documents = load_and_chunk_txt_files(TXT_FOLDER)

    print("\n🔢 Embedding and saving to Chroma...")
    embed_and_store(documents, CHROMA_DB_DIR)

    print("\n📝 Saving chunk info to CSV...")
    save_chunk_info_to_csv(documents, CSV_OUTPUT_PATH)

    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
