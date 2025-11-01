import os
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import should_index, file_hash

def build_or_load_vectorstore(project_path: str):
    persist_dir = os.path.join(project_path, ".codeflow_index")
    meta_path = os.path.join(persist_dir, "index_meta.json")
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # Load existing metadata
    old_meta = {}
    if Path(meta_path).exists():
        try:
            old_meta = json.loads(Path(meta_path).read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load index metadata: {e}")

    new_docs = []
    new_meta = {}

    print("Scanning project files...")
    for file_path in Path(project_path).rglob("*"):
        if not should_index(file_path):
            continue

        current_hash = file_hash(file_path)
        file_key = str(file_path.resolve())

        if file_key in old_meta and old_meta[file_key] == current_hash:
            continue  # Unchanged

        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_key
            new_docs.extend(docs)
            new_meta[file_key] = current_hash
        except Exception as e:
            print(f"Skipped {file_path}: {e}")

    # Build or update
    if new_docs:
        print(f"Indexing {len(new_docs)} changed files...")
        chunks = splitter.split_documents(new_docs)
        vectorstore = Chroma.from_documents(
            chunks, embeddings, persist_directory=persist_dir
        )
        print(f"Indexed {len(chunks)} chunks.")
    else:
        print("No changes. Loading existing index.")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Save meta - merge old and new metadata
    old_meta.update(new_meta)
    Path(meta_path).write_text(json.dumps(old_meta, indent=2))
    return vectorstore