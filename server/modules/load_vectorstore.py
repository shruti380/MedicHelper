import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------- ENV SETUP --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env file")

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------- PINECONE INIT --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_ENV
)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="dotproduct",
        spec=spec
    )

    # Wait until index is ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# -------------------- VECTORSTORE LOADER --------------------
def load_vectorstore(uploaded_files):
    """
    Process uploaded PDFs and store embeddings in Pinecone
    """

    # HuggingFace embedding model (local, no API key)
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    file_paths = []

    # Save uploaded files locally
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # Process PDFs
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]

        # Metadata (IMPORTANT for retrieval)
        metadatas = []
        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta["text"] = chunk.page_content
            meta["source"] = Path(file_path).name
            metadatas.append(meta)

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        batch_size = 100

        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            for i in range(0, len(embeddings), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]

                index.upsert(
                    vectors=zip(batch_ids, batch_embeddings, batch_metadatas)
                )

                progress.update(len(batch_ids))

        print(f"✅ Upload complete for {file_path}")

    return {
        "message": f"Successfully processed {len(file_paths)} file(s)"
    }
