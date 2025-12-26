from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query: {question}")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
        index = pc.Index(index_name)
        
        # Create HuggingFace embeddings (same as upload!)
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Embed the query
        logger.debug("Embedding query...")
        embedded_query = embed_model.embed_query(question)
        
        # Query Pinecone
        logger.debug("Querying Pinecone...")
        res = index.query(
            vector=embedded_query, 
            top_k=5,  # Increased from 3 for better results
            include_metadata=True
        )
        
        # Check if we got results
        if not res.get("matches"):
            logger.warning("No matches found in Pinecone")
            return JSONResponse(
                status_code=200,
                content={
                    "response": "I couldn't find any relevant information in the uploaded documents. Please make sure documents are uploaded first.",
                    "sources": []
                }
            )
        
        # Create documents from results - now "text" exists in metadata!
        docs = []
        for match in res["matches"]:
            text = match["metadata"].get("text", "")
            if text:  # Only add if text exists
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": match["metadata"].get("source", "Unknown"),
                            "page": match["metadata"].get("page", 0),
                            "score": match.get("score", 0)
                        }
                    )
                )
        
        if not docs:
            logger.warning("No documents with text content found")
            return JSONResponse(
                status_code=200,
                content={
                    "response": "Found matches but no text content. Please re-upload your documents.",
                    "sources": []
                }
            )
        
        logger.info(f"Found {len(docs)} relevant documents")
        
        # Simple retriever class
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)
            
            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs
        
        # Create retriever and chain
        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)
        
        logger.info("Query successful")
        return result
        
    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error processing question: {str(e)}"}
        )