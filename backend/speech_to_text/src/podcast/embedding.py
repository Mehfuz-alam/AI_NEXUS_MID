from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
pinecone_index_name = "podcast-transcripts"

def store_embeddings(documents):
    docsearch = PineconeVectorStore.from_documents(documents, embedding_function, index_name=pinecone_index_name)
    return docsearch