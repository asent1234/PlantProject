from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import requests
import os
import shutil
import nltk
import json

# Set up NLTK data path and download required data
nltk_data_dir = 'C:/Users/aakas/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

class OllamaEmbeddings:
    def __init__(self, model_name="deepseek-70b-r1"):
        self.model_name = model_name
        
    def embed_documents(self, texts):
        return [self.get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        return self.get_embedding(text)
    
    def get_embedding(self, text):
        try:
            response = requests.post('http://localhost:11434/api/embeddings', 
                                   json={
                                       "model": self.model_name,
                                       "prompt": text
                                   })
            response.raise_for_status()
            result = response.json()
            if 'embedding' not in result:
                raise ValueError(f"No embedding found in response: {result}")
            return result['embedding']
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid response from Ollama: {str(e)}")

def main():
    documents = loadDocuments()
    chunks = textChunks(documents)
    
    if os.path.exists(chromaPath):
        shutil.rmtree(chromaPath)
        
    embeddings = OllamaEmbeddings()
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=chromaPath
    )
    db.persist()
    
    print(f"Saved {len(chunks)} chunks to {chromaPath}.")


from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader

def loadDocuments():
    # Load markdown files
    md_loader = DirectoryLoader(dataPath, glob="*.md")
    md_documents = md_loader.load()
    # Load PDF files
    pdf_loader = DirectoryLoader(
        dataPath, glob="*.pdf",
        loader_cls=UnstructuredPDFLoader
    )
    pdf_documents = pdf_loader.load()
    # Combine
    return md_documents + pdf_documents


def textChunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


chromaPath = "chroma"
dataPath = os.path.join(os.path.dirname(__file__), 'data')

if __name__ == "__main__":
    main()
