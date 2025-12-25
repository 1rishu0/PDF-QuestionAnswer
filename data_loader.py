# This is where we use llamaindex to load in PDF documents and to embedd them
# Imports the OpenAI client used to interact with OpenAI models and APIs
# Imports a reader to load and extract text content from PDF files
# Imports a utility to split text into sentence-based chunks for indexing
# Imports a helper to load environment variables from a .env file
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text_embedding-3-large"
# embed dim should match the dimensions inside vector_db.py QuadrantStorage class
EMBED_DIM = 3072

# Chunk overlap is how much of the end of one chunk is included in the beginning of another chunk.
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    # we are going to get all of the text content for every single document inside our documents, if this document has some text attribute
    texts = [d.text for d in docs if getattr(d, 'text', None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_texts(t))

    return chunks

# It is going to send a request to OpenAI and it is going to pass all of text that we have chunked already and it going to embed them which we can store in the vector database.
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]