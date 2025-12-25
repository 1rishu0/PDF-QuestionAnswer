# QdrantClient is used to Imports the client used to connect and interact with a Qdrant vector database
# VectorParams Imports configuration settings for defining vector size and similarity metric
# Distance Imports distance metrics (e.g., cosine, dot, euclidean) for vector comparison
# PointStruct Imports the structure used to store vectors along with their IDs and metadata
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# for creating the vector database locally we have to create class
class QdrantStorage:
    # we are adding docs to collection because this is going to be the collection where we are storing the information essentially
    # dim is dimensions means it is the number of values that we have inside our vector
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        # create client with timeout feature so if we don't connect in 30s , we essentially crash this  program
        self.client = QdrantClient(url=url, timeout=30)
        # we are going to create a new collection in our database inside of this qdrant storage folder
        self.collection = collection
        # if we don't create collection by ourselves it will create automatically
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name= self.collection,
                # Distance.cosine is a formula for calculating the distance between different points in our vector database.
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    # we are going to create a new function which is upsert , which is essentialy insert and update
    # In a vector database, the payload stores the actual contextual data associated with a vector, while the vector itself is used only for similarity search.
    # It is going to get all of the associated IDs, vectors, and the payloads from below three lists effectively and a point structure which is what we need to create in order to insert this into our vector database.
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        # we are going to pass a series of IDs which is a list of a bunch of vectors which is kind of the vectorized version that's going to be in a dimension of 3072 and payload that is going to be real data , real human readable data that kind of represents the information that we have vectorized.
        # we are going to convert all these three things and convert this into point structure which is just what's  required for it quadrant
        self.client.upsert(self.collection, points=points)

    # next important thing is the searching for the vectors
    # top_k means we are looking for this many results from the vector database
    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            colelction_name = self.collection,
            query_vector = query_vector,
            with_payload = True,
            limit = top_k,
        )
        # the reason for this variable because we need to get all of the context or information
        contexts = []
        # we need to get the sources to the documents that we pulled this information from
        sources = set()

        # It is going to search our vector database and it is going to get the relevant results based on the query_vector, and then we are going to pull out all the sources and the context and return that
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get('text', None)
            source = payload.get('source', "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
