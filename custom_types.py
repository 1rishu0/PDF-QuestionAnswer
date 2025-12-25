# The Reason for this so that we can make application a little bit more readable and we can import and use pydantic which is supported in ingest.
import pydantic

# This type is essentially represent the result after we chunk and get the source for a particular PDF document.
class RAGChunkAndSrc(pydantic.BaseModel):
    # chunks: list[list[float]]
    chunks: list[str]
    source_id: str = None

# this is going to be the result after we upsert the document
class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

# when we are searching for some text that's  what we are going to have here
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

# This is different from the search result class from above this is the query that the user is actually sending to the endpoint
class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int

