# Used to log messages for debugging or tracking application behavior
# FastAPI framework for building web APIs
# Inngest is a serverless tool that lets you run background tasks and workflows (like sending emails, processing data, or scheduled jobs) reliably with automatic retries and no need to manage servers or queues.
# Inngest client library for defining and running event-driven workflows
# Provides FastAPI integration helpers for Inngest and it directly connects with the fastapi
# Experimental AI utilities from Inngest for workflow or agent features , meaning these utilities are provided from the inngest itself
# Loads environment variables from a .env file
# Generates unique identifiers, often used for request IDs or resource tracking
# Provides operating system utilities like reading environment variables
# Used to work with dates and timestamps
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import *

# load_dotenv loads the environment variables inside of the .env file.
load_dotenv()

# Now we are also going to start creating some of our clients
inngest_client = inngest.Inngest(
    # define the name of app
    app_id="rag_app",
    # Gets the logger instance named "uvicorn" for logging application messages
    # Uvicorn is a fast ASGI(Asynchronous Server Gateway Interface) server used to run FastAPI applications
    logger=logging.getLogger("uvicorn"),
    # is_production=False means the application is running in development mode, not in a live production environment.
    is_production=False,
    # Uses Inngest's PydanticSerializer to convert Pydantic models to JSON for event handling.
    # A Serializer is used to convert data into a format that can be stored, sent or transferred (usually JSON), and then convert it back when needed.
    # The Inngest PydanticSerializer is used to automatically convert Pydantic models to JSON and from JSON when sending or receiving events in inngest workflows.
    # Pydantic models are Python data classes that automatically validate and structure data using type hints - ensuring your data is always in the correct format.
    # age: int
    serializer=inngest.PydanticSerializer()
)

# We make an inngest function because we have this line right here which is "inngest.fast_api.server(app, inngest_client, []" , inngest will automatically kind of serve that function for us and it will connect to the inngest development server.
@inngest_client.create_function(
    # we give id a Readable Human Name
    fn_id="RAG: Ingest PDF",
    # we are going to specify the trigger, now the way the function is trigger or called is by some event being issued to the ingest server. It can be triggered from client which is frontend or another function.
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    # Allows at most 2 function runs to start per minute, excess events are queued and processed later (FIFO)
    throttle=inngest.Throttle(
        count=2, period=datetime.timedelta(minutes=1)
    ),
    # Strictly enforces 1 run per 4 hours per unique source_id, excess events are dropped/skipped (hard limit)
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
# ctx is context
# we created this function that's going to be effectively controlled by inngest and the development server.
async def rag_ingest_pdf(ctx: inngest.Context):
    # step 1: This step is going to be for loading the PDF.
    # Step 2: This step is going to be for embedding it and kind of chunking it or not chunking it , but adding it to the vector database.
    # The Idea is I have these two individual steps that I want to run inside of this function (rag_ingest_pdf) and then we need to load and to add to the vector database.
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        # load_and_chunk_pdf function is from data_loader.py module
        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(chunks = chunks, source_id = source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks) # from data_loader.py
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL,f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [ {"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage.upsert(ids = ids, vectors = vecs , payloads = payloads)

        # Number of chunks we ended up with ingested
        return RAGUpsertResult(ingested = len(chunks))

    # Rather than just calling the load function directly, which is what you would do if you are just working kind of standardly in python, we call await and first write a human readable name and then we put the function that we want to call and we have the ability to specify the output type
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type = RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type = RAGUpsertResult)

    # what it does is it just takes our pydantic model and convert it into JSON or a Python Dictionary.
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        # The reason for this is that if I want to query my database I need to do it with a vector, So whatever the question is that the user asked I need to embed that, So it is in the same format as everything in the vector database.
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run('embed-and-search',lambda: _search(question, top_k), output_type = RAGSearchResult)

    # I am just taking all of the context in a list and converting it into a string
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using the context above."
    )

    # call the ingest to call the ai model
    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model='gpt-4o-mini'
    )

    # response for the ai
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature":0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()

    return {"answer": answer, "sources": found.sources,"num_contexts": len(found.contexts)}

# creating the fastapi
app = FastAPI()

# Normally the client would sent the request directly to the api but here instead of doing that now we sent request to inngest then inngest sent the data in the desired format to the api.
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])