import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# Configure Streamlit page layout and title
st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# Cached singleton to reuse the same Inngest client across reruns
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

# Save the uploaded PDF file to a local "uploads" directory and return its path
def save_uploaded_pdf(file)-> Path:
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

# Send an asynchronous event to Inngest to trigger PDF ingestion
async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

# Main section: Upload a PDF and trigger ingestion
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader(label="Choose a PDF", type="PDF", accept_multiple_files=False)

if uploaded is not None:
    with st.spinner(text="Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # kick off the event and block until the send completes
        asyncio.run(send_rag_ingest_event(path))
        # small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()

# Main section: Ask questions about ingested PDFs
st.title("Ask a question about your PDFs")

# Send an event to Inngest to trigger a RAG query and return the event ID
async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]

# Helper to get the base URL for the local Inngest dev server (configurable via env)
def _inngest_api_base() -> str:
    # Local dev server default configurable via env
    return os.getenv("INNGEST_API_BASE","http://127.0.0.1:8288/v1")

# Fetch all function runs associated with a specific event ID from the Inngest API
def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data",[])

# Poll the Inngest API until the function run completes, fails, or times out
def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

# Form for submitting a question and retrieving an AI-genearted answer
with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner(text="Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observablitiy/workflow
            # Send the query event to Inngest
            event_id = asyncio.run(send_rag_ingest_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's  output
            # Wait for the function to complete and retrieve its output
            output = wait_for_run_output(event_id=event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")