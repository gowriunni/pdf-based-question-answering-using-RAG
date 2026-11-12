
from django.shortcuts import render
from django.conf import settings

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from functools import lru_cache

import os


# Load API key from the project's .env file
load_dotenv(settings.BASE_DIR / ".env")


# Path where FAISS database will be stored
FAISS_INDEX_PATH = os.path.join(
    settings.BASE_DIR,
    "faiss_index"
)


# STEP 1: Load embedding model only once
@lru_cache(maxsize=1)
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# STEP 2: Load Groq LLM only once
@lru_cache(maxsize=1)
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


# STEP 3: Main Django view
def home(request):

    message = ""
    answer = ""
    retrieved_chunks = []

    if request.method == 'POST':

        # Get uploaded PDF and question
        pdf_file = request.FILES.get('pdf_file')
        question = request.POST.get('question', '').strip()

        # --------------------------------
        # STEP 4: PDF UPLOAD AND PROCESSING
        # --------------------------------

        if pdf_file:

            try:

                # Read PDF
                reader = PdfReader(pdf_file)

                # Extract text from PDF
                extracted_text = ""

                for page in reader.pages:
                    extracted_text += page.extract_text() or ""

                if extracted_text.strip():

                    # Split text into smaller chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )

                    chunks = text_splitter.split_text(
                        extracted_text
                    )

                    # Generate embeddings
                    embedding_model = get_embedding_model()

                    # Store embeddings in FAISS
                    vector_store = FAISS.from_texts(
                        chunks,
                        embedding_model
                    )

                    # Save FAISS database
                    vector_store.save_local(
                        FAISS_INDEX_PATH
                    )

                    message = (
                        "PDF uploaded and embeddings "
                        "generated successfully!"
                    )

                else:

                    message = (
                        "PDF uploaded, but no readable "
                        "text was found."
                    )

            except Exception as e:

                message = f"Error processing PDF: {e}"

        # --------------------------------
        # STEP 5: RETRIEVE RELEVANT CHUNKS
        # --------------------------------

        elif question:

            try:

                # Check whether FAISS index exists
                if os.path.exists(
                    os.path.join(
                        FAISS_INDEX_PATH,
                        "index.faiss"
                    )
                ):

                    # Load saved FAISS database
                    vector_store = FAISS.load_local(
                        FAISS_INDEX_PATH,
                        get_embedding_model(),
                        allow_dangerous_deserialization=True
                    )

                    # Search for relevant PDF chunks
                    retrieved_chunks = (
                        vector_store.similarity_search(
                            question,
                            k=3
                        )
                    )

                    if not retrieved_chunks:

                        message = (
                            "No relevant information found."
                        )

                    else:

                        # --------------------------------
                        # STEP 6: PREPARE CONTEXT
                        # --------------------------------

                        context = "\n\n".join(
                            chunk.page_content
                            for chunk in retrieved_chunks
                        )

                        # --------------------------------
                        # STEP 7: GENERATE ANSWER USING LLM
                        # --------------------------------

                        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the
information provided in the context.

Do not make up information.

If the answer is not available in the context,
say:
"The answer is not available in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""

                        # Get Groq LLM
                        llm = get_llm()

                        # Generate response
                        response = llm.invoke(prompt)

                        answer = response.content

                        message = (
                            "Answer generated successfully!"
                        )

                else:

                    message = (
                        "Please upload a PDF before asking "
                        "a question."
                    )

            except Exception as e:

                message = f"Error generating answer: {e}"

        else:

            message = (
                "Please upload a PDF or enter a question."
            )

    # --------------------------------
    # STEP 8: DISPLAY RESULTS
    # --------------------------------

    return render(request, 'home.html', {

        'message': message,

        'answer': answer,

        'retrieved_chunks': retrieved_chunks

    })

