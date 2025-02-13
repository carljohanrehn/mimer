#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: gemini-rag-chroma-app.py

Description:
    This module facilitates interaction between a Chroma database and the Gemini Generative AI model.
    It provides functionality to execute embedding-based queries on the database, generate
    context-aware responses from the AI, and retrieve relevant source documents for enhanced
    user interaction in a Retrieval-Augmented Generation (RAG) workflow.

Features:
    - Read and persist collections in the Chroma database.
    - Interface with the Google Gemini Generative AI model for content generation.
    - Construct and send context-specific prompts to the AI model.
    - Retrieve and display source documents along with AI-generated responses.
    - Support for command-line arguments to configure:
        - Persist directory for Chroma collections.
        - Collection name to query and store data.
        - Gemini model details for response generation.
    - API key configuration for seamless interaction with the Google Gemini service.

Dependencies:
    - chromadb
    - google-generative-ai

Author:
    Carl Johan Rehn
    Email: care02@gmail.com

Version:
    1.0.0

Date:
    2025-01-22
"""

import argparse
import os

from dotenv import load_dotenv
from typing import Any, Callable, Optional

import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

from load_config import load_config

load_dotenv('config.env')

# Define a global constant for the default model name
DEFAULT_MODEL_NAME: str = "gemini-2.0-flash-exp"

# TODO Exempel på prompter.
# Vad kan du berätta om Bungenäs på Gotland, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Vad kan du berätta om Skånes kust, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Vad kan du berätta om Skånesleden, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Vad kan du berätta om Dalarnas natur, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Vad kan du berätta om naturern kring platåberget Kinnekulle i Västergötland, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Vad kan du berätta om vandringslederna vid Kinnekulle i Västergötland, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Vad kan du berätta om vandringslederna i Stockholms skärgård, sök i lokal kontext först, saknas information använd din egen kunskap. Svara på svenska.
# Mälarmården var skogslandet i Södermanland, kan du beskriva denna del av landskapet utifrån den lokala kontexten och dina egna kunskaper? Svara på svenska.
# Hur skulle du beskriva landskapet kring sjön Tåkern i Östergötland utifrån den lokala kontexten och dina egna kunskaper? Svara på svenska.

# Var ligger Brantevik i Skåne? Komplettera med din egen kunskap.
# Kan du kortfattat beskriva naturen kring Brantevik i Skåne och på Österlen? Komplettera med din egen kunskap. Skriv ett utkast till en essä.

# Beskriv Sala silvergruva kortfattat och sakligt, håll dig till de centrala aspekterna. Använd sakprosa och essä som språklig stil.


def build_prompt(
        query: str,
        context: list[str],
        model: genai.GenerativeModel
) -> str:
    """
    Builds a prompt for the LLM.

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    Args:
        query: The original query.
        context: The context of the query, returned by embedding search.
        model: The generative model that will be used.

    Returns:
        A prompt for the LLM (str).
    """

    base_prompt = (
        "Du är en hjälpsam assistent. Vänligen besvara frågan nedan."
        "\n\n"
        "Prioritera alltid att använda den tillhandahållna 'kontextinformationen' i ditt svar. "
        "Om kontextinformationen inte är tillräcklig för att helt besvara frågan, "
        "komplettera med din egen kunskap och ange tydligt vilka delar som bygger på din interna kunskap. "
        "Svara alltid på svenska. "
        "Gör en tydlig skillnad mellan information som kommer från kontexten och din egen kunskap."
        "\n\n"
        "Om det inte finns relevant information från vare sig kontexten eller din kunskap, säg det tydligt."
    )

    context_section = (
        "Context Documents:\n" + "\n".join(context)
        if context else "No relevant documents were retrieved from the database."
    )

    return f"{base_prompt}\n\n{context_section}\n\nQuestion: {query}"


def get_gemini_response(
        query: str,
        context: list[str],
        model: genai.GenerativeModel
) -> str:
    """
    Queries the Gemini API to get a response to the question.

    Args:
        query: The original query.
        context: The context of the query, returned by embedding search.
        model: The generative model that will be used.

    Returns:
        A response to the question.
    """

    context_missing = not context or len(context) == 0

    # Step 2: Build the prompt
    if context_missing:
        # If no context exists, explicitly instruct the LLM to generate from its knowledge
        context = ["No relevant context was provided."]  # Placeholder to shift focus
    else:
        pass  # Use retrieved documents

    # Build the prompt
    prompt: str = build_prompt(query, context, model)

    # Get response from the LLM
    response = model.generate_content(prompt)

    return response.text


def main(
    collection_name: str = "documents_collection",
    persist_directory: str = "chroma_storage",
    model_name: str = DEFAULT_MODEL_NAME
) -> None:
    """
    Main function to interact with the Chroma database and Gemini model.

    Args:
      collection_name: The name of the Chroma collection to query.
      persist_directory: The directory where Chroma DB is persisted.
      model_name: The Gemini model name.
    """
    # Initialize the model with the value provided by the user, or default.
    model: genai.GenerativeModel = genai.GenerativeModel(model_name)

    # Check if the GOOGLE_API_KEY environment variable is set. Prompt the user to set it if not.
    google_api_key: Optional[str] = None

    if "GOOGLE_API_KEY" not in os.environ:
        gapikey: str = input("Please enter your Google API Key: ")
        genai.configure(api_key=gapikey)
        google_api_key = gapikey
    else:
        google_api_key: str = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=google_api_key)

    # Instantiate a persistent chroma client in the persist_directory.
    # This will automatically load any previously saved collections.
    # Learn more at docs.trychroma.com
    client: chromadb.PersistentClient = chromadb.PersistentClient(path=persist_directory)

    # create embedding function
    embedding_function: embedding_functions.GoogleGenerativeAiEmbeddingFunction = (
        embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=google_api_key,
            task_type="RETRIEVAL_QUERY"
        )
    )

    # Get the collection.
    collection: chromadb.Collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # We use a simple input loop.
    while True:
        # Get the user's query
        query: str = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        # Query the collection to get the 5 most relevant results
        results: chromadb.QueryResult = collection.query(
            query_texts=[query],
#            n_results=5,
            n_results=25,
            include=["documents", "metadatas"]
        )

        # TODO Filter metadatas here...
        # sources: str = "\n".join(
        #     [
        #         f"{result['filename']}: line {result['line_number']}"
        #         for result in results["metadatas"][0]  # type: ignore
        #     ]
        # )

        # Get local context from Chroma DB
        context: list[str] = list(dict.fromkeys(results["documents"][0]))

        # Get the response from Gemini
        response: str = get_gemini_response(query, context, model)

        # Output, with sources
        print(response)
        print("\n")

        # TODO ...
        # print(f"Source documents:\n{sources}")
        # print("\n")


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Use document collection from Chroma collection with LLM model"
    )

    # Add arguments
    parser.add_argument(
        "--config_path",
        type=str,
        default='./chroma/rosenberg/config.yaml',
        help="The path to your YAML configuration file",
    )
    parser.add_argument(
        "--active_tag",
        type=str,
        default='',
        help="The active tag for which a Chroma collection will be created",
    )
    parser.add_argument(
      "--model_name",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help="The name of the Gemini model to use",
    )

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    config_path: str = args.config_path
    active_tag: str = args.active_tag

    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist.")

    config: dict[str, dict[str, Any]] = load_config(config_path, active_tag)

    active_tag_dict: dict[str, Any] = config[active_tag]

    main(
        collection_name=active_tag_dict["collection_name"],
        persist_directory=active_tag_dict["persist_directory"],
        model_name = args.model_name
    )
