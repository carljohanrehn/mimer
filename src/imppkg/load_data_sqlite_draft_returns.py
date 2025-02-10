#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module Name: gemini-rag-chroma-app.py

Description:
    This module is designed to load documents from a specified directory into
    a Chroma collection for use in a retrieval-augmented generation (RAG) pipeline.
    It utilizes Google's Generative AI API to generate embeddings for efficient
    semantic search and retrieval.

Features:
    - Scans a directory for text-based documents.
    - Generates embeddings using Google's Generative AI API.
    - Stores embeddings in a Chroma collection for future retrieval.

Dependencies:
    - chromadb
    - google-generative-ai

Usage:
    Run the script and specify the directory containing the documents to be processed
    and indexed into the Chroma collection. Adjust settings as needed for document
    preprocessing or embedding configuration.

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
import ast
import time

from dotenv import load_dotenv
from typing import Any, Callable, Optional

from returns.result import Result, Success, Failure
from returns.pipeline import is_successful

import pandas as pd
import sqlite3

from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

from load_config import load_config

# Global constant or environment variable for embedding provider type
load_dotenv()
EMBEDDING_PROVIDERS: list[str] = ['google', 'openai', 'huggingface']
EMBEDDING_PROVIDER: str = os.getenv('EMBEDDING_PROVIDER', 'google')


def get_embedding_function(api_key: str) -> Result[Callable, str]:
    """
    Creates and retrieves an embedding function based on the configured embedding provider.

    The embedding provider must be one of the predefined types ('google', 'openai', or 'huggingface')
    to ensure compatibility with supported embedding APIs. If an unsupported provider is set,
    the function returns a failure result.

    Args:
        api_key (str): The API key used to authenticate with the embedding provider.

    Returns:
        Result[Callable, str]: A `Success` result containing the embedding function
        if the provider is supported, or a `Failure` result with an error message
        if the provider is unsupported.
    """

    if EMBEDDING_PROVIDER == 'google':
        return Success(embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key))
    elif EMBEDDING_PROVIDER == 'openai':
        return Success(embedding_functions.OpenAiEmbeddingFunction(api_key=api_key))
    elif EMBEDDING_PROVIDER == 'huggingface':
        return Success(embedding_functions.HuggingFaceEmbeddingFunction(api_key=api_key))
    else:
        return Failure(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def query_database(database_path: str, sql_query: str) -> Result[pd.DataFrame, str]:
    """
    Executes a query on a SQLite database and retrieves the results as a pandas DataFrame.

    Args:
        database_path (str): The file path to the SQLite database.
        sql_query (str): The SQL query to execute on the database.

    Returns:
        Result[pd.DataFrame, str]:
            - Success: Contains the DataFrame with the query results.
            - Failure: Contains an error message if the database query fails.

    Exceptions:
        Catches general exceptions such as database connection or query errors.
    """

    try:
        with sqlite3.connect(database_path) as conn:
            data_frame: pd.DataFrame = pd.read_sql_query(sql_query, conn)
        return Success(data_frame)
    except Exception as e:
        return Failure(f"Failed to query database: {e}")


def process_dataframe(
        data_frame: pd.DataFrame,
        document_column: str = 'content',
        metadata_columns: Optional[list[str]] = None,
        show_progress: bool = True
) -> Result[tuple[list[str], list[dict[str, Any]]], str]:
    """
    Processes a pandas DataFrame containing documents and optionally metadata, extracting and
    validating the specified columns.

    Args:
        data_frame (pd.DataFrame): The input pandas DataFrame containing document records.
        document_column (str, optional): The name of the column in the DataFrame that contains
            the document text. Defaults to 'content'.
        metadata_columns (Optional[list[str]], optional): A list of column names in the DataFrame
            to be extracted as metadata. Defaults to None.
        show_progress (bool, optional): Whether to display a progress bar during processing. Defaults to True.

    Returns:
        Result[tuple[list[str], list[dict[str, Any]]], str]:
            - Success: A tuple containing:
                - A list of valid document texts extracted from the DataFrame.
                - A list of metadata dictionaries corresponding to the documents.
            - Failure: An error message, if the specified document column or metadata columns
              are missing or invalid.
    """

    # Validate document column
    if document_column not in data_frame.columns:
        return Failure(f"'{document_column}' not found in database columns: {list(data_frame.columns)}")

    # Validate all metadata columns exist
    if metadata_columns:
        missing_columns = [col for col in metadata_columns if col not in data_frame.columns]
        if missing_columns:
            return Failure(
                f"Metadata columns {missing_columns} not found in database columns: {list(data_frame.columns)}"
            )

    # Extract documents and metadata
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for _, row in tqdm(
            data_frame.iterrows(),
            total=len(data_frame),
            desc="Processing SQLite rows",
            disable=not show_progress,
    ):
        # Skip rows with invalid or empty document content
        document = row.get(document_column)
        if pd.isna(document) or not isinstance(document, str):
            continue
        documents.append(document)

        # Extract metadata if specified
        metadata = {}
        if metadata_columns:
            metadata = {col: row.get(col) for col in metadata_columns}
        metadatas.append(metadata)

    # Return Success with the processed data
    return Success((documents, metadatas))


def read_documents(
        database_path: str,
        sql_query: Optional[str] = None,
        document_column: str = 'content',
        metadata_columns: Optional[list[str]] = None
) -> Result[tuple[list[str], list[dict[str, Any]]], str]:
    """
    Reads records from a SQLite database file, extracts document content and optional metadata,
    and returns them in a structured format.

    Args:
        database_path (str): Path to the SQLite database file (.sqlite) to be queried.
        sql_query (Optional[str]): SQL query to filter and retrieve specific rows from the database.
                                   Defaults to `SELECT * FROM documents` if not provided.
        document_column (str): The name of the column in the query result to extract as
                               the main document content. Defaults to 'content'.
        metadata_columns (Optional[list[str]]): A list of additional column names (from the query result)
                                                to extract as metadata associated with content.

    Returns:
        Result[tuple[list[str], list[dict[str, Any]]], str]:
            - Success (tuple):
                - documents: A list containing the extracted document texts.
                - metadatas: A list of dictionaries, each holding metadata for the corresponding document.
            - Failure: Contains an error message if reading or processing fails.
    """

    # Validate the database path
    if not os.path.isfile(database_path) or not database_path.endswith('.sqlite'):
        return Failure(
            f"Invalid database file: '{database_path}'. Ensure the file exists and has a '.sqlite' extension."
        )

    # Set the default SQL query if not provided
    sql_query = sql_query or 'SELECT * FROM documents'

    # Connect to the database and read data into a Pandas DataFrame
    result_df: Result[pd.DataFrame, str] = query_database(
        database_path,
        sql_query
    )

    # Bind result of query to the final `Result`
    return result_df.bind(
        lambda df: process_dataframe(df, document_column, metadata_columns)
    )


def get_api_key(
        provider: str,
        env_var: Callable[[], Optional[str]],
        input_provider: Callable[[], Optional[str]]
) -> Result[str, str]:
    """
    Retrieves an API key dynamically for a given provider (i.e., the external service requiring the API key)
    from either an environment variable or user input, wrapped in a `Result` monad for railway-oriented error handling.

    Args:
        provider (str): The name of the external service requiring the API key.
        env_var (Callable[[], Optional[str]]): A callable that retrieves the API key from environment variables, expected
            to return the API key as a string or `None` if not available.
        input_provider (Callable[[], Optional[str]]): A callable that retrieves the API key from user input, expected
            to return the API key as a string or `None` if not provided.

    Returns:
        Result[str, str]: A `Success` with the valid API key, or a `Failure` with the error message if it is missing.
    """

    api_key: Optional[str] = env_var()

    if api_key is None or not api_key:
        api_key = input_provider()

    if api_key:
        return Success(api_key)
    else:
        return Failure(f'{provider.capitalize()} API key not provided. Ensure it is set in the environment or input.')


def create_chroma_collection(
        client: chromadb.PersistentClient,
        collection_name: str,
        api_key: str
) -> Result[chromadb.Collection, str]:
    """
     Creates or retrieves a Chroma collection and configures it with a dynamically generated embedding function.

    Args:
        client (chromadb.PersistentClient): An instance of Chroma's PersistentClient to manage database interactions.
        collection_name (str): The unique name assigned to the Chroma collection.
        api_key (str): The API key used for authentication and embedding function initialization.

    Returns:
        Result[chromadb.Collection, str]:
            - Success: Contains the Chroma collection object if successfully created or retrieved.
            - Failure: Contains an error message detailing the failure reason.
        or a `Failure` with an error message.
    """

    # Dynamically fetch the embedding function, which is wrapped in a `Result`
    try:
        embedding_fn_result: Result[Callable, str] = get_embedding_function(api_key)
    except Exception as e:
        return Failure(f"Failed to create EmbeddingFunction: {e}")

    try:
        # Get or create the collection with the embedding function
        collection: chromadb.Collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn_result.unwrap()  # Safely unwrap the embedding function
        )
        return Success(collection)
    except Exception as e:
        # Handle any issues at this stage and return as `Failure`
        return Failure(f"Failed to create Chroma collection: {str(e)}")


def process_batch(
        collection: chromadb.Collection,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        count: int,
        batch_index: int
) -> Result[None, str]:
    """
    Processes a batch of documents, metadata, and IDs to be added to the specified
    ChromaDB collection. Handles metadata normalization by ensuring all `None` values
    are replaced with empty strings. Each batch is limited to a size of 100 entries,
    and unique IDs are generated for the provided documents.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to which
            the documents, metadata, and IDs will be added.
        documents (list[str]): A list of document strings to be added to the
            collection. The number of documents in this list should correspond
            to the metadata provided.
        metadatas (list[dict[str, Any]]): A list of dictionaries containing the
            metadata for each document. Each dictionary corresponds to a single
            document.
        count (int): The starting count used to generate unique IDs for the
            documents. This ensures newly added documents have unique identifiers.
        batch_index (int): The starting index of the batch being processed. This
            determines which subset of documents, metadata, and IDs will be handled.

    Returns:
        Result[None, str]: A `Success` result if the batch was processed and added
            to the collection successfully. A `Failure` result containing an error
            message if an exception occurred during processing.

    Raises:
        Exception: Captures any runtime exception during the batch processing and
            converts it into a `Failure` result with an appropriate error message.
    """

    try:
        fixed_metadatas: list[dict[str, str]] = [
            {
                key: (value if value is not None else "") for key, value in metadata.items()
            } for metadata in metadatas[batch_index: batch_index + 100]
        ]

        # Generate unique IDs for the new documents
        ids: list[str] = [str(i) for i in range(count, count + len(documents))]

        collection.add(
            ids=ids[batch_index: batch_index + 100],
            documents=documents[batch_index: batch_index + 100],
            metadatas=fixed_metadatas,
        )

        return Success(None)
    except Exception as e:
        return Failure(f"Error processing batch at index {batch_index}: {str(e)}")


def process_batches_with_progress(
        elements: list[tuple[str, dict[str, Any]]],
        batch_size: int,
        process_batch_fn: Callable[[list[tuple[str, dict[str, Any]]]], Result[None, str]],
        description: str = "Processing Batches"
) -> Result[None, str]:
    """
    Processes elements in batches with a progress bar using tqdm.
    Stops processing immediately upon the first failure.

    Args:
        elements (list[tuple[str, dict[str, Any]]]): List of items (documents and metadatas) to process in batches.
        batch_size (int): Maximum size of each batch.
        process_batch_fn (Callable[[list[tuple[str, dict[str, Any]]]], Result[None, str]]): A function to process each batch.
        description (str): Description for the progress bar.

    Returns:
        Result[None, str]: Success if all batches process without error, or Failure with the first error message.
    """

    for start_index in tqdm(
            range(0, len(elements), batch_size),
            desc=description,
            unit="batch",
            leave=False
    ):
        # Get the current batch
        batch = elements[start_index:start_index + batch_size]

        # Process the batch
        result = process_batch_fn(batch)

        # Stop processing and return failure if any batch fails
        if not is_successful(result):
            return result

    # If all batches succeed
    return Success(None)


def load_documents_into_collection(
        collection: chromadb.Collection,
        documents: list[str],
        metadatas: list[dict[str, Any]]
) -> Result[None, str]:
    """
    Loads a set of documents and their corresponding metadata into a Chroma collection
    in batches. The function processes the input data in chunks with a progress bar
    to optimize memory usage and ensure efficient transaction handling. If any batch
    fails during the process, it immediately stops further execution and returns an error.

    Args:
        collection (chromadb.Collection): The Chroma collection where the documents
            and metadata will be added.
        documents (list[str]): A list of string documents to be added to the collection.
        metadatas (list[dict[str, Any]]): A list of metadata dictionaries corresponding
            to each document in the documents list.

    Returns:
        Result[None, str]: Success with the number of documents added, or Failure with an error message.
    """

    # Process the batches with progress
    def process_single_batch(
            batch: list[tuple[str, dict[str, Any]]]
    ) -> Result[None, str]:

        # Unpack the batch into documents and metadata for processing
        batch_documents = [item[0] for item in batch]
        batch_metadatas = [item[1] for item in batch]
        batch_index = initial_count  # Adjust batch_index logic if necessary

        return process_batch(
            collection,
            batch_documents,
            batch_metadatas,
            initial_count,
            batch_index
        )

    try:
        # Get the initial count from the collection
        initial_count: int = collection.count()

        # Pair documents and metadatas for batch processing
        elements = list(zip(documents, metadatas))

        # Use the helper function with tqdm for batching
        result = process_batches_with_progress(
            elements=elements,
            batch_size=100,
            process_batch_fn=process_single_batch,
            description="Adding documents"
        )

        # Check result of the batch processing
        if not is_successful(result):
            return result  # Return the Failure directly

        # Get the final count and return success
        final_count = collection.count()
        return Success(f"Added {final_count - initial_count} documents")

    except Exception as e:
        return Failure(f"Error loading documents into collection: {str(e)}")


def get_api_key_provider() -> Result[str, str]:
    """
    Prompt user or fetch API key from environment variables functionally.

    Returns:
        Result[str, str]: Success with the API key if found/provided,
                          otherwise Failure with an error message.
    """
    # Fetch the key from environment variables
    maybe_api_key: Optional[str] = os.getenv(f"{EMBEDDING_PROVIDER.upper()}_API_KEY")

    if maybe_api_key:
        return Success(maybe_api_key)  # Return the API key if found

    try:
        # Prompt the user for API key only if it wasn't in the environment
        api_key_from_input = input(f"Please enter your {EMBEDDING_PROVIDER.capitalize()} API Key: ").strip()
        if api_key_from_input:
            return Success(api_key_from_input)
        else:
            return Failure("No API key was provided.")
    except Exception as e:
        return Failure(f"Error occurred while fetching API key: {e}")


def create_client(
        persist_directory: str
) -> Result[chromadb.PersistentClient, str]:
    """
    Creates a PersistentClient object for managing persistence in a given directory.

    This function attempts to initialize a `PersistentClient` from the `chromadb` library
    using the provided `persist_directory`. If successful, it returns the `PersistentClient`
    wrapped in a `Success` result. If an exception is raised during the initialization, the
    function returns a `Failure` result along with a descriptive error message.

    Args:
        persist_directory (str): Path to the directory where persistence should be stored.

    Returns:
        Result[chromadb.PersistentClient, str]: A `Success` result containing a
        `PersistentClient` instance if creation succeeds, or a `Failure` result with an
        error message if initialization fails.
    """
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        return Success(client)
    except Exception as e:
        return Failure(f"Failed to create PersistentClient: {e}")


def create_collection_workflow(
        client_result: Result[chromadb.PersistentClient, str],
        api_key_result: Result[str, str],
        collection_name: str
) -> Result[chromadb.Collection, str]:
    """
    Chains the series of tasks to create or retrieve a ChromaDB collection:
    - Resolves `PersistentClient`.
    - Fetches or validates API key.
    - Ultimately produces or retrieves the ChromaDB collection.

    Args:
        client_result: Result of creating the ChromaDB PersistentClient.
        api_key_result: Result of fetching/validating the API key.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Result[chromadb.Collection, str]: Either a Success wrapping the collection,
        or Failure with an error message.
    """
    return client_result.bind(
        lambda client: api_key_result.bind(
            lambda api_key: create_chroma_collection(client, collection_name, api_key)
        )
    )


def create_final_result_workflow(
        collection_workflow_result: Result[chromadb.Collection, str],
        documents_and_metadata_result: Result[tuple[list[str], list[dict[str, Any]]], str]
) -> Result[None, str]:
    """
    Chains the series of tasks to create the final result by:
    - Resolving a ChromaDB collection (via collection workflow).
    - Loading documents and metadata into the resolved collection.

    Args:
        collection_workflow_result: Result of the collection workflow, producing a `chromadb.Collection`.
        documents_and_metadata_result: Result containing documents and their corresponding metadata.

    Returns:
        Result[None, str]: Success if all tasks complete successfully, or Failure with the relevant error message.
    """
    return collection_workflow_result.bind(
        lambda collection: documents_and_metadata_result.bind(
            lambda documents_and_metadata: load_documents_into_collection(
                collection, *documents_and_metadata
            )
        )
    )


def main(
        collection_name: str = 'documents_collection',
        persist_directory: str = 'chroma_storage',
        database_path: str = '',
        sql_query: Optional[str] = None,
        document_column: str = 'content',
        metadata_columns: Optional[list[str]] = None,
) -> Result[None, str]:
    """
    Main function to execute the document loading and management workflow for ChromaDB.
    This includes creating a collection, reading documents and metadata,
    and loading the documents into the collection.

    Args:
        collection_name: The name of the collection in ChromaDB.
        persist_directory: Directory for ChromaDB data persistence.
        database_path: Path to the database file used as a source.
        sql_query: SQL query to fetch documents and metadata from the database.
        document_column: Column containing document content.
        metadata_columns: List of database columns for metadata.

    Returns:
        Result[None, str]: Success if the workflow completes, or Failure with the error message.
    """

    # Step 1: Fetch documents and their corresponding metadata from the database.
    result_documents_and_metadata = read_documents(
        database_path,
        sql_query,
        document_column,
        metadata_columns
    )

    # Step 2: Obtain the API key for the embedding provider,
    # either from an environment variable or user input.
    result_api_key = get_api_key(
        provider=EMBEDDING_PROVIDER,
        env_var=lambda: os.getenv(f"{EMBEDDING_PROVIDER.upper()}_API_KEY"),
        input_provider=get_api_key_provider,
    )

    # Step 3: Initialize a persistent ChromaDB client for database operations.
    result_client = create_client(persist_directory)

    # Step 4: Perform a workflow to create or retrieve the specified ChromaDB collection.
    # This involves validating the provided API key and ensuring the collection is ready for use.
    result_collection_workflow = create_collection_workflow(
        result_client,
        result_api_key,
        collection_name
    )

    # Step 5: Execute the workflow to finalize the process.
    # This workflow loads the fetched documents and metadata into the resolved ChromaDB collection,
    # ensuring all steps are completed and the final result is generated.
    final_result = create_final_result_workflow(
        result_collection_workflow,
        result_documents_and_metadata
    )

    return final_result


if __name__ == '__main__':

    # TODO read from different "books", structure folders and files

    # Read the data directory, collection name, and persist directory
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Load documents from a SQLite database into a Chroma collection'
    )

    # Add arguments
    parser.add_argument(
        '--config_path',
        type=str,
        default='./chroma/rosenberg/config.yaml',
        help='The path to your YAML configuration file',
    )
    parser.add_argument(
        '--active_tag',
        type=str,
        default='',
        help='The active tag for which a Chroma collection will be created',
    )

    # Parse arguments
    args: argparse.Namespace = parser.parse_args()

    config_path: str = args.config_path
    active_tag: str = args.active_tag
    active_tag = 'vastmanland'

    if not os.path.exists(config_path):
        raise ValueError(f'{config_path} does not exist.')

    config: dict[str, dict[str, Any]] = load_config(config_path, active_tag)

    skip_tag: set[str] = {}
    # skip_tag: set[str] = {
    #     'blekinge',
    #     'gotland',
    #     'gavleborg',
    #     'goteborg_bohus',
    #     'halland',
    #     'jamtland',
    #     'jonkoping',
    #     'kalmar',
    #     'kopparberg',
    #     'kristianstad',
    #     'kronoberg',
    #     'malmohus',
    #     'norrbotten',
    #     'skaraborg',
    #     'sodermanland',
    #     'stockholm',
    #     'uppsala',
    #     'varmland',
    #     'vasterbotten',
    #     'vasternorrland',
    #     'vastmanland',
    #     'alvsborg',
    #     'orebro',
    # }

    for tag, config in config.items():
        if not tag in skip_tag:

            print(f'Loading documents for tag: {tag}')

            main(
                database_path=config['database_path'],
                collection_name=config['collection_name'],
                persist_directory=config['persist_directory'],
                sql_query=config['sql_query'],
                document_column=config['document_column'],
                metadata_columns=ast.literal_eval(str(config['metadata_columns'])),
            )

            # Delay between iterations to let Chroma finish
            time.sleep(1)  # Adjust the duration as needed (e.g., start with 1 second).

