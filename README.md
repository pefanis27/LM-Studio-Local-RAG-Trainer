# LM Studio Local RAG Trainer

A polished Streamlit-based local RAG application for LM Studio that lets you build reusable knowledge collections from your own files, retrieve relevant context with embeddings, and generate grounded answers through a clean multi-tab interface.

## Overview

LM Studio Local RAG Trainer is a single-file Python application designed for local Retrieval-Augmented Generation (RAG) workflows. It connects to LM Studio, manages available LLM and embedding models, creates local knowledge collections from user files, and answers questions using retrieved context instead of model fine-tuning.

The application is focused on practical local document-based question answering. It supports both reusable knowledge collections and one-off attachment-based questioning, making it suitable for personal knowledge bases, technical notes, code review support, study material exploration, and private local document search.

## Key Features

- Local RAG workflow on top of your own files
- Clean Streamlit interface with sidebar controls and tabbed navigation
- Automatic browser launch when the app starts
- Automatic LM Studio service detection and startup
- Optional background-first LM Studio startup logic
- Model discovery from the LM Studio API, with CLI fallback when needed
- LLM model selection and service model loading
- Embedding model discovery and automatic resolution
- Reusable local knowledge collections stored on disk
- Chunking and embedding generation for semantic retrieval
- Question answering against:
  - an existing knowledge collection, or
  - uploaded question attachments only
- Support for attachment-aware prompting
- Top-K retrieval control
- Temperature and max token controls
- Formatted answer panel with markdown rendering
- Copy-friendly plain text output
- Answer download as `.txt`
- Clear source display for retrieved passages and attachments
- Collection browsing, metadata display, filtering, and deletion

## How It Works

1. Launch the app.
2. The app opens in Streamlit and can automatically open in your default browser.
3. Configure the LM Studio Base URL, API key, and timeout from the sidebar.
4. Refresh available models or start the LM Studio service automatically.
5. Build a knowledge collection by uploading supported files and choosing an embedding model.
6. The app extracts text, normalizes it, splits it into chunks, creates embeddings, and stores the collection locally.
7. Open the Ask tab and choose either a knowledge collection or attachment-only mode.
8. Ask a question, optionally attach extra files, and generate an answer grounded in the retrieved context.
9. Review the formatted answer, inspect the sources used, copy the text, or download the result.

## Interface Structure

### Sidebar

The sidebar is used for service and model management. It includes:

- Base URL configuration
- API key configuration
- Timeout settings
- Model refresh actions
- Service status display
- Service model selection
- Service startup and model loading actions
- Cached lists of LLM and embedding models

### Tabs

#### 1. Overview
The dashboard tab provides a quick summary of the application, the latest answer preview, and the recommended usage flow.

#### 2. Build Knowledge
This tab is used to create or update a local knowledge collection.

You can:
- define a collection name
- choose an embedding model
- upload files
- configure chunk size
- configure chunk overlap
- configure embedding batch size
- build the collection locally

#### 3. Ask
This tab is used for question answering.

You can:
- choose a collection
- use attachment-only mode without a collection
- choose an LLM model
- choose an embedding model for retrieval
- set Top-K retrieval
- set temperature
- set max tokens
- write a question
- upload additional attachments as extra context
- generate a formatted answer with references

#### 4. Collections
This tab is used to manage saved collections.

You can:
- browse available collections
- inspect collection metadata
- see document counts
- see chunk counts
- view the embedding model used
- view creation timestamps
- delete collections

## Supported File Types

The application supports a broad range of text, document, data, and code files.

### Text and markup
- `.txt`, `.md`,`.markdown`,`.rst`,`.log`

### Documents
- `.pdf`,`.docx`,`.rtf`

### Data and structured text
- `.json`,`.jsonl`,`.csv`,`.tsv`,`.xml`,`.yaml`,`.yml`,`.toml`,`.ini`
- `.cfg`,`.env`,`.sql`,`.tex`,`.html`,`.htm`

### Code and notebooks
- `.py`,`.ipynb`,`.js`,`.ts`,`.tsx`,`.jsx`,`.java`
- `.c`,`.cpp`,`.cc`,`.h`,`.hpp`,`.cs`,`.go`,`.rs`
- `.php`,`.rb`,`.swift``.kt`,`.scala`,`.sh`,`.bat`,`.ps1`

## Local Storage Layout

Knowledge collections are stored locally under the application data directory.

Each collection contains:
- original uploaded source files
- collection metadata
- chunk definitions
- normalized embedding vectors

This makes collections reusable between sessions without rebuilding them every time.

## Answering Behavior

When you ask a question using a saved collection, the application:

- embeds the query
- performs semantic search on stored chunk embeddings
- selects the top relevant chunks
- builds a context block from those results
- optionally appends uploaded question attachments
- sends the final prompt to LM Studio
- returns the generated answer together with the supporting sources

When no collection is selected, the app can still answer using readable uploaded attachments as direct context.

## Model Handling

The application includes flexible model handling for LM Studio:

- reads available models from the API when the service is active
- falls back to the `lms` CLI when needed
- distinguishes between LLM and embedding models
- supports automatic model selection
- can request model loading before building collections or answering questions

## Requirements

### Software
- Python 3.10+
- LM Studio installed locally
- At least one LLM model downloaded in LM Studio
- At least one embedding model downloaded in LM Studio

### Python packages
```bash
pip install streamlit numpy requests python-docx pypdf markdown
