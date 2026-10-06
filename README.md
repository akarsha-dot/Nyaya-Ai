# Nyaya AI - Legal Assistant using RAG

## 📌 Project Overview

Nyaya AI is an AI-powered legal assistant that uses **Retrieval-Augmented Generation (RAG)** to provide answers to legal questions based on uploaded legal documents.

The system retrieves relevant information from legal documents and uses an AI model to generate context-aware responses.

## 🎯 Objectives

- Provide an AI-based assistant for legal queries.
- Retrieve relevant information from legal documents.
- Generate accurate and context-aware answers.
- Reduce the time required to search through large legal documents.
- Make legal information easier to access and understand.

## 🔄 How It Works

The project follows a Retrieval-Augmented Generation pipeline:

1. User uploads a legal document.
2. The document is converted into text.
3. The text is divided into smaller chunks.
4. Text chunks are converted into embeddings.
5. Embeddings are stored in a vector database.
6. User asks a legal question.
7. The system retrieves the most relevant document sections.
8. The retrieved information is provided to the AI model.
9. The AI generates an answer based on the retrieved context.

## 🏗️ Architecture

User
↓
Web Interface
↓
Document Upload
↓
Text Extraction
↓
Text Chunking
↓
Embeddings
↓
Vector Database
↓
Relevant Document Retrieval
↓
AI / LLM
↓
Generated Answer

## 🛠️ Technologies Used

- Python
- Flask
- Retrieval-Augmented Generation (RAG)
- Natural Language Processing (NLP)
- Large Language Models (LLM)
- PDF Document Processing
- Vector Database
- HTML
- CSS
- JavaScript

## 📂 Project Structure

```text
Nyaya-Ai/
│
├── project.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── data/
│   └── legal_documents/
│
└── uploads/
