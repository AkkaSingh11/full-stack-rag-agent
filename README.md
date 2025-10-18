# Gemini Fullstack LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent is designed to perform comprehensive research on a user's query by dynamically generating search terms, querying the web using Google Search, reflecting on the results to identify knowledge gaps, and iteratively refining its search until it can provide a well-supported answer with citations. This application serves as an example of building research-augmented conversational AI using LangGraph and Google's Gemini models.

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend.
- 🧠 Powered by a LangGraph agent for advanced research and conversational AI.
- 🔍 Dynamic search query generation using Google Gemini models.
- 🌐 Integrated web research via Google Search API.
- 🤔 Reflective reasoning to identify knowledge gaps and refine searches.
- 📄 Generates answers with citations from gathered sources.
- 🔄 Hot-reloading for both frontend and backend during development.

## Project Structure

The project is divided into two main directories:

-   `frontend/`: Contains the React application built with Vite.
-   `backend/`: Contains the LangGraph/FastAPI application, including the research agent logic.

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.11+
-   **`GEMINI_API_KEY`**: The backend agent requires a Google Gemini API key.
    1.  Navigate to the `backend/` directory.
    2.  Create a file named `.env` by copying the `backend/.env.example` file.
    3.  Open the `.env` file and add your Gemini API key: `GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"`

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
pip install .
```

**Frontend:**

```bash
cd frontend
npm install
```

**3. Run Development Servers:**

**Backend & Frontend:**

```bash
make dev
```
This will run the backend and frontend development servers.    Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).

_Alternatively, you can run the backend and frontend development servers separately. For the backend, open a terminal in the `backend/` directory and run `langgraph dev`. The backend API will be available at `http://127.0.0.1:2024`. It will also open a browser window to the LangGraph UI. For the frontend, open a terminal in the `frontend/` directory and run `npm run dev`. The frontend will be available at `http://localhost:5173`._

## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It follows these steps:

1.  **Route Query:** Classifies the user's intent as conversational, RAG (document-based), or research (web search)
2.  **RAG Path (if applicable):** Retrieves relevant documents from the local knowledge base using hybrid search (semantic + keyword), evaluates sufficiency, and generates an answer or falls back to web search
3.  **Generate Search Queries:** Based on your input, generates optimized search queries using a Gemini model
4.  **Web Research:** For each query, uses the Gemini model with Google Search API to find relevant web pages
5.  **Reflection & Knowledge Gap Analysis:** Analyzes search results to determine if information is sufficient or if there are knowledge gaps
6.  **Iterative Refinement:** If gaps are found, generates follow-up queries and repeats the web research and reflection steps (up to a configured maximum number of loops)
7.  **Finalize Answer:** Synthesizes gathered information into a coherent answer with citations from web sources or local documents

## Deployment

For detailed deployment instructions including Docker setup, environment variables, and production configuration, see [DEPLOYMENT.md](DEPLOYMENT.md).

Quick start:

```bash
# Build Docker image
docker build -t gemini-fullstack-langgraph -f Dockerfile .

# Run with docker-compose
GEMINI_API_KEY=<key> LANGSMITH_API_KEY=<key> docker-compose up
```

Access the application at `http://localhost:8123/app/`

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface.
- [Tailwind CSS](https://tailwindcss.com/) - For styling.
- [Shadcn UI](https://ui.shadcn.com/) - For components.
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent.
- [Google Gemini](https://ai.google.dev/models/gemini) - LLM for query generation, reflection, and answer synthesis.

## Key Features

### Intelligent Query Routing
The agent uses a 3-way routing system:
- **Conversational**: Handles greetings and chitchat without web search
- **RAG**: Queries local knowledge base with hybrid retrieval (semantic + keyword search)
- **Research**: Performs comprehensive web research for factual queries

### Hybrid RAG System
- Combines semantic (dense) and keyword (sparse) search strategies
- Configurable alpha parameter for fine-tuning (0.0 = pure keyword, 1.0 = pure semantic)
- Intelligent sufficiency evaluation with automatic web fallback
- Supports PDF and DOCX document ingestion

### Document Ingestion
Add your own documents to the knowledge base:

```bash
# Create docs directory and add files
mkdir -p docs
cp /path/to/your/documents/*.pdf docs/

# Run ingestion
cd backend
python scripts/ingest_documents.py
```

## Architecture Details

For comprehensive architecture documentation and implementation details, see:
- [CLAUDE.md](CLAUDE.md) - Complete project documentation and development guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
