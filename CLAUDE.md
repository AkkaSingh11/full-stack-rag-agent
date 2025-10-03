# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fullstack RAG (Retrieval-Augmented Generation) application with a React frontend and LangGraph backend. The backend implements an iterative research agent using Google's Gemini models that:
1. Generates optimized search queries from user questions
2. Performs web research using Google Search API
3. Reflects on results to identify knowledge gaps
4. Iteratively refines searches until sufficient information is gathered
5. Synthesizes a final answer with citations

## Development Commands

### Running the Application

**Development (both frontend & backend):**
```bash
make dev
```

**Backend only:**
```bash
cd backend
langgraph dev
# Runs on http://127.0.0.1:2024 and opens LangGraph UI
```

**Frontend only:**
```bash
cd frontend
npm run dev
# Runs on http://localhost:5173
```

**CLI research (one-off questions):**
```bash
cd backend
python examples/cli_research.py "your question here"
# Optional flags: --initial-queries, --max-loops, --reasoning-model
```

### Backend Commands

**Install dependencies:**
```bash
cd backend
pip install .
```

**Run tests:**
```bash
cd backend
make test                    # Run all tests
make test TEST_FILE=path     # Run specific test file
make test_watch              # Run tests in watch mode
```

**Linting & formatting:**
```bash
cd backend
make format                  # Format code with ruff
make lint                    # Run ruff and mypy checks
```

### Frontend Commands

**Install dependencies:**
```bash
cd frontend
npm install
```

**Build:**
```bash
cd frontend
npm run build     # TypeScript compilation + Vite build
```

**Lint:**
```bash
cd frontend
npm run lint      # ESLint
```

## Architecture

### Backend Structure

**Core agent logic:** [backend/src/agent/graph.py](backend/src/agent/graph.py)

The LangGraph agent uses a state graph with these nodes:
- `generate_query`: Creates initial search queries from user input (Gemini 2.0 Flash)
- `web_research`: Executes web searches using Google Search API (runs in parallel for multiple queries)
- `reflection`: Analyzes results for knowledge gaps and generates follow-up queries (Gemini 2.5 Flash)
- `evaluate_research`: Routes to either more research or finalization
- `finalize_answer`: Synthesizes final answer with citations (Gemini 2.5 Pro by default)

**State management:** [backend/src/agent/state.py](backend/src/agent/state.py)
- `OverallState`: Main graph state with messages, queries, results, sources
- `ReflectionState`: Reflection output (sufficiency, knowledge gaps, follow-ups)
- `QueryGenerationState`: Search query generation output
- `WebSearchState`: Individual web search parameters

**Configuration:** [backend/src/agent/configuration.py](backend/src/agent/configuration.py)
- Models: `query_generator_model`, `reflection_model`, `answer_model`
- Parameters: `number_of_initial_queries` (default: 3), `max_research_loops` (default: 2)
- Can be configured via environment variables or RunnableConfig

**API server:** [backend/src/agent/app.py](backend/src/agent/app.py)
- FastAPI app that serves the LangGraph agent
- Also serves static frontend build in production

**Entry points:**
- LangGraph API: Configured in [backend/langgraph.json](backend/langgraph.json)
- Graph: `./src/agent/graph.py:graph`
- HTTP app: `./src/agent/app.py:app`

### Frontend Structure

**Main app:** [frontend/src/App.tsx](frontend/src/App.tsx)
- Uses `@langchain/langgraph-sdk` for streaming agent events
- Switches API URL based on environment (dev: localhost:2024, prod: localhost:8123)
- Processes streaming events from agent nodes into timeline activities

**Key components:**
- `WelcomeScreen`: Initial landing screen
- `ChatMessagesView`: Displays conversation messages
- `ActivityTimeline`: Shows real-time agent activities (search, reflection, etc.)
- `InputForm`: User input for research questions

**Tech stack:**
- React 19 with TypeScript
- Vite for dev server and builds
- Tailwind CSS + Shadcn UI components
- React Router for routing

### Environment Setup

**Required:** Create [backend/.env](backend/.env) from [backend/.env.example](backend/.env.example):
```
GEMINI_API_KEY="your_api_key_here"
```

**Optional for deployment:**
- `LANGSMITH_API_KEY`: For LangSmith tracing/monitoring
- `REDIS_URI`: Redis for pub-sub streaming (required in production)
- `POSTGRES_URI`: Postgres for state persistence (required in production)

## Deployment

**Build Docker image:**
```bash
docker build -t gemini-fullstack-langgraph -f Dockerfile .
```

**Run with docker-compose:**
```bash
GEMINI_API_KEY=<key> LANGSMITH_API_KEY=<key> docker-compose up
# App available at http://localhost:8123/app/
```

The Dockerfile:
1. Builds optimized frontend (Node.js stage)
2. Combines with backend in LangGraph API base image
3. Backend serves static frontend files in production

## Key Implementation Details

**Parallel web research:** The agent uses LangGraph's `Send` API to spawn multiple `web_research` nodes in parallel, one per search query.

**Citation handling:** Uses Google's native search grounding metadata. URLs are resolved to short URLs during research to save tokens, then replaced with original URLs in the final answer.

**Iterative refinement:** The reflection node evaluates if gathered information is sufficient. If not, it generates follow-up queries and continues research (up to `max_research_loops`).

**Model selection:** Different Gemini models for different tasks:
- Query generation: Fast model (2.0 Flash)
- Reflection: Reasoning model (2.5 Flash)
- Final answer: High-quality model (2.5 Pro)

**URL management:** See [backend/src/agent/utils.py](backend/src/agent/utils.py) for citation extraction, URL resolution, and marker insertion logic.
