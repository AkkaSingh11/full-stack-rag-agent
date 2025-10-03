# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A fullstack RAG (Retrieval-Augmented Generation) application with a React frontend and LangGraph backend. The backend implements an intelligent research agent with an orchestrator router pattern using Google's Gemini models that:
1. **Routes queries intelligently** - Classifies user input as conversational (greetings, chitchat) or research (complex questions)
2. **Generates optimized search queries** from user questions (research path only)
3. **Performs web research** using Google Search API (research path only)
4. **Reflects on results** to identify knowledge gaps and generates follow-up queries
5. **Iteratively refines searches** until sufficient information is gathered
6. **Synthesizes final answer** with citations (research path) or provides direct conversational responses
7. **Traces all operations** with LangSmith for debugging and monitoring (optional)

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

**Testing router (new feature):**
```bash
cd backend
python test_router_simple.py   # Quick test with "Hi"
python test_router.py           # Full test suite
```

### Backend Commands

**Install dependencies:**
```bash
cd backend
pip install -e .  # Use -e for editable install during development
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
- `route_query`: **NEW** - Classifies user intent as "conversational" or "research" (Gemini 2.5 Flash)
- `conversational_response`: **NEW** - Generates friendly responses for greetings/chitchat (Gemini 2.5 Flash)
- `generate_query`: Creates initial search queries from user input (Gemini 2.5 Flash)
- `web_research`: Executes web searches using Google Search API (runs in parallel for multiple queries)
- `reflection`: Analyzes results for knowledge gaps and generates follow-up queries (Gemini 2.5 Flash)
- `evaluate_research`: Routes to either more research or finalization
- `finalize_answer`: Synthesizes final answer with citations (Gemini 2.5 Flash by default)

**Graph Flow:**
```
START → route_query → {
  conversational → conversational_response → END
  research → generate_query → web_research → reflection → evaluate_research → finalize_answer → END
}
```

**State management:** [backend/src/agent/state.py](backend/src/agent/state.py)
- `OverallState`: Main graph state with messages, queries, results, sources, **route_decision**
- `ReflectionState`: Reflection output (sufficiency, knowledge gaps, follow-ups)
- `QueryGenerationState`: Search query generation output
- `WebSearchState`: Individual web search parameters

**Schemas:** [backend/src/agent/tools_and_schemas.py](backend/src/agent/tools_and_schemas.py)
- `SearchQueryList`: Structured output for search queries
- `Reflection`: Structured output for reflection analysis
- `RouteDecision`: **NEW** - Structured output for routing (intent + reasoning)

**Configuration:** [backend/src/agent/configuration.py](backend/src/agent/configuration.py)
- Models: `router_model`, `conversational_model`, `query_generator_model`, `reflection_model`, `answer_model`
- Parameters: `number_of_initial_queries` (default: 3), `max_research_loops` (default: 2)
- Can be configured via environment variables or RunnableConfig
- All models default to `gemini-2.5-flash`

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
- **NEW events:** `route_query` (shows routing decision), `conversational_response` (direct responses)

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

**Optional for LangSmith tracing (recommended for debugging):**
```
LANGSMITH_API_KEY="your_langsmith_key"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=fullstack-rag-agent
```

**Optional for deployment:**
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

**Orchestrator Router Pattern (NEW):** The agent now intelligently routes queries:
- **Conversational queries** (greetings, chitchat) → Direct LLM response (no web search)
- **Research queries** (factual questions) → Full web research flow
- Uses LLM-based classification with structured output (`RouteDecision`)
- Reduces latency and API costs for simple interactions

**Parallel web research:** The agent uses LangGraph's `Send` API to spawn multiple `web_research` nodes in parallel, one per search query.

**Citation handling:** Uses Google's native search grounding metadata. URLs are resolved to short URLs during research to save tokens, then replaced with original URLs in the final answer.

**Iterative refinement:** The reflection node evaluates if gathered information is sufficient. If not, it generates follow-up queries and continues research (up to `max_research_loops`).

**Model selection:** Different Gemini models for different tasks:
- Router: Fast classification (2.5 Flash)
- Conversational: Friendly responses (2.5 Flash)
- Query generation: Fast model (2.5 Flash)
- Reflection: Reasoning model (2.5 Flash)
- Final answer: High-quality model (2.5 Flash)

**LangSmith Tracing:** Auto-enabled when `LANGSMITH_API_KEY` is set in environment. Traces all graph executions for debugging and monitoring. View traces at https://smith.langchain.com/

**URL management:** See [backend/src/agent/utils.py](backend/src/agent/utils.py) for citation extraction, URL resolution, and marker insertion logic.

## Recent Updates

### Orchestrator Router Pattern (October 2025)
- **Branch:** `feature/orchestrator-router-langsmith`
- **Implementation:** Added intelligent routing to avoid web searches for conversational queries
- **New nodes:** `route_query`, `conversational_response`
- **New prompts:** `router_instructions`, `conversational_instructions`
- **Test coverage:** `test_router.py`, `test_router_simple.py`
- **Documentation:** See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for full details

**Benefits:**
- ✅ Faster responses for greetings and chitchat (no web search latency)
- ✅ Reduced API costs (no Google Search calls for simple queries)
- ✅ Better UX with instant conversational responses
- ✅ Full observability with LangSmith tracing

**Testing the router:**
```bash
cd backend
python test_router_simple.py  # Should show: Route=conversational, Response="Hi there! 👋..."
```
