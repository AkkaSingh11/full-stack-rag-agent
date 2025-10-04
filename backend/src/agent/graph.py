import os

from agent.tools_and_schemas import SearchQueryList, Reflection, RouteDecision, RagJudge
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from typing import Literal

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    router_instructions,
    conversational_instructions,
    rag_judge_instructions,
    rag_answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.vector_store import get_retriever

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Enable LangSmith tracing if API key is provided
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "fullstack-rag-agent"

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def route_query(state: OverallState, config: RunnableConfig) -> dict:
    """LangGraph node that routes user queries to conversational or research paths.

    Analyzes the user's message to determine if it requires web research or can be
    answered conversationally (greetings, chitchat, clarifications).

    Args:
        state: Current graph state containing the user's messages
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with route_decision key containing "conversational" or "research"
    """
    configurable = Configuration.from_runnable_config(config)

    # Get the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return {"route_decision": "conversational"}

    latest_message = user_messages[-1].content

    # Get conversation context (last few messages for context)
    conversation_context = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state["messages"][-3:]]
    ) if len(state["messages"]) > 1 else "No previous context"

    # Format the prompt
    formatted_prompt = router_instructions.format(
        user_message=latest_message,
        conversation_context=conversation_context,
    )

    # Initialize router model
    llm = ChatGoogleGenerativeAI(
        model=configurable.router_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(RouteDecision)

    # Get routing decision
    result = structured_llm.invoke(formatted_prompt)

    return {"route_decision": result.intent}


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


def conversational_response(state: OverallState, config: RunnableConfig):
    """LangGraph node that generates friendly conversational responses.

    Handles greetings, chitchat, and simple questions without triggering web research.

    Args:
        state: Current graph state containing conversation history
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with messages key containing the conversational response
    """
    configurable = Configuration.from_runnable_config(config)

    # Get the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    latest_message = user_messages[-1].content if user_messages else ""

    # Format conversation history for context
    conversation_history = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][-5:]]
    )

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = conversational_instructions.format(
        current_date=current_date,
        conversation_history=conversation_history,
        user_message=latest_message,
    )

    # Initialize conversational model
    llm = ChatGoogleGenerativeAI(
        model=configurable.conversational_model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    result = llm.invoke(formatted_prompt)

    return {"messages": [AIMessage(content=result.content)]}


def rag_lookup(state: OverallState, config: RunnableConfig):
    """LangGraph node that retrieves documents from the knowledge base.

    Performs semantic search on the vector store using the user's question.

    Args:
        state: Current graph state containing the user's question
        config: Configuration for the runnable, including retrieval settings

    Returns:
        Dictionary with rag_chunks and rag_sources keys
    """
    configurable = Configuration.from_runnable_config(config)

    # Get the latest user message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    query = user_messages[-1].content if user_messages else ""

    # Get retriever with configured top_k
    retriever = get_retriever(k=configurable.rag_top_k)

    if retriever is None:
        # No vector store available, return empty results
        return {
            "rag_chunks": "",
            "rag_sources": [],
            "rag_sufficient": False,
        }

    # Retrieve documents
    docs = retriever.invoke(query)

    if not docs:
        # No documents found
        return {
            "rag_chunks": "",
            "rag_sources": [],
            "rag_sufficient": False,
        }

    # Format retrieved documents
    chunks_text = "\n\n---\n\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    # Extract sources
    sources = [
        {
            "source_file": doc.metadata.get("source_file", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in docs
    ]

    return {
        "rag_chunks": chunks_text,
        "rag_sources": sources,
    }


def judge_sufficiency(state: OverallState, config: RunnableConfig):
    """LangGraph node that judges if retrieved documents are sufficient.

    Uses an LLM to evaluate whether the retrieved documents adequately answer
    the user's question.

    Args:
        state: Current graph state with rag_chunks
        config: Configuration for the runnable

    Returns:
        Dictionary with rag_sufficient key
    """
    configurable = Configuration.from_runnable_config(config)

    # Get the user's question
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    query = user_messages[-1].content if user_messages else ""

    # Format the prompt
    formatted_prompt = rag_judge_instructions.format(
        question=query,
        documents=state.get("rag_chunks", "No documents retrieved"),
    )

    # Initialize judge model
    llm = ChatGoogleGenerativeAI(
        model=configurable.judge_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(RagJudge)

    # Get judgment
    result = structured_llm.invoke(formatted_prompt)

    return {"rag_sufficient": result.sufficient}


def finalize_rag_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that generates final answer from RAG documents.

    Synthesizes the retrieved documents into a comprehensive answer.

    Args:
        state: Current graph state with rag_chunks
        config: Configuration for the runnable

    Returns:
        Dictionary with messages key containing the final answer
    """
    configurable = Configuration.from_runnable_config(config)

    # Get the user's question
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    query = user_messages[-1].content if user_messages else ""

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = rag_answer_instructions.format(
        current_date=current_date,
        question=query,
        documents=state.get("rag_chunks", "No documents retrieved"),
    )

    # Initialize RAG answer model
    llm = ChatGoogleGenerativeAI(
        model=configurable.rag_model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    result = llm.invoke(formatted_prompt)

    return {"messages": [AIMessage(content=result.content)]}


def decide_route(state: OverallState) -> Literal["conversational_response", "rag_lookup", "generate_query"]:
    """Routing function that decides the next node based on intent classification.

    Args:
        state: Current graph state containing the route_decision

    Returns:
        String literal indicating next node: "conversational_response", "rag_lookup", or "generate_query"
    """
    route = state.get("route_decision")
    if route == "conversational":
        return "conversational_response"
    elif route == "rag":
        return "rag_lookup"
    else:
        return "generate_query"


def decide_after_rag(state: OverallState) -> Literal["finalize_rag_answer", "generate_query"]:
    """Routing function after RAG lookup to decide if documents are sufficient.

    Args:
        state: Current graph state containing rag_sufficient

    Returns:
        String literal indicating next node: "finalize_rag_answer" if sufficient, "generate_query" for web fallback
    """
    if state.get("rag_sufficient", False):
        return "finalize_rag_answer"
    else:
        return "generate_query"


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define all nodes
builder.add_node("route_query", route_query)
builder.add_node("conversational_response", conversational_response)
builder.add_node("rag_lookup", rag_lookup)
builder.add_node("judge_sufficiency", judge_sufficiency)
builder.add_node("finalize_rag_answer", finalize_rag_answer)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `route_query`
# This is the first node called to determine conversational vs rag vs research path
builder.add_edge(START, "route_query")

# Add conditional routing based on intent classification (3-way routing)
builder.add_conditional_edges(
    "route_query",
    decide_route,
    {
        "conversational_response": "conversational_response",
        "rag_lookup": "rag_lookup",
        "generate_query": "generate_query",
    },
)

# Conversational path ends directly
builder.add_edge("conversational_response", END)

# RAG path: lookup -> judge -> either finalize or fallback to web research
builder.add_edge("rag_lookup", "judge_sufficiency")
builder.add_conditional_edges(
    "judge_sufficiency",
    decide_after_rag,
    {
        "finalize_rag_answer": "finalize_rag_answer",
        "generate_query": "generate_query",
    },
)
builder.add_edge("finalize_rag_answer", END)

# Research path continues with existing flow
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
