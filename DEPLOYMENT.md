# Deployment Guide

This guide covers how to deploy the Gemini Fullstack LangGraph application in production.

## Production Architecture

In production, the backend server serves the optimized static frontend build. LangGraph requires:
- **Redis**: Used as a pub-sub broker to enable streaming real-time output from background runs
- **PostgreSQL**: Used to store assistants, threads, runs, persist thread state and long-term memory, and to manage the state of the background task queue with 'exactly once' semantics

For more details on deployment options, see the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/).

## Prerequisites

Before deploying, ensure you have:
- Docker and Docker Compose installed
- A Google Gemini API key ([Get one here](https://ai.google.dev/))
- A LangSmith API key ([Get one here](https://smith.langchain.com/settings)) - Optional but recommended for monitoring
- Redis instance (included in docker-compose.yml below)
- PostgreSQL instance (included in docker-compose.yml below)

## Environment Variables

### Required
- `GEMINI_API_KEY`: Your Google Gemini API key for LLM operations

### Optional
- `LANGSMITH_API_KEY`: For LangSmith tracing and monitoring (highly recommended)
- `LANGCHAIN_TRACING_V2`: Set to `true` to enable LangSmith tracing
- `LANGCHAIN_PROJECT`: Project name in LangSmith (default: `fullstack-rag-agent`)
- `TAVILY_API_KEY`: For web search fallback when RAG documents are insufficient

### Production-Specific
- `REDIS_URI`: Redis connection URI (e.g., `redis://redis:6379`)
- `POSTGRES_URI`: PostgreSQL connection URI (e.g., `postgres://user:pass@postgres:5432/dbname`)

## Deployment with Docker

### Step 1: Build the Docker Image

From the **project root directory**, run:

```bash
docker build -t gemini-fullstack-langgraph -f Dockerfile .
```

This will:
1. Build the optimized frontend (Node.js stage)
2. Combine it with the backend in the LangGraph API base image
3. Configure the backend to serve static frontend files

### Step 2: Run with Docker Compose

The `docker-compose.yml` file includes:
- The application server
- Redis for pub-sub
- PostgreSQL for state persistence

Run the stack:

```bash
GEMINI_API_KEY=<your_gemini_api_key> LANGSMITH_API_KEY=<your_langsmith_api_key> docker-compose up
```

Or create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_actual_key_here
LANGSMITH_API_KEY=your_actual_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=fullstack-rag-agent
```

Then run:

```bash
docker-compose up
```

### Step 3: Access the Application

- **Frontend**: http://localhost:8123/app/
- **API**: http://localhost:8123
- **LangGraph Studio**: The LangGraph API exposes endpoints for graph visualization and debugging

## Configuration for Different Environments

### Updating API URL

If you're NOT using the docker-compose.yml example or exposing the backend server to a different host, update the `apiUrl` in `frontend/src/App.tsx`:

```typescript
// Development
const apiUrl = "http://localhost:2024"

// Docker Compose (default)
const apiUrl = "http://localhost:8123"

// Production (custom domain)
const apiUrl = "https://your-domain.com"
```

## Production Checklist

- [ ] Set all required environment variables
- [ ] Configure Redis with persistence and backups
- [ ] Configure PostgreSQL with regular backups
- [ ] Set up SSL/TLS certificates for HTTPS
- [ ] Configure proper firewall rules
- [ ] Enable LangSmith monitoring
- [ ] Set up logging and error tracking
- [ ] Configure rate limiting
- [ ] Review and adjust resource limits (CPU, memory)
- [ ] Set up health checks and monitoring
- [ ] Configure domain and DNS
- [ ] Test RAG document ingestion pipeline

## RAG Document Management

Before deploying, ensure you've ingested your documents:

```bash
# Create docs directory
mkdir -p docs

# Add your PDF or DOCX files to the docs directory
cp /path/to/your/documents/*.pdf docs/

# Run ingestion (before building Docker image)
cd backend
python scripts/ingest_documents.py
```

The ChromaDB database will be included in the Docker image at `backend/chroma_db/`.

## Scaling Considerations

### Horizontal Scaling
- The application can be scaled horizontally by running multiple instances
- Redis and PostgreSQL must be shared across all instances
- Use a load balancer (e.g., nginx, AWS ALB) to distribute traffic

### Database Scaling
- **Redis**: Consider Redis Cluster for high availability
- **PostgreSQL**: Set up read replicas for better performance
- **ChromaDB**: For large-scale RAG, consider distributed vector databases like Milvus or Pinecone

### Performance Optimization
- Enable database connection pooling
- Configure appropriate timeout values
- Monitor API rate limits (Gemini API)
- Cache frequently accessed data
- Optimize chunk size and overlap for RAG retrieval

## Troubleshooting

### Common Issues

**Container fails to start:**
- Check environment variables are set correctly
- Verify Redis and PostgreSQL are running and accessible
- Check Docker logs: `docker-compose logs -f`

**Frontend can't connect to backend:**
- Verify the `apiUrl` in `frontend/src/App.tsx` matches your deployment
- Check CORS settings if deploying to different domains
- Verify firewall/security group rules allow traffic

**RAG retrieval returns no results:**
- Verify documents were ingested successfully
- Check ChromaDB directory exists and has data
- Review ingestion logs for errors
- Test retrieval with known document content

**LangSmith tracing not working:**
- Verify `LANGSMITH_API_KEY` is set
- Ensure `LANGCHAIN_TRACING_V2=true` is set
- Check LangSmith project name is correct

## Monitoring and Observability

### LangSmith
- View all graph executions and traces
- Monitor token usage and costs
- Debug agent decision-making
- Access at: https://smith.langchain.com/

### Application Logs
```bash
# View logs
docker-compose logs -f app

# View only errors
docker-compose logs -f app | grep ERROR
```

### Health Checks
The backend exposes health check endpoints (if configured):
- `/health`: Basic health check
- `/ready`: Readiness check (database connections, etc.)

## Security Best Practices

1. **Never commit secrets**: Use environment variables or secret management tools
2. **Use HTTPS**: Always use SSL/TLS in production
3. **API Key Rotation**: Regularly rotate API keys
4. **Database Security**: Use strong passwords, enable encryption at rest
5. **Network Security**: Use private networks for database connections
6. **Input Validation**: The application validates user inputs
7. **Rate Limiting**: Implement rate limiting to prevent abuse
8. **Regular Updates**: Keep dependencies and base images updated

## Support and Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Docker Documentation](https://docs.docker.com/)

For issues specific to this application, please check the main README.md or CLAUDE.md files.
