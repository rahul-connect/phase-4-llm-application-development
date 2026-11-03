# Phase 4 — LLM Application Development

A collection of hands-on exercises and backend integrations exploring **Large Language Model (LLM) application development** using Python, Groq, FastAPI, Pydantic, and PostgreSQL.

This repository is part of my **AI Engineering Learning Roadmap** and documents my progression from making basic LLM API calls to building backend applications that manage conversations, validate structured responses, and persist chat history.

The focus is on understanding how LLMs interact with application code, how to manage their inputs and outputs, and how to integrate them into maintainable backend systems.

**Status:** Lessons 01–09 covered. This repository documents the completed lesson work and does not include a Phase 4 capstone.

---

## Getting Started

### Prerequisites

- Python 3.13 or later
- [uv — Python Package Manager](https://docs.astral.sh/uv/)
- [Groq API](https://console.groq.com/) key
- PostgreSQL for Lessons 08–09
- Git and a code editor such as VS Code

### 1. Clone the Repository

```bash
git clone https://github.com/rahul-connect/phase-4-llm-application-development.git
cd phase-4-llm-application-development
```

### 2. Check Your Environment

```bash
python3 --version
uv --version
```

### 3. Install Lesson Dependencies

Each lesson may contain its own `pyproject.toml`.

For example:

```bash
cd lesson-01-first-api-call
uv sync
```

Repeat the setup in whichever lesson directory you want to explore.

### 4. Configure Environment Variables

The required environment variables vary by lesson.

**Lessons 01–06 — Groq API integration**

Create a `.env` file in the relevant lesson directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Lesson 07 — FastAPI integration**

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=openai/gpt-oss-120b
APP_NAME=Chat API
```

**Lessons 08–09 — PostgreSQL integration**

```env
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/sql_learning
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=openai/gpt-oss-120b
APP_NAME=Chat API
```

The database URL is an example. Update the username, password, host, port, and database name for your PostgreSQL installation.

You may use a local PostgreSQL instance, a Docker-based database, or a compatible hosted PostgreSQL service.

The configured model must be available through your Groq account.

**Security:** Never commit real API keys, database passwords, or `.env` files to GitHub.

### 5. Run a Lesson

For standalone Python exercises, run the appropriate script from its lesson directory:

```bash
uv run python main.py
```

Use the actual script filename if the lesson has a different entry point.

For FastAPI lessons, if `main.py` exposes a FastAPI application named `app`:

```bash
uv run uvicorn main:app --reload
```

For package-based applications, adjust the Uvicorn import path to match the source code.

FastAPI's interactive documentation is available at:

http://127.0.0.1:8000/docs

---

## Curriculum — Lesson by Lesson

| Lesson | Topic | Key Concepts |
|---|---|---|
| 01 | [First LLM API Call](lesson-01-first-api-call/) | Groq SDK, authentication, chat completions |
| 02 | [Prompt Engineering](lesson-02-prompt-engineering/) | System prompts, instructions, few-shot prompting |
| 03 | [Conversations](lesson-03-conversations/) | Message roles, chat history, context management |
| 04 | [Structured Outputs](lesson-04-structured-outputs/) | JSON generation, Pydantic validation |
| 05 | [Streaming](lesson-05-streaming/) | Streaming responses, generators, incremental output |
| 06 | [Tool Use](lesson-06-tool-use/) | Function schemas, tool calls, argument handling |
| 07 | [FastAPI Integration](lesson-07-fastapi-integration/) | LLM-powered API endpoints, configuration |
| 08 | [PostgreSQL Persistence](lesson-08-postgres-persistence/) | Conversation storage, SQLAlchemy, database integration |
| 09 | [Production Concerns](lesson-09-production-concerns/) | Reliability, errors, configuration, operational considerations |

---

## Part I — LLM Fundamentals and Integration

Lessons 01–06 focus on understanding and experimenting with LLM capabilities using Python and the Groq API.

### Lesson 01 — First LLM API Call

**Folder:** `lesson-01-first-api-call/`

Introduction to interacting with a hosted language model through an API.

**Concepts:**

- Understanding hosted LLM inference
- API keys and authentication
- Installing and using the Groq SDK
- Creating chat completion requests
- System and user messages
- Reading model-generated responses
- Model selection and basic inference parameters

### Lesson 02 — Prompt Engineering

**Folder:** `lesson-02-prompt-engineering/`

Understanding how prompts influence LLM behavior.

**Concepts:**

- System prompts
- User instructions
- Prompt structure
- Role-based messages
- Few-shot prompting
- Examples and constraints
- Improving consistency of model responses
- Understanding the limitations of prompt instructions

### Lesson 03 — Conversations

**Folder:** `lesson-03-conversations/`

Building conversation-aware LLM interactions.

**Concepts:**

- User and assistant message roles
- Multi-turn conversations
- Managing conversation history
- Passing previous messages to the model
- Context windows
- Recent-message sliding windows
- Token usage considerations
- Conversation summarization concepts

### Lesson 04 — Structured Outputs

**Folder:** `lesson-04-structured-outputs/`

Working with machine-readable LLM responses.

**Concepts:**

- JSON-formatted model responses
- Structured output schemas
- Pydantic models
- Field validation
- Required and optional fields
- Parsing generated JSON
- Handling invalid or unexpected model output
- Classification use cases

### Lesson 05 — Streaming LLM Responses

**Folder:** `lesson-05-streaming/`

Understanding how LLM responses can be processed incrementally.

**Concepts:**

- Streaming chat completions
- `stream=True`
- Python generators
- `yield`
- Iterating over response chunks
- Extracting streamed content
- Handling empty or missing content
- Streaming versus non-streaming responses

### Lesson 06 — Tool Use and Function Calling

**Folder:** `lesson-06-tool-use/`

Understanding how an LLM can request actions from application-defined tools.

**Concepts:**

- Tool and function definitions
- JSON schemas for tool arguments
- Tool selection
- `tool_choice`
- Reading tool call requests
- Parsing function arguments
- Executing application-controlled functions
- Returning tool results to the LLM
- Handling invalid arguments and tool failures
- Security boundaries between model output and application execution

Tool calling enables an LLM to request operations, but the application remains responsible for validating arguments, authorizing actions, and executing functions.

---

## Part II — Backend Integration

Lessons 07–09 build on the standalone LLM exercises by integrating language models into backend applications.

### Lesson 07 — FastAPI Integration

**Folder:** `lesson-07-fastapi-integration/`

Connecting LLM functionality to a Python web backend.

**Concepts:**

- FastAPI route handlers
- Request and response models
- Calling LLM services from API endpoints
- Separating API routes from LLM logic
- Dependency management
- Pydantic Settings
- Environment-based configuration
- API error handling

**Environment variables:**

- `GROQ_API_KEY`
- `MODEL_NAME`
- `APP_NAME`

### Lesson 08 — PostgreSQL Conversation Persistence

**Folder:** `lesson-08-postgres-persistence/`

Extending LLM applications with persistent conversation storage.

**Concepts:**

- PostgreSQL integration
- SQLAlchemy ORM models
- Conversations and messages
- Database sessions
- Saving user and assistant messages
- Retrieving conversation history
- Maintaining message order
- Repository and service layers
- Database-backed LLM context

**Environment variables:**

- `DATABASE_URL`
- `GROQ_API_KEY`
- `MODEL_NAME`
- `APP_NAME`

This lesson connects LLM application development with the database engineering concepts introduced in Phase 3.

### Lesson 09 — Production Concerns

**Folder:** `lesson-09-production-concerns/`

Exploring engineering considerations involved in building more reliable LLM-powered backends.

**Concepts:**

- LLM API failure handling
- Configuration management
- Protecting API credentials
- Timeouts and retry strategies
- Token usage monitoring
- Logging and observability
- Response validation
- Managing conversation context
- Understanding inference latency and costs
- Reliability considerations for external API dependencies

These are areas of study and engineering practice; this repository should not be interpreted as a fully production-hardened application.

---

## LLM Application Architecture

The later lessons explore a backend architecture connecting FastAPI, LLM inference, and PostgreSQL.

```text
API Client
    |
    v
FastAPI Routes
    |
    v
Application Service
    |
    +------------------+
    |                  |
    v                  v
LLM Service       Repository
    |                  |
    v                  v
Groq API           SQLAlchemy
                       |
                       v
                   PostgreSQL
```

The application service coordinates the workflow, while LLM integration and database access are separated into dedicated components.

This separation helps keep API routes focused on HTTP concerns and makes the application easier to understand and extend.

---

## Key Engineering Skills

### LLM Engineering

- Hosted LLM API integration
- Chat completion requests
- System prompts and few-shot prompting
- Conversation context management
- Streaming concepts
- Structured outputs
- Tool calling concepts
- Token usage awareness

### Backend Engineering

- FastAPI development
- Request and response validation
- Dependency injection
- Configuration management
- Error handling
- Modular application architecture

### Database Engineering

- PostgreSQL
- SQLAlchemy ORM
- Database sessions
- Persistent conversation storage
- Repository pattern
- Retrieving ordered conversation history

### Reliability and Security

- Environment-based secrets management
- Input and output validation
- Handling external API failures
- Understanding LLM limitations
- Logging and operational awareness
- Safe tool execution principles

---

## Project Structure

```text
phase-4-llm-application-development/
│
├── lesson-01-first-api-call/
├── lesson-02-prompt-engineering/
├── lesson-03-conversations/
├── lesson-04-structured-outputs/
├── lesson-05-streaming/
├── lesson-06-tool-use/
├── lesson-07-fastapi-integration/
├── lesson-08-postgres-persistence/
├── lesson-09-production-concerns/
│
├── .gitignore
└── README.md
```

Individual lessons may have their own dependencies, configuration files, and application structure.

---

## Learning Progression

This repository represents **Phase 4 of my AI Engineering Learning Roadmap**.

**Phase 0 — Python Foundations**

Python syntax, data structures, functions, file handling, and CLI applications.

**Phase 1 — Modern Python Development**

uv, type hints, modules, Pydantic, classes, testing, asynchronous programming, and refactoring.

**Phase 2 — FastAPI Backend Development**

REST APIs, routing, CRUD operations, dependency injection, middleware, and API testing.

**Phase 3 — PostgreSQL & Database Engineering**

SQL, relational modeling, SQLAlchemy, Alembic, database integration, and persistent backend applications.

**Phase 4 — LLM Application Development**

LLM API integration, prompt engineering, conversation management, structured outputs, streaming, tool calling, and integration with FastAPI and PostgreSQL.

**Next — Retrieval-Augmented Generation (RAG)**

Embeddings, vector search, document retrieval, retrieval evaluation, and building knowledge-grounded AI applications.

---

## Project Status

**Phase 4: Lessons 01–09**

This repository documents the practical exercises and application development work completed during Phase 4.

Individual lessons explore different LLM capabilities, while the later lessons focus on backend integration and persistent conversations.

Not every standalone exercise is necessarily integrated into the final backend application.

This repository is intended for educational and portfolio purposes, not as a production-ready customer support system.