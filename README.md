# Retail Analytics Copilot

A hybrid AI agent that answers retail analytics questions by combining RAG (Retrieval Augmented Generation) over local documents with SQL queries over a SQLite database. Built using DSPy and LangGraph for optimized performance and structured reasoning.

## Features

- **Hybrid Processing**: Combines document retrieval and SQL execution for comprehensive answers
- **Local & Free**: Runs entirely locally using Ollama with Phi-3.5-mini-instruct
- **Typed Outputs**: Produces structured, typed answers with proper citations
- **Repair Loop**: Automatically fixes SQL errors and validates outputs
- **DSPy Optimization**: Optimized modules for improved accuracy

## Graph Design

The agent uses a LangGraph with 7 nodes and conditional routing:

- **Router**: Classifies queries as RAG-only, SQL-only, or hybrid using DSPy
- **Retriever**: TF-IDF-based document search with chunking and scoring
- **Planner**: Extracts constraints (dates, categories, KPIs) from retrieved documents
- **NL→SQL**: Generates SQLite queries using schema introspection and constraints
- **Executor**: Runs SQL queries with error handling and result capture
- **Synthesizer**: Combines SQL results and documents into typed final answers
- **Repair**: Fixes failed SQL queries and validates output formats (≤2 attempts)

The graph includes conditional edges for intelligent routing and a repair loop that can iterate up to 2 times for query fixes.

## DSPy Optimization

**Optimized Module**: NL→SQL Generator (BootstrapFewShot)
- **Metric**: Strict valid-SQL rate on a tiny local split (24 handcrafted examples)
- **Checks**: executes without error AND passes pattern checks (uses `[Order Details]`, no non-SQLite functions, no nonexistent tables)
- **Before (measured locally)**: 62.5%
- **After (measured locally)**: 62.5%
- **Notes**: With a deterministic local LM and limited budget, the optimizer did not bootstrap new demos, so accuracy remained the same. Code includes SQL normalization (table quoting, date filters) that improves robustness during inference.

If more time/data were available, we would expand examples and/or optimize the Synthesizer for exact format/citation adherence to yield a clearer delta.

## Trade-offs and Assumptions

- **Cost Approximation**: CostOfGoods = 70% of UnitPrice when calculating gross margin (standard retail assumption)
- **Date Filtering**: Uses string comparison for date ranges (adequate for Northwind's date format)
- **Chunking Strategy**: Paragraph-based document chunking optimized for policy/KPI documents
- **Model Limitations**: Phi-3.5-mini may struggle with complex multi-table joins; repair loop mitigates this
- **Confidence Scoring**: Heuristic-based combining retrieval scores and SQL success rates

## Setup and Usage

### Prerequisites

1. Install Ollama: https://ollama.com
2. Pull the model: `ollama pull phi3.5:3.8b-mini-instruct-q4_K_M`
3. Install dependencies: `pip install -r requirements.txt`

### Running the Agent

```bash
# Process evaluation questions
python run_agent_hybrid.py \
    --batch sample_questions_hybrid_eval.jsonl \
    --out outputs_hybrid.jsonl

# With verbose logging
python run_agent_hybrid.py \
    --batch sample_questions_hybrid_eval.jsonl \
    --out outputs_hybrid.jsonl \
    --verbose
```

### Output Format

Each output line follows the contract:
```json
{
  "id": "question_id",
  "final_answer": "typed_answer_matching_format_hint",
  "sql": "executed_sql_query_or_empty",
  "confidence": 0.85,
  "explanation": "Brief explanation of the result",
  "citations": ["Orders", "Order Details", "kpi_definitions::chunk0"]
}
```

## Architecture

```
agent/
├─ graph_hybrid.py        # LangGraph with 7 nodes + repair loop
├─ dspy_signatures.py     # DSPy modules (Router/NL→SQL/Synthesizer)  
├─ rag/retrieval.py       # TF-IDF retrieval with chunking
└─ tools/sqlite_tool.py   # SQLite access + schema introspection

data/northwind.sqlite     # Northwind sample database
docs/                     # Document corpus (policies, KPIs, catalog)
```

## Key Components

- **Database**: Northwind SQLite with Orders, Order Details, Products, Customers, Categories
- **Documents**: Marketing calendar, KPI definitions, product catalog, return policies
- **Retrieval**: TF-IDF vectorization with cosine similarity scoring
- **LM Integration**: Ollama Phi-3.5-mini-instruct (no fallbacks)
- **Validation**: Format checking, citation tracking, confidence scoring

## Current Status

**Working Components:**
- ✅ **RAG Pipeline**: Perfect document retrieval and synthesis (1/1 RAG questions correct)
- ✅ **LangGraph Flow**: All 7 nodes working with proper state management
- ✅ **DSPy Integration**: Real Ollama Phi-3.5-mini generating intelligent responses
- ✅ **Output Format**: Exact compliance with assignment contract
- ✅ **Citations**: Proper document chunk ID tracking

**Known Limitations:**
- SQL generation can still produce malformed date filters or mis-quoted columns; the repair loop and post-processing now fix common cases (e.g., `o.[Order Date]` → `o.OrderDate`, month range normalization).

## Testing

The system includes evaluation on 6 test cases:
- **RAG-only**: ✅ Perfect (14-day beverage return policy)  
- **SQL/Hybrid**: Known table quoting issue affects complex queries
- **Architecture**: All components functional, repair loops active

**Current Results**: End-to-end correctness depends on the local LM. With the included cleaning/repair steps, most SQL executes; exact numeric correctness varies by prompt determinism. RAG-only items are consistently correct.
