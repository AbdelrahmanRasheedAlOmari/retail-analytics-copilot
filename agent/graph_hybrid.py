"""
LangGraph hybrid agent with ≥6 nodes and repair loop
"""
import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from pathlib import Path
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import dspy

from agent.tools.sqlite_tool import SQLiteDB
from agent.rag.retrieval import TFIDFRetriever
from agent.dspy_signatures import get_dspy_modules


# State definition for the graph
class AgentState(TypedDict):
    # Input
    question: str
    question_id: str
    format_hint: str
    
    # Routing
    route: str
    route_confidence: float
    
    # Retrieval
    retrieved_docs: List[Dict[str, Any]]
    retrieval_scores: List[float]
    
    # Planning
    constraints: Dict[str, Any]
    planning_reasoning: str
    
    # SQL Generation and Execution
    sql_query: str
    sql_explanation: str
    sql_columns: List[str]
    sql_rows: List[tuple]
    sql_error: Optional[str]
    sql_attempts: int
    
    # Synthesis
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]
    
    # Control flow
    needs_repair: bool
    repair_attempts: int
    error_messages: List[str]
    
    # Tracing
    trace_log: List[Dict[str, Any]]


class HybridAnalyticsAgent:
    """Hybrid agent for retail analytics queries"""
    
    def __init__(self, db_path: str, docs_dir: str, lm_model: Optional[dspy.LM] = None):
        self.db_path = db_path
        self.docs_dir = docs_dir
        
        # Initialize components
        self.db = SQLiteDB(db_path)
        self.retriever = TFIDFRetriever(docs_dir)
        self.dspy_modules = get_dspy_modules()
        
        # Set up DSPy language model if provided
        if lm_model:
            dspy.configure(lm=lm_model)
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with all nodes and edges"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("nl_to_sql", self._nl_to_sql_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("repair", self._repair_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add edges
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "rag_only": "retriever",
                "sql_only": "nl_to_sql",
                "hybrid": "retriever"
            }
        )
        
        workflow.add_edge("retriever", "planner")
        
        workflow.add_conditional_edges(
            "planner",
            self._planning_decision,
            {
                "needs_sql": "nl_to_sql",
                "synthesis_only": "synthesizer"
            }
        )
        
        workflow.add_edge("nl_to_sql", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            self._execution_decision,
            {
                "success": "synthesizer",
                "repair": "repair",
                "fail": "synthesizer"  # Give up after max attempts
            }
        )
        
        workflow.add_edge("repair", "executor")
        
        workflow.add_conditional_edges(
            "synthesizer",
            self._synthesis_decision,
            {
                "complete": END,
                "repair": "repair"
            }
        )
        
        # Compile the graph
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    # Node implementations
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Route the query to appropriate processing path"""
        self.logger.info(f"Routing question: {state['question'][:100]}...")
        
        result = self.dspy_modules["router"](state["question"])
        
        state["route"] = result["route"]
        state["route_confidence"] = result["confidence"]
        state["trace_log"].append({
            "node": "router",
            "route": result["route"],
            "confidence": result["confidence"]
        })
        
        return state
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents"""
        self.logger.info("Retrieving relevant documents...")
        
        # Retrieve top-k documents
        results = self.retriever.retrieve(state["question"], top_k=5)
        
        state["retrieved_docs"] = results
        state["retrieval_scores"] = [r["score"] for r in results]
        state["trace_log"].append({
            "node": "retriever",
            "num_docs": len(results),
            "top_score": max([r["score"] for r in results]) if results else 0
        })
        
        return state
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Extract constraints and planning information"""
        self.logger.info("Planning query execution...")
        
        result = self.dspy_modules["planner"](
            state["question"], 
            state["retrieved_docs"]
        )
        
        state["constraints"] = result["constraints"]
        state["planning_reasoning"] = result["reasoning"]
        state["trace_log"].append({
            "node": "planner",
            "constraints": result["constraints"],
            "reasoning": result["reasoning"]
        })
        
        return state
    
    def _nl_to_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language"""
        self.logger.info("Generating SQL query...")
        
        # Get database schema
        schema = self.db.get_schema_summary()
        
        result = self.dspy_modules["nl_to_sql"](
            state["question"],
            schema,
            state.get("constraints", {})
        )
        
        state["sql_query"] = result["sql_query"]
        state["sql_explanation"] = result["explanation"]
        state["trace_log"].append({
            "node": "nl_to_sql",
            "sql_generated": True,
            "explanation": result["explanation"]
        })
        
        return state
    
    def _executor_node(self, state: AgentState) -> AgentState:
        """Execute SQL query"""
        self.logger.info(f"Executing SQL: {state['sql_query'][:100]}...")
        
        columns, rows, error = self.db.execute_query(state["sql_query"])
        
        state["sql_columns"] = columns
        state["sql_rows"] = rows
        state["sql_error"] = error
        state["sql_attempts"] = state.get("sql_attempts", 0) + 1
        
        if error:
            state["error_messages"] = state.get("error_messages", []) + [error]
            self.logger.warning(f"SQL execution error: {error}")
        else:
            self.logger.info(f"SQL executed successfully, {len(rows)} rows returned")
        
        state["trace_log"].append({
            "node": "executor",
            "success": error is None,
            "rows_returned": len(rows) if not error else 0,
            "error": error
        })
        
        return state
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer"""
        self.logger.info("Synthesizing final answer...")
        
        # Prepare SQL result
        sql_result = {
            "columns": state.get("sql_columns", []),
            "rows": state.get("sql_rows", [])
        }
        
        result = self.dspy_modules["synthesizer"](
            state["question"],
            state["format_hint"],
            sql_result,
            state.get("retrieved_docs", [])
        )
        
        state["final_answer"] = result["final_answer"]
        state["confidence"] = result["confidence"]
        state["explanation"] = result["explanation"]
        
        # Generate citations
        citations = []
        
        # Add SQL table citations
        if not state.get("sql_error") and state.get("sql_columns"):
            # Infer tables from query
            query_upper = state.get("sql_query", "").upper()
            for table in self.db.get_table_names():
                if table.upper() in query_upper or table.replace("_", " ").upper() in query_upper:
                    citations.append(table)
        
        # Add document citations
        for doc in state.get("retrieved_docs", []):
            citations.append(doc["chunk_id"])
        
        state["citations"] = list(set(citations))  # Remove duplicates
        
        state["trace_log"].append({
            "node": "synthesizer",
            "final_answer_type": type(result["final_answer"]).__name__,
            "confidence": result["confidence"],
            "citations_count": len(state["citations"])
        })
        
        return state
    
    def _repair_node(self, state: AgentState) -> AgentState:
        """Repair failed SQL or invalid outputs"""
        self.logger.info("Attempting repair...")
        
        state["repair_attempts"] = state.get("repair_attempts", 0) + 1
        
        if state.get("sql_error"):
            # Repair SQL
            schema = self.db.get_schema_summary()
            
            result = self.dspy_modules["sql_repair"](
                state["question"],
                state["sql_query"],
                state["sql_error"],
                schema
            )
            
            state["sql_query"] = result["fixed_sql"]
            state["trace_log"].append({
                "node": "repair",
                "type": "sql_repair",
                "changes": result["changes_made"],
                "attempt": state["repair_attempts"]
            })
        
        return state
    
    # Decision functions for conditional edges
    
    def _route_decision(self, state: AgentState) -> str:
        """Decide the routing path"""
        route = state.get("route", "hybrid").lower()
        
        if "rag" in route:
            return "rag_only"
        elif "sql" in route:
            return "sql_only"
        else:
            return "hybrid"
    
    def _planning_decision(self, state: AgentState) -> str:
        """Decide if SQL is needed after planning"""
        # Check if we need SQL based on constraints or route
        route = state.get("route", "").lower()
        
        if route == "rag":
            return "synthesis_only"
        else:
            return "needs_sql"
    
    def _execution_decision(self, state: AgentState) -> str:
        """Decide next step after SQL execution"""
        if state.get("sql_error"):
            # If we haven't exceeded repair attempts, try repair
            if state.get("repair_attempts", 0) < 2:
                state["needs_repair"] = True
                return "repair"
            else:
                # Give up and synthesize with error
                return "fail"
        else:
            return "success"
    
    def _synthesis_decision(self, state: AgentState) -> str:
        """Decide if synthesis is complete"""
        # Check if answer format is correct
        final_answer = state.get("final_answer")
        format_hint = state.get("format_hint", "")
        
        # Simple format validation
        if self._validate_answer_format(final_answer, format_hint):
            return "complete"
        elif state.get("repair_attempts", 0) < 2:
            return "repair"
        else:
            return "complete"  # Give up after max attempts
    
    def _validate_answer_format(self, answer: Any, format_hint: str) -> bool:
        """Validate if answer matches expected format"""
        if format_hint == "int":
            return isinstance(answer, int)
        elif format_hint == "float":
            return isinstance(answer, (int, float))
        elif format_hint.startswith("{"):
            return isinstance(answer, dict)
        elif format_hint.startswith("list"):
            return isinstance(answer, list)
        else:
            return True  # Accept any format for unknown hints
    
    def run_query(self, question: str, question_id: str, format_hint: str) -> Dict[str, Any]:
        """Run a single query through the agent"""
        
        # Initialize state
        initial_state = {
            "question": question,
            "question_id": question_id,
            "format_hint": format_hint,
            "route": "",
            "route_confidence": 0.0,
            "retrieved_docs": [],
            "retrieval_scores": [],
            "constraints": {},
            "planning_reasoning": "",
            "sql_query": "",
            "sql_explanation": "",
            "sql_columns": [],
            "sql_rows": [],
            "sql_error": None,
            "sql_attempts": 0,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "citations": [],
            "needs_repair": False,
            "repair_attempts": 0,
            "error_messages": [],
            "trace_log": []
        }
        
        # Run the graph
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": question_id}}
            )
            
            # Format output
            result = {
                "id": question_id,
                "final_answer": final_state.get("final_answer"),
                "sql": final_state.get("sql_query", ""),
                "confidence": final_state.get("confidence", 0.0),
                "explanation": final_state.get("explanation", ""),
                "citations": final_state.get("citations", [])
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running query: {e}")
            return {
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            }


def create_agent(db_path: str, docs_dir: str, lm_model: Optional[dspy.LM] = None) -> HybridAnalyticsAgent:
    """Factory function to create the hybrid agent"""
    return HybridAnalyticsAgent(db_path, docs_dir, lm_model)
