#!/usr/bin/env python3
"""
Main entrypoint for the hybrid retail analytics agent
CLI contract: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
"""
import json
import click
from pathlib import Path
from typing import List, Dict, Any
import logging
import time

# Set up DSPy with Ollama
import dspy

from agent.graph_hybrid import create_agent


def setup_ollama_lm():
    """Set up DSPy with Ollama Phi-3.5-mini model as specified in assignment"""
    # Use DSPy's LM class with LiteLLM format for Ollama
    lm = dspy.LM(
        model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M",
        base_url="http://localhost:11434",
        temperature=0.0,
        max_tokens=2000
    )
    
    # Test the connection with a simple query
    test_response = lm("Say hello")
    logging.info("Successfully connected to Ollama Phi-3.5-mini")
    logging.info(f"Test response: {test_response}")
    return lm


def load_questions(input_file: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file"""
    questions = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSONL file"""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
@click.option('--db', default='data/northwind.sqlite', help='Path to SQLite database')
@click.option('--docs', default='docs', help='Path to documents directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def main(batch: str, out: str, db: str, docs: str, verbose: bool):
    """
    Retail Analytics Copilot - Hybrid agent for retail analytics queries
    
    This agent combines RAG over local documents with SQL queries over a SQLite database
    to answer retail analytics questions with typed, auditable answers.
    """
    
    # Set up logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Retail Analytics Copilot")
    logger.info(f"Database: {db}")
    logger.info(f"Documents: {docs}")
    logger.info(f"Input: {batch}")
    logger.info(f"Output: {out}")
    
    # Validate input files
    if not Path(batch).exists():
        raise click.ClickException(f"Input file not found: {batch}")
    
    if not Path(db).exists():
        raise click.ClickException(f"Database not found: {db}")
    
    if not Path(docs).exists():
        raise click.ClickException(f"Documents directory not found: {docs}")
    
    # Set up language model (force reconfiguration to override any previous settings)
    logger.info("Setting up language model...")
    lm = setup_ollama_lm()
    
    # FORCE DSPy reconfiguration (clears any previous dummy LM configs)
    dspy.configure(lm=lm)
    logger.info(f"DSPy configured with: {dspy.settings.lm}")
    
    # Verify real Ollama is working
    if hasattr(lm, 'model') and 'ollama' in str(lm.model):
        logger.info("Ollama model confirmed")
    else:
        logger.error(f"Wrong LM configured: {lm}")
        raise ValueError("Expected Ollama model but got something else")
    
    # Create the agent
    logger.info("Initializing agent...")
    agent = create_agent(db, docs, lm)
    
    # Load questions
    logger.info(f"Loading questions from {batch}")
    questions = load_questions(batch)
    logger.info(f"Loaded {len(questions)} questions")
    
    # Process questions
    results = []
    for i, question_data in enumerate(questions, 1):
        question_id = question_data["id"]
        question = question_data["question"]
        format_hint = question_data.get("format_hint", "str")
        
        logger.info(f"Processing question {i}/{len(questions)}: {question_id}")
        logger.info(f"Question: {question}")
        
        start_time = time.time()
        
        try:
            result = agent.run_query(question, question_id, format_hint)
            processing_time = time.time() - start_time
            
            logger.info(f"Completed in {processing_time:.2f}s")
            logger.info(f"Answer: {result['final_answer']}")
            logger.info(f"Confidence: {result['confidence']}")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            results.append({
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            })
    
    # Save results
    logger.info(f"Saving results to {out}")
    save_results(results, out)
    
    # Summary
    successful = sum(1 for r in results if r["final_answer"] is not None)
    logger.info(f"Completed: {successful}/{len(questions)} questions successful")
    
    click.echo(f"Processed {len(questions)} questions")
    click.echo(f"{successful} successful, {len(questions) - successful} failed")
    click.echo(f"Results saved to {out}")


if __name__ == "__main__":
    main()
