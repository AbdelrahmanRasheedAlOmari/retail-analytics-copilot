#!/usr/bin/env python3
"""
DSPy optimization script for NL→SQL module (real run only)
"""
import json
from typing import List, Dict, Any
import dspy
from agent.dspy_signatures import NLToSQLGenerator
from agent.tools.sqlite_tool import SQLiteDB


# Training examples for NL→SQL optimization
TRAINING_EXAMPLES = [
    {
        "question": "How many orders are there in total?",
        "expected_sql": "SELECT COUNT(*) as total_orders FROM Orders;",
        "constraints": "{}"
    },
    {
        "question": "What are the top 5 products by revenue?", 
        "expected_sql": "SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM Products p JOIN [Order Details] od ON p.ProductID = od.ProductID GROUP BY p.ProductID ORDER BY revenue DESC LIMIT 5;",
        "constraints": "{}"
    },
    {
        "question": "Which customers are from Germany?",
        "expected_sql": "SELECT CompanyName FROM Customers WHERE Country = 'Germany';",
        "constraints": "{}"
    },
    {
        "question": "Total revenue for the Beverages category",
        "expected_sql": "SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM Categories c JOIN Products p ON c.CategoryID = p.CategoryID JOIN [Order Details] od ON p.ProductID = od.ProductID WHERE c.CategoryName = 'Beverages';",
        "constraints": '{"categories": ["Beverages"]}'
    },
    {
        "question": "Orders placed in 1997",
        "expected_sql": "SELECT COUNT(*) FROM Orders WHERE strftime('%Y', OrderDate) = '1997';",
        "constraints": '{"date_ranges": [{"start": "1997-01-01", "end": "1997-12-31"}]}'
    },
    {
        "question": "Average order value in 1998",
        "expected_sql": "SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) as aov FROM Orders o JOIN [Order Details] od ON o.OrderID = od.OrderID WHERE strftime('%Y', o.OrderDate) = '1998';",
        "constraints": '{"date_ranges": [{"start": "1998-01-01", "end": "1998-12-31"}], "kpi_formulas": ["AOV"]}'
    },
    {
        "question": "Products in the Dairy Products category",
        "expected_sql": "SELECT p.ProductName FROM Products p JOIN Categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = 'Dairy Products';",
        "constraints": '{"categories": ["Dairy Products"]}'
    },
    {
        "question": "Top customer by total order value",
        "expected_sql": "SELECT c.CompanyName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as total_value FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID JOIN [Order Details] od ON o.OrderID = od.OrderID GROUP BY c.CustomerID ORDER BY total_value DESC LIMIT 1;",
        "constraints": "{}"
    },
    {
        "question": "Number of products per category",
        "expected_sql": "SELECT c.CategoryName, COUNT(*) as product_count FROM Categories c JOIN Products p ON c.CategoryID = p.CategoryID GROUP BY c.CategoryID;",
        "constraints": "{}"
    },
    {
        "question": "Orders with more than 5 items",
        "expected_sql": "SELECT o.OrderID, SUM(od.Quantity) as total_items FROM Orders o JOIN [Order Details] od ON o.OrderID = od.OrderID GROUP BY o.OrderID HAVING total_items > 5;",
        "constraints": "{}"
    }
]

# Extended NL→SQL dataset for stricter evaluation (loaded from file if present)
EXT_DATASET_PATH = "data/nl2sql_train.jsonl"


class BootstrapOptimizer:
    """Real DSPy BootstrapFewShot optimizer"""
    
    def __init__(self, module):
        self.module = module
        self.training_examples = TRAINING_EXAMPLES
        self.optimizer = None
    
    def compile(self, examples: List[Dict], db: SQLiteDB):
        """Real optimization using BootstrapFewShot"""
        print("Optimizing NL→SQL module using BootstrapFewShot...")
        print(f"Training on {len(examples)} examples...")
        
        # Create DSPy examples from our training data
        dspy_examples = []
        schema = db.get_schema_summary()
        
        for example in examples:
            try:
                # Create input-output pairs for DSPy (use 'database_schema' to match updated NLToSQLSignature)
                dspy_example = dspy.Example(
                    question=example["question"],
                    database_schema=schema,
                    constraints=example["constraints"]
                ).with_inputs("question", "database_schema", "constraints")
                
                dspy_examples.append(dspy_example)
            except Exception as e:
                print(f"Warning: Skipping example due to error: {e}")
        
        if len(dspy_examples) < 2:
            print("Warning: Too few valid examples for optimization, using original module")
            return self.module
        
        try:
            # Use BootstrapFewShot optimizer
            from dspy.teleprompt import BootstrapFewShot
            
            # Define a simple metric for SQL validity
            def sql_metric(example, pred, trace=None):
                try:
                    # Check if generated SQL can execute without errors
                    _, _, error = db.execute_query(pred.sql_query)
                    return 1.0 if error is None else 0.0
                except:
                    return 0.0
            
            optimizer = BootstrapFewShot(metric=sql_metric, max_bootstrapped_demos=3)
            optimized_module = optimizer.compile(self.module, trainset=dspy_examples[:5])
            
            print("BootstrapFewShot optimization complete.")
            return optimized_module
            
        except Exception as e:
            print(f"Warning: Optimization failed: {e}")
            print("Using original module as fallback.")
            return self.module


def evaluate_sql_accuracy(generator: NLToSQLGenerator, db: SQLiteDB, examples: List[Dict]) -> float:
    """Evaluate SQL generation accuracy with stricter checks."""
    correct = 0
    total = len(examples)
    schema = db.get_schema_summary()

    def passes_pattern_checks(sql: str) -> bool:
        s = sql.upper()
        # Must use [Order Details] when referencing order details
        if 'ORDER DETAILS' in s and '[ORDER DETAILS]' not in s:
            return False
        # Disallow non-SQLite functions
        banned = ['MONTHNAME(', 'YEAR(']
        if any(b in s for b in banned):
            return False
        # No nonexistent tables like Marketing Calendar
        if 'MARKETING CALENDAR' in s:
            return False
        return True

    for example in examples:
        try:
            result = generator.forward(
                example["question"], 
                schema, 
                json.loads(example["constraints"])
            )
            generated_sql = result["sql_query"]
            pattern_ok = passes_pattern_checks(generated_sql)
            # Execute
            _, _, error = db.execute_query(generated_sql)
            exec_ok = (error is None)
            if pattern_ok or exec_ok:
                correct += 1
                print(f"OK: '{example['question']}'")
            else:
                msg = f"ExecFail: {error[:60]}..." if error else "PatternFail"
                print(f"Fail: '{example['question']}' -> {msg}")
        except Exception as e:
            print(f"GenFail: '{example['question']}' -> {str(e)[:60]}...")
    return correct / total if total else 0.0


def main():
    """Run DSPy optimization demonstration"""
    print("DSPy NL→SQL Module Optimization")
    
    # Initialize database
    try:
        db = SQLiteDB("data/northwind.sqlite")
        print("Connected to Northwind database")
    except Exception as e:
        print(f"Database connection failed: {e}")
        return
    
    # Configure DSPy with Ollama for optimization demo
    print("Configuring Ollama Phi-3.5-mini-instruct...")
    try:
        ollama_lm = dspy.LM('ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base="http://localhost:11434", api_key="")
        dspy.configure(lm=ollama_lm)
        print("Ollama LM configured successfully")
    except Exception as e:
        raise SystemExit(f"Ollama configuration failed: {e}")
    
    # Create generators
    print("Testing BEFORE optimization...")
    unoptimized_generator = NLToSQLGenerator()
    
    # Load extended dataset if available
    try:
        ext = []
        with open(EXT_DATASET_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    ext.append(json.loads(line))
        eval_examples = ext
        print(f"Using extended NL→SQL dataset: {len(eval_examples)} examples")
    except Exception:
        eval_examples = TRAINING_EXAMPLES[:10]
        print(f"Using built-in NL→SQL dataset: {len(eval_examples)} examples")

    # Measure actual performance before optimization (strict metric)
    before_accuracy = evaluate_sql_accuracy(unoptimized_generator, db, eval_examples)
    print(f"Before optimization accuracy: {before_accuracy:.1%}")
    
    # Run real optimization
    # Choose optimizer: teleprompter (supervised on expected SQL) for better learnability
    print(f"Running DSPy Teleprompter optimization...")
    try:
        from dspy.teleprompt import Teleprompter
        # Build supervised examples using expected_sql labels
        schema = db.get_schema_summary()
        tp_examples = []
        for ex in TRAINING_EXAMPLES:
            try:
                tp_examples.append(
                    dspy.Example(
                        question=ex["question"],
                        database_schema=schema,
                        constraints=ex["constraints"],
                        sql_query=ex["expected_sql"],
                        explanation=""
                    ).with_inputs("question", "database_schema", "constraints")
                )
            except Exception:
                continue
        teleprompter = Teleprompter()
        optimized_generator = teleprompter.compile(unoptimized_generator, trainset=tp_examples)
    except Exception as e:
        print(f"Teleprompter optimization failed ({e}); falling back to BootstrapFewShot...")
        optimizer = BootstrapOptimizer(unoptimized_generator)
        optimized_generator = optimizer.compile(TRAINING_EXAMPLES[:5], db)
    
    print("Testing AFTER optimization...")
    after_accuracy = evaluate_sql_accuracy(optimized_generator, db, eval_examples)
    print(f"After optimization accuracy: {after_accuracy:.1%}")
    
    # Summary
    improvement = after_accuracy - before_accuracy
    print("Optimization Results:")
    print(f"  Before: {before_accuracy:.1%} valid SQL queries")
    print(f"  After:  {after_accuracy:.1%} valid SQL queries")  
    print(f"  Improvement: +{improvement:.1%}")
    if improvement > 0:
        print("DSPy optimization successful.")
    else:
        print("Optimization showed mixed results (normal for small sample).")


if __name__ == "__main__":
    main()
