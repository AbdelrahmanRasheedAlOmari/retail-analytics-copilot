#!/usr/bin/env python3
"""
Test script to verify basic functionality without external dependencies
"""
import sys
import sqlite3
from pathlib import Path


def test_database():
    """Test basic database connectivity"""
    print("Testing database connectivity...")
    
    db_path = "data/northwind.sqlite"
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT COUNT(*) FROM Orders")
        count = cursor.fetchone()[0]
        print(f"✅ Orders table has {count} records")
        
        # Test tables exist
        required_tables = ["Orders", "Order Details", "Products", "Customers", "Categories"]
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        missing_tables = [t for t in required_tables if t not in tables]
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        print("✅ All required tables present")
        
        # Test a join query
        cursor.execute("""
            SELECT COUNT(*) 
            FROM Orders o 
            JOIN [Order Details] od ON o.OrderID = od.OrderID 
            JOIN Products p ON od.ProductID = p.ProductID
        """)
        join_count = cursor.fetchone()[0]
        print(f"✅ Join query successful: {join_count} order detail records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def test_documents():
    """Test document files exist and are readable"""
    print("\nTesting documents...")
    
    docs_dir = Path("docs")
    required_files = [
        "marketing_calendar.md",
        "kpi_definitions.md", 
        "catalog.md",
        "product_policy.md"
    ]
    
    success = True
    for filename in required_files:
        filepath = docs_dir / filename
        if not filepath.exists():
            print(f"❌ Missing document: {filename}")
            success = False
        else:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:
                    print(f"✅ {filename} ({len(content)} chars)")
                else:
                    print(f"❌ {filename} is empty")
                    success = False
    
    return success


def test_evaluation_file():
    """Test evaluation file exists and is properly formatted"""
    print("\nTesting evaluation file...")
    
    eval_file = "sample_questions_hybrid_eval.jsonl"
    if not Path(eval_file).exists():
        print(f"❌ Evaluation file not found: {eval_file}")
        return False
    
    try:
        import json
        questions = []
        with open(eval_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        question = json.loads(line)
                        questions.append(question)
                        
                        # Validate required fields
                        required_fields = ["id", "question", "format_hint"]
                        missing_fields = [f for f in required_fields if f not in question]
                        if missing_fields:
                            print(f"❌ Line {line_num} missing fields: {missing_fields}")
                            return False
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ Line {line_num} invalid JSON: {e}")
                        return False
        
        print(f"✅ Evaluation file has {len(questions)} valid questions")
        
        # Test specific question IDs
        expected_ids = [
            "rag_policy_beverages_return_days",
            "hybrid_top_category_qty_summer_1997", 
            "hybrid_aov_winter_1997",
            "sql_top3_products_by_revenue_alltime",
            "hybrid_revenue_beverages_summer_1997",
            "hybrid_best_customer_margin_1997"
        ]
        
        actual_ids = [q["id"] for q in questions]
        missing_ids = [id for id in expected_ids if id not in actual_ids]
        
        if missing_ids:
            print(f"❌ Missing question IDs: {missing_ids}")
            return False
        
        print("✅ All expected question IDs present")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation file error: {e}")
        return False


def test_project_structure():
    """Test project directory structure"""
    print("\nTesting project structure...")
    
    required_files = [
        "run_agent_hybrid.py",
        "requirements.txt",
        "README.md",
        "agent/__init__.py",
        "agent/graph_hybrid.py",
        "agent/dspy_signatures.py",
        "agent/rag/__init__.py",
        "agent/rag/retrieval.py",
        "agent/tools/__init__.py",
        "agent/tools/sqlite_tool.py"
    ]
    
    success = True
    for filepath in required_files:
        if not Path(filepath).exists():
            print(f"❌ Missing file: {filepath}")
            success = False
        else:
            print(f"✅ {filepath}")
    
    return success


def main():
    """Run all tests"""
    print("🧪 Testing Retail Analytics Copilot Components\n")
    
    tests = [
        test_project_structure,
        test_database,
        test_documents,
        test_evaluation_file
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ All tests passed! The system is ready to run.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install Ollama and pull model: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")
        print("3. Run the agent: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
