"""
SQLite tools for database access and schema introspection
"""
import sqlite3
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class SQLiteDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema = {}
            for table in tables:
                # Get column info (properly quote table names with spaces)
                quoted_table = f"[{table}]" if " " in table else table
                cursor.execute(f"PRAGMA table_info({quoted_table})")
                columns = []
                for col_info in cursor.fetchall():
                    columns.append({
                        'name': col_info[1],
                        'type': col_info[2],
                        'not_null': bool(col_info[3]),
                        'default': col_info[4],
                        'pk': bool(col_info[5])
                    })
                
                # Get foreign keys (properly quote table names with spaces)
                cursor.execute(f"PRAGMA foreign_key_list({quoted_table})")
                foreign_keys = []
                for fk_info in cursor.fetchall():
                    foreign_keys.append({
                        'column': fk_info[3],
                        'references_table': fk_info[2],
                        'references_column': fk_info[4]
                    })
                
                schema[table] = {
                    'columns': columns,
                    'foreign_keys': foreign_keys
                }
        
        return schema
    
    def get_sample_data(self, table: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample data from a table"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    def execute_query(self, query: str) -> Tuple[List[str], List[Tuple], Optional[str]]:
        """
        Execute a SQL query and return results
        Returns: (column_names, rows, error_message)
        """
        try:
            # Auto-fix common SQL issues with table names
            fixed_query = self._fix_table_names(query)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(fixed_query)
                
                columns = [description[0] for description in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                
                # Convert Row objects to tuples for JSON serialization
                rows = [tuple(row) for row in rows]
                
                return columns, rows, None
        except Exception as e:
            return [], [], str(e)
    
    def _fix_table_names(self, query: str) -> str:
        """Fix common table name issues in SQL queries"""
        # Fix unquoted "Order Details" references
        import re
        
        # Don't fix if already properly quoted
        if '[Order Details]' in query:
            return query
            
        # Replace unquoted Order Details with proper bracketed version
        # Use negative lookbehind/lookahead to avoid double-quoting
        pattern = r'(?<!\[)(?<!")(?<!`)Order\s+Details(?!\])(?!")(?!`)'
        fixed_query = re.sub(pattern, '[Order Details]', query, flags=re.IGNORECASE)
        
        return fixed_query
    
    def validate_query_syntax(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query syntax without executing
        Returns: (is_valid, error_message)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Use EXPLAIN to check syntax without executing
                cursor.execute(f"EXPLAIN {query}")
                return True, None
        except Exception as e:
            return False, str(e)
    
    def get_table_names(self) -> List[str]:
        """Get all table names"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            return [row[0] for row in cursor.fetchall()]
    
    def get_schema_summary(self) -> str:
        """Get a human-readable schema summary"""
        schema = self.get_schema_info()
        summary = []
        
        summary.append("CRITICAL: Only these 5 tables exist in the database:")
        summary.append("  - Orders (no spaces)")
        summary.append("  - [Order Details] (HAS SPACES - MUST use square brackets!)")  
        summary.append("  - Products (no spaces)")
        summary.append("  - Customers (no spaces)")
        summary.append("  - Categories (no spaces)")
        summary.append("")
        summary.append("NEVER reference: Marketing Calendar, KPI tables, document tables - they don't exist!")
        summary.append("Marketing data comes from DOCUMENTS, not database tables.")
        summary.append("")
        
        for table_name, table_info in schema.items():
            # Show proper quoting for table names with spaces
            display_name = f"[{table_name}]" if " " in table_name else table_name
            summary.append(f"\nTable: {display_name}")
            
            # Columns
            for col in table_info['columns']:
                pk_indicator = " (PK)" if col['pk'] else ""
                not_null = " NOT NULL" if col['not_null'] else ""
                summary.append(f"  - {col['name']}: {col['type']}{pk_indicator}{not_null}")
            
            # Foreign keys
            if table_info['foreign_keys']:
                summary.append("  Foreign Keys:")
                for fk in table_info['foreign_keys']:
                    summary.append(f"    - {fk['column']} -> {fk['references_table']}.{fk['references_column']}")
        
        return "\n".join(summary)


def create_sqlite_tool(db_path: str) -> SQLiteDB:
    """Factory function to create SQLiteDB instance"""
    return SQLiteDB(db_path)


if __name__ == "__main__":
    # Test the SQLite tool
    db = SQLiteDB("../../data/northwind.sqlite")
    print("Schema Summary:")
    print(db.get_schema_summary())
    
    print("\nSample Orders:")
    sample_orders = db.get_sample_data("Orders", 3)
    for order in sample_orders:
        print(order)
    
    print("\nTest Query:")
    columns, rows, error = db.execute_query("SELECT COUNT(*) as total_orders FROM Orders")
    if error:
        print(f"Error: {error}")
    else:
        print(f"Columns: {columns}")
        print(f"Result: {rows}")
