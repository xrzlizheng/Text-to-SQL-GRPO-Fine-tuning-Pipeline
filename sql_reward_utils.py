import re
import os
import tempfile
import sqlite3
import logging
import sqlparse
import torch
import copy
from sqlglot import parse, transpile, ParseError
from sqlglot.expressions import Column, Table  # Import available types
from typing import List, Dict, Tuple, Any, Optional, Set, Union

# ================== Logging Setup ==================
# More detailed format, log level controlled by DEBUG_MODE
log_level = logging.DEBUG if os.environ.get("SQL_DEBUG_MODE") == "1" else logging.CRITICAL + 1  # disables all logging
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Global Configuration ==================
REWARD_WEIGHTS = {
    "format": 1.0,
    "sql_correctness": 1.2,
    "complexity": 0.6,
    "reasoning": 0.7,
}

# Control detailed logging via environment variable or direct setting
DEBUG_MODE = os.environ.get("SQL_DEBUG_MODE") == "1"

# Constants for Error Categories
ERR_SYNTAX = "syntax_error"
ERR_MISSING_TABLE = "missing_table"
ERR_MISSING_COLUMN = "missing_column"
ERR_AMBIGUOUS_COLUMN = "ambiguous_column"
ERR_TYPE_MISMATCH = "type_mismatch"
ERR_CONSTRAINT = "constraint_violation"
ERR_FUNCTION = "function_error"
ERR_RESOURCE = "resource_error"
ERR_OTHER = "other_error"
ERR_SCHEMA_SETUP = "schema_setup_error"
ERR_CONVERSION = "sql_conversion_error"
ERR_EXECUTION = "sql_execution_error"
ERR_SCHEMA_VALIDATION = "schema_validation_error"

 
# ================== Helper Functions (Parsing, Normalization, etc.) ==================

def _get_response_text(completion: Any) -> str:
    """Extract text content from various completion formats."""
    response_text = ""
    if isinstance(completion, str):
        response_text = completion
    elif isinstance(completion, list) and completion:
        if isinstance(completion[0], dict):
            response_text = completion[0].get('content', completion[0].get('generated_text', ''))
        elif isinstance(completion[0], str):
            response_text = completion[0]
    elif isinstance(completion, dict):
        response_text = completion.get('content', completion.get('generated_text', ''))
    else:
        try:
            response_text = str(completion)
        except Exception:
            response_text = ""
            logger.debug("Could not convert completion to string: %s", type(completion))
    return response_text


def extract_sql(text: str) -> str:
    """Extract SQL query from model output using regex for robustness."""
    if not text:
        return ""
    match = re.search(r"<sql>(.*?)</sql>", text, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Basic cleaning: remove comments that might interfere with parsing/execution
        sql = re.sub(r"^\s*--.*?\n", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"\n--.*?\s*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        return sql.strip()
    else:
        # Fallback attempt (less reliable)
        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ", "DROP ", "WITH "]
        text_upper = text.upper()
        sql_start_index = -1
        for keyword in sql_keywords:
            idx = text_upper.find(keyword)
            if idx != -1 and (sql_start_index == -1 or idx < sql_start_index):
                sql_start_index = idx
        if sql_start_index != -1:
            potential_sql = text[sql_start_index:]
            # Simple termination heuristics
            potential_sql = potential_sql.split("</sql>", 1)[0]
            potential_sql = potential_sql.split("</reasoning>", 1)[0]
            if ";" in potential_sql:
                potential_sql = potential_sql.split(";", 1)[0] + ";"
            logger.debug("Extracted SQL using fallback method.")
            return potential_sql.strip()
        logger.debug("Could not extract SQL using primary or fallback methods.")
        return ""


def extract_reasoning(text: str) -> str:
    """Extract reasoning from model output."""
    if not text: return ""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def calculate_sql_complexity(sql: str) -> float:
    """Calculate a heuristic complexity score for an SQL query."""
    if not sql: return 0.0
    try:
        sql_upper = sql.upper()
        score = 1.0
        score += sql_upper.count(" JOIN ") * 0.6
        score += sql_upper.count(" UNION ") * 0.8 + sql_upper.count(" INTERSECT ") * 0.8 + sql_upper.count(" EXCEPT ") * 0.8
        score += sql_upper.count("(SELECT") * 1.0 # Subqueries
        score += sql_upper.count(" WITH ") * 0.8   # CTEs
        if " WHERE " in sql_upper: score += 0.2
        if " GROUP BY " in sql_upper: score += 0.5
        if " HAVING " in sql_upper: score += 0.7
        if " ORDER BY " in sql_upper: score += 0.3
        if " LIMIT " in sql_upper: score += 0.1
        # Functions and Operators
        agg_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
        score += sum(sql_upper.count(agg) for agg in agg_functions) * 0.3
        score += sql_upper.count(" DISTINCT ") * 0.3
        score += sql_upper.count(" CASE ") * 0.4
        score += sql_upper.count(" OVER(") * 1.0 # Window functions
        # Basic WHERE clause complexity
        where_match = re.search(r" WHERE (.*?)(?: GROUP BY | ORDER BY | LIMIT | OFFSET |$)", sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            score += where_clause.count(" AND ") * 0.15 + where_clause.count(" OR ") * 0.20
            score += where_clause.count(" IN ") * 0.2 + where_clause.count(" LIKE ") * 0.1
            score += where_clause.count(" BETWEEN ") * 0.2 + where_clause.count(" EXISTS ") * 0.3
        return max(0.0, score)
    except Exception as e:
        logger.warning(f"Error calculating complexity for '{sql[:50]}...': {e}")
        return 1.0 # Default complexity on error


def identify_sql_statement_type(sql: str) -> str:
    """Identify the type of SQL statement (SELECT, INSERT, UPDATE, DELETE, etc.)."""
    if not sql: return "UNKNOWN"
    # Basic cleaning and get first word
    clean_sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE).strip()
    clean_sql = re.sub(r'/\*.*?\*/', '', clean_sql, flags=re.DOTALL).strip()
    if not clean_sql: return "UNKNOWN"
    first_word = clean_sql.split(None, 1)[0].upper()

    if first_word == "SELECT": return "SELECT"
    if first_word == "INSERT": return "INSERT"
    if first_word == "UPDATE": return "UPDATE"
    if first_word == "DELETE": return "DELETE"
    if first_word == "CREATE":
        # Basic check for TABLE/VIEW/INDEX after CREATE
        if re.search(r"CREATE\s+(TABLE|VIEW|INDEX)", clean_sql[:30], re.IGNORECASE):
             # More specific type if needed (e.g., CREATE_TABLE)
             second_word = clean_sql.split(None, 2)[1].upper() if len(clean_sql.split()) > 1 else ""
             if second_word == "TABLE": return "CREATE_TABLE"
             if second_word == "VIEW": return "CREATE_VIEW"
             if second_word == "INDEX": return "CREATE_INDEX"
        return "CREATE_OTHER" # Default create type
    if first_word == "DROP": return "DROP"
    if first_word == "ALTER": return "ALTER"
    if first_word == "WITH": # Could be CTE for SELECT/INSERT/UPDATE/DELETE
        # Look for the actual DML/SELECT keyword after the CTE definition
        match = re.search(r'\)\s*(SELECT|INSERT|UPDATE|DELETE)', clean_sql, re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).upper()
        return "WITH_UNKNOWN" # Cannot determine underlying type easily
    return "OTHER"


# ================== Database Interaction Helpers ==================

def list_all_tables(conn: sqlite3.Connection) -> List[str]:
    """List all tables and views in a SQLite database connection."""
    try:
        cursor = conn.cursor()
        # sqlite_master includes tables and views
        cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error listing tables/views: {e}")
        return []

def get_table_schema(conn: sqlite3.Connection, table_name: str) -> List[Tuple]:
    """Get schema information for a specific table."""
    try:
        cursor = conn.cursor()
        # Use parameterized query for safety, though table names are usually controlled
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.warning(f"Error getting schema for table {table_name}: {e}")
        return []

def check_table_exists(conn: sqlite3.Connection, table_name: str) -> Tuple[bool, bool, Optional[str]]:
    """Check if table/view exists. Returns: (exists_exact, exists_case_insensitive, correct_case_name)."""
    try:
        cursor = conn.cursor()
        # Exact match
        cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name=?;", (table_name,))
        exact_match = cursor.fetchone()
        if exact_match:
            return True, True, table_name

        # Case-insensitive match
        cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND lower(name)=lower(?);", (table_name,))
        insensitive_match = cursor.fetchone()
        if insensitive_match:
            return False, True, insensitive_match[0]

        return False, False, None
    except sqlite3.Error as e:
        logger.warning(f"Error checking existence for table/view {table_name}: {e}")
        return False, False, None

def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get column names for a table, preserving original case."""
    schema = get_table_schema(conn, table_name)
    return [col[1] for col in schema] # Column name is at index 1

def extract_tables_from_query(sql: str) -> Set[str]:
    """Extract table names referenced in a SQL query using sqlglot (more robust)."""
    tables = set()
    if not sql: return tables
    try:
        # Use sqlglot's parsing capabilities
        parsed_expression = parse(sql, read="sqlite") # Assume SQLite or target dialect
        
        # Handle both single expression or list of expressions
        if isinstance(parsed_expression, list):
            # Process each expression in the list
            for expr in parsed_expression:
                if hasattr(expr, 'find_all'):
                    for node in expr.find_all():
                        if hasattr(node, 'name') and hasattr(node, 'is_table') and node.is_table:
                            table_name = node.name
                            if table_name:
                                tables.add(table_name) # sqlglot usually handles quotes/case
        else:
            # Find all table expressions in the AST
            for node in parsed_expression.find_all():
                if hasattr(node, 'name') and hasattr(node, 'is_table') and node.is_table:
                    table_name = node.name
                    if table_name:
                        tables.add(table_name) # sqlglot usually handles quotes/case
    except ParseError as e:
        logger.warning(f"sqlglot failed to parse for table extraction: {e}. Falling back to regex.")
        # Fallback Regex (less reliable)
        # Look for FROM/JOIN clauses, handle aliases simply
        pattern = r'(?:FROM|JOIN)\s+([`"\[]?\w+[`"\]]?)(?:\s+(?:AS\s+)?(\w+))?'
        for match in re.finditer(pattern, sql, re.IGNORECASE):
             table = match.group(1).strip('`"[]')
             tables.add(table)
    except Exception as e:
         logger.error(f"Unexpected error during table extraction: {e}")

    # Note: sqlglot typically returns identifiers in lowercase unless quoted.
    # If case sensitivity is critical and differs from source, more handling might be needed.
    return tables


# ================== SQL Conversion and Fixing Helpers ==================

def convert_sql_to_sqlite(sql: str, source_dialect: str = "mysql") -> Optional[str]:
    """Convert SQL to SQLite using sqlglot. Returns None on critical failure."""
    if not sql or not sql.strip():
        return sql
    try:
        if DEBUG_MODE: logger.debug(f"Converting SQL from {source_dialect} to sqlite: {sql[:150]}...")
        # Handle dialect-specific conversion issues
        if source_dialect == "postgresql":
            # First try direct conversion
            try:
                converted = transpile(sql, read="postgres", write="sqlite")
            except ParseError as pg_err:
                logger.warning(f"PostgreSQL parse error: {pg_err}, trying fallback conversion...")
                # PostgreSQL-specific preprocessing before conversion
                modified_sql = sql
                # Handle PostgreSQL specific constructs
                # 1. Replace :: type casts with SQLite compatible CAST
                modified_sql = re.sub(r'(\w+)::\w+', r'CAST(\1 AS TEXT)', modified_sql)
                # 2. Replace RETURNING clauses which SQLite doesn't support
                modified_sql = re.sub(r'\s+RETURNING\s+.*?$', '', modified_sql, flags=re.IGNORECASE)
                # Try again with preprocessed SQL
                try:
                    converted = transpile(modified_sql, read="postgres", write="sqlite")
                except ParseError:
                    # If still failing, try with generic SQL
                    logger.warning("Falling back to generic SQL parsing...")
                    converted = transpile(modified_sql, read="generic", write="sqlite")
        else:
            # Standard conversion for MySQL and other dialects
            converted = transpile(sql, read=source_dialect, write="sqlite")
        if converted and isinstance(converted, list) and converted[0]:
             if DEBUG_MODE: logger.debug(f"Converted SQL: {converted[0][:150]}...")
             return converted[0]
        else:
             logger.warning(f"sqlglot transpile returned empty result for: {sql[:100]}...")
             return sql # Return original as fallback if conversion yields nothing
    except ParseError as e:
        logger.warning(f"sqlglot ParseError during conversion: {e}. SQL: {sql[:150]}...")
        # Simple fallback attempts for common issues (e.g., AUTO_INCREMENT)
        if 'AUTO_INCREMENT' in sql.upper():
            modified_sql = re.sub(r'AUTO_INCREMENT', 'AUTOINCREMENT', sql, flags=re.IGNORECASE)
            # Ensure INTEGER PRIMARY KEY AUTOINCREMENT format
            modified_sql = re.sub(r'(\w+)\s+(?:INT|INTEGER)\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
                               r'\1 INTEGER PRIMARY KEY AUTOINCREMENT', modified_sql, flags=re.IGNORECASE)
            logger.debug("Applied manual AUTO_INCREMENT fix attempt.")
            return modified_sql
        return sql # Return original if parse error and no simple fix applied
    except Exception as e:
        logger.error(f"Unexpected error during sqlglot conversion: {e}", exc_info=DEBUG_MODE)
        return sql # Return original on unexpected errors


def fix_case_sensitivity_in_sql(conn: sqlite3.Connection, sql: str) -> str:
    """Attempt to fix table and column case sensitivity issues in SQL for SQLite."""
    if not sql: return sql

    corrected_sql = sql
    all_db_tables = list_all_tables(conn)
    if not all_db_tables: return sql # Cannot fix if we don't know the tables

    table_case_map = {t.lower(): t for t in all_db_tables}

    # 1. Fix Table Names
    referenced_tables = extract_tables_from_query(corrected_sql) # Use improved extraction
    needs_table_fix = False
    for table in referenced_tables:
        table_lower = table.lower()
        if table not in table_case_map.values() and table_lower in table_case_map:
            correct_case_table = table_case_map[table_lower]
            logger.debug(f"Case Fix: Replacing table '{table}' with '{correct_case_table}'")
            # Use regex with word boundaries to avoid partial replacements
            # Be careful with quotes/backticks if sqlglot didn't handle them
            corrected_sql = re.sub(r'\b' + re.escape(table) + r'\b', correct_case_table, corrected_sql, flags=re.IGNORECASE)
            needs_table_fix = True

    if needs_table_fix:
         logger.debug(f"SQL after table case correction: {corrected_sql[:150]}...")

    # 2. Fix Column Names (more complex, do it per table)
    # Re-extract tables from potentially corrected SQL
    current_referenced_tables = extract_tables_from_query(corrected_sql)
    needs_column_fix = False

    # Use sqlglot's parser if possible to find column references more accurately
    try:
        parsed_exp = parse(corrected_sql, read="sqlite")
        
        # Handle both single expression or list of expressions
        if isinstance(parsed_exp, list):
            # If we got a list of expressions, process each one
            all_col_refs = []
            for expr in parsed_exp:
                if hasattr(expr, 'find_all'):
                    all_col_refs.extend(expr.find_all(Column))
        else:
            # Single expression
            all_col_refs = parsed_exp.find_all(Column)
            
        for col_exp in all_col_refs:
            col_name = col_exp.name
            table_alias_or_name = col_exp.table # Might be alias or actual table name

            # Determine the actual table this column belongs to (this is hard without full context)
            # Simple approach: If table is specified, use it. Otherwise, check all referenced tables.
            target_table = None
            if table_alias_or_name:
                 # TODO: Need alias resolution here for robustness. For now, assume it's a real table name.
                 if table_alias_or_name.lower() in table_case_map:
                      target_table = table_case_map[table_alias_or_name.lower()]
            else:
                 # Unqualified column - check all referenced tables (potential ambiguity)
                 # For simplicity, we'll check against known columns later
                 pass # Handled below by checking all table schemas

            if target_table:
                 db_columns = get_column_names(conn, target_table)
                 col_case_map = {c.lower(): c for c in db_columns}
                 if col_name not in db_columns and col_name.lower() in col_case_map:
                      correct_case_col = col_case_map[col_name.lower()]
                      logger.debug(f"Case Fix: Replacing column '{table_alias_or_name}.{col_name}' with '{table_alias_or_name}.{correct_case_col}'")
                      # Regex replacement is tricky here due to potential alias/table mismatch
                      # Pattern: table_or_alias . col_name (with word boundaries)
                      pattern = r'\b' + re.escape(table_alias_or_name) + r'\s*\.\s*' + re.escape(col_name) + r'\b'
                      replacement = f"{table_alias_or_name}.{correct_case_col}"
                      corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                      needs_column_fix = True

            # Handle unqualified columns by checking all referenced tables
            # This is prone to error if column names overlap between tables!
            elif not table_alias_or_name: # Unqualified column
                possible_corrections = []
                for ref_table_name_lower in current_referenced_tables:
                     actual_ref_table = table_case_map.get(ref_table_name_lower.lower())
                     if actual_ref_table:
                          db_columns = get_column_names(conn, actual_ref_table)
                          col_case_map = {c.lower(): c for c in db_columns}
                          if col_name not in db_columns and col_name.lower() in col_case_map:
                               possible_corrections.append(col_case_map[col_name.lower()])

                if len(possible_corrections) == 1: # Only correct if unambiguous
                    correct_case_col = possible_corrections[0]
                    logger.debug(f"Case Fix: Replacing unqualified column '{col_name}' with '{correct_case_col}'")
                    # Pattern: non-dot word boundary, column name, word boundary
                    pattern = r'(?<![\w\.])\b' + re.escape(col_name) + r'\b'
                    corrected_sql = re.sub(pattern, correct_case_col, corrected_sql, flags=re.IGNORECASE)
                    needs_column_fix = True
                elif len(possible_corrections) > 1:
                    logger.warning(f"Ambiguous case correction for unqualified column '{col_name}'. Found in multiple tables. Skipping.")

    except ParseError as e:
         logger.warning(f"sqlglot failed to parse for column case fixing: {e}. Column fix might be incomplete.")
    except Exception as e:
         logger.error(f"Unexpected error during column case fixing: {e}", exc_info=DEBUG_MODE)


    if needs_column_fix:
        logger.debug(f"SQL after column case correction: {corrected_sql[:150]}...")

    return corrected_sql


def fix_ambiguous_columns(sql: str, conn: Optional[sqlite3.Connection] = None) -> str:
    """Minimal heuristic attempt to fix ambiguous columns by qualifying with the first table found."""
    # Note: This is very basic and might be incorrect. A proper fix requires schema knowledge
    # and understanding aliases, which is better handled by robust parsers or during generation.
    if " JOIN " not in sql.upper():
        return sql # Only apply to joins

    try:
        parsed_exp = parse(sql, read="sqlite")
        # Identify potential unqualified columns often causing ambiguity
        common_ambiguous = {'id', 'name', 'date', 'code'} # Example set
        first_table_alias = None
        
        # Handle both single expression or list of expressions
        if isinstance(parsed_exp, list):
            tables = []
            for expr in parsed_exp:
                if hasattr(expr, 'find_all'):
                    for node in expr.find_all():
                        if hasattr(node, 'name') and hasattr(node, 'is_table') and node.is_table:
                            tables.append(node)
        else:
            tables = [node for node in parsed_exp.find_all() 
                     if hasattr(node, 'name') and hasattr(node, 'is_table') and node.is_table]
            
        if tables:
             # Try to get alias if present, else table name
             first_table_alias = tables[0].alias_or_name

        if not first_table_alias:
             return sql # Cannot qualify if no table/alias identified

        fixed_sql = sql
        modified = False
        
        # Process column references
        if isinstance(parsed_exp, list):
            all_col_refs = []
            for expr in parsed_exp:
                if hasattr(expr, 'find_all'):
                    all_col_refs.extend(expr.find_all(Column))
        else:
            all_col_refs = parsed_exp.find_all(Column)
            
        for col_exp in all_col_refs:
             if not col_exp.table and col_exp.name.lower() in common_ambiguous:
                  # Found unqualified common column
                  logger.debug(f"Ambiguity Fix: Qualifying '{col_exp.name}' with '{first_table_alias}'")
                  # Simple regex replace - potential for errors!
                  # Pattern: non-dot word boundary, column name, word boundary
                  pattern = r'(?<![\w\.])\b' + re.escape(col_exp.name) + r'\b'
                  replacement = f"{first_table_alias}.{col_exp.name}"
                  fixed_sql = re.sub(pattern, replacement, fixed_sql, flags=re.IGNORECASE)
                  modified = True

        if modified:
             logger.debug(f"SQL after ambiguity fix attempt: {fixed_sql[:150]}...")
             return fixed_sql
        else:
             return sql

    except ParseError:
         logger.warning("Failed to parse SQL for ambiguity fixing. Returning original.")
         return sql
    except Exception as e:
        logger.error(f"Error during ambiguity fixing: {e}", exc_info=DEBUG_MODE)
        return sql


# ================== Error Categorization ==================

def categorize_sql_error(error_msg: str) -> Tuple[str, float]:
    """Categorize SQLite error messages and assign partial credit."""
    # Returns (error_category, partial_credit_multiplier)
    if not error_msg: return ERR_OTHER, 0.0
    error_lower = error_msg.lower()
    if DEBUG_MODE: logger.debug(f"Categorizing SQL error: {error_msg}")

    if "syntax error" in error_lower: return ERR_SYNTAX, 0.0
    if "no such table" in error_lower: return ERR_MISSING_TABLE, 0.0 # Severe
    if "no such column" in error_lower: return ERR_MISSING_COLUMN, 0.1 # Potentially typo/alias
    if "ambiguous column" in error_lower: return ERR_AMBIGUOUS_COLUMN, 0.2 # Join structure likely ok
    if "datatype mismatch" in error_lower: return ERR_TYPE_MISMATCH, 0.15
    if "constraint failed" in error_lower or "constraint violation" in error_lower:
         # e.g., NOT NULL, UNIQUE
         return ERR_CONSTRAINT, 0.1
    if "no such function" in error_lower: return ERR_FUNCTION, 0.05
    if "too many terms in compound select" in error_lower: return ERR_SYNTAX, 0.0 # Structure issue
    if "subquery returned more than 1 row" in error_lower: return ERR_EXECUTION, 0.1 # Logical error
    # Add more specific SQLite errors as needed

    return ERR_OTHER, 0.0

 
# ================== Format & Reasoning Rewards ==================

def strict_format_reward_func(prompts, completions, references=None, **kwargs) -> list[float]:
    """Strict format: <reasoning>non-empty</reasoning>\\s*<sql>non-empty</sql>"""
    strict_pattern = r"<reasoning>(.+?)</reasoning>\s*<sql>(.+?)</sql>"
    base_reward = REWARD_WEIGHTS.get("format", 1.0)
    rewards = []
    for completion in completions:
        response_text = _get_response_text(completion)
        match = re.search(strict_pattern, response_text, re.IGNORECASE | re.DOTALL)
        rewards.append(base_reward if match else 0.0)
    return rewards

def soft_format_reward_func(prompts, completions, references=None, **kwargs) -> list[float]:
    """Soft format: <reasoning>any</reasoning>\\s*<sql>any</sql>"""
    soft_pattern = r"<reasoning>(.*?)</reasoning>\s*<sql>(.*?)</sql>"
    base_reward = REWARD_WEIGHTS.get("format", 1.0)
    rewards = []
    for completion in completions:
        response_text = _get_response_text(completion)
        match = re.search(soft_pattern, response_text, re.IGNORECASE | re.DOTALL)
        rewards.append(base_reward if match else 0.0)
    return rewards

def extract_tables_columns(sql_context: str) -> tuple[set[str], set[str]]:
    """Extracts table/view and column names from CREATE statements using regex."""
    tables = set()
    columns = set()
    if not sql_context: return tables, columns

    # Patterns (simplified, consider sqlglot for full schema parsing if needed)
    create_table_pattern = r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[`\"\[]?(\w+)[`\"\]]?)\s*\((.*?)\);"
    create_view_pattern = r"CREATE\s+VIEW\s+(?:[`\"\[]?(\w+)[`\"\]]?)\s+AS"
    column_pattern = r"^\s*([`\"\[]?\w+[`\"\]]?)" # Matches the first word, potential column name

    try:
        statements = sqlparse.split(sql_context)
        for stmt in statements:
            stmt_clean = stmt.strip()
            # CREATE TABLE
            table_match = re.search(create_table_pattern, stmt_clean, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if table_match:
                table_name = table_match.group(1).lower()
                tables.add(table_name)
                cols_text = table_match.group(2)
                # Simple comma splitting for columns (fragile for complex types/constraints)
                for part in re.split(r',(?![^\(]*\))', cols_text): # Split on comma unless inside parentheses
                    col_match = re.match(column_pattern, part.strip())
                    if col_match:
                        columns.add(col_match.group(1).strip('`"[]').lower())
            # CREATE VIEW
            view_match = re.search(create_view_pattern, stmt_clean, re.IGNORECASE)
            if view_match:
                view_name = view_match.group(1).lower()
                tables.add(view_name) # Treat views like tables for reasoning check

    except Exception as e:
        logger.warning(f"Could not parse schema elements from context: {e}")

    return tables, columns


def reasoning_quality_reward(prompts, completions, references=None, **kwargs) -> list[float]:
    """Heuristic reward for reasoning quality."""
    rewards = []
    schema_cache = {} # Cache per batch

    for i, completion in enumerate(completions):
        response_text = _get_response_text(completion)
        reasoning = extract_reasoning(response_text)
        reward_components = {}

        if not reasoning:
            rewards.append(0.0)
            continue

        reasoning_lower = reasoning.lower()
        words = reasoning.split()
        lines = [line for line in reasoning.split("\n") if line.strip()]

        # Length Score
        len_score = 0.0
        if len(words) >= 50: len_score = 0.20
        elif len(words) >= 25: len_score = 0.15
        elif len(words) >= 10: len_score = 0.10
        reward_components['length'] = len_score

        # SQL Terminology Score
        sql_terms = ["table", "column", "join", "select", "where", "group by", "order by", "filter", "aggregate", "schema", "database"]
        term_count = sum(1 for term in sql_terms if term in reasoning_lower)
        term_score = min(0.20, term_count * 0.03) # Slightly higher weight per term
        reward_components['terms'] = term_score

        # Structure Score (lines)
        structure_score = 0.0
        if len(lines) >= 3: structure_score = 0.15
        elif len(lines) >= 2: structure_score = 0.10
        reward_components['structure'] = structure_score

        # Step Indicators Score
        step_score = 0.0
        # Simplified check for common step patterns
        if re.search(r'(step 1|first|start|initial|begin)', reasoning_lower) and \
           re.search(r'(step 2|next|then|second|final|last|subsequent)', reasoning_lower):
            step_score = 0.15
        reward_components['steps'] = step_score

        # Schema Element Mentions Score
        schema_mention_score = 0.0
        sql_context = None
        try:
            if references and i < len(references) and references[i] and isinstance(references[i], list) and references[i][0]:
                 sql_context = references[i][0].get('sql_context')
        except IndexError:
             logger.warning(f"IndexError accessing references at index {i}")

        if sql_context:
            if i not in schema_cache:
                 schema_cache[i] = extract_tables_columns(sql_context) if isinstance(sql_context, str) else (set(), set())
            tables, columns = schema_cache[i]

            if tables or columns:
                 # Use word boundaries for matching
                 mentioned_tables = sum(1 for t in tables if re.search(r'\b' + re.escape(t) + r'\b', reasoning_lower))
                 mentioned_cols = sum(1 for c in columns if re.search(r'\b' + re.escape(c) + r'\b', reasoning_lower))
                 total_mentions = mentioned_tables + mentioned_cols
                 schema_mention_score = min(0.30, total_mentions * 0.05) # Capped reward
        reward_components['schema'] = schema_mention_score

        # Combine scores
        total_unscaled_reward = sum(reward_components.values())
        final_reward = min(1.0, total_unscaled_reward) * REWARD_WEIGHTS.get("reasoning", 0.7)
        rewards.append(final_reward)
        if DEBUG_MODE: logger.debug(f"Reasoning Scores (Comp {i}): {reward_components} -> Total Raw: {total_unscaled_reward:.3f} -> Final: {final_reward:.3f}")

    return rewards


def complexity_reward(prompts, completions, references, **kwargs) -> list[float]:
    """Rewards based on complexity similarity using Gaussian on log ratio."""
    rewards = []
    base_weight = REWARD_WEIGHTS.get("complexity", 0.6)

    for i, completion in enumerate(completions):
        response_text = _get_response_text(completion)
        gen_sql = extract_sql(response_text)
        reward = 0.0

        gold_sql = ""
        try:
             if references and i < len(references) and references[i] and isinstance(references[i], list) and references[i][0]:
                 gold_sql = references[i][0].get('gold_sql', '')
        except IndexError:
              logger.warning(f"IndexError accessing references at index {i}")

        if not gen_sql:
            rewards.append(0.0)
            continue

        try:
            gen_complexity = calculate_sql_complexity(gen_sql)

            if not gold_sql: # Baseline reward if no gold SQL
                # Reward if complexity is within a 'reasonable' range
                reward = (0.4 if 1.5 <= gen_complexity <= 8.0 else 0.1) * base_weight
                if DEBUG_MODE: logger.debug(f"Complexity (Comp {i}): No Gold SQL. Gen={gen_complexity:.2f}. Reward={reward:.3f}")
            else:
                gold_complexity = calculate_sql_complexity(gold_sql)
                if gold_complexity < 0.1: # Trivial gold query
                    rel_score = 1.0 if gen_complexity < 0.1 else 0.0
                else:
                    ratio = max(1e-3, min(gen_complexity / gold_complexity, 1e3)) # Clamp ratio
                    log_ratio = torch.log(torch.tensor(ratio))
                    rel_score = torch.exp(-0.5 * (log_ratio**2)).item() # Gaussian similarity
                reward = rel_score * base_weight
                if DEBUG_MODE: logger.debug(f"Complexity (Comp {i}): Gen={gen_complexity:.2f}, Gold={gold_complexity:.2f}, Ratio={ratio:.2f}, Score={rel_score:.3f}, Reward={reward:.3f}")

            rewards.append(max(0.0, reward))

        except Exception as e:
            logger.warning(f"Error in complexity reward calculation: {e}")
            rewards.append(0.0)

    return rewards

def dump_database_schema(conn):
    """
    Dump the complete schema of the database for debugging purposes.
    """
    try:
        cursor = conn.cursor()
        tables = list_all_tables(conn)
        
        schema_info = {}
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Format column information
            column_info = []
            for col in columns:
                col_id, name, col_type, not_null, default_val, is_pk = col
                col_desc = f"{name} ({col_type})"
                if is_pk:
                    col_desc += " PRIMARY KEY"
                if not_null:
                    col_desc += " NOT NULL"
                if default_val is not None:
                    col_desc += f" DEFAULT {default_val}"
                column_info.append(col_desc)
            
            schema_info[table] = column_info
            
            # Check for indexes
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()
            if indexes:
                schema_info[f"{table}_indexes"] = []
                for idx in indexes:
                    idx_name = idx[1]
                    cursor.execute(f"PRAGMA index_info({idx_name})")
                    idx_columns = cursor.fetchall()
                    idx_cols = [info[2] for info in idx_columns]  # Column names in the index
                    schema_info[f"{table}_indexes"].append(f"{idx_name} ({', '.join(idx_cols)})")
        
        return schema_info
    except Exception as e:
        logger.warning(f"Error dumping database schema: {e}")
        return {"error": str(e)}

# ================== Main Execution Reward Function ==================
def execute_query_reward_func(prompts, completions, references, **kwargs) -> list[float]:
    """
    Universal reward function that handles all SQL statement types.
    Evaluates correctness by comparing the results or effects of SQL statements.
    Uses sqlglot for SQL dialect conversion with enhanced debugging.
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        # Extract generated SQL
        response_text = _get_response_text(completion)
        gen_sql = extract_sql(response_text)
        
        # Extract gold SQL and context from references
        gold_sql = ""
        sql_context = ""
        try:
            if references and i < len(references) and references[i] and isinstance(references[i], list) and references[i][0]:
                gold_sql = references[i][0].get('gold_sql', '')
                sql_context = references[i][0].get('sql_context', '')
                
                # Log the reference information in debug mode
                if DEBUG_MODE:
                    logger.debug(f"Reference {i}: Gold SQL = {gold_sql[:100]}...")
                    logger.debug(f"Reference {i}: Context SQL  = {sql_context}")
        except IndexError:
            logger.warning(f"IndexError accessing references at index {i}")
        
        # Default reward (if execution fails)
        reward = 0.0
        
        if not gen_sql or not gold_sql or not sql_context:
            logger.warning(f"Missing SQL data for completion {i}: gen_sql={bool(gen_sql)}, gold_sql={bool(gold_sql)}, sql_context={bool(sql_context)}")
            rewards.append(reward)
            continue
        
        # 1. Identify SQL statement types
        gold_type = identify_sql_statement_type(gold_sql)
        gen_type = identify_sql_statement_type(gen_sql)
        
        # Add small reward if statement types match
        if gold_type == gen_type:
            reward += 0.1 * REWARD_WEIGHTS["sql_correctness"]
        
        # Log the statement types
        if DEBUG_MODE:
            logger.debug(f"Gold SQL type: {gold_type}")
            logger.debug(f"Generated SQL type: {gen_type}")
        
        # Create a completely fresh database for each completion
        conn = None
        temp_db_file = None
        try:
            # Create temporary database in a unique file
            temp_db_file = tempfile.NamedTemporaryFile(delete=False).name
            conn = sqlite3.connect(temp_db_file, timeout=5)
            conn.isolation_level = None  # Enable autocommit mode
            cursor = conn.cursor()
            
            create_table_statements = []
            create_view_statements = []  # New list for view statements
            other_statements = []

            for stmt in sqlparse.split(sql_context):
                stmt = stmt.strip()
                if not stmt:
                    continue
                
                stmt_upper = stmt.upper()
                if stmt_upper.startswith('CREATE TABLE'):
                    create_table_statements.append(stmt)
                elif stmt_upper.startswith('CREATE VIEW'):
                    create_view_statements.append(stmt)  # Separate view statements
                else:
                    other_statements.append(stmt)

            if DEBUG_MODE:
                logger.debug(f"Found {len(create_table_statements)} CREATE TABLE statements, "
                            f"{len(create_view_statements)} CREATE VIEW statements, and "
                            f"{len(other_statements)} other statements")
                

            # Execute all CREATE TABLE statements first - CONVERT USING SQLGLOT
            tables_created = []
            for stmt in create_table_statements:
                try:
                    # Extract table name for reference
                    table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)', 
                                        stmt, re.IGNORECASE)
                    table_name = table_match.group(1).strip('`"[]') if table_match else "unknown"
                    
                    # Use sqlglot to convert to SQLite syntax
                    converted_stmt = convert_sql_to_sqlite(stmt)
                    
                    if DEBUG_MODE:
                        logger.debug(f"Creating table {table_name} with statement: {converted_stmt[:100]}...")
                    
                    # Execute the statement
                    cursor.execute(converted_stmt)
                    tables_created.append(table_name)
                    
                    # Verify the table was created
                    exists_exact, exists_case_insensitive, correct_case = check_table_exists(conn, table_name)
                    if exists_exact:
                        if DEBUG_MODE:
                            logger.debug(f"Table {table_name} created successfully")
                            # Log the schema of the created table
                            schema = get_table_schema(conn, table_name)
                            logger.debug(f"Schema for {table_name}: {schema}")
                    else:
                        logger.warning(f"Table {table_name} creation failed silently")
                    
                except sqlite3.Error as e:
                    logger.warning(f"Error in CREATE TABLE statement: {e}")
                    logger.warning(f"Table name: {table_name if 'table_name' in locals() else 'unknown'}")
                    logger.warning(f"Original statement: {stmt[:200]}...")
                    logger.warning(f"Converted statement: {converted_stmt[:200]}..." if 'converted_stmt' in locals() else "conversion failed")

            # Execute all CREATE VIEW statements SECOND - after tables are created
            views_created = []
            for stmt in create_view_statements:
                try:
                    # Extract view name for reference
                    view_match = re.search(r'CREATE\s+VIEW\s+([^\s(]+)', stmt, re.IGNORECASE)
                    view_name = view_match.group(1).strip('`"[]') if view_match else "unknown"
                    
                    # Use sqlglot to convert to SQLite syntax
                    converted_stmt = convert_sql_to_sqlite(stmt)
                    
                    if DEBUG_MODE:
                        logger.debug(f"Creating view {view_name} with statement: {converted_stmt[:100]}...")
                    
                    # Execute the statement
                    cursor.execute(converted_stmt)
                    views_created.append(view_name)
                    
                    # Verify the view was created (views appear as tables in SQLite)
                    exists_exact, exists_case_insensitive, correct_case = check_table_exists(conn, view_name)
                    if exists_exact:
                        if DEBUG_MODE:
                            logger.debug(f"View {view_name} created successfully")
                    else:
                        logger.warning(f"View {view_name} creation failed silently")
                    
                except sqlite3.Error as e:
                    logger.warning(f"Error in CREATE VIEW statement: {e}")
                    logger.warning(f"View name: {view_name if 'view_name' in locals() else 'unknown'}")
                    logger.warning(f"Original statement: {stmt[:200]}...")
                    logger.warning(f"Converted statement: {converted_stmt[:200]}..." if 'converted_stmt' in locals() else "conversion failed")


            # Then execute all other statements - CONVERT USING SQLGLOT
            for stmt in other_statements:
                try:
                    # Simple heuristic to check if the statement might be trying to insert data
                    is_insert_like = stmt.upper().startswith("INSERT") or "INSERT INTO" in stmt.upper()
                    
                    # Use sqlglot to convert to SQLite syntax
                    converted_stmt = convert_sql_to_sqlite(stmt)
                    
                    if DEBUG_MODE and is_insert_like:
                        logger.debug(f"Executing insert-like statement: {converted_stmt[:100]}...")
                    
                    cursor.execute(converted_stmt)
                except sqlite3.Error as e:
                    logger.warning(f"Error in non-CREATE statement: {e}")
                    logger.warning(f"Statement causing error: {stmt[:200]}...")
            
            # After executing all statements, dump the database schema
            if DEBUG_MODE:
                schema_info = dump_database_schema(conn)
                logger.debug(f"Database schema after setup: {schema_info}")
                
                # List all tables to verify what was created
                all_tables = list_all_tables(conn)
                logger.debug(f"All tables in database: {all_tables}")
            
            # Extract tables mentioned in the generated query
            referenced_tables = extract_tables_from_query(gen_sql)
            if DEBUG_MODE:
                logger.debug(f"Tables referenced in generated query: {referenced_tables}")
                
                # Check if the referenced tables exist with case-sensitivity handling
                for table in referenced_tables:
                    exists_exact, exists_case_insensitive, correct_case = check_table_exists(conn, table)
                    if exists_exact:
                        logger.debug(f"Table '{table}' referenced in query exists exactly as specified")
                    elif exists_case_insensitive:
                        logger.debug(f"Table '{table}' exists but with different case: '{correct_case}'")
                    else:
                        logger.debug(f"Table '{table}' does not exist in any case form")
            
            # If some referenced tables don't exist, try case-insensitive matching
            existing_tables = list_all_tables(conn)
            existing_tables_lower = [t.lower() for t in existing_tables]
            missing_tables = [table for table in referenced_tables if table.lower() not in existing_tables_lower]
            case_mismatch_tables = [table for table in referenced_tables if table not in existing_tables and table.lower() in existing_tables_lower]
            
            if case_mismatch_tables:
                logger.warning(f"Case-mismatch in table references: {case_mismatch_tables}")
                
                # Create a mapping from lowercase to actual case
                case_mapping = {t.lower(): t for t in existing_tables}
                
                # Fix the case in the SQL query
                for wrong_case in case_mismatch_tables:
                    correct_case = case_mapping[wrong_case.lower()]
                    logger.debug(f"Fixing case: '{wrong_case}' → '{correct_case}'")
                    
                    # Replace the table name with proper case, preserving word boundaries
                    gen_sql = re.sub(r'\b' + re.escape(wrong_case) + r'\b', 
                                    correct_case, 
                                    gen_sql,
                                    flags=re.IGNORECASE)
                
                logger.debug(f"Adjusted SQL with correct case: {gen_sql[:200]}...")
                
            if missing_tables:
                logger.warning(f"Tables genuinely missing (not just case mismatch): {missing_tables}")
            
            # For SELECT statements, use enhanced result comparison
            if gold_type == "SELECT" and gen_type == "SELECT":
                try:
                    # First, try to fix any ambiguous columns in the generated SQL
                    fixed_gen_sql = fix_ambiguous_columns(gen_sql)
                    
                    # If fixed SQL is different, log it for debugging
                    if fixed_gen_sql != gen_sql:
                        logger.debug(f"Fixed ambiguous columns in generated SQL")
                        logger.debug(f"Original SQL: {gen_sql[:200]}...")
                        logger.debug(f"Fixed SQL: {fixed_gen_sql[:200]}...")
                        gen_sql = fixed_gen_sql
                    
                    # Execute gold query and get results (convert first!)
                    converted_gold_sql = convert_sql_to_sqlite(gold_sql)
                    logger.debug(f"Executing gold SQL: {converted_gold_sql[:200]}...")
                    
                    cursor.execute(converted_gold_sql)
                    gold_columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    gold_result = cursor.fetchmany(1000)
                    
                    logger.debug(f"Gold SQL execution successful, returned {len(gold_result)} rows")
                    if gold_result and len(gold_result) > 0:
                        logger.debug(f"First row of gold result: {gold_result[0]}")
                    
                    # Fix case sensitivity issues in generated SQL
                    gen_sql_fixed = fix_case_sensitivity_in_sql(conn, gen_sql)
                    
                    # If case fixes were applied, use the fixed version
                    if gen_sql_fixed != gen_sql:
                        logger.debug(f"Fixed case sensitivity issues in generated SQL")
                        gen_sql = gen_sql_fixed
                    
                    # Execute generated query and get results (convert first!)
                    converted_gen_sql = convert_sql_to_sqlite(gen_sql)
                    logger.debug(f"Executing generated SQL: {converted_gen_sql[:200]}...")
                    
                    cursor.execute(converted_gen_sql)
                    gen_columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    gen_result = cursor.fetchmany(1000)
                    
                    logger.debug(f"Generated SQL execution successful, returned {len(gen_result)} rows")
                    if gen_result and len(gen_result) > 0:
                        logger.debug(f"First row of generated result: {gen_result[0]}")
                    
                    # Award base reward for successful execution
                    base_reward = 0.3 * REWARD_WEIGHTS["sql_correctness"]
                    reward = base_reward
                    
                    # ---------- IMPROVED REWARD CALCULATION START ----------
                    # Convert results for comparison
                    gold_rows = set(tuple(row) for row in gold_result)
                    gen_rows = set(tuple(row) for row in gen_result)
                    
                    # 1. Exact match case (highest reward)
                    if gold_rows == gen_rows and gold_columns == gen_columns:
                        reward = REWARD_WEIGHTS["sql_correctness"]
                        logger.debug(f"Results and columns match exactly!")
                    # 2. Content similarity metrics
                    elif gold_rows and gen_rows:
                        # Jaccard similarity of result rows
                        # Calculate Jaccard similarity with proper column handling
                        if gold_columns == gen_columns:
                            # If columns match exactly (or are in same order), use direct comparison
                            intersection = len(gold_rows.intersection(gen_rows))
                            union = len(gold_rows.union(gen_rows))
                            jaccard = intersection / union if union > 0 else 0
                        else:
                            # Columns differ - need to find common columns and project results
                            gold_cols_lower = [c.lower() for c in gold_columns]
                            gen_cols_lower = [c.lower() for c in gen_columns]
                            common_columns_indices = []
                            
                            # Find positions of common columns in both result sets
                            for i, gold_col in enumerate(gold_cols_lower):
                                if gold_col in gen_cols_lower:
                                    j = gen_cols_lower.index(gold_col)
                                    common_columns_indices.append((i, j))
                            
                            # Calculate similarity only on common columns
                            if common_columns_indices:
                                # Project results to only include common columns
                                gold_projected = [{i: row[i] for i, _ in common_columns_indices} 
                                                for row in gold_result]
                                gen_projected = [{j: row[j] for _, j in common_columns_indices}
                                                for row in gen_result]
                                
                                # Convert to hashable form for set operations
                                gold_proj_rows = {tuple(sorted(d.items())) for d in gold_projected}
                                gen_proj_rows = {tuple(sorted(d.items())) for d in gen_projected}
                                
                                # Calculate Jaccard on projected data
                                intersection = len(gold_proj_rows.intersection(gen_proj_rows))
                                union = len(gold_proj_rows.union(gen_proj_rows))
                                jaccard = intersection / union if union > 0 else 0
                                
                                # Log the number of common columns considered
                                if DEBUG_MODE:
                                    logger.debug(f"Similarity calculated on {len(common_columns_indices)} common columns")
                            else:
                                # No common columns at all
                                jaccard = 0.0
                        
                        # Row count similarity
                        row_count_ratio = min(len(gen_rows), len(gold_rows)) / max(len(gen_rows), len(gold_rows)) if max(len(gen_rows), len(gold_rows)) > 0 else 0
                        
                        # Column similarity score
                        col_similarity = 0.0
                        if gold_columns and gen_columns:
                            gold_cols_set = set(c.lower() for c in gold_columns)
                            gen_cols_set = set(c.lower() for c in gen_columns)
                            col_intersection = len(gold_cols_set.intersection(gen_cols_set))
                            col_union = len(gold_cols_set.union(gen_cols_set))
                            col_similarity = col_intersection / col_union if col_union > 0 else 0
                        
                        # Data accuracy - percentage of gold rows that appear in generated results
                        data_accuracy = len(gold_rows.intersection(gen_rows)) / len(gold_rows) if gold_rows else 0
                        
                        # Combine different metrics with appropriate weights
                        content_similarity = (
                            0.40 * jaccard +         # Similarity of overall result sets
                            0.20 * row_count_ratio + # Similar number of rows returned
                            0.25 * col_similarity +  # Right columns selected
                            0.15 * data_accuracy     # Correct data returned
                        )
                        
                        # Scale by weight parameter
                        reward = REWARD_WEIGHTS["sql_correctness"] * content_similarity
                        
                        # Detailed logging of reward calculation
                        if DEBUG_MODE:
                            logger.debug(f"Reward calculation: jaccard={jaccard:.3f}, row_ratio={row_count_ratio:.3f}, " + 
                                      f"col_sim={col_similarity:.3f}, data_acc={data_accuracy:.3f}, " +
                                      f"content_sim={content_similarity:.3f}, final_reward={reward:.3f}")
                        
                        # Minimum reward for successful execution with any matching data
                        if intersection > 0 and reward < 0.3 * REWARD_WEIGHTS["sql_correctness"]:
                            reward = 0.3 * REWARD_WEIGHTS["sql_correctness"]
                    
                    # 3. Minimum reward for successful execution, even with empty results
                    if reward <= base_reward and gen_result is not None:
                        reward = max(reward, 0.2 * REWARD_WEIGHTS["sql_correctness"])
                    # ---------- IMPROVED REWARD CALCULATION END ----------
                    
                except sqlite3.Error as e:
                    error_msg = str(e)
                    # Categorize error and get partial credit
                    error_type, partial_credit = categorize_sql_error(error_msg)
                    
                    # Apply partial credit (if any)
                    if partial_credit > 0:
                        reward = partial_credit * REWARD_WEIGHTS["sql_correctness"]
                    
                    logger.warning(f"Error executing SELECT statement ({error_type}): {error_msg}")
                    logger.warning(f"Generated SQL: {gen_sql[:200]}...")
                    if 'converted_gen_sql' in locals():  # Only log if it was successfully converted
                        logger.warning(f"Converted SQL: {converted_gen_sql[:200]}...")
            
            # For DML statements (INSERT, UPDATE, DELETE)
            elif gen_type in ["INSERT", "UPDATE", "DELETE"]:
                try:
                    # For non-SELECT statements, we need special care with JOINs
                    # Check if the generated SQL contains JOIN - not typically valid in DML
                    if "JOIN" in gen_sql.upper() and gen_type != "SELECT":
                        logger.warning(f"JOIN detected in {gen_type} statement - may cause issues")
                        
                        # Try to extract the main table if this is a common pattern
                        if gen_type == "INSERT":
                            table_match = re.search(r'INSERT\s+INTO\s+([^\s(]+)', gen_sql, re.IGNORECASE)
                            if table_match:
                                main_table = table_match.group(1)
                                logger.debug(f"Main table for INSERT: {main_table}")
                        elif gen_type == "UPDATE":
                            table_match = re.search(r'UPDATE\s+([^\s(]+)', gen_sql, re.IGNORECASE)
                            if table_match:
                                main_table = table_match.group(1)
                                logger.debug(f"Main table for UPDATE: {main_table}")
                        elif gen_type == "DELETE":
                            table_match = re.search(r'DELETE\s+FROM\s+([^\s(]+)', gen_sql, re.IGNORECASE)
                            if table_match:
                                main_table = table_match.group(1)
                                logger.debug(f"Main table for DELETE: {main_table}")
                                
                        # Check if this table exists
                        if 'main_table' in locals():
                            exists = check_table_exists(conn, main_table)
                            logger.debug(f"Main table '{main_table}' exists: {exists}")
                    
                    # Fix case sensitivity issues in the SQL first
                    gen_sql_fixed = fix_case_sensitivity_in_sql(conn, gen_sql)
                    
                    # If case fixes were applied, use the fixed version
                    if gen_sql_fixed != gen_sql:
                        logger.debug(f"Fixed case sensitivity issues in DML statement")
                        gen_sql = gen_sql_fixed
                    
                    # Convert and execute the DML statement
                    converted_gen_sql = convert_sql_to_sqlite(gen_sql)
                    logger.debug(f"Executing DML statement: {converted_gen_sql[:200]}...")
                    
                    cursor.execute(converted_gen_sql)
                    reward = 0.5 * REWARD_WEIGHTS["sql_correctness"]  # Reward for successful execution
                    
                    # We could compare against gold standard execution results for more precise rewards
                    
                except sqlite3.Error as e:
                    error_msg = str(e)
                    logger.warning(f"Error executing DML statement: {error_msg}")
                    logger.warning(f"Generated SQL: {gen_sql[:200]}...")
                    
                    # Check if it's a table not found error
                    if "no such table" in error_msg.lower():
                        # Extract the table name that wasn't found
                        table_match = re.search(r"no such table: (\w+)", error_msg, re.IGNORECASE)
                        if table_match:
                            missing_table = table_match.group(1)
                            logger.debug(f"Missing table: {missing_table}")
                            
                            # Log all available tables for comparison
                            all_tables = list_all_tables(conn)
                            logger.debug(f"Available tables: {all_tables}")
                            
                            # Create case-insensitive mapping
                            case_mapping = {t.lower(): t for t in all_tables}
                            
                            # Check for case mismatches
                            if missing_table.lower() in case_mapping:
                                correct_case = case_mapping[missing_table.lower()]
                                logger.debug(f"Case mismatch detected! '{missing_table}' vs '{correct_case}'")
                                
                                # Try fixing the SQL with the correct case - use regex for word boundaries
                                corrected_sql = re.sub(r'\b' + re.escape(missing_table) + r'\b', 
                                                     correct_case, 
                                                     gen_sql, 
                                                     flags=re.IGNORECASE)
                                                     
                                logger.debug(f"Corrected SQL: {corrected_sql[:200]}...")
                                
                                try:
                                    # Convert and execute with the fixed table name
                                    converted_corrected = convert_sql_to_sqlite(corrected_sql)
                                    cursor.execute(converted_corrected)
                                    reward = 0.4 * REWARD_WEIGHTS["sql_correctness"]  # Slightly lower reward for needing correction
                                    logger.debug(f"Execution successful after case correction!")
                                except sqlite3.Error as e2:
                                    logger.warning(f"Still failed after case correction: {e2}")
                                    # Log the new error for more debugging
                                    logger.debug(f"New error after case correction: {e2}")
                                    logger.debug(f"Converted corrected SQL: {converted_corrected[:200]}...")
            
            rewards.append(reward)
            
        except Exception as e:
            logger.warning(f"Error in execution reward calculation: {e}")
            import traceback
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            rewards.append(reward)
        finally:
            # Log the final reward
            logger.debug(f"Final reward for completion {i}: {reward}")
            
            # Cleanup resources
            if conn:
                try:
                    conn.close()
                except:
                    pass
                
            # Remove the temporary database file
            try:
                if temp_db_file and os.path.exists(temp_db_file):
                    os.unlink(temp_db_file)
            except:
                pass
    
    return rewards


# Example usage pattern (for testing)
if __name__ == "__main__":
    # Set debug mode for detailed logs when running directly
    DEBUG_MODE = True
    log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
    logger.info("Running example usage...")

    # Example Data (replace with actual data)
    prompts_example = ["Show names of dogs older than 5 years."]
    completions_example = [
         "<reasoning>Find dogs table. Filter by age > 5. Select name column.</reasoning>\n<sql>SELECT name FROM dogs WHERE age > 5;</sql>"
    ]
    # Example with slight error (case sensitivity)
    # completions_example = [
    #      "<reasoning>Find dogs table. Filter by age > 5. Select name column.</reasoning>\n<sql>SELECT NAME FROM DOGS WHERE AGE > 5;</sql>"
    # ]
    # Example with execution error (wrong column)
    # completions_example = [
    #      "<reasoning>Find dogs table. Filter by age > 5. Select name column.</reasoning>\n<sql>SELECT names FROM dogs WHERE age > 5;</sql>"
    # ]
     # Example with syntax error
    # completions_example = [
    #      "<reasoning>Find dogs table. Filter by age > 5. Select name column.</reasoning>\n<sql>SELECT name FROM dogs WHERE age > 5 ORDER BY ;</sql>"
    # ]

    references_example = [[{
        "gold_sql": "SELECT name FROM dogs WHERE age > 5 ORDER BY dog_id;", # Gold has ordering
        "sql_context": """
        CREATE TABLE dogs (dog_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO dogs (name, age) VALUES ('Buddy', 7);
        INSERT INTO dogs (name, age) VALUES ('Lucy', 4);
        INSERT INTO dogs (name, age) VALUES ('Max', 8);
        """,
        "question": prompts_example[0]
    }]]

    # --- Test Execution Reward ---
    print("\n--- Testing execute_query_reward_func (Order Ignored) ---")
    exec_rewards_order_ignored = execute_query_reward_func(
        prompts_example, completions_example, references_example, 
        source_dialect_dataset="mysql", 
        source_dialect_generated="postgresql",
        order_matters=False, 
        validate_schema=True
    )
    print(f"Execution Rewards (Order Ignored): {exec_rewards_order_ignored}")

    print("\n--- Testing execute_query_reward_func (Order Matters) ---")
    exec_rewards_order_matters = execute_query_reward_func(
        prompts_example, completions_example, references_example, 
        source_dialect_dataset="mysql", 
        source_dialect_generated="postgresql",
        order_matters=True, 
        validate_schema=True
    )
    print(f"Execution Rewards (Order Matters): {exec_rewards_order_matters}")


    # --- Test Other Rewards ---
    print("\n--- Testing Format Rewards ---")
    strict_format_rewards = strict_format_reward_func(prompts_example, completions_example)
    soft_format_rewards = soft_format_reward_func(prompts_example, completions_example)
    print(f"Strict Format Rewards: {strict_format_rewards}")
    print(f"Soft Format Rewards: {soft_format_rewards}")

    print("\n--- Testing Complexity Reward ---")
    complexity_rewards = complexity_reward(prompts_example, completions_example, references_example)
    print(f"Complexity Rewards: {complexity_rewards}")

    print("\n--- Testing Reasoning Quality Reward ---")
    reasoning_rewards = reasoning_quality_reward(prompts_example, completions_example, references_example)
    print(f"Reasoning Quality Rewards: {reasoning_rewards}")