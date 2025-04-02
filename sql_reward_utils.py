import re
import os
import tempfile
import sqlite3
import logging
import sqlparse
import torch
import copy
from sqlglot import parse, transpile, ParseError
from sqlglot.expressions import Column, Table
from typing import List, Dict, Tuple, Any, Optional, Set, Union

log_level = logging.DEBUG if os.environ.get(
    "SQL_DEBUG_MODE") == "1" else logging.CRITICAL + 1
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

REWARD_WEIGHTS = {
    "format": 1.0,
    "sql_correctness": 1.2,
    "complexity": 0.6,
    "reasoning": 0.7,
}

DEBUG_MODE = os.environ.get("SQL_DEBUG_MODE") == "1"

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


def _get_response_text(completion: Any) -> str:
    response_text = ""
    if isinstance(completion, str):
        response_text = completion
    elif isinstance(completion, list) and completion:
        if isinstance(completion[0], dict):
            response_text = completion[0].get(
                'content', completion[0].get('generated_text', ''))
        elif isinstance(completion[0], str):
            response_text = completion[0]
    elif isinstance(completion, dict):
        response_text = completion.get(
            'content', completion.get('generated_text', ''))
    else:
        try:
            response_text = str(completion)
        except Exception:
            response_text = ""
            logger.debug(
                "Could not convert completion to string: %s", type(completion))
    return response_text


def extract_sql(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<sql>(.*?)</sql>", text, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        sql = re.sub(r"^\s*--.*?\n", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"\n--.*?\s*$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        return sql.strip()
    else:
        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ",
                        "DELETE ", "CREATE ", "ALTER ", "DROP ", "WITH "]
        text_upper = text.upper()
        sql_start_index = -1
        for keyword in sql_keywords:
            idx = text_upper.find(keyword)
            if idx != -1 and (sql_start_index == -1 or idx < sql_start_index):
                sql_start_index = idx
        if sql_start_index != -1:
            potential_sql = text[sql_start_index:]
            potential_sql = potential_sql.split("</sql>", 1)[0]
            potential_sql = potential_sql.split("</reasoning>", 1)[0]
            if ";" in potential_sql:
                potential_sql = potential_sql.split(";", 1)[0] + ";"
            logger.debug("Extracted SQL using fallback method.")
            return potential_sql.strip()
        logger.debug(
            "Could not extract SQL using primary or fallback methods.")
        return ""


def extract_reasoning(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<reasoning>(.*?)</reasoning>",
                      text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def calculate_sql_complexity(sql: str) -> float:
    if not sql:
        return 0.0
    try:
        sql_upper = sql.upper()
        score = 1.0
        score += sql_upper.count(" JOIN ") * 0.6
        score += sql_upper.count(" UNION ") * 0.8 + sql_upper.count(
            " INTERSECT ") * 0.8 + sql_upper.count(" EXCEPT ") * 0.8
        score += sql_upper.count("(SELECT") * 1.0
        score += sql_upper.count(" WITH ") * 0.8
        if " WHERE " in sql_upper:
            score += 0.2
        if " GROUP BY " in sql_upper:
            score += 0.5
        if " HAVING " in sql_upper:
            score += 0.7
        if " ORDER BY " in sql_upper:
            score += 0.3
        if " LIMIT " in sql_upper:
            score += 0.1
        agg_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
        score += sum(sql_upper.count(agg) for agg in agg_functions) * 0.3
        score += sql_upper.count(" DISTINCT ") * 0.3
        score += sql_upper.count(" CASE ") * 0.4
        score += sql_upper.count(" OVER(") * 1.0
        where_match = re.search(
            r" WHERE (.*?)(?: GROUP BY | ORDER BY | LIMIT | OFFSET |$)", sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            score += where_clause.count(" AND ") * \
                0.15 + where_clause.count(" OR ") * 0.20
            score += where_clause.count(" IN ") * \
                0.2 + where_clause.count(" LIKE ") * 0.1
            score += where_clause.count(" BETWEEN ") * \
                0.2 + where_clause.count(" EXISTS ") * 0.3
        return max(0.0, score)
    except Exception as e:
        logger.warning(
            f"Error calculating complexity for '{sql[:50]}...': {e}")
        return 1.0


def identify_sql_statement_type(sql: str) -> str:
    if not sql:
        return "UNKNOWN"
    clean_sql = re.sub(r'--.*?$', '', sql, flags=re.MULTILINE).strip()
    clean_sql = re.sub(r'/\*.*?\*/', '', clean_sql, flags=re.DOTALL).strip()
    if not clean_sql:
        return "UNKNOWN"
    first_word = clean_sql.split(None, 1)[0].upper()

    if first_word == "SELECT":
        return "SELECT"
    if first_word == "INSERT":
        return "INSERT"
    if first_word == "UPDATE":
        return "UPDATE"
    if first_word == "DELETE":
        return "DELETE"
    if first_word == "CREATE":
        if re.search(r"CREATE\s+(TABLE|VIEW|INDEX)", clean_sql[:30], re.IGNORECASE):
            second_word = clean_sql.split(None, 2)[1].upper() if len(
                clean_sql.split()) > 1 else ""
            if second_word == "TABLE":
                return "CREATE_TABLE"
            if second_word == "VIEW":
                return "CREATE_VIEW"
            if second_word == "INDEX":
                return "CREATE_INDEX"
        return "CREATE_OTHER"
    if first_word == "DROP":
        return "DROP"
    if first_word == "ALTER":
        return "ALTER"
    if first_word == "WITH":
        match = re.search(r'\)\s*(SELECT|INSERT|UPDATE|DELETE)',
                          clean_sql, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()
        return "WITH_UNKNOWN"
    return "OTHER"


def list_all_tables(conn: sqlite3.Connection) -> List[str]:
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"Error listing tables/views: {e}")
        return []


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> List[Tuple]:
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        return cursor.fetchall()
    except sqlite3.Error as e:
        logger.warning(f"Error getting schema for table {table_name}: {e}")
        return []


def check_table_exists(conn: sqlite3.Connection, table_name: str) -> Tuple[bool, bool, Optional[str]]:
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name=?;", (table_name,))
        exact_match = cursor.fetchone()
        if exact_match:
            return True, True, table_name

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND lower(name)=lower(?);", (table_name,))
        insensitive_match = cursor.fetchone()
        if insensitive_match:
            return False, True, insensitive_match[0]

        return False, False, None
    except sqlite3.Error as e:
        logger.warning(
            f"Error checking existence for table/view {table_name}: {e}")
        return False, False, None


def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
    schema = get_table_schema(conn, table_name)
    return [col[1] for col in schema]


def extract_tables_from_query(sql: str) -> Set[str]:
    tables = set()
    if not sql:
        return tables
    try:
        parsed_expression = parse(sql, read="sqlite")

        if isinstance(parsed_expression, list):
            for expr in parsed_expression:
                if hasattr(expr, 'find_all'):
                    for node in expr.find_all():
                        if hasattr(node, 'name') and hasattr(node, 'is_table') and node.is_table:
                            table_name = node.name
                            if table_name:
                                tables.add(table_name)
        else:
            for node in parsed_expression.find_all():
                if hasattr(node, 'name') and hasattr(node, 'is_table') and node.is_table:
                    table_name = node.name
                    if table_name:
                        tables.add(table_name)
    except ParseError as e:
        logger.warning(
            f"sqlglot failed to parse for table extraction: {e}. Falling back to regex.")
        pattern = r'(?:FROM|JOIN)\s+([`"\[]?\w+[`"\]]?)(?:\s+(?:AS\s+)?(\w+))?'
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            table = match.group(1).strip('`"[]')
            tables.add(table)
    except Exception as e:
        logger.error(f"Unexpected error during table extraction: {e}")

    return tables


def convert_sql_to_sqlite(sql: str, source_dialect: str = "mysql") -> Optional[str]:
    if not sql or not sql.strip():
        return sql
    try:
        if DEBUG_MODE:
            logger.debug(
                f"Converting SQL from {source_dialect} to sqlite: {sql[:150]}...")
        if source_dialect == "postgresql":
            try:
                converted = transpile(sql, read="postgres", write="sqlite")
            except ParseError as pg_err:
                logger.warning(
                    f"PostgreSQL parse error: {pg_err}, trying fallback conversion...")
                modified_sql = sql
                modified_sql = re.sub(
                    r'(\w+)::\w+', r'CAST(\1 AS TEXT)', modified_sql)
                modified_sql = re.sub(
                    r'\s+RETURNING\s+.*?$', '', modified_sql, flags=re.IGNORECASE)
                try:
                    converted = transpile(
                        modified_sql, read="postgres", write="sqlite")
                except ParseError:
                    logger.warning("Falling back to generic SQL parsing...")
                    converted = transpile(
                        modified_sql, read="generic", write="sqlite")
        else:
            converted = transpile(sql, read=source_dialect, write="sqlite")
        if converted and isinstance(converted, list) and converted[0]:
            if DEBUG_MODE:
                logger.debug(f"Converted SQL: {converted[0][:150]}...")
            return converted[0]
        else:
            logger.warning(
                f"sqlglot transpile returned empty result for: {sql[:100]}...")
            return sql
    except ParseError as e:
        logger.warning(
            f"sqlglot ParseError during conversion: {e}. SQL: {sql[:150]}...")
        if 'AUTO_INCREMENT' in sql.upper():
            modified_sql = re.sub(
                r'AUTO_INCREMENT', 'AUTOINCREMENT', sql, flags=re.IGNORECASE)
            modified_sql = re.sub(r'(\w+)\s+(?:INT|INTEGER)\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
                                  r'\1 INTEGER PRIMARY KEY AUTOINCREMENT', modified_sql, flags=re.IGNORECASE)
            logger.debug("Applied manual AUTO_INCREMENT fix attempt.")
            return modified_sql
        return sql
    except Exception as e:
        logger.error(
            f"Unexpected error during sqlglot conversion: {e}", exc_info=DEBUG_MODE)
        return sql


def fix_case_sensitivity_in_sql(conn: sqlite3.Connection, sql: str) -> str:
    if not sql:
        return sql

    corrected_sql = sql
    all_db_tables = list_all_tables(conn)
    if not all_db_tables:
        return sql

    table_case_map = {t.lower(): t for t in all_db_tables}

    referenced_tables = extract_tables_from_query(corrected_sql)
    needs_table_fix = False
    for table in referenced_tables:
        table_lower = table.lower()
        if table not in table_case_map.values() and table_lower in table_case_map:
            correct_case_table = table_case_map[table_lower]
            logger.debug(
                f"Case Fix: Replacing table '{table}' with '{correct_case_table}'")
            corrected_sql = re.sub(r'\b' + re.escape(table) + r'\b',
                                   correct_case_table, corrected_sql, flags=re.IGNORECASE)
            needs_table_fix = True

    if needs_table_fix:
        logger.debug(
            f"SQL after table case correction: {corrected_sql[:150]}...")

    current_referenced_tables = extract_tables_from_query(corrected_sql)
    needs_column_fix = False

    try:
        parsed_exp = parse(corrected_sql, read="sqlite")

        if isinstance(parsed_exp, list):
            all_col_refs = []
            for expr in parsed_exp:
                if hasattr(expr, 'find_all'):
                    all_col_refs.extend(expr.find_all(Column))
        else:
            all_col_refs = parsed_exp.find_all(Column)

        for col_exp in all_col_refs:
            col_name = col_exp.name
            table_alias_or_name = col_exp.table

            target_table = None
            if table_alias_or_name:
                if table_alias_or_name.lower() in table_case_map:
                    target_table = table_case_map[table_alias_or_name.lower()]
            else:
                pass

            if target_table:
                db_columns = get_column_names(conn, target_table)
                col_case_map = {c.lower(): c for c in db_columns}
                if col_name not in db_columns and col_name.lower() in col_case_map:
                    correct_case_col = col_case_map[col_name.lower()]
                    logger.debug(
                        f"Case Fix: Replacing column '{table_alias_or_name}.{col_name}' with '{table_alias_or_name}.{correct_case_col}'")
                    pattern = r'\b' + \
                        re.escape(table_alias_or_name) + \
                        r'\s*\.\s*' + re.escape(col_name) + r'\b'
                    replacement = f"{table_alias_or_name}.{correct_case_col}"
                    corrected_sql = re.sub(
                        pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                    needs_column_fix = True

            elif not table_alias_or_name:
                possible_corrections = []
                for ref_table_name_lower in current_referenced_tables:
                    actual_ref_table = table_case_map.get(
                        ref_table_name_lower.lower())
                    if actual_ref_table:
                        db_columns = get_column_names(conn, actual_ref_table)
                        col_case_map = {c.lower(): c for c in db_columns}
                        if col_name not in db_columns and col_name.lower() in col_case_map:
                            possible_corrections.append(
                                col_case_map[col_name.lower()])

                if len(possible_corrections) == 1:
                    correct_case_col = possible_corrections[0]
                    logger.debug(
                        f"Case Fix: Replacing unqualified column '{col_name}' with '{correct_case_col}'")
                    pattern = r'(?<![\w\.])\b' + re.escape(col_name) + r'\b'
                    corrected_sql = re.sub(
                        pattern, correct_case_col, corrected_sql, flags=re.IGNORECASE)
                    needs_column_fix = True
                elif len(possible_corrections) > 1:
                    logger.warning(
                        f"Ambiguous case correction for unqualified column '{col_name}'. Found in multiple tables. Skipping.")

    except ParseError as e:
        logger.warning(
            f"sqlglot failed to parse for column case fixing: {e}. Column fix might be incomplete.")
    except Exception as e:
        logger.error(
            f"Unexpected error during column case fixing: {e}", exc_info=DEBUG_MODE)

    if needs_column_fix:
        logger.debug(
            f"SQL after column case correction: {corrected_sql[:150]}...")

    return corrected_sql


def fix_ambiguous_columns(sql: str, conn: Optional[sqlite3.Connection] = None) -> str:
    if " JOIN " not in sql.upper():
        return sql

    try:
        parsed_exp = parse(sql, read="sqlite")
        common_ambiguous = {'id', 'name', 'date', 'code'}
        first_table_alias = None

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
            first_table_alias = tables[0].alias_or_name

        if not first_table_alias:
            return sql

        fixed_sql = sql
        modified = False

        if isinstance(parsed_exp, list):
            all_col_refs = []
            for expr in parsed_exp:
                if hasattr(expr, 'find_all'):
                    all_col_refs.extend(expr.find_all(Column))
        else:
            all_col_refs = parsed_exp.find_all(Column)

        for col_exp in all_col_refs:
            if not col_exp.table and col_exp.name.lower() in common_ambiguous:
                logger.debug(
                    f"Ambiguity Fix: Qualifying '{col_exp.name}' with '{first_table_alias}'")
                pattern = r'(?<![\w\.])\b' + re.escape(col_exp.name) + r'\b'
                replacement = f"{first_table_alias}.{col_exp.name}"
                fixed_sql = re.sub(pattern, replacement,
                                   fixed_sql, flags=re.IGNORECASE)
                modified = True

        if modified:
            logger.debug(
                f"SQL after ambiguity fix attempt: {fixed_sql[:150]}...")
            return fixed_sql
        else:
            return sql

    except ParseError:
        logger.warning(
            "Failed to parse SQL for ambiguity fixing. Returning original.")
        return sql
    except Exception as e:
        logger.error(
            f"Error during ambiguity fixing: {e}", exc_info=DEBUG_MODE)
        return sql


def categorize_sql_error(error_msg: str) -> Tuple[str, float]:
    if not error_msg:
        return ERR_OTHER, 0.0
    error_lower = error_msg.lower()
    if DEBUG_MODE:
        logger.debug(f"Categorizing SQL error: {error_msg}")

    if "syntax error" in error_lower:
        return ERR_SYNTAX, 0.0
    if "no such table" in error_lower:
        return ERR_MISSING_TABLE, 0.0
    if "no such column" in error_lower:
        return ERR_MISSING_COLUMN, 0.1
    if "ambiguous column" in error_lower:
        return ERR_AMBIGUOUS_COLUMN, 0.2
    if "datatype mismatch" in error_lower:
        return ERR_TYPE_MISMATCH, 0.15
    if "constraint failed" in error_lower or "constraint violation" in error_lower:
        return ERR_CONSTRAINT, 0.1
    if "no such function" in error_lower:
        return ERR_FUNCTION, 0.05
    if "too many terms in compound select" in error_lower:
        return ERR_SYNTAX, 0.0
    if "subquery returned more than 1 row" in error_lower:
        return ERR_EXECUTION, 0.1

    return ERR_OTHER, 0.0


def strict_format_reward_func(prompts, completions, references=None, **kwargs) -> list[float]:
    strict_pattern = r"<reasoning>(.+?)</reasoning>\s*<sql>(.+?)</sql>"
    base_reward = REWARD_WEIGHTS.get("format", 1.0)
    rewards = []
    for completion in completions:
        response_text = _get_response_text(completion)
        match = re.search(strict_pattern, response_text,
                          re.IGNORECASE | re.DOTALL)
        rewards.append(base_reward if match else 0.0)
    return rewards


def soft_format_reward_func(prompts, completions, references=None, **kwargs) -> list[float]:
    soft_pattern = r"<reasoning>(.*?)</reasoning>\s*<sql>(.*?)</sql>"
    base_reward = REWARD_WEIGHTS.get("format", 1.0)
    rewards = []
    for completion in completions:
        response_text = _get_response_text(completion)
        match = re.search(soft_pattern, response_text,
                          re.IGNORECASE | re.DOTALL)
        rewards.append(base_reward if match else 0.0)
    return rewards


def extract_tables_columns(sql_context: str) -> tuple[set[str], set[str]]:
    tables = set()
    columns = set()
    if not sql_context:
        return tables, columns

    create_table_pattern = r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[`\"\[]?(\w+)[`\"\]]?)\s*\((.*?)\);"
    create_view_pattern = r"CREATE\s+VIEW\s+(?:[`\"\[]?(\w+)[`\"\]]?)\s+AS"
    column_pattern = r"^\s*([`\"\[]?\w+[`\"\]]?)"

    try:
        statements = sqlparse.split(sql_context)
        for stmt in statements:
            stmt_clean = stmt.strip()
            table_match = re.search(
                create_table_pattern, stmt_clean, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if table_match:
                table_name = table_match.group(1).lower()
                tables.add(table_name)
                cols_text = table_match.group(2)
                for part in re.split(r',(?![^\(]*\))', cols_text):
                    col_match = re.match(column_pattern, part.strip())
                    if col_match:
                        columns.add(col_match.group(1).strip('`"[]').lower())
            view_match = re.search(create_view_pattern,
                                   stmt_clean, re.IGNORECASE)
            if view_match:
                view_name = view_match.group(1).lower()
                tables.add(view_name)

    except Exception as e:
        logger.warning(f"Could not parse schema elements from context: {e}")

    return tables, columns


def reasoning_quality_reward(prompts, completions, references=None, **kwargs) -> list[float]:
    rewards = []
    schema_cache = {}

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

        len_score = 0.0
        if len(words) >= 50:
            len_score = 0.20
        elif len(words) >= 25:
            len_score = 0.15
        elif len(words) >= 10:
            len_score = 0.10
        reward_components['length'] = len_score

        sql_terms = ["table", "column", "join", "select", "where",
                     "group by", "order by", "filter", "aggregate", "schema", "database"]
        term_count = sum(1 for term in sql_terms if term in reasoning_lower)
        term_score = min(0.20, term_count * 0.03)
        reward_components['terms'] = term_score

        structure_score = 0.0
        if len(lines) >= 3:
            structure_score = 0.15
        elif len(lines) >= 2:
            structure_score = 0.10
        reward_components['structure'] = structure_score

        step_score = 0.0
        if re.search(r'(step 1|first|start|initial|begin)', reasoning_lower) and \
           re.search(r'(step 2|next|then|second|final|last|subsequent)', reasoning_lower):
            step_score = 0.15
        reward_components['steps'] = step_score

        schema_mention_score = 0.0
        sql_context = None
        try:
            if references and i < len(references) and references[i] and isinstance(references[i], list) and references[i][0]:
                sql_context = references[i][0].get('sql_context')
        except IndexError:
            logger.warning(f"IndexError accessing references at index {i}")

        if sql_context:
            if i not in schema_cache:
                schema_cache[i] = extract_tables_columns(
                    sql_context) if isinstance(sql_context, str) else (set(), set())
            tables, columns = schema_cache[i]

            if tables or columns:
                mentioned_tables = sum(1 for t in tables if re.search(
                    r'\b' + re.escape(t) + r'\b', reasoning_lower))
                mentioned_cols = sum(1 for c in columns if re.search(
                    r'\b' + re.escape(c) + r'\b', reasoning_lower))
                total_mentions = mentioned_tables + mentioned_cols
                schema_mention_score = min(0.30, total_mentions * 0.05)
        reward_components['schema'] = schema_mention_score

        total_unscaled_reward = sum(reward_components.values())
        final_reward = min(1.0, total_unscaled_reward) * \
            REWARD_WEIGHTS.get("reasoning", 0.7)
        rewards.append(final_reward)
        if DEBUG_MODE:
            logger.debug(
                f"Reasoning Scores (Comp {i}): {reward_components} -> Total Raw: {total_unscaled_reward:.3f} -> Final: {final_reward:.3f}")

    return rewards


def complexity_reward(prompts, completions, references, **kwargs) -> list[float]:
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

            if not gold_sql:
                reward = (0.4 if 1.5 <= gen_complexity <=
                          8.0 else 0.1) * base_weight
                if DEBUG_MODE:
                    logger.debug(
                        f"Complexity (Comp {i}): No Gold SQL. Gen={gen_complexity:.2f}. Reward={reward:.3f}")
            else:
                gold_complexity = calculate_sql_complexity(gold_sql)
                if gold_complexity < 0.1:
                    rel_score = 1.0 if gen_complexity < 0.1 else 0.0
                else:
                    ratio = max(
                        1e-3, min(gen_complexity / gold_complexity, 1e3))
                    log_ratio = torch.log(torch.tensor(ratio))
                    rel_score = torch.exp(-0.5 * (log_ratio**2)).item()
                reward = rel_score * base_weight
                if DEBUG_MODE:
                    logger.debug(
                        f"Complexity (Comp {i}): Gen={gen_complexity:.2f}, Gold={gold_complexity:.2f}, Ratio={ratio:.2f}, Score={rel_score:.3f}, Reward={reward:.3f}")

            rewards.append(max(0.0, reward))

        except Exception as e:
            logger.warning(f"Error in complexity reward calculation: {e}")
            rewards.append(0.0)

    return rewards


def dump_database_schema(conn):
    try:
        cursor = conn.cursor()
        tables = list_all_tables(conn)

        schema_info = {}

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

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

            cursor.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()
            if indexes:
                schema_info[f"{table}_indexes"] = []
                for idx in indexes:
                    idx_name = idx[1]
                    cursor.execute(f"PRAGMA index_info({idx_name})")
                    idx_columns = cursor.fetchall()
                    idx_cols = [info[2] for info in idx_columns]
                    schema_info[f"{table}_indexes"].append(
                        f"{idx_name} ({', '.join(idx_cols)})")

        return schema_info
    except Exception as e:
        logger.warning(f"Error dumping database schema: {e}")
        return {"error": str(e)}


def execute_query_reward_func(prompts, completions, references, **kwargs) -> list[float]:
    rewards = []

    for i, completion in enumerate(completions):
        response_text = _get_response_text(completion)
        gen_sql = extract_sql(response_text)

        gold_sql = ""
        sql_context = ""
        try:
            if references and i < len(references) and references[i] and isinstance(references[i], list) and references[i][0]:
                gold_sql = references[i][0].get('gold_sql', '')
                sql_context = references[i][0].get('sql_context', '')

                if DEBUG_MODE:
                    logger.debug(
                        f"Reference {i}: Gold SQL = {gold_sql[:100]}...")
                    logger.debug(
                        f"Reference {i}: Context SQL  = {sql_context}")
        except IndexError:
            logger.warning(f"IndexError accessing references at index {i}")

        reward = 0.0

        if not gen_sql or not gold_sql or not sql_context:
            logger.warning(
                f"Missing SQL data for completion {i}: gen_sql={bool(gen_sql)}, gold_sql={bool(gold_sql)}, sql_context={bool(sql_context)}")
            rewards.append(reward)
            continue

        gold_type = identify_sql_statement_type(gold_sql)
        gen_type = identify_sql_statement_type(gen_sql)

        if gold_type == gen_type:
            reward += 0.1 * REWARD_WEIGHTS["sql_correctness"]

        if DEBUG_MODE:
            logger.debug(f"Gold SQL type: {gold_type}")
            logger.debug(f"Generated SQL type: {gen_type}")

        conn = None
        temp_db_file = None
        try:
            temp_db_file = tempfile.NamedTemporaryFile(delete=False).name
            conn = sqlite3.connect(temp_db_file, timeout=5)
            conn.isolation_level = None
            cursor = conn.cursor()

            create_table_statements = []
            create_view_statements = []
            other_statements = []

            for stmt in sqlparse.split(sql_context):
                stmt = stmt.strip()
                if not stmt:
                    continue

                stmt_upper = stmt.upper()
                if stmt_upper.startswith('CREATE TABLE'):
                    create_table_statements.append(stmt)
                elif stmt_upper.startswith('CREATE VIEW'):
                    create_view_statements.append(stmt)
                else:
                    other_statements.append(stmt)

            if DEBUG_MODE:
                logger.debug(f"Found {len(create_table_statements)} CREATE TABLE statements, "
                             f"{len(create_view_statements)} CREATE VIEW statements, and "
                             f"{len(other_statements)} other statements")

            tables_created = []
            for stmt in create_table_statements:
                try:
                    table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)',
                                            stmt, re.IGNORECASE)
                    table_name = table_match.group(1).strip(
                        '`"[]') if table_match else "unknown"

                    converted_stmt = convert_sql_to_sqlite(stmt)

                    if DEBUG_MODE:
                        logger.debug(
                            f"Creating table {table_name} with statement: {converted_stmt[:100]}...")

                    cursor.execute(converted_stmt)
                    tables_created.append(table_name)

                    exists_exact, exists_case_insensitive, correct_case = check_table_exists(
                        conn, table_name)
                    if exists_exact:
                        if DEBUG_MODE:
                            logger.debug(
                                f"Table {table_name} created successfully")
                            schema = get_table_schema(conn, table_name)
                            logger.debug(f"Schema for {table_name}: {schema}")
                    else:
                        logger.warning(
                            f"Table {table_name} creation failed silently")

                except sqlite3.Error as e:
                    logger.warning(f"Error in CREATE TABLE statement: {e}")
                    logger.warning(
                        f"Table name: {table_name if 'table_name' in locals() else 'unknown'}")
                    logger.warning(f"Original statement: {stmt[:200]}...")
                    logger.warning(f"Converted statement: {converted_stmt[:200]}..." if 'converted_stmt' in locals(
                    ) else "conversion failed")

            views_created = []
            for stmt in create_view_statements:
                try:
                    view_match = re.search(
                        r'CREATE\s+VIEW\s+([^\s(]+)', stmt, re.IGNORECASE)
                    view_name = view_match.group(1).strip(
                        '`"[]') if view_match else "unknown"

                    converted_stmt = convert_sql_to_sqlite(stmt)

                    if DEBUG_MODE:
                        logger.debug(
                            f"Creating view {view_name} with statement: {converted_stmt[:100]}...")

                    cursor.execute(converted_stmt)
                    views_created.append(view_name)

                    exists_exact, exists_case_insensitive, correct_case = check_table_exists(
                        conn, view_name)
                    if exists_exact:
                        if DEBUG_MODE:
                            logger.debug(
                                f"View {view_name} created successfully")
                    else:
                        logger.warning(
                            f"View {view_name} creation failed silently")

                except sqlite3.Error as e:
                    logger.warning(f"Error in CREATE VIEW statement: {e}")
                    logger.warning(
                        f"View name: {view_name if 'view_name' in locals() else 'unknown'}")
                    logger.warning(f"Original statement: {stmt[:200]}...")
                    logger.warning(f"Converted statement: {converted_stmt[:200]}..." if 'converted_stmt' in locals(
                    ) else "conversion failed")

            for stmt in other_statements:
                try:
                    is_insert_like = stmt.upper().startswith(
                        "INSERT") or "INSERT INTO" in stmt.upper()

                    converted_stmt = convert_sql_to_sqlite(stmt)

                    if DEBUG_MODE and is_insert_like:
                        logger.debug(
                            f"Executing insert-like statement: {converted_stmt[:100]}...")

                    cursor.execute(converted_stmt)
                except sqlite3.Error as e:
                    logger.warning(f"Error in non-CREATE statement: {e}")
                    logger.warning(f"Statement causing error: {stmt[:200]}...")

            if DEBUG_MODE:
                schema_info = dump_database_schema(conn)
                logger.debug(f"Database schema after setup: {schema_info}")

                all_tables = list_all_tables(conn)
                logger.debug(f"All tables in database: {all_tables}")

            referenced_tables = extract_tables_from_query(gen_sql)
            if DEBUG_MODE:
                logger.debug(
                    f"Tables referenced in generated query: {referenced_tables}")

                for table in referenced_tables:
                    exists_exact, exists_case_insensitive, correct_case = check_table_exists(
                        conn, table)
                    if exists_exact:
                        logger.debug(
                            f"Table '{table}' referenced in query exists exactly as specified")
                    elif exists_case_insensitive:
                        logger.debug(
                            f"Table '{table}' exists but with different case: '{correct_case}'")
                    else:
                        logger.debug(
                            f"Table '{table}' does not exist in any case form")

            existing_tables = list_all_tables(conn)
            existing_tables_lower = [t.lower() for t in existing_tables]
            missing_tables = [table for table in referenced_tables if table.lower(
            ) not in existing_tables_lower]
            case_mismatch_tables = [
                table for table in referenced_tables if table not in existing_tables and table.lower() in existing_tables_lower]

            if case_mismatch_tables:
                logger.warning(
                    f"Case-mismatch in table references: {case_mismatch_tables}")

                case_mapping = {t.lower(): t for t in existing_tables}

                for wrong_case in case_mismatch_tables:
                    correct_case = case_mapping[wrong_case.lower()]
                    logger.debug(
                        f"Fixing case: '{wrong_case}' → '{correct_case}'")

                    gen_sql = re.sub(r'\b' + re.escape(wrong_case) + r'\b',
                                     correct_case,
                                     gen_sql,
                                     flags=re.IGNORECASE)

                logger.debug(
                    f"Adjusted SQL with correct case: {gen_sql[:200]}...")

            if missing_tables:
                logger.warning(
                    f"Tables genuinely missing (not just case mismatch): {missing_tables}")

            if gold_type == "SELECT" and gen_type == "SELECT":
                try:
                    fixed_gen_sql = fix_ambiguous_columns(gen_sql)

                    if fixed_gen_sql != gen_sql:
                        logger.debug(
                            f"Fixed ambiguous columns in generated SQL")
                        logger.debug(f"Original SQL: {gen_sql[:200]}...")
                        logger.debug(f"Fixed SQL: {fixed_gen_sql[:200]}...")
                        gen_sql = fixed_gen_sql

                    converted_gold_sql = convert_sql_to_sqlite(gold_sql)
                    logger.debug(
                        f"Executing gold SQL: {converted_gold_sql[:200]}...")

                    cursor.execute(converted_gold_sql)
                    gold_columns = [
                        desc[0] for desc in cursor.description] if cursor.description else []
                    gold_result = cursor.fetchmany(1000)

                    logger.debug(
                        f"Gold SQL execution successful, returned {len(gold_result)} rows")
                    if gold_result and len(gold_result) > 0:
                        logger.debug(
                            f"First row of gold result: {gold_result[0]}")

                    gen_sql_fixed = fix_case_sensitivity_in_sql(conn, gen_sql)

                    if gen_sql_fixed != gen_sql:
                        logger.debug(
                            f"Fixed case sensitivity issues in generated SQL")
                        gen_sql = gen_sql_fixed

                    converted_gen_sql = convert_sql_to_sqlite(gen_sql)
                    logger.debug(
                        f"Executing generated SQL: {converted_gen_sql[:200]}...")

                    cursor.execute(converted_gen_sql)
                    gen_columns = [
                        desc[0] for desc in cursor.description] if cursor.description else []
                    gen_result = cursor.fetchmany(1000)

                    logger.debug(
                        f"Generated SQL execution successful, returned {len(gen_result)} rows")
                    if gen_result and len(gen_result) > 0:
                        logger.debug(
                            f"First row of generated result: {gen_result[0]}")

                    base_reward = 0.3 * REWARD_WEIGHTS["sql_correctness"]
                    reward = base_reward

                    gold_rows = set(tuple(row) for row in gold_result)
                    gen_rows = set(tuple(row) for row in gen_result)

                    if gold_rows == gen_rows and gold_columns == gen_columns:
                        reward = REWARD_WEIGHTS["sql_correctness"]
                        logger.debug(f"Results and columns match exactly!")
                    elif gold_rows and gen_rows:
                        if gold_columns == gen_columns:
                            intersection = len(
                                gold_rows.intersection(gen_rows))
                            union = len(gold_rows.union(gen_rows))
                            jaccard = intersection / union if union > 0 else 0
                        else:
                            gold_cols_lower = [c.lower() for c in gold_columns]
                            gen_cols_lower = [c.lower() for c in gen_columns]
                            common_columns_indices = []

                            for i, gold_col in enumerate(gold_cols_lower):
                                if gold_col in gen_cols_lower:
                                    j = gen_cols_lower.index(gold_col)
                                    common_columns_indices.append((i, j))

                            if common_columns_indices:
                                gold_projected = [{i: row[i] for i, _ in common_columns_indices}
                                                  for row in gold_result]
                                gen_projected = [{j: row[j] for _, j in common_columns_indices}
                                                 for row in gen_result]

                                gold_proj_rows = {
                                    tuple(sorted(d.items())) for d in gold_projected}
                                gen_proj_rows = {
                                    tuple(sorted(d.items())) for d in gen_projected}

                                intersection = len(
                                    gold_proj_rows.intersection(gen_proj_rows))
                                union = len(
                                    gold_proj_rows.union(gen_proj_rows))
                                jaccard = intersection / union if union > 0 else 0

                                if DEBUG_MODE:
                                    logger.debug(
                                        f"Similarity calculated on {len(common_columns_indices)} common columns")
                            else:
                                jaccard = 0.0

                        row_count_ratio = min(len(gen_rows), len(gold_rows)) / max(
                            len(gen_rows), len(gold_rows)) if max(len(gen_rows), len(gold_rows)) > 0 else 0

                        col_similarity = 0.0
                        if gold_columns and gen_columns:
                            gold_cols_set = set(c.lower()
                                                for c in gold_columns)
                            gen_cols_set = set(c.lower() for c in gen_columns)
                            col_intersection = len(
                                gold_cols_set.intersection(gen_cols_set))
                            col_union = len(gold_cols_set.union(gen_cols_set))
                            col_similarity = col_intersection / col_union if col_union > 0 else 0

                        data_accuracy = len(gold_rows.intersection(
                            gen_rows)) / len(gold_rows) if gold_rows else 0

                        content_similarity = (
                            0.40 * jaccard +
                            0.20 * row_count_ratio +
                            0.25 * col_similarity +
                            0.15 * data_accuracy
                        )

                        reward = REWARD_WEIGHTS["sql_correctness"] * \
                            content_similarity

                        if DEBUG_MODE:
                            logger.debug(f"Reward calculation: jaccard={jaccard:.3f}, row_ratio={row_count_ratio:.3f}, " +
                                         f"col_sim={col_similarity:.3f}, data_acc={data_accuracy:.3f}, " +
                                         f"content_sim={content_similarity:.3f}, final_reward={reward:.3f}")

                        if intersection > 0 and reward < 0.3 * REWARD_WEIGHTS["sql_correctness"]:
                            reward = 0.3 * REWARD_WEIGHTS["sql_correctness"]

                    if reward <= base_reward and gen_result is not None:
                        reward = max(
                            reward, 0.2 * REWARD_WEIGHTS["sql_correctness"])

                except sqlite3.Error as e:
                    error_msg = str(e)
                    error_type, partial_credit = categorize_sql_error(
                        error_msg)

                    if partial_credit > 0:
                        reward = partial_credit * \
                            REWARD_WEIGHTS["sql_correctness"]

                    logger.warning(
                        f"Error executing SELECT statement ({error_type}): {error_msg}")
                    logger.warning(f"Generated SQL: {gen_sql[:200]}...")
                    if 'converted_gen_sql' in locals():
                        logger.warning(
                            f"Converted SQL: {converted_gen_sql[:200]}...")

            elif gen_type in ["INSERT", "UPDATE", "DELETE"]:
                try:
                    if "JOIN" in gen_sql.upper() and gen_type != "SELECT":
                        logger.warning(
                            f"JOIN detected in {gen_type} statement - may cause issues")

                        if gen_type == "INSERT":
                            table_match = re.search(
                                r'INSERT\s+INTO\s+([^\s(]+)', gen_sql, re.IGNORECASE)
                            if table_match:
                                main_table = table_match.group(1)
                                logger.debug(
                                    f"Main table for INSERT: {main_table}")
                        elif gen_type == "UPDATE":
                            table_match = re.search(
                                r'UPDATE\s+([^\s(]+)', gen_sql, re.IGNORECASE)
                            if table_match:
                                main_table = table_match.group(1)
                                logger.debug(
                                    f"Main table for UPDATE: {main_table}")
                        elif gen_type == "DELETE":
                            table_match = re.search(
                                r'DELETE\s+FROM\s+([^\s(]+)', gen_sql, re.IGNORECASE)
                            if table_match:
                                main_table = table_match.group(1)
                                logger.debug(
                                    f"Main table for DELETE: {main_table}")

                        if 'main_table' in locals():
                            exists = check_table_exists(conn, main_table)
                            logger.debug(
                                f"Main table '{main_table}' exists: {exists}")

                    gen_sql_fixed = fix_case_sensitivity_in_sql(conn, gen_sql)

                    if gen_sql_fixed != gen_sql:
                        logger.debug(
                            f"Fixed case sensitivity issues in DML statement")
                        gen_sql = gen_sql_fixed

                    converted_gen_sql = convert_sql_to_sqlite(gen_sql)
                    logger.debug(
                        f"Executing DML statement: {converted_gen_sql[:200]}...")

                    cursor.execute(converted_gen_sql)
                    reward = 0.5 * REWARD_WEIGHTS["sql_correctness"]

                except sqlite3.Error as e:
                    error_msg = str(e)
                    logger.warning(
                        f"Error executing DML statement: {error_msg}")
                    logger.warning(f"Generated SQL: {gen_sql[:200]}...")

                    if "no such table" in error_msg.lower():
                        table_match = re.search(
                            r"no such table: (\w+)", error_msg, re.IGNORECASE)
                        if table_match:
                            missing_table = table_match.group(1)
                            logger.debug(f"Missing table: {missing_table}")

                            all_tables = list_all_tables(conn)
                            logger.debug(f"Available tables: {all_tables}")

                            case_mapping = {t.lower(): t for t in all_tables}

                            if missing_table.lower() in case_mapping:
                                correct_case = case_mapping[missing_table.lower(
                                )]
                                logger.debug(
                                    f"Case mismatch detected! '{missing_table}' vs '{correct_case}'")

                                corrected_sql = re.sub(r'\b' + re.escape(missing_table) + r'\b',
                                                       correct_case,
                                                       gen_sql,
                                                       flags=re.IGNORECASE)

                                logger.debug(
                                    f"Corrected SQL: {corrected_sql[:200]}...")

                                try:
                                    converted_corrected = convert_sql_to_sqlite(
                                        corrected_sql)
                                    cursor.execute(converted_corrected)
                                    reward = 0.4 * \
                                        REWARD_WEIGHTS["sql_correctness"]
                                    logger.debug(
                                        f"Execution successful after case correction!")
                                except sqlite3.Error as e2:
                                    logger.warning(
                                        f"Still failed after case correction: {e2}")
                                    logger.debug(
                                        f"New error after case correction: {e2}")
                                    logger.debug(
                                        f"Converted corrected SQL: {converted_corrected[:200]}...")

            rewards.append(reward)

        except Exception as e:
            logger.warning(f"Error in execution reward calculation: {e}")
            import traceback
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            rewards.append(reward)
        finally:
            logger.debug(f"Final reward for completion {i}: {reward}")

            if conn:
                try:
                    conn.close()
                except:
                    pass

            try:
                if temp_db_file and os.path.exists(temp_db_file):
                    os.unlink(temp_db_file)
            except:
                pass

    return rewards


if __name__ == "__main__":
    DEBUG_MODE = True
    log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
    logger.info("Running example usage...")

    prompts_example = ["Show names of dogs older than 5 years."]
    completions_example = [
        "<reasoning>Find dogs table. Filter by age > 5. Select name column.</reasoning>\n<sql>SELECT name FROM dogs WHERE age > 5;</sql>"
    ]
    references_example = [[{
        "gold_sql": "SELECT name FROM dogs WHERE age > 5 ORDER BY dog_id;",
        "sql_context": """
        CREATE TABLE dogs (dog_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO dogs (name, age) VALUES ('Buddy', 7);
        INSERT INTO dogs (name, age) VALUES ('Lucy', 4);
        INSERT INTO dogs (name, age) VALUES ('Max', 8);
        """,
        "question": prompts_example[0]
    }]]

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

    print("\n--- Testing Format Rewards ---")
    strict_format_rewards = strict_format_reward_func(
        prompts_example, completions_example)
    soft_format_rewards = soft_format_reward_func(
        prompts_example, completions_example)
    print(f"Strict Format Rewards: {strict_format_rewards}")
    print(f"Soft Format Rewards: {soft_format_rewards}")

    print("\n--- Testing Complexity Reward ---")
    complexity_rewards = complexity_reward(
        prompts_example, completions_example, references_example)
    print(f"Complexity Rewards: {complexity_rewards}")

    print("\n--- Testing Reasoning Quality Reward ---")
    reasoning_rewards = reasoning_quality_reward(
        prompts_example, completions_example, references_example)
    print(f"Reasoning Quality Rewards: {reasoning_rewards}")
