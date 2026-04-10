"""
DataCleanerAgent — specializes in CSV data cleaning tasks.

Strategies it can employ (emergent via soil trails):
  - pandas_dropna: drop null rows, deduplicate
  - regex_cleaning: apply regex substitutions
  - schema_validation: validate against a JSON schema
  - fill_missing: fill NaN with median/mode
"""

import io
import json
import logging
import time
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from base.agent import EmergentAgent, TaskOutcome

logger = logging.getLogger(__name__)


class DataCleanerAgent(EmergentAgent):
    def __init__(self, agent_id: Optional[str] = None):
        genome = {
            "specialization": "data_cleaning",
            "default_approach": {
                "method": "pandas_dropna",
                "drop_duplicates": True,
                "fill_strategy": None,
            },
            "fallback_approach": {
                "method": "regex_cleaning",
                "patterns": ["\\s+", "\\x00"],
            },
        }
        super().__init__(agent_id=agent_id, genome=genome)

    @property
    def task_domain(self) -> str:
        return "data_cleaning"

    def clean(self, file_path: str, schema: Optional[dict] = None) -> dict:
        """
        Clean a CSV file using the strategy from the Soil.
        Returns a result dict with metadata about what was done.
        """
        task_desc = f"clean CSV file: {Path(file_path).name}"
        if schema:
            task_desc += f" with schema: {list(schema.keys())}"

        strategy = self.before_task(task_desc)
        start = time.time()

        try:
            result = self._execute_strategy(file_path, strategy, schema)
            elapsed_ms = (time.time() - start) * 1000

            self.after_task(
                task_desc,
                outcome=TaskOutcome(success=True, data=result),
                resources={"cpu_ms": elapsed_ms, "memory_mb": 50.0},
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning(f"[{self.id}] Clean failed: {e}")

            self.after_task(
                task_desc,
                outcome=TaskOutcome(success=False, error=str(e)),
                resources={"cpu_ms": elapsed_ms},
            )

            # Retry once with mutated approach
            if self.experience_buffer:
                self.current_strategy = self.mutate_approach(task_desc)
                try:
                    result = self._execute_strategy(file_path, self.current_strategy, schema)
                    self.after_task(
                        task_desc,
                        outcome=TaskOutcome(success=True, data=result),
                        resources={"cpu_ms": (time.time() - start) * 1000},
                    )
                    return result
                except Exception as e2:
                    logger.error(f"[{self.id}] Retry also failed: {e2}")

            return {"success": False, "error": str(e)}

    def _execute_strategy(self, file_path: str, strategy: dict, schema: Optional[dict]) -> dict:
        method = strategy.get("method", "pandas_dropna")

        if method == "pandas_dropna":
            return self._pandas_dropna(file_path, strategy)

        elif method == "regex_cleaning":
            return self._regex_cleaning(file_path, strategy)

        elif method == "fill_missing":
            return self._fill_missing(file_path, strategy)

        elif method == "schema_validation":
            return self._schema_validation(file_path, schema or {})

        else:
            # Unknown method — fall back to pandas_dropna
            logger.warning(f"[{self.id}] Unknown method '{method}', falling back to pandas_dropna")
            return self._pandas_dropna(file_path, strategy)

    def _pandas_dropna(self, file_path: str, strategy: dict) -> dict:
        df = pd.read_csv(file_path)
        original_rows = len(df)
        original_cols = len(df.columns)

        df = df.dropna()

        if strategy.get("drop_duplicates", True):
            df = df.drop_duplicates()

        output_path = file_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=False)

        return {
            "success": True,
            "method": "pandas_dropna",
            "original_rows": original_rows,
            "cleaned_rows": len(df),
            "rows_removed": original_rows - len(df),
            "columns": original_cols,
            "output": output_path,
        }

    def _regex_cleaning(self, file_path: str, strategy: dict) -> dict:
        patterns = strategy.get("patterns", [r"\s+", r"\x00"])

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        original_len = len(content)
        for pattern in patterns:
            content = re.sub(pattern, " ", content)

        output_path = file_path.replace(".csv", "_regex_cleaned.csv")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "success": True,
            "method": "regex_cleaning",
            "original_chars": original_len,
            "cleaned_chars": len(content),
            "output": output_path,
        }

    def _fill_missing(self, file_path: str, strategy: dict) -> dict:
        df = pd.read_csv(file_path)
        fill_strategy = strategy.get("fill_strategy", "median")

        for col in df.select_dtypes(include=["number"]).columns:
            if fill_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_strategy == "zero":
                df[col] = df[col].fillna(0)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("UNKNOWN")

        output_path = file_path.replace(".csv", "_filled.csv")
        df.to_csv(output_path, index=False)

        return {
            "success": True,
            "method": "fill_missing",
            "fill_strategy": fill_strategy,
            "output": output_path,
        }

    def _schema_validation(self, file_path: str, schema: dict) -> dict:
        df = pd.read_csv(file_path)
        violations = []

        for col, expected_type in schema.items():
            if col not in df.columns:
                violations.append(f"Missing column: {col}")
                continue

            if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                violations.append(f"Column '{col}' expected numeric, got {df[col].dtype}")

        return {
            "success": len(violations) == 0,
            "method": "schema_validation",
            "violations": violations,
            "rows": len(df),
        }
