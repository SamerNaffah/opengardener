"""
CodeGeneratorAgent — specializes in LLM-powered code generation tasks.

Strategies (emergent via soil trails):
  - llm_direct: ask LLM to write the code directly
  - llm_with_context: include soil trail context in the prompt
  - template_based: fill in a code template from genome
  - iterative_refinement: generate, then ask LLM to improve
"""

import logging
import time
from typing import Optional

from base.agent import EmergentAgent, TaskOutcome

logger = logging.getLogger(__name__)


class CodeGeneratorAgent(EmergentAgent):
    def __init__(self, agent_id: Optional[str] = None):
        genome = {
            "specialization": "code_generation",
            "default_approach": {
                "method": "llm_direct",
                "language": "python",
                "include_tests": False,
            },
            "fallback_approach": {
                "method": "template_based",
                "language": "python",
            },
            "templates": {
                "function": "def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    pass\n",
                "class": "class {name}:\n    def __init__(self):\n        pass\n",
            },
        }
        super().__init__(agent_id=agent_id, genome=genome)

    @property
    def task_domain(self) -> str:
        return "code_generation"

    def generate(self, task_description: str, language: str = "python") -> dict:
        """
        Generate code for a given task description.
        Returns a result dict with the generated code.
        """
        strategy = self.before_task(task_description)
        start = time.time()

        # Override language from strategy if specified
        lang = strategy.get("language", language)

        try:
            result = self._execute_strategy(task_description, strategy, lang)
            elapsed_ms = (time.time() - start) * 1000

            self.after_task(
                task_description,
                outcome=TaskOutcome(success=bool(result.get("code")), data=result),
                resources={"cpu_ms": elapsed_ms},
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning(f"[{self.id}] Code generation failed: {e}")

            self.after_task(
                task_description,
                outcome=TaskOutcome(success=False, error=str(e)),
                resources={"cpu_ms": elapsed_ms},
            )
            return {"success": False, "error": str(e), "code": ""}

    def _execute_strategy(self, task_description: str, strategy: dict, language: str) -> dict:
        method = strategy.get("method", "llm_direct")

        if method == "llm_direct":
            return self._llm_direct(task_description, language, strategy)

        elif method == "llm_with_context":
            return self._llm_with_context(task_description, language, strategy)

        elif method == "iterative_refinement":
            return self._iterative_refinement(task_description, language, strategy)

        elif method == "template_based":
            return self._template_based(task_description, language)

        else:
            return self._llm_direct(task_description, language, strategy)

    def _llm_direct(self, task: str, language: str, strategy: dict) -> dict:
        include_tests = strategy.get("include_tests", False)

        system = (
            f"You are an expert {language} programmer. "
            "Write clean, working code. Respond with ONLY the code, no explanation."
        )

        prompt = f"Write {language} code for the following task:\n\n{task}"
        if include_tests:
            prompt += "\n\nAlso include unit tests."

        code = self.llm.complete(prompt, system=system, max_tokens=1024)

        # Strip markdown code fences if present
        code = self._strip_code_fences(code)

        return {
            "success": bool(code),
            "method": "llm_direct",
            "language": language,
            "code": code,
            "task": task,
        }

    def _llm_with_context(self, task: str, language: str, strategy: dict) -> dict:
        # Include successful past approaches as context
        context_approaches = [
            e["approach"].get("rationale", "")
            for e in self.experience_buffer
            if e["success"] and "rationale" in e.get("approach", {})
        ][:3]

        context = "\n".join(f"- {c}" for c in context_approaches) if context_approaches else "None available"

        system = (
            f"You are an expert {language} programmer with context from past successful approaches. "
            "Write clean, working code. Respond with ONLY the code."
        )

        prompt = (
            f"Task: {task}\n\n"
            f"Context from past successes:\n{context}\n\n"
            f"Write {language} code:"
        )

        code = self.llm.complete(prompt, system=system, max_tokens=1024)
        code = self._strip_code_fences(code)

        return {
            "success": bool(code),
            "method": "llm_with_context",
            "language": language,
            "code": code,
            "task": task,
        }

    def _iterative_refinement(self, task: str, language: str, strategy: dict) -> dict:
        # Step 1: Initial generation
        result = self._llm_direct(task, language, strategy)
        code = result.get("code", "")

        if not code:
            return result

        # Step 2: Ask LLM to review and improve
        system = f"You are a {language} code reviewer. Improve the following code for correctness and efficiency. Respond with ONLY the improved code."
        prompt = f"Original task: {task}\n\nCode to improve:\n```{language}\n{code}\n```"

        improved = self.llm.complete(prompt, system=system, max_tokens=1024)
        improved = self._strip_code_fences(improved)

        return {
            "success": bool(improved or code),
            "method": "iterative_refinement",
            "language": language,
            "code": improved or code,
            "original_code": code,
            "task": task,
        }

    def _template_based(self, task: str, language: str) -> dict:
        # Simple fallback: return a stub template
        template = self.genome["templates"].get("function", "# TODO: implement\ndef solution():\n    pass\n")

        code = template.format(
            name="solution",
            params="*args, **kwargs",
            docstring=task[:100],
        )

        return {
            "success": True,
            "method": "template_based",
            "language": language,
            "code": code,
            "task": task,
            "note": "Template fallback — LLM not available",
        }

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        import re
        # Remove ```python ... ``` or ``` ... ``` fences
        code = re.sub(r"^```[a-z]*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```$", "", code, flags=re.MULTILINE)
        return code.strip()
