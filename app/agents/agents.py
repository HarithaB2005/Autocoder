"""
Agent Definitions
=================
Each agent wraps one GPU model with a role-specific system prompt
AND enforces its own temperature bounds.

Temperature philosophy:
  - Code agents (gen/refactor/debug): 0.0–0.3  — deterministic, fewer hallucinations
  - Validation agent:                 0.0–0.1  — PASS/FAIL must be consistent
  - Explanation agent:                0.0–0.8  — fluent prose needs some creativity
  - Metadata agent:                   0.0–0.2  — structured JSON must be stable

Any user-supplied temperature is CLAMPED (not rejected) into the agent's range.
"""

import ast
import sys
import uuid
import logging
import textwrap
import traceback
import subprocess
import tempfile
import json
import re
from pathlib import Path
from typing import Optional
from app.core.llm_registry import registry
from app.core.config import settings

logger = logging.getLogger(__name__)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


# ─────────────────────────────────────────────────────────────────────────────
# Gibberish Detection
# ─────────────────────────────────────────────────────────────────────────────
# FLAG: Set GIBBERISH_CHECK_ENABLED = False to disable instantly if needed.
# This is purely CPU-based — no GPU call, no overhead.
GIBBERISH_CHECK_ENABLED = True

def is_gibberish(content: str) -> tuple[bool, str]:
    """
    Lightweight CPU-only gibberish detector. No ML, no GPU.
    Returns (is_gibberish: bool, reason: str).

    Three signals:
      1. Printable ratio   — binary/garbage files fail this
      2. Alphabetic ratio  — pure symbol soup fails this
      3. Avg word length   — unbroken random strings fail this

    To disable: set GIBBERISH_CHECK_ENABLED = False above.
    """
    if not GIBBERISH_CHECK_ENABLED:
        return False, ""

    if not content or len(content.strip()) == 0:
        return True, "Empty content."

    sample = content[:3000]

    printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
    if printable / len(sample) < 0.85:
        return True, "Too many non-printable characters — likely a binary file."

    alpha = sum(1 for c in sample if c.isalpha())
    if alpha / len(sample) < 0.08:
        return True, "Too few alphabetic characters — content appears to be pure symbols or numbers."

    words = sample.split()
    if not words:
        return True, "No readable words found."
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 30:
        return True, "Average word length too high — content appears to be random characters."

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox Executor
# ─────────────────────────────────────────────────────────────────────────────

class SandboxExecutor:
    """
    Executes Python code safely in a subprocess with:
      - Hard timeout (default 10s)
      - No network access (best-effort: no import allowed for requests/socket/urllib)
      - Captured stdout/stderr
      - Auto-generated test cases from the task description
    """

    TIMEOUT_SECONDS = 10

    # Dangerous imports we block in generated code before even running it
    BLOCKED_IMPORTS = {
        "subprocess", "os.system", "shutil.rmtree",
        "socket", "requests", "urllib", "httpx",
        "__import__", "eval(", "exec(",
    }

    def run(self, code: str, task_description: str) -> dict:
        """
        Run code in sandbox and return:
        {
            "ran": bool,
            "stdout": str,
            "stderr": str,
            "timed_out": bool,
            "blocked": bool,
            "block_reason": str,
            "test_results": list[dict],
            "execution_score": float  # 0.0 – 1.0
        }
        """
        # ── Safety pre-check ─────────────────────────────────────────────────
        for blocked in self.BLOCKED_IMPORTS:
            if blocked in code:
                return {
                    "ran": False, "stdout": "", "stderr": "",
                    "timed_out": False, "blocked": True,
                    "block_reason": f"Blocked import/call detected: '{blocked}'",
                    "test_results": [], "execution_score": 0.0,
                }

        # ── Syntax check first (free, instant) ───────────────────────────────
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "ran": False, "stdout": "", "stderr": str(e),
                "timed_out": False, "blocked": False, "block_reason": "",
                "test_results": [], "execution_score": 0.0,
            }

        # ── Auto-generate test harness ────────────────────────────────────────
        test_harness = self._build_test_harness(code, task_description)
        full_code = code + "\n\n" + test_harness

        # ── Write to temp file and execute ────────────────────────────────────
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(full_code)
                tmp_path = f.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            timed_out = False

        except subprocess.TimeoutExpired:
            stdout, stderr, timed_out = "", "Execution timed out.", True
        except Exception as e:
            stdout, stderr, timed_out = "", str(e), False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        # ── Parse test results from stdout ────────────────────────────────────
        test_results, score = self._parse_test_results(stdout)

        return {
            "ran": not timed_out and not stderr,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "blocked": False,
            "block_reason": "",
            "test_results": test_results,
            "execution_score": score,
        }

    def _build_test_harness(self, code: str, task: str) -> str:
        """
        Heuristically generate test cases from the task description.
        Looks for function definitions in the code and builds simple assertions.
        """
        try:
            tree = ast.parse(code)
        except Exception:
            return ""

        # Find all top-level function definitions
        funcs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        if not funcs:
            # No functions — just run the code, no assertions
            return (
                "print('TEST:no_functions:PASS:Code ran without errors')"
            )

        # Build test cases per function using type hints + name heuristics
        lines = ["", "# ── Auto-generated test harness ──────────────────────────────────────────", "import traceback as _tb", "_passed = 0", "_failed = 0", ""]

        for func in funcs[:5]:  # limit to 5 functions
            fname = func.name
            args  = [a.arg for a in func.args.args]
            nargs = len(args)
            cases = self._infer_test_cases(fname, args, task)

            for i, (inputs, expected, label) in enumerate(cases[:3]):  # max 3 cases per func
                test_id = f"{fname}_case{i+1}"
                lines.append(f"try:")
                if expected is None:
                    # Just check it runs without error
                    lines.append(f"    {fname}({inputs})")
                    lines.append(f"    print('TEST:{test_id}:PASS:{label}')")
                    lines.append(f"    _passed += 1")
                else:
                    lines.append(f"    _result = {fname}({inputs})")
                    lines.append(f"    assert _result == {repr(expected)}, f'Got {{_result}}, expected {repr(expected)}'")
                    lines.append(f"    print('TEST:{test_id}:PASS:{label}')")
                    lines.append(f"    _passed += 1")
                lines.append(f"except Exception as _e:")
                lines.append(f"    print(f'TEST:{test_id}:FAIL:{label} | {{_e}}')")
                lines.append(f"    _failed += 1")
                lines.append("")

        lines.append("print(f'SUMMARY:passed={{_passed}}:failed={{_failed}}')")
        return "\n".join(lines)

    def _infer_test_cases(self, fname: str, args: list, task: str) -> list:
        """
        Infer basic test cases from function name and task description.
        Returns list of (inputs_str, expected_value_or_None, label).
        """
        task_lower = task.lower()
        cases = []

        # Number operations
        if any(w in fname.lower() for w in ["add", "sum", "plus"]):
            cases = [("2, 3", 5, "add 2+3=5"), ("0, 0", 0, "add 0+0=0"), ("-1, 1", 0, "add -1+1=0")]
        elif any(w in fname.lower() for w in ["subtract", "minus", "diff"]):
            cases = [("5, 3", 2, "subtract 5-3=2"), ("0, 0", 0, "subtract 0-0=0")]
        elif any(w in fname.lower() for w in ["multiply", "product", "mul"]):
            cases = [("3, 4", 12, "multiply 3*4=12"), ("0, 5", 0, "multiply 0*5=0")]
        elif any(w in fname.lower() for w in ["divide", "div"]):
            cases = [("10, 2", 5.0, "divide 10/2=5"), ("0, 1", 0.0, "divide 0/1=0")]
        elif any(w in fname.lower() for w in ["factorial"]):
            cases = [("5", 120, "factorial(5)=120"), ("0", 1, "factorial(0)=1")]
        elif any(w in fname.lower() for w in ["fibonacci", "fib"]):
            cases = [("1", 1, "fib(1)=1"), ("5", 5, "fib(5)=5")]
        elif any(w in fname.lower() for w in ["palindrome"]):
            cases = [("'racecar'", True, "palindrome racecar=True"), ("'hello'", False, "palindrome hello=False")]
        elif any(w in fname.lower() for w in ["reverse"]):
            if "str" in task_lower or "string" in task_lower:
                cases = [("'hello'", "olleh", "reverse hello=olleh")]
            else:
                cases = [("[1,2,3]", [3,2,1], "reverse list")]
        elif any(w in fname.lower() for w in ["sort"]):
            cases = [("[3,1,2]", [1,2,3], "sort [3,1,2]=[1,2,3]")]
        elif any(w in fname.lower() for w in ["max", "maximum"]):
            cases = [("[1,5,3]", 5, "max of [1,5,3]=5")]
        elif any(w in fname.lower() for w in ["min", "minimum"]):
            cases = [("[1,5,3]", 1, "min of [1,5,3]=1")]
        elif any(w in fname.lower() for w in ["is_even", "even"]):
            cases = [("4", True, "4 is even"), ("3", False, "3 is not even")]
        elif any(w in fname.lower() for w in ["is_prime", "prime"]):
            cases = [("7", True, "7 is prime"), ("4", False, "4 is not prime")]
        else:
            # Generic — just run with sensible defaults based on arg count
            if len(args) == 0:
                cases = [(""  , None, "no-arg call")]
            elif len(args) == 1:
                cases = [("'test'", None, "single string arg"), ("42", None, "single int arg")]
            elif len(args) == 2:
                cases = [("1, 2", None, "two int args"), ("'a', 'b'", None, "two string args")]
            else:
                cases = [(", ".join(["1"] * len(args)), None, "all-int args")]

        return cases

    def _parse_test_results(self, stdout: str) -> tuple[list, float]:
        """Parse TEST: lines from stdout into structured results + score."""
        results = []
        passed = failed = 0

        for line in stdout.splitlines():
            if line.startswith("TEST:"):
                parts = line.split(":", 3)
                if len(parts) >= 3:
                    _, test_id, status = parts[0], parts[1], parts[2]
                    label = parts[3] if len(parts) > 3 else ""
                    results.append({"test": test_id, "status": status, "detail": label})
                    if status == "PASS":
                        passed += 1
                    else:
                        failed += 1
            elif line.startswith("SUMMARY:"):
                # Parse summary line for accuracy
                m = re.search(r"passed=(\d+):failed=(\d+)", line)
                if m:
                    passed = int(m.group(1))
                    failed = int(m.group(2))

        total = passed + failed
        score = (passed / total) if total > 0 else 0.0
        return results, score


sandbox = SandboxExecutor()


# ─────────────────────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    name:         str   = "Base Agent"
    model:        str   = "llama-3-70b"
    role_prompt:  str   = "You are a helpful AI assistant."
    temp_default: float = 0.2
    temp_min:     float = 0.0
    temp_max:     float = 1.0

    def execute(
        self,
        task:        str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> str:
        raw_temp   = temperature if temperature is not None else self.temp_default
        final_temp = _clamp(raw_temp, self.temp_min, self.temp_max)

        if temperature is not None and final_temp != temperature:
            logger.warning(
                "[%s] Temperature %.2f clamped to %.2f (agent bounds: %.2f–%.2f)",
                self.name, temperature, final_temp, self.temp_min, self.temp_max,
            )

        final_tokens = min(max_tokens, settings.MAX_TOKENS_CEILING) if max_tokens else None
        full_prompt  = f"{self.role_prompt}\n\nTask:\n{task}"
        logger.info(
            "[%s] → model=%s  max_tokens=%s  temperature=%.2f",
            self.name, self.model, final_tokens or "model-default", final_temp,
        )

        result = registry.generate(
            model_name=self.model,
            prompt=full_prompt,
            max_tokens=final_tokens,
            temperature=final_temp,
        )
        logger.info("[%s] ← %d chars received", self.name, len(result))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Specialised Agents — enterprise-grade prompts
# ─────────────────────────────────────────────────────────────────────────────

class GeneralAssistantAgent(BaseAgent):
    name         = "Assistant"
    model        = "llama-3-70b"
    temp_default = 0.3
    temp_min     = 0.0
    temp_max     = 0.8
    role_prompt  = (
        "You are a knowledgeable, concise, and professional AI assistant. "
        "Respond clearly and directly to the user's request. "
        "For greetings, reply briefly and warmly. "
        "For questions, give accurate, well-structured answers. "
        "Never fabricate facts. If unsure, say so honestly."
    )


class CodeGenerationAgent(BaseAgent):
    name         = "Code Generator"
    model        = "qwen-coder-32b"
    temp_default = 0.15
    temp_min     = 0.0
    temp_max     = 0.3
    role_prompt  = textwrap.dedent("""\
        You are a senior software engineer at a top-tier technology company.
        Your code is production-ready, enterprise-grade, and reviewed by principal engineers.

        STRICT REQUIREMENTS — every response must follow all of these:
        1. Return ONLY raw code. No markdown fences, no explanations, no preamble.
        2. Include a module-level docstring describing what the code does.
        3. Every function/class must have a Google-style docstring.
        4. Use type hints on ALL function signatures.
        5. Handle all edge cases explicitly (None inputs, empty collections, division by zero, etc.).
        6. Use meaningful variable names — never single letters except loop indices.
        7. No dead code, no TODO comments, no placeholder logic.
        8. If the task involves I/O, include proper error handling with specific exception types.
        9. Follow PEP 8 strictly.
        10. If external libraries are needed, use only well-known stdlib or popular packages.\
    """)


class RefactorAgent(BaseAgent):
    name         = "Refactorer"
    model        = "llama-3-70b"
    temp_default = 0.15
    temp_min     = 0.0
    temp_max     = 0.3
    role_prompt  = textwrap.dedent("""\
        You are an expert software architect specialising in code quality and maintainability.

        REFACTORING RULES:
        1. Return ONLY the refactored code. No prose, no markdown fences.
        2. Preserve 100% of existing behaviour — do not add or remove features.
        3. Apply SOLID principles where applicable.
        4. Eliminate code duplication (DRY principle).
        5. Replace magic numbers/strings with named constants.
        6. Break down functions longer than 20 lines into smaller, single-purpose functions.
        7. Improve naming to be self-documenting.
        8. Add/improve type hints and docstrings.
        9. Add an inline comment for every non-obvious change explaining WHY, not just what.\
    """)


class DebuggingAgent(BaseAgent):
    name         = "Debugger"
    model        = "llama-3-70b"
    temp_default = 0.1
    temp_min     = 0.0
    temp_max     = 0.3
    role_prompt  = textwrap.dedent("""\
        You are a principal engineer and debugging specialist.

        DEBUGGING PROTOCOL:
        1. Return ONLY the corrected code. No prose, no markdown fences.
        2. Identify the ROOT CAUSE of each bug — not just symptoms.
        3. For every fix, add a short inline comment: # FIX: <what was wrong and why>
        4. Do not introduce new bugs while fixing existing ones.
        5. Do not change logic unrelated to the bug.
        6. If the error is an exception trace, find the exact line causing it.
        7. Check for: off-by-one errors, None dereferences, wrong types, incorrect logic operators,
           missing return statements, unhandled exceptions, and incorrect variable scope.\
    """)


class ExplanationAgent(BaseAgent):
    name         = "Explainer"
    model        = "gemma-2-9b"
    temp_default = 0.5
    temp_min     = 0.0
    temp_max     = 0.8
    role_prompt  = textwrap.dedent("""\
        You are a senior technical educator and documentation specialist.

        EXPLANATION GUIDELINES — enterprise standard:
        1. Match depth to the request: quick question = concise answer; deep dive = thorough breakdown.
        2. Use plain language first, then introduce technical terms with brief definitions.
        3. Use concrete, minimal examples to illustrate abstract concepts.
        4. Structure longer explanations with clear sections.
        5. If explaining code: describe WHAT it does, HOW it works, and WHY design decisions were made.
        6. Never hallucinate or fabricate facts. If uncertain, say so clearly.
        7. Cover edge cases and limitations where relevant.\
    """)


class MetadataExtractionAgent(BaseAgent):
    name         = "Metadata Extractor"
    model        = "gemma-2-9b"
    temp_default = 0.1
    temp_min     = 0.0
    temp_max     = 0.2
    role_prompt  = textwrap.dedent("""\
        You are a structured-data extraction engine.

        OUTPUT REQUIREMENTS:
        1. Return ONLY valid JSON — no prose, no markdown fences, no trailing commas.
        2. Use exactly these keys:
           {
             "language": string,
             "dependencies": [{"name": string, "version": string|null, "type": "stdlib|third_party|local"}],
             "entry_points": [string],
             "exported_symbols": [{"name": string, "type": "function|class|constant|variable"}],
             "file_purpose": string,
             "complexity_estimate": "low|medium|high",
             "detected_patterns": [string]
           }
        3. If a field cannot be determined, use null or [].
        4. Be exhaustive — list every import, every exported symbol.\
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Validation Agent — Multi-phase intelligent validator
# ─────────────────────────────────────────────────────────────────────────────

class ValidationAgent(BaseAgent):
    """
    4-phase intelligent validator:
      Phase 1 — Static AST syntax check         (free, instant, 100% reliable)
      Phase 2 — Sandbox execution + test cases  (subprocess, isolated, scored)
      Phase 3 — LLM requirement coverage check  (does code satisfy the task?)
      Phase 4 — LLM code quality review         (hallucinated imports, edge cases)

    Scoring:
      - Each phase contributes to a weighted score (0.0–1.0)
      - Passes if score >= PASS_THRESHOLD
      - Feedback is always specific enough for the generator to self-correct

    Temperature pinned near-zero: PASS/FAIL must be deterministic.
    """

    PASS_THRESHOLD  = 0.75   # weighted score needed to pass
    SANDBOX_WEIGHT  = 0.40   # execution result weight
    LLM_REQ_WEIGHT  = 0.35   # requirement coverage weight
    LLM_QA_WEIGHT   = 0.25   # code quality weight

    name         = "Validator"
    model        = "gemma-2-9b"
    temp_default = 0.05
    temp_min     = 0.0
    temp_max     = 0.1
    role_prompt  = textwrap.dedent("""\
        You are a strict QA engineer and code reviewer at a top-tier technology company.
        You review generated code against original requirements with zero tolerance for:
        1. Hallucinated imports (libraries that don't exist or aren't installed)
        2. Missing requirements (task asked for X, code doesn't do X)
        3. Logical errors (code runs but produces wrong results)
        4. Unsafe patterns (eval, exec, unhandled exceptions on common inputs)
        5. Placeholder code (pass, TODO, NotImplementedError in non-abstract classes)
        Be specific in your feedback — the generator needs to know exactly what to fix.\
    """)

    def validate(self, code: str, original_task: str) -> dict:
        """
        Returns:
          {
            "valid":    bool,
            "feedback": str,
            "score":    float,   # 0.0–1.0
            "phases":   dict     # per-phase results for transparency
          }
        """
        phases   = {}
        feedback = []
        score    = 0.0

        # ── Phase 1: AST Syntax Check (free) ─────────────────────────────────
        syntax_ok, syntax_msg = self._check_syntax(code)
        phases["syntax"] = {"passed": syntax_ok, "detail": syntax_msg}
        if not syntax_ok:
            # Hard fail — no point running further phases
            return {
                "valid":    False,
                "feedback": f"Syntax error — fix this before anything else: {syntax_msg}",
                "score":    0.0,
                "phases":   phases,
            }

        # ── Phase 2: Sandbox Execution + Auto Test Cases ──────────────────────
        sandbox_result = sandbox.run(code, original_task)
        exec_score     = sandbox_result["execution_score"]
        phases["sandbox"] = sandbox_result

        if sandbox_result["blocked"]:
            return {
                "valid":    False,
                "feedback": f"Unsafe code detected: {sandbox_result['block_reason']}",
                "score":    0.0,
                "phases":   phases,
            }
        if sandbox_result["timed_out"]:
            feedback.append("Code execution timed out — likely an infinite loop or blocking call.")
            exec_score = 0.0
        elif sandbox_result["stderr"]:
            feedback.append(f"Runtime error: {sandbox_result['stderr'][:300]}")
            exec_score = 0.0
        elif sandbox_result["test_results"]:
            failed_tests = [t for t in sandbox_result["test_results"] if t["status"] == "FAIL"]
            if failed_tests:
                details = "; ".join(f"{t['test']}: {t['detail']}" for t in failed_tests[:3])
                feedback.append(f"Test failures: {details}")

        score += exec_score * self.SANDBOX_WEIGHT

        # ── Phase 3: LLM Requirement Coverage ────────────────────────────────
        req_score, req_feedback = self._check_requirements(code, original_task)
        phases["requirements"] = {"score": req_score, "feedback": req_feedback}
        if req_feedback:
            feedback.append(req_feedback)
        score += req_score * self.LLM_REQ_WEIGHT

        # ── Phase 4: LLM Code Quality Review ─────────────────────────────────
        qa_score, qa_feedback = self._check_quality(code, original_task)
        phases["quality"] = {"score": qa_score, "feedback": qa_feedback}
        if qa_feedback:
            feedback.append(qa_feedback)
        score += qa_score * self.LLM_QA_WEIGHT

        # ── Final verdict ─────────────────────────────────────────────────────
        passed           = score >= self.PASS_THRESHOLD
        combined_feedback = " | ".join(feedback) if feedback else "All checks passed."

        logger.info(
            "[Validator] score=%.3f (sandbox=%.2f req=%.2f qa=%.2f) → %s",
            score, exec_score, req_score, qa_score, "PASS" if passed else "FAIL",
        )

        return {
            "valid":    passed,
            "feedback": combined_feedback,
            "score":    round(score, 3),
            "phases":   phases,
        }

    # ── Private phase helpers ──────────────────────────────────────────────────

    @staticmethod
    def _check_syntax(code: str) -> tuple[bool, str]:
        try:
            ast.parse(code)
            return True, "Syntax OK."
        except SyntaxError as e:
            detail = "".join(traceback.format_exception_only(type(e), e)).strip()
            return False, detail

    def _check_requirements(self, code: str, task: str) -> tuple[float, str]:
        """Ask LLM: does the code fully satisfy the task requirements?"""
        prompt = textwrap.dedent(f"""\
            {self.role_prompt}

            ORIGINAL TASK:
            {task}

            GENERATED CODE:
            {code}

            Does the generated code fully and correctly satisfy ALL requirements in the original task?

            Respond in this exact format:
            SCORE: <number 0-10>
            ISSUES: <one sentence describing the main issue, or "None" if fully satisfied>
        """)
        try:
            response = registry.generate(
                model_name=self.model,
                prompt=prompt,
                max_tokens=256,
                temperature=self.temp_default,
            )
            score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            issue_match = re.search(r"ISSUES:\s*(.+)", response, re.IGNORECASE)

            raw_score = float(score_match.group(1)) if score_match else 5.0
            norm_score = min(raw_score / 10.0, 1.0)
            issue = issue_match.group(1).strip() if issue_match else ""
            issue = "" if issue.lower() in ("none", "none.", "n/a", "") else issue

            return norm_score, issue
        except Exception as e:
            logger.warning("[Validator] Requirement check failed: %s", e)
            return 0.7, ""  # neutral fallback

    def _check_quality(self, code: str, task: str) -> tuple[float, str]:
        """Ask LLM: are there quality issues, hallucinated imports, unsafe patterns?"""
        prompt = textwrap.dedent(f"""\
            {self.role_prompt}

            Review this code for quality issues:
            {code}

            Check specifically for:
            1. Hallucinated imports (packages that don't exist or aren't standard)
            2. Placeholder logic (pass, TODO, NotImplementedError without reason)
            3. Obvious logical bugs (wrong operator, off-by-one, None dereference)
            4. Missing edge case handling

            Respond in this exact format:
            SCORE: <number 0-10, where 10 is perfect quality>
            ISSUES: <one sentence, or "None">
        """)
        try:
            response = registry.generate(
                model_name=self.model,
                prompt=prompt,
                max_tokens=256,
                temperature=self.temp_default,
            )
            score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
            issue_match = re.search(r"ISSUES:\s*(.+)", response, re.IGNORECASE)

            raw_score = float(score_match.group(1)) if score_match else 5.0
            norm_score = min(raw_score / 10.0, 1.0)
            issue = issue_match.group(1).strip() if issue_match else ""
            issue = "" if issue.lower() in ("none", "none.", "n/a", "") else issue

            return norm_score, issue
        except Exception as e:
            logger.warning("[Validator] Quality check failed: %s", e)
            return 0.7, ""


# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY: dict[str, BaseAgent] = {
    "general_chat":        GeneralAssistantAgent(),
    "generation":          CodeGenerationAgent(),
    "refactoring":         RefactorAgent(),
    "debugging":           DebuggingAgent(),
    "explanation":         ExplanationAgent(),
    "metadata_extraction": MetadataExtractionAgent(),
}

VALIDATION_AGENT = ValidationAgent()
