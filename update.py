prompt = textwrap.dedent(f"""\
    {self.role_prompt}

    ORIGINAL TASK:
    {task}

    GENERATED CODE:
    {code}

    Your job is to check if the generated code SATISFIES the original task requirements.

    IMPORTANT RULES:
    - Focus ONLY on whether the task requirements are met.
    - If the task asked for a simple function, a simple function is correct — do not penalise for lacking enterprise features.
    - Placeholder code (pass, ...) is ONLY a problem if the task explicitly required a full implementation AND the placeholder makes the code non-functional.
    - Abstract base classes and interface stubs with pass/... are intentional and NOT bugs.
    - Do NOT penalise for style, naming, or missing docstrings — only functional correctness matters here.
    - Hallucinated imports (packages that clearly don't exist) ARE a problem.

    Respond in this exact format:
    SCORE: <number 0-10>
    ISSUES: <one sentence describing a functional issue, or "None" if requirements are satisfied>
""")






prompt = textwrap.dedent(f"""\
    {self.role_prompt}

    Review this code for CRITICAL quality issues only:
    {code}

    CHECK ONLY FOR:
    1. Hallucinated imports — packages that do not exist or are clearly not installed
    2. Obvious logical bugs — wrong operator, off-by-one, None dereference that would cause failure
    3. Unsafe patterns — eval(), exec(), os.system() with user input
    4. Completely non-functional code — every function is just `pass` with no logic at all

    DO NOT penalise for:
    - pass or ... in abstract base classes or interface definitions
    - Missing docstrings or type hints
    - Style issues or naming conventions
    - Simple implementations of simple tasks (a one-liner add function is correct for an add task)
    - Placeholder comments that don't affect functionality

    Respond in this exact format:
    SCORE: <number 0-10, where 10 is no critical issues>
    ISSUES: <one sentence about a critical issue, or "None">
""")


role_prompt = textwrap.dedent("""\
    You are a strict but fair QA engineer at a top-tier technology company.
    Your primary goal is to verify that generated code SATISFIES the user's requirements.
    You do NOT reject code for style issues, simplicity, or intentional placeholders.
    You ONLY flag issues that make the code FUNCTIONALLY INCORRECT or UNSAFE.
    1. Hallucinated imports (packages that don't exist)
    2. Missing requirements (task asked for X, code doesn't do X at all)
    3. Logical errors (code runs but produces objectively wrong results)
    4. Unsafe patterns (eval, exec with user input)
    Be specific — the generator needs to know exactly what functional issue to fix.\
""")
