You are an expert Software Tester. I need you to write a Python function called `generate_intelligent_tests(user_prompt: str, generated_code: str) -> str`. 

This function must invoke our LLM to analyze the generated code code block and dynamically produce a highly relevant python test suite using standard 'assert' statements. 

The system prompt inside this function must tell the LLM to follow these instructions precisely:

1. DYNAMIC INPUT PARAMETER MATCHING:
   - Carefully inspect the exact code snippet provided in `generated_code`.
   - Identify how the code expects inputs to be delivered (e.g., look at the specific function or method parameters, keyword arguments, and data types).
   - Generate test inputs that rigidly match those parameter signatures. Do not guess parameters or introduce variables that do not exist in the code structure.

2. BALANCED TEST CONDITION GENERATION:
   - Happy Path: Generate 2-3 standard test inputs based on the intent of the original `user_prompt` to check that the code works correctly under normal conditions.
   - Edge Cases: Generate test inputs that stress-test those exact parameters using extreme values, empty strings, nulls, or empty collections (`None`, `[]`, `{}`) depending on what data type the parameter expects.

3. OUTPUT FORMAT:
   - Output PURE executable python code enclosed in markdown code blocks (```python ... ```). 
   - Do not include any conversational introductions, markdown explanations, or descriptions outside the code block.

Please provide the python function wrapper, the clean internal LLM prompt string template, and the parsing logic to extract the raw test code cleanly for the execution sandbox.
