You are an expert Principal Software Engineer and Technical Writer. I am halfway through developing this project and need to hand it off to another developer (or AI) to complete the second half. 

Your task is to analyze my current codebase and generate a comprehensive, highly technical, and logically precise `README.md` file. This document must act as a seamless handoff guide so the next developer understands exactly what exists and what needs to be built next.

Please read all the code files I provide and structure the README.md with the following exact sections:

### 1. Architecture & Technical Workflow
* Provide a high-level technical overview of the system architecture.
* Explain the logical data flow and control flow across the application.
* List the complete technology stack, frameworks, libraries, and runtime environments used.

### 2. Deep-Dive Function & Module Registry
For every single code file, list every function, class, and method present. For each item, explicitly document:
* **Function Signature & Name:** (e.g., `calculateMetrics(userId: string, Range: DateRange)`)
* **Input Parameters:** Exact data types, formats, and structural expectations.
* **Return Values:** Exact data types, objects, or payload structures returned.
* **Core Logic & Capabilities:** A dense, technical description of its internal logic, algorithms, side effects, and abilities.
* **Dependencies:** What other functions, APIs, or internal modules it invokes.

### 3. Current Project State (What is Done)
* Detail all fully implemented features, modules, and database schemas.
* Confirm which components are thoroughly tested, structurally complete, and production-ready.

### 4. Remaining Roadmap (What is Planned & Remaining)
* Map out the incomplete half of the project with logical precision.
* Break down the remaining features into explicit, sequential technical tasks.
* Highlight any unwritten functions, missing API endpoints, or unconfigured database models that are planned but not yet built.
* Specify how the new code must hook into the existing functions documented in Section 2.

### 5. Setup & Local Execution
* Step-by-step instructions to install dependencies and configure environment variables.
* Exact commands needed to run, test, and debug the application locally.

---
Strict Instructions for Output:
- Use precise engineering terminology (do not abstract away details; use actual types, design patterns, and architectural terms).
- Ensure 100% coverage of the codebase; do not skip minor helper functions or utility classes.
- Maintain a direct, objective, and highly actionable tone.

Here are my project files:
[PASTE YOUR CODE FILES / ARCHITECTURE NOTES HERE]
