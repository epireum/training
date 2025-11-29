# Week 1: Python for Agentic AI
[back](readme.md)
## Student Reading Material

### Introduction

Welcome to Week 1 of the Agentic AI program. This week focuses on Python fundamentals that form the foundation for building intelligent AI agents. By mastering these concepts, you'll be equipped to create tools and utilities that power agentic systems.

### Why Python for AI?

Python has become the language of choice for AI development, and understanding why will help you appreciate its role in building agents:

**1. Extensive AI Library Support**  
Most artificial intelligence and machine learning libraries are written in Python. This includes frameworks specifically designed for building AI agents, such as LangChain, CrewAI, and LlamaIndex.

**2. Easy to Learn and Flexible**  
Python's clean, readable syntax allows you to focus on solving problems rather than wrestling with complex language rules. This flexibility is crucial when rapidly prototyping and iterating on agent behaviors.

**3. Seamless Integration**  
Python integrates effortlessly with machine learning frameworks and agent-based systems, making it the ideal glue language for connecting different AI components.

**4. Rich Ecosystem**  
The Python ecosystem includes powerful libraries for every aspect of AI development:
- Agent frameworks: langchain, crewai, llamaindex
- Data manipulation: numpy, pandas
- API interactions: requests
- And thousands more available through pip

### Learning Outcomes

By the end of this week, you will be able to:
- Work confidently with Python's core data types and structures
- Parse and manipulate JSON data
- Import and utilize external Python modules
- Create reusable functions that serve as agent tools
- Make HTTP API calls to external services
- Read from and write to files
- Build a utility module for your AI agent projects

---

## Core Concepts

### 1. Variables and Data Types

Python uses dynamic typing, meaning you don't need to declare variable types explicitly. The interpreter determines the type based on the value assigned.

**Basic Data Types:**
- `int`: Whole numbers (e.g., 42, -17, 0)
- `float`: Decimal numbers (e.g., 3.14, -0.5, 2.0)
- `str`: Text strings (e.g., "Hello", 'Python')
- `bool`: Boolean values (True or False)

**Collections:**
- **Lists**: Ordered, mutable sequences enclosed in square brackets `[]`
  ```python
  fruits = ["apple", "banana", "cherry"]
  ```
  
- **Dictionaries**: Key-value pairs enclosed in curly braces `{}`
  ```python
  student = {"name": "Alex", "score": 89}
  ```
  
Dictionaries are particularly important for AI agents because they mirror JSON structure, which is the standard format for data exchange between systems.

### 2. Functions

Functions are reusable blocks of code that perform specific tasks. In the context of AI agents, functions become "tools" that agents can use to accomplish goals.

**Basic function syntax:**
```python
def add(a, b):
    return a + b
```

This function takes two parameters (a and b) and returns their sum. When building agents, you'll create functions that perform actions like searching databases, calling APIs, or processing data.

### 3. Modules and Imports

Python's power comes from its ability to use code written by others through modules. A module is simply a Python file containing functions, classes, and variables that you can import into your program.

**Common imports for AI work:**
```python
import requests  # For making HTTP requests
import json      # For working with JSON data
```

You can also import specific functions:
```python
from json import loads, dumps
```

### 4. JSON Basics

JSON (JavaScript Object Notation) is the universal language of data exchange on the web. AI agents constantly work with JSON when communicating with APIs and storing data.

Python's json module provides two key functions:
- `json.loads()`: Converts a JSON string into a Python object
- `json.dumps()`: Converts a Python object into a JSON string

**Example:**
```python
import json
data = json.loads('{"city": "Berlin"}')
print(data["city"])  # Output: Berlin
```

The structure of JSON maps directly to Python dictionaries, making it natural to work with in Python.

### 5. Calling APIs

APIs (Application Programming Interfaces) allow your agent to interact with external services. The requests library makes this straightforward.

**Example using the PokeAPI:**
```python
import requests
response = requests.get("https://pokeapi.co/api/v2/pokemon/pikachu")
print(response.json())
```

This code:
1. Sends an HTTP GET request to the PokeAPI
2. Receives a response containing Pikachu's data
3. Parses the JSON response and prints it

For AI agents, API calls are essential tools. Your agent might call weather APIs, search APIs, database APIs, or even other AI services.

### 6. File Handling

Agents often need to persist data or read configuration files. Python's file handling uses context managers (the 'with' statement) to ensure files are properly closed.

**Writing to a file:**
```python
with open("data.json", "w") as f:
    json.dump(student, f)
```

**Reading from a file:**
```python
with open("data.json", "r") as f:
    data = json.load(f)
```

The 'with' statement automatically handles closing the file, even if an error occurs.

---

## Building Toward Agents

Each concept you're learning this week directly supports agent development:
- **Variables and data types** let you store agent state and information
- **Functions** become tools your agent can use
- **Modules** provide pre-built capabilities
- **JSON** enables communication between your agent and external systems
- **API calls** allow your agent to access real-world data and services
- **File handling** lets your agent remember information between sessions

In the coming weeks, you'll combine these fundamentals to create agents that can reason, plan, and take action autonomously.

---

## ASSIGNMENT (Week 1):

### Assignment 1 — Build 3 Tools

Create a file tools.py with:

* get_weather(city) — Use weather API (Open-Meteo – no key needed)

* get_crypto_price(symbol) — Use CoinGecko API

* get_pokemon(name) — Pokémon API

Store results to results.json.

### Assignment 2 — Build a CLI Menu

Example:

1. Weather  
2. Crypto  
3. Pokémon  
4. Exit


Call the functions and store outputs in logs/output.json.

---

## Preparation for Class

Before attending class, please:
1. Install Python 3.8 or higher on your system
2. Set up a code editor (VS Code, PyCharm, or similar)
3. Install the requests library: `pip install requests`
4. Review this material and note any questions
5. Try running the example code snippets on your machine

### Practice Exercise

To reinforce your learning, try this before class:
1. Create a Python script that fetches data from a public API
2. Parse the JSON response
3. Extract specific information and save it to a file

This exercise combines all the concepts from Week 1 and prepares you for building agent tools.

---

## Looking Ahead

Week 1 establishes the foundation. In subsequent weeks, you'll build on these skills to create increasingly sophisticated agent systems that can interact with the world, make decisions, and accomplish complex tasks autonomously.
