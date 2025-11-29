# Week 3.1: MCP Server
[back](README.md)
## Student Reading Material

### Introduction

This supplementary session introduces the Model Context Protocol (MCP) Server, a standardized way to provide context and tools to AI agents. MCP enables agents to interact with external systems, databases, and services through a unified protocol, making agent development more modular and scalable.

### What is MCP Server?

The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to Large Language Models (LLMs). An MCP Server acts as a bridge between AI agents and external resources.

**Key Concepts:**

**1. Protocol-Based Communication**
- Standardized interface for agent-resource interaction
- Language and framework agnostic
- JSON-RPC based messaging

**2. Context Providers**
- Servers expose resources (data, tools, prompts)
- Agents consume these resources through the protocol
- Decouples agent logic from resource implementation

**3. Tool Exposure**
- Functions are exposed as MCP tools
- Agents can discover and invoke tools dynamically
- Type-safe parameter passing

### Why MCP Server?

**Traditional Approach Problems:**
- Tight coupling between agents and tools
- Duplicate code across projects
- Difficult to share tools between teams
- Hard to version and maintain tools

**MCP Server Benefits:**
- **Modularity**: Tools are independent services
- **Reusability**: One server, many agents
- **Discoverability**: Agents can list available tools
- **Maintainability**: Update tools without changing agent code
- **Security**: Centralized access control
- **Scalability**: Distribute tools across servers

### Learning Outcomes

By the end of this session, you will be able to:
- Understand the MCP protocol architecture
- Identify use cases for MCP servers
- Build a basic MCP server
- Expose tools through MCP
- Connect agents to MCP servers
- Implement practical MCP server applications

---

## Basics of MCP Server

### Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Agent     │ ◄─────► │ MCP Server  │ ◄─────► │  Resources  │
│   (Client)  │   MCP   │             │         │  (DB, APIs) │
└─────────────┘ Protocol└─────────────┘         └─────────────┘
```

**Components:**

1. **MCP Client (Agent)**: Consumes tools and resources
2. **MCP Server**: Exposes tools and manages resources
3. **Resources**: Databases, APIs, files, services

### Core Concepts

**1. Resources**

Data or content that can be read by agents.

```python
# Example: Expose documents as resources
resources = {
    "document://readme": {
        "uri": "document://readme",
        "name": "README",
        "mimeType": "text/plain",
        "content": "This is the README content"
    }
}
```

**2. Tools**

Functions that agents can invoke.

```python
# Example: Weather tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {
        "city": city,
        "temperature": 22,
        "condition": "sunny"
    }
```

**3. Prompts**

Pre-defined prompt templates.

```python
# Example: Summarization prompt
prompts = {
    "summarize": {
        "name": "summarize",
        "description": "Summarize text",
        "template": "Summarize the following text:\n\n{text}"
    }
}
```

### MCP Protocol Messages

**Tool Discovery:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
}
```

**Tool Invocation:**
```json
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "get_weather",
        "arguments": {
            "city": "Berlin"
        }
    },
    "id": 2
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "content": [
            {
                "type": "text",
                "text": "{\"city\": \"Berlin\", \"temperature\": 22}"
            }
        ]
    },
    "id": 2
}
```

---

## MCP Server Use Cases

### 1. Database Access

Expose database queries as MCP tools.

**Use Case:** Agent needs to query customer data

```python
from mcp.server import Server
import sqlite3

server = Server("database-server")

@server.tool()
def query_customers(name: str) -> list:
    """Search customers by name."""
    conn = sqlite3.connect("customers.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE name LIKE ?", (f"%{name}%",))
    results = cursor.fetchall()
    conn.close()
    return results

@server.tool()
def get_customer_orders(customer_id: int) -> list:
    """Get all orders for a customer."""
    conn = sqlite3.connect("customers.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE customer_id = ?", (customer_id,))
    results = cursor.fetchall()
    conn.close()
    return results
```

### 2. API Integration

Wrap external APIs as MCP tools.

**Use Case:** Agent needs weather, crypto, and stock data

```python
import requests

server = Server("api-server")

@server.tool()
def get_weather(city: str) -> dict:
    """Get current weather."""
    response = requests.get(f"https://api.weather.com/v1/weather?city={city}")
    return response.json()

@server.tool()
def get_crypto_price(symbol: str) -> dict:
    """Get cryptocurrency price."""
    response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd")
    return response.json()

@server.tool()
def get_stock_price(ticker: str) -> dict:
    """Get stock price."""
    response = requests.get(f"https://api.stocks.com/v1/quote/{ticker}")
    return response.json()
```

### 3. File System Operations

Provide file access through MCP.

**Use Case:** Agent needs to read/write files

```python
import os

server = Server("filesystem-server")

@server.tool()
def read_file(path: str) -> str:
    """Read file contents."""
    with open(path, 'r') as f:
        return f.read()

@server.tool()
def write_file(path: str, content: str) -> dict:
    """Write content to file."""
    with open(path, 'w') as f:
        f.write(content)
    return {"status": "success", "path": path}

@server.tool()
def list_directory(path: str) -> list:
    """List directory contents."""
    return os.listdir(path)
```

### 4. ML Model Serving

Expose ML models as MCP tools.

**Use Case:** Agent needs predictions from ML models

```python
import pickle

server = Server("ml-server")

@server.tool()
def predict_salary(years_experience: float) -> dict:
    """Predict salary based on experience."""
    model = pickle.load(open("salary_model.pkl", "rb"))
    prediction = model.predict([[years_experience]])
    return {
        "years_experience": years_experience,
        "predicted_salary": float(prediction[0])
    }

@server.tool()
def classify_iris(sepal_length: float, sepal_width: float, 
                  petal_length: float, petal_width: float) -> dict:
    """Classify iris species."""
    model = pickle.load(open("iris_model.pkl", "rb"))
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    species = ["setosa", "versicolor", "virginica"]
    return {"species": species[prediction[0]]}
```

### 5. RAG System

Expose document search as MCP tool.

**Use Case:** Agent needs to query knowledge base

```python
from langchain.vectorstores import Chroma

server = Server("rag-server")

@server.tool()
def search_documents(query: str, k: int = 3) -> list:
    """Search document knowledge base."""
    vectorstore = Chroma(persist_directory="./rag_vectorstore")
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in results
    ]

@server.tool()
def answer_from_docs(question: str) -> str:
    """Answer question using RAG."""
    from langchain.chains import RetrievalQA
    vectorstore = Chroma(persist_directory="./rag_vectorstore")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    result = qa_chain({"query": question})
    return result["result"]
```

---

## Practical Implementation

### Building a Basic MCP Server

**Step 1: Install MCP SDK**

```bash
pip install mcp
```

**Step 2: Create Server**

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio

# Initialize server
server = Server("my-first-mcp-server")

# Define tools
@server.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@server.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@server.tool()
def get_greeting(name: str) -> str:
    """Generate a greeting message."""
    return f"Hello, {name}! Welcome to MCP Server."

# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Configure Server**

Create `mcp_config.json`:

```json
{
    "mcpServers": {
        "my-first-mcp-server": {
            "command": "python",
            "args": ["mcp_server.py"]
        }
    }
}
```

### Complete Example: Multi-Tool MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio
import requests
import pickle

server = Server("agent-tools-server")

# Weather Tool
@server.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    try:
        # Replace with actual API call
        return {
            "city": city,
            "temperature": 22,
            "condition": "sunny",
            "humidity": 65
        }
    except Exception as e:
        return {"error": str(e)}

# Crypto Tool
@server.tool()
def get_crypto_price(symbol: str) -> dict:
    """Get cryptocurrency price in USD."""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ML Prediction Tool
@server.tool()
def predict_salary(years: float) -> dict:
    """Predict salary based on years of experience."""
    try:
        model = pickle.load(open("salary_model.pkl", "rb"))
        prediction = model.predict([[years]])
        return {
            "years_experience": years,
            "predicted_salary": float(prediction[0])
        }
    except Exception as e:
        return {"error": str(e)}

# Calculator Tool
@server.tool()
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

# File Tool
@server.tool()
def read_file(path: str) -> dict:
    """Read contents of a file."""
    try:
        with open(path, 'r') as f:
            content = f.read()
        return {
            "path": path,
            "content": content,
            "size": len(content)
        }
    except Exception as e:
        return {"error": str(e)}

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Connecting Agent to MCP Server

```python
from mcp.client import Client
import asyncio

async def use_mcp_tools():
    # Connect to MCP server
    client = Client("my-agent")
    await client.connect("stdio", command="python", args=["mcp_server.py"])
    
    # List available tools
    tools = await client.list_tools()
    print("Available tools:", [tool.name for tool in tools])
    
    # Call a tool
    result = await client.call_tool("get_weather", {"city": "Berlin"})
    print("Weather result:", result)
    
    # Call another tool
    result = await client.call_tool("predict_salary", {"years": 5})
    print("Salary prediction:", result)
    
    await client.close()

# Run
asyncio.run(use_mcp_tools())
```

### Integrating MCP with LangChain Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import OpenAI
from mcp.client import Client
import asyncio

class MCPToolWrapper:
    def __init__(self, mcp_client, tool_name):
        self.client = mcp_client
        self.tool_name = tool_name
    
    async def run(self, **kwargs):
        return await self.client.call_tool(self.tool_name, kwargs)

async def create_agent_with_mcp():
    # Connect to MCP server
    mcp_client = Client("langchain-agent")
    await mcp_client.connect("stdio", command="python", args=["mcp_server.py"])
    
    # Get available tools from MCP
    mcp_tools = await mcp_client.list_tools()
    
    # Convert MCP tools to LangChain tools
    langchain_tools = []
    for mcp_tool in mcp_tools:
        tool = Tool(
            name=mcp_tool.name,
            func=lambda **kwargs: asyncio.run(
                MCPToolWrapper(mcp_client, mcp_tool.name).run(**kwargs)
            ),
            description=mcp_tool.description
        )
        langchain_tools.append(tool)
    
    # Create agent
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        langchain_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Use agent
    response = agent.run("What's the weather in Berlin and predict salary for 5 years experience?")
    print(response)
    
    await mcp_client.close()

asyncio.run(create_agent_with_mcp())
```

---

## Best Practices

### 1. Error Handling

Always handle errors gracefully in tools.

```python
@server.tool()
def safe_tool(param: str) -> dict:
    """Tool with proper error handling."""
    try:
        # Tool logic
        result = process(param)
        return {"status": "success", "result": result}
    except ValueError as e:
        return {"status": "error", "error": "Invalid parameter", "details": str(e)}
    except Exception as e:
        return {"status": "error", "error": "Internal error", "details": str(e)}
```

### 2. Input Validation

Validate parameters before processing.

```python
@server.tool()
def validated_tool(age: int, name: str) -> dict:
    """Tool with input validation."""
    if age < 0 or age > 150:
        return {"error": "Age must be between 0 and 150"}
    
    if not name or len(name) < 2:
        return {"error": "Name must be at least 2 characters"}
    
    return {"status": "success", "message": f"{name} is {age} years old"}
```

### 3. Documentation

Provide clear descriptions for tools.

```python
@server.tool()
def well_documented_tool(city: str, units: str = "metric") -> dict:
    """
    Get weather information for a city.
    
    Args:
        city: Name of the city (e.g., "Berlin", "Tokyo")
        units: Temperature units - "metric" (Celsius) or "imperial" (Fahrenheit)
    
    Returns:
        Dictionary with weather data including temperature, condition, and humidity
    
    Example:
        get_weather("Berlin", "metric")
        Returns: {"city": "Berlin", "temperature": 22, "condition": "sunny"}
    """
    # Implementation
    pass
```

### 4. Security

Implement access control and rate limiting.

```python
from functools import wraps
import time

# Rate limiting
call_times = {}

def rate_limit(calls_per_minute=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if func.__name__ not in call_times:
                call_times[func.__name__] = []
            
            # Remove old calls
            call_times[func.__name__] = [
                t for t in call_times[func.__name__] 
                if now - t < 60
            ]
            
            if len(call_times[func.__name__]) >= calls_per_minute:
                return {"error": "Rate limit exceeded"}
            
            call_times[func.__name__].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@server.tool()
@rate_limit(calls_per_minute=10)
def rate_limited_tool(param: str) -> dict:
    """Tool with rate limiting."""
    return {"result": f"Processed {param}"}
```

---

## Hands-On Exercise

### Build Your Own MCP Server

Create an MCP server that exposes all your previous week's tools:

**Requirements:**
1. Weather API tool
2. Crypto price tool
3. Salary prediction tool (Week 2)
4. Iris classification tool (Week 2)
5. RAG search tool (Week 3)

**Implementation Steps:**
1. Create `agent_mcp_server.py`
2. Define all tools with proper error handling
3. Add input validation
4. Document each tool
5. Test with a client script

**Bonus:**
- Add logging for all tool calls
- Implement rate limiting
- Create a health check tool
- Add metrics collection

---

## Summary

MCP Server provides:
- **Standardization**: Common protocol for tool exposure
- **Modularity**: Independent, reusable tools
- **Scalability**: Distribute tools across servers
- **Maintainability**: Update tools without changing agents
- **Discoverability**: Agents can find available tools

This architecture enables building sophisticated agent systems that can leverage diverse tools and resources through a unified interface.

---

## Preparation

Before the next session:
1. Install MCP SDK: `pip install mcp`
2. Review async/await in Python
3. Test your existing tools (ML models, RAG system)
4. Read MCP documentation: https://modelcontextprotocol.io
5. Prepare questions about MCP implementation

---

## Looking Ahead

MCP Server skills will be essential as you build more complex agent systems. In upcoming weeks, you'll use MCP to create distributed agent architectures where multiple agents access shared tool servers, enabling true multi-agent collaboration at scale.
