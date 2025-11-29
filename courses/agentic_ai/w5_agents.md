# Week 4: Agent Architecture
## Student Reading Material

### Introduction

Welcome to Week 4 of the Agentic AI program. This week brings together everything you've learned—Python fundamentals, machine learning, and RAG—into a unified agent architecture. You'll build autonomous agents that can think, plan, act, and learn from their interactions with the world.

### What is an AI Agent?

An AI agent is more than just a chatbot or a single-purpose tool. It's an autonomous system that can:

**Core Components:**
1. **LLM (Large Language Model)**: The "brain" that provides reasoning and language understanding
2. **Tools**: Functions the agent can invoke to interact with the world
3. **Multi-step reasoning**: Ability to break down complex tasks into steps
4. **Memory**: Capacity to remember past interactions and context
5. **Autonomy**: Self-directed decision-making without constant human guidance

**What Makes It "Agentic":**

Unlike traditional software that follows predetermined paths, agents:
- Decide which tools to use based on the task
- Plan sequences of actions to achieve goals
- Adapt their approach based on results
- Learn from experience through memory

Think of an agent as a digital assistant that can actually get things done, not just answer questions.

### Learning Outcomes

By the end of this week, you will be able to:
- Understand the agent reasoning loop (Think-Plan-Act-Observe)
- Design and implement multi-tool agent architectures
- Integrate diverse tools (APIs, ML models, RAG systems)
- Implement agent memory systems (short-term and long-term)
- Build agents that autonomously select and use appropriate tools
- Debug and optimize agent decision-making
- Create production-ready agent applications

---

## The Agent Loop

Agents operate in a continuous cycle of reasoning and action:

### 1. Think

The agent analyzes the user's request and its current context.

```python
# Agent receives input
user_query = "What's the weather in Berlin and should I bring an umbrella?"

# Agent thinks about the query
# - This requires weather information
# - Need to check current conditions
# - Need to reason about umbrella necessity
```

### 2. Plan

The agent determines which tools to use and in what order.

```python
# Agent creates a plan
plan = [
    "Use weather_tool to get Berlin weather",
    "Analyze precipitation data",
    "Provide recommendation about umbrella"
]
```

### 3. Act

The agent executes the planned actions by calling tools.

```python
# Agent acts
weather_data = weather_tool("Berlin")
# Returns: {"temp": 15, "condition": "rainy", "precipitation": 80}
```

### 4. Observe

The agent examines the results of its actions.

```python
# Agent observes
# - Temperature is 15°C
# - Condition is rainy
# - 80% chance of precipitation
# - Umbrella is definitely needed
```

### 5. Repeat

If the goal isn't achieved, the agent loops back to thinking and planning.

```python
# Agent formulates response
response = "The weather in Berlin is rainy with 15°C and 80% chance of precipitation. Yes, you should definitely bring an umbrella."
```

**Complete Agent Loop Example:**

```python
class Agent:
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
        self.memory = []
    
    def run(self, user_query):
        # Think
        context = self._build_context(user_query)
        
        # Plan
        plan = self.llm.create_plan(context, self.tools)
        
        # Act & Observe
        results = []
        for action in plan:
            tool_name = action["tool"]
            tool_input = action["input"]
            result = self.tools[tool_name](tool_input)
            results.append(result)
        
        # Generate response
        response = self.llm.synthesize(user_query, results)
        
        # Update memory
        self.memory.append({"query": user_query, "response": response})
        
        return response
```

---

## Agent Tools

Tools are the agent's interface to the world. Each tool is a Python function that performs a specific task.

### Tool Categories

**1. API Tools**

Connect to external services for real-time data.

```python
import requests

def weather_tool(city):
    """Get current weather for a city."""
    api_key = "your_api_key"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return {
        "city": city,
        "temp": data["main"]["temp"],
        "condition": data["weather"][0]["main"]
    }

def crypto_tool(symbol):
    """Get cryptocurrency price."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    response = requests.get(url)
    return response.json()

def pokemon_tool(name):
    """Get Pokemon information."""
    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    response = requests.get(url)
    data = response.json()
    return {
        "name": data["name"],
        "height": data["height"],
        "weight": data["weight"],
        "types": [t["type"]["name"] for t in data["types"]]
    }
```

**2. ML Model Tools**

Use trained models for predictions (from Week 2).

```python
import pickle

def salary_predictor_tool(years):
    """Predict salary based on years of experience."""
    model = pickle.load(open("salary_model.pkl", "rb"))
    prediction = model.predict([[years]])
    return {"years": years, "predicted_salary": prediction[0]}

def iris_classifier_tool(measurements):
    """Classify iris flower species."""
    model = pickle.load(open("iris_model.pkl", "rb"))
    prediction = model.predict([measurements])
    species = ["setosa", "versicolor", "virginica"]
    return {"species": species[prediction[0]]}
```

**3. RAG Tools**

Query knowledge bases (from Week 3).

```python
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def rag_tool(question):
    """Answer questions using document knowledge base."""
    vectorstore = Chroma(
        persist_directory="./rag_vectorstore",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain({"query": question})
    return result["result"]
```

### Tool Registration

Agents need to know what tools are available and how to use them:

```python
tools = {
    "weather": {
        "function": weather_tool,
        "description": "Get current weather for a city",
        "parameters": {"city": "string"}
    },
    "crypto": {
        "function": crypto_tool,
        "description": "Get cryptocurrency price in USD",
        "parameters": {"symbol": "string"}
    },
    "salary_predictor": {
        "function": salary_predictor_tool,
        "description": "Predict salary based on years of experience",
        "parameters": {"years": "number"}
    },
    "rag": {
        "function": rag_tool,
        "description": "Answer questions from document knowledge base",
        "parameters": {"question": "string"}
    }
}
```

---

## Agent Memory

Memory enables agents to maintain context and learn from interactions.

### Memory Types

**1. Short-Term Memory**

Stores the current conversation context.

```python
class ShortTermMemory:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_context(self):
        return self.messages
```

**Usage:**

```python
memory = ShortTermMemory(max_messages=10)
memory.add("user", "What's the weather in Berlin?")
memory.add("assistant", "The weather in Berlin is sunny, 22°C")
memory.add("user", "What about Paris?")
# Agent can reference "Berlin" from previous context
```

**2. Long-Term Memory (Vector Store)**

Stores historical interactions for semantic retrieval.

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class LongTermMemory:
    def __init__(self, persist_directory="./agent_memory"):
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def store(self, interaction):
        """Store a conversation interaction."""
        text = f"User: {interaction['query']}\nAgent: {interaction['response']}"
        self.vectorstore.add_texts([text])
    
    def recall(self, query, k=3):
        """Retrieve relevant past interactions."""
        return self.vectorstore.similarity_search(query, k=k)
```

**3. Task Memory**

Tracks the current task's progress and intermediate results.

```python
class TaskMemory:
    def __init__(self):
        self.current_task = None
        self.steps_completed = []
        self.intermediate_results = {}
    
    def start_task(self, task_description):
        self.current_task = task_description
        self.steps_completed = []
        self.intermediate_results = {}
    
    def record_step(self, step, result):
        self.steps_completed.append(step)
        self.intermediate_results[step] = result
    
    def get_progress(self):
        return {
            "task": self.current_task,
            "completed": self.steps_completed,
            "results": self.intermediate_results
        }
```

---

## Building a Multi-Tool Agent

Here's a complete agent that integrates multiple tools:

```python
from openai import OpenAI
import json

class MultiToolAgent:
    def __init__(self, tools):
        self.tools = tools
        self.client = OpenAI()
        self.memory = ShortTermMemory()
    
    def run(self, user_query):
        # Add user query to memory
        self.memory.add("user", user_query)
        
        # Build tool descriptions for LLM
        tool_descriptions = self._format_tools()
        
        # Think & Plan: Ask LLM which tool to use
        messages = [
            {"role": "system", "content": f"You are an AI agent with access to these tools:\n{tool_descriptions}\nSelect the appropriate tool and parameters."},
            *self.memory.get_context()
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=self._get_function_schemas()
        )
        
        # Act: Execute tool if LLM chose one
        if response.choices[0].message.function_call:
            function_name = response.choices[0].message.function_call.name
            arguments = json.loads(response.choices[0].message.function_call.arguments)
            
            # Call the tool
            result = self.tools[function_name]["function"](**arguments)
            
            # Observe: Get final response with tool result
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(result)
            })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            answer = final_response.choices[0].message.content
        else:
            answer = response.choices[0].message.content
        
        # Update memory
        self.memory.add("assistant", answer)
        
        return answer
    
    def _format_tools(self):
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool['description']}")
        return "\n".join(descriptions)
    
    def _get_function_schemas(self):
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "name": name,
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool["parameters"]
                }
            })
        return schemas

# Initialize agent with all tools
agent = MultiToolAgent({
    "weather": {
        "function": weather_tool,
        "description": "Get current weather for a city",
        "parameters": {"city": {"type": "string"}}
    },
    "crypto": {
        "function": crypto_tool,
        "description": "Get cryptocurrency price",
        "parameters": {"symbol": {"type": "string"}}
    },
    "salary_predictor": {
        "function": salary_predictor_tool,
        "description": "Predict salary based on experience",
        "parameters": {"years": {"type": "number"}}
    }
})

# Use the agent
response = agent.run("What's the weather in Berlin?")
print(response)
```

---

## Hands-On Practice

### Saturday Workshop Activities

During the hands-on session, you'll build a comprehensive multi-tool agent:

**Tools to Integrate:**
1. **Weather Tool**: Real-time weather data
2. **Crypto Tool**: Cryptocurrency prices
3. **Pokémon Tool**: Pokémon information
4. **Salary ML Model**: Predictions from Week 2
5. **Iris ML Model**: Classifications from Week 2
6. **RAG Bot**: Document Q&A from Week 3

**Agent Capabilities:**

Your agent should handle queries like:
- "What's the weather in Tokyo?"
- "What's the current price of Bitcoin?"
- "Tell me about Pikachu"
- "What salary can I expect with 7 years of experience?"
- "Classify this iris: 5.1, 3.5, 1.4, 0.2"
- "What does the document say about machine learning?"

---

## Assignments

### Assignment 1: Build a Working Agent

**Objective**: Create an autonomous agent that selects and uses appropriate tools

**Requirements:**
- Implement an agent that can handle diverse user queries
- Agent must autonomously select the correct tool based on the query
- Integrate at least 4 different tools
- Provide combined reasoning in responses (not just raw tool output)
- Handle cases where no tool is appropriate

**Example Interactions:**

```
User: "What's the weather in Paris and the price of Ethereum?"
Agent: [Uses weather_tool and crypto_tool]
Response: "The weather in Paris is currently sunny at 18°C. Ethereum is trading at $2,450 USD."

User: "If I have 5 years of experience, what salary should I expect?"
Agent: [Uses salary_predictor_tool]
Response: "Based on the model, with 5 years of experience, you can expect a salary of approximately $75,000."
```

**Deliverables:**
- `agent.py` with complete agent implementation
- Tool integration for all required tools
- Test script demonstrating various queries
- Documentation of agent architecture

### Assignment 2: Add Memory

**Objective**: Implement conversation memory for context awareness

**Requirements:**
- Store the previous 10 conversations
- Agent should reference past interactions when relevant
- Implement both short-term memory (current session) and persistent storage
- Handle memory retrieval efficiently

**Example with Memory:**

```
User: "What's the weather in Berlin?"
Agent: "The weather in Berlin is rainy, 15°C."

User: "What about the same city's crypto scene?"
Agent: [Remembers "Berlin" from previous query]
Response: "I can check cryptocurrency prices, but they're global, not city-specific. Would you like to know about a specific cryptocurrency?"
```

**Deliverables:**
- Enhanced `agent.py` with memory implementation
- Memory persistence across sessions
- Demonstration of context-aware responses
- Analysis of how memory improves agent performance

---

## Agent Design Patterns

### Pattern 1: ReAct (Reasoning + Acting)

Agent alternates between reasoning and action:

```python
def react_loop(query):
    thought = llm.think(query)
    action = llm.decide_action(thought)
    observation = execute_tool(action)
    
    if task_complete(observation):
        return llm.synthesize(query, observation)
    else:
        return react_loop(query)  # Continue loop
```

### Pattern 2: Plan-and-Execute

Agent creates a complete plan before executing:

```python
def plan_and_execute(query):
    plan = llm.create_plan(query, available_tools)
    results = []
    
    for step in plan:
        result = execute_tool(step)
        results.append(result)
    
    return llm.synthesize(query, results)
```

### Pattern 3: Reflexion

Agent reflects on its actions and improves:

```python
def reflexion_loop(query):
    response = agent.run(query)
    evaluation = llm.evaluate(query, response)
    
    if evaluation["quality"] < threshold:
        feedback = llm.generate_feedback(evaluation)
        return agent.run(query, feedback)
    
    return response
```

---

## Debugging Agents

Common issues and solutions:

**Issue 1: Agent Chooses Wrong Tool**

```python
# Add logging
def run(self, query):
    print(f"Query: {query}")
    tool_choice = self.select_tool(query)
    print(f"Selected tool: {tool_choice}")
    # Continue...
```

**Issue 2: Infinite Loops**

```python
# Add iteration limit
def run(self, query, max_iterations=5):
    for i in range(max_iterations):
        if task_complete():
            break
        # Continue loop
```

**Issue 3: Tool Errors**

```python
# Add error handling
def execute_tool(self, tool_name, params):
    try:
        return self.tools[tool_name](**params)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}
```

---

## Best Practices

1. **Clear Tool Descriptions**: Help the LLM understand when to use each tool
2. **Error Handling**: Tools should gracefully handle failures
3. **Logging**: Track agent decisions for debugging
4. **Memory Management**: Don't let memory grow unbounded
5. **Cost Awareness**: Monitor LLM API calls
6. **Testing**: Test with diverse queries
7. **User Feedback**: Allow users to correct agent mistakes

---

## Connecting to Production

Considerations for production agents:

- **Scalability**: Handle multiple concurrent users
- **Reliability**: Implement retries and fallbacks
- **Security**: Validate tool inputs, protect API keys
- **Monitoring**: Track performance and errors
- **Cost Management**: Optimize LLM calls
- **User Experience**: Provide progress indicators for long tasks

---

## Preparation for Class

Before attending class, please:
1. Review Weeks 1-3 materials (Python, ML, RAG)
2. Ensure all previous tools are working (ML models, RAG system)
3. Install OpenAI library: `pip install openai`
4. Obtain an OpenAI API key
5. Test your API tools (weather, crypto, Pokémon)
6. Read through this material and prepare questions

### Pre-Class Exercise

Try designing an agent on paper:
- What tools would it need?
- How would it decide which tool to use?
- What memory would be helpful?
- What could go wrong?

---

## Looking Ahead

Week 4 brings together all your skills into autonomous agents. In Week 5, you'll explore advanced agent frameworks like LangChain and CrewAI, learning how to build multi-agent systems where specialized agents collaborate to solve complex problems.
