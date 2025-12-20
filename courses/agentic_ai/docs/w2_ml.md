# Week 2: Machine Learning Essentials
[back](../README.md)
## Student Reading Material

### Introduction

Welcome to Week 2 of the Agentic AI program. This week bridges the gap between basic Python programming and intelligent agent behavior by introducing machine learning fundamentals. You'll learn how to train models that enable agents to make predictions, classify data, and reason about the world.

### What is Machine Learning?

Machine learning (ML) is a subset of artificial intelligence where algorithms learn patterns from data rather than following explicitly programmed rules. Instead of telling a computer exactly what to do in every situation, you provide examples and let it discover the patterns.

**Key Characteristics:**
- **Learning from data**: Models improve their performance by analyzing examples
- **Predictive intelligence**: Trained models can make predictions on new, unseen data
- **Agent reasoning**: ML enables agents to rank options, prioritize tasks, and make informed decisions

For AI agents, machine learning is crucial. An agent might use ML to:
- Predict the best course of action based on past outcomes
- Classify incoming requests to route them appropriately
- Rank search results by relevance
- Estimate resource requirements for tasks

### Learning Outcomes

By the end of this week, you will be able to:
- Understand the fundamentals of machine learning and its role in AI agents
- Implement linear regression for continuous predictions
- Follow the complete ML workflow from data loading to model deployment
- Save and load trained models using pickle
- Create reusable ML tools that agents can invoke
- Integrate ML predictions into agent decision-making

---

## Core Concepts

### Two Essential ML Models

This week focuses on two foundational machine learning models that cover the most common prediction scenarios:

**1. Linear Regression**  
Used for predicting continuous numerical values. Examples include predicting salaries, temperatures, prices, or any quantity that can take a range of values.

**2. Classification (Logistic Regression)**  
Used for predicting categories or classes. Examples include determining flower species, classifying emails as spam or not spam, or identifying customer segments.

These two models form the foundation for understanding more complex ML techniques you'll encounter later.

### The Machine Learning Workflow

Every ML project follows a standard workflow. Understanding this process is essential for building reliable agent tools:

**1. Load Data**  
Import your dataset, typically from CSV files or databases. The data contains examples with known inputs and outputs.

```python
import pandas as pd
data = pd.read_csv('Salary_Data.csv')
```

**2. Split Data (Train/Test)**  
Divide your data into two sets:
- **Training set**: Used to teach the model (typically 70-80% of data)
- **Test set**: Used to evaluate model performance on unseen data (typically 20-30%)

This split prevents overfitting and ensures your model generalizes well to new situations.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**3. Train Model**  
Feed the training data to the ML algorithm, which learns patterns and relationships between inputs and outputs.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

**4. Evaluate**  
Test the model on the test set to measure its accuracy and reliability. This step reveals how well the model will perform in real-world agent scenarios.

```python
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")
```

**5. Save Model**  
Persist the trained model to disk so it can be loaded and used without retraining.

```python
import pickle
pickle.dump(model, open("salary_model.pkl", "wb"))
```

**6. Load and Use in Agent**  
Import the saved model into your agent's tools, enabling it to make predictions on demand.

**Complete Workflow Example:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load Data
data = pd.read_csv('Salary_Data.csv')
X = data[['YearsExperience']]
y = data['Salary']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")

# 5. Save Model
pickle.dump(model, open("salary_model.pkl", "wb"))

# 6. Load and Use
loaded_model = pickle.load(open("salary_model.pkl", "rb"))
prediction = loaded_model.predict([[5]])
print(f"Predicted salary for 5 years experience: {prediction[0]}")
```

---

## Practical Applications

### Linear Regression: Salary Prediction

Linear regression finds the best-fit line through data points to predict continuous values.

**Use Case**: Predict salary based on years of experience

This type of model helps agents make numerical estimates. For example, an HR agent might use salary prediction to:
- Suggest competitive compensation packages
- Estimate budget requirements for new hires
- Analyze salary trends across departments

**How it works:**  
The model learns the relationship between experience (input) and salary (output) from historical data. Once trained, it can predict salaries for any experience level.

**Code Example:**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load salary data
data = pd.read_csv('Salary_Data.csv')
X = data[['YearsExperience']]
y = data['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict([[5]])  # Predict salary for 5 years experience
print(f"Predicted salary: ${prediction[0]:,.2f}")
```

---

## Model Persistence with Pickle

Training ML models can be time-consuming and resource-intensive. The `pickle` module allows you to save trained models and load them instantly when needed.

### Saving a Model

```python
import pickle

# After training your model
pickle.dump(model, open("model.pkl", "wb"))
```

This serializes the model object and writes it to a file. The `"wb"` mode means "write binary."

### Loading a Model

```python
model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict([[5.4, 3.1, 1.7, 0.2]])
```

This deserializes the model from the file, making it ready to use immediately. The `"rb"` mode means "read binary."

**Why this matters for agents:**  
Agents need to respond quickly. Loading a pre-trained model takes milliseconds, while training from scratch could take minutes or hours. This enables real-time agent responses.

---

## Building ML Tools for Agents

The goal this week is to create reusable functions that wrap your ML models, making them easy for agents to invoke.

### Creating ml_tools.py

You'll build a module containing functions like:

```python
def predict_salary(years):
    """
    Predict salary based on years of experience.
    
    Args:
        years: Number of years of experience
        
    Returns:
        Predicted salary as a float
    """
    model = pickle.load(open("salary_model.pkl", "rb"))
    prediction = model.predict([[years]])
    return prediction[0]

def predict_flower(specs):
    """
    Predict iris flower species based on measurements.
    
    Args:
        specs: List of [sepal_length, sepal_width, petal_length, petal_width]
        
    Returns:
        Predicted species name as a string
    """
    model = pickle.load(open("iris_model.pkl", "rb"))
    prediction = model.predict([specs])
    return prediction[0]
```

These functions encapsulate the complexity of loading models and making predictions, providing a clean interface for agents to use.

---

## Hands-On Practice

### Sunday Workshop Activities

During the hands-on session, you'll:

1. **Train a Salary Predictor**
   - Load the Salary_Data.csv dataset
   - Build and train a linear regression model
   - Evaluate its accuracy
   - Save the model as salary_model.pkl


2. **Create ml_tools.py**
   - Implement `predict_salary(years)` function
   - Test the functions with sample inputs



---

## Assignments

### Assignment 1: Train & Save a Regression Model

**Objective**: Build a salary prediction model

**Requirements:**
- Use the provided Salary_Data.csv dataset
- Train a linear regression model
- Save the trained model as salary_model.pkl
- Document your model's accuracy metrics

### Assignment 2: Build ML Tools

**Objective**: Create reusable prediction functions

**Requirements:**
- Create ml_tools.py module
- Implement `predict_salary(years)` function
- Include proper error handling and documentation

### Assignment 3: Test with CLI

**Objective**: Integrate ML tools into a command-line interface

**Requirements:**
- Extend your Week 1 CLI application
- Add menu options for salary prediction
- Add menu options for iris classification
- Demonstrate the tools working end-to-end

---

## Connecting to Agents

This week's skills directly enable agent intelligence:

- **Prediction capabilities**: Agents can forecast outcomes and make data-driven decisions
- **Classification abilities**: Agents can categorize inputs and route them appropriately
- **Tool creation**: You're building the functions that agents will invoke
- **Model persistence**: Agents can access pre-trained intelligence instantly

In upcoming weeks, you'll connect these ML tools to agent frameworks like LangChain and CrewAI, creating agents that leverage machine learning for sophisticated reasoning and decision-making.

---

## Preparation for Class

Before attending class, please:
1. Review Week 1 Python fundamentals, especially functions and modules
2. Install scikit-learn: `pip install scikit-learn`
3. Install pandas: `pip install pandas`
4. Ensure you have the Salary_Data.csv file (will be provided)
5. Read through this material and prepare questions

### Recommended Pre-Reading

Familiarize yourself with these concepts:
- What is supervised learning?
- The difference between regression and classification
- How to evaluate model performance (accuracy, error metrics)

---

## Looking Ahead

Week 2 adds intelligence to your agent toolkit. In Week 3, you'll explore more advanced ML techniques and begin integrating these capabilities into full-fledged agent systems that can learn, adapt, and make autonomous decisions.
