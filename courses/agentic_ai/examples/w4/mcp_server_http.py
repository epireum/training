#!/usr/bin/env python3
from flask import Flask, request, jsonify

app = Flask(__name__)

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "inputSchema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Calculate a math expression",
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]

def get_weather(location):
    return f"Weather in {location}: 72°F, Sunny"

def calculate(expression):
    return str(eval(expression))

@app.route('/', methods=['POST'])
def handle_request():
    data = request.json
    method = data.get("method")
    
    if method == "tools/list":
        return jsonify({"tools": TOOLS})
    
    if method == "tools/call":
        params = data.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        
        if name == "get_weather":
            return jsonify({"content": [{"type": "text", "text": get_weather(args["location"])}]})
        elif name == "calculate":
            return jsonify({"content": [{"type": "text", "text": calculate(args["expression"])}]})
    
    return jsonify({"error": "Unknown method"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
