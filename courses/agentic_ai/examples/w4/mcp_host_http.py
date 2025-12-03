#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify
import google.generativeai as genai
from mcp_client import MCPClient

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = Flask(__name__)
client = MCPClient("http://localhost:8080")

class MCPHost:
    def __init__(self, mcp_client):
        self.client = mcp_client

        # Step-2: input to MCP Host: list of tools
        # Step-3: output from MCP Host: loaded tools
        self.tools = self._load_tools()

        # Initialize Gemini model with MCP tools
        # step-4 tools listed in the model
        self.model = genai.GenerativeModel(
            "gemini-3-pro-preview",
            tools=[self._convert_to_gemini_tool(t) for t in self.tools]
        )
    
    def _load_tools(self):
        response = self.client.list_tools()
        return response.get("tools", [])
    
    def _convert_to_gemini_tool(self, mcp_tool):
        return genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name=mcp_tool["name"],
                    description=mcp_tool["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            k: genai.protos.Schema(type=genai.protos.Type.STRING)
                            for k in mcp_tool["inputSchema"]["properties"]
                        },
                        required=mcp_tool["inputSchema"].get("required", [])
                    )
                )
            ]
        )
    
    def chat(self, user_message):
        chat = self.model.start_chat()

        # step-4 :  input to LLM: User context
        # step-5: output from LLM: api and payload
        response = chat.send_message(user_message)

        
        while response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                args = {}
                if fc.args:
                    for k in fc.args:
                        args[k] = fc.args[k]
                # Step-6 : input : Call MCP tool with api and payload
                # Step-7 : output : tool response
                result = self.client.call_tool(fc.name, args)
                tool_response = result["content"][0]["text"]
                
                # step-8: input to LLM: tool response
                # step-9: output from LLM: final response to user
                response = chat.send_message(
                    genai.protos.Content(parts=[
                        genai.protos.Part(function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response={"result": tool_response}
                        ))
                    ])
                )
            else:
                return part.text
        
        return "No response"

host = MCPHost(client)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message")
    if not message:
        return jsonify({"error": "message required"}), 400
    
    # step-1: input from user
    # step-10: final response to user
    response = host.chat(message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
