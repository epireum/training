#!/usr/bin/env python3
import requests

class MCPClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
    
    def call(self, method, params=None):
        request = {"method": method}
        if params:
            request["params"] = params
        
        response = requests.post(self.server_url, json=request)
        return response.json()
    
    def list_tools(self):
        return self.call("tools/list")
    
    def call_tool(self, name, arguments):
        return self.call("tools/call", {"name": name, "arguments": arguments})
    
    def close(self):
        pass
