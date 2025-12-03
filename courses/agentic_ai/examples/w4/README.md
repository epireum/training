# MCP Server Example

A minimal Model Context Protocol server that provides sample tools.

## Usage

### Terminal 1 (MCP Server)
```bash
python mcp_server_http.py

```


### Terminal 2 (MCP HOST)
```bash
export GEMINI_API_KEY=your-api-key
python mcp_host_http.py

```

### Test Remove Agent
```bash
# Test from remote agent
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"method":"tools/list"}'

curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"method":"tools/call","params":{"name":"get_weather","arguments":{"location":"Seattle"}}}'
```


### Call MCP host (HTTP)
```bash
# Start HTTP server
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the weather in Tokyo?"}'


## Run call.http
```


## Tools Available

- `get_weather`: Returns mock weather data for a location
- `calculate`: Evaluates a mathematical expression
