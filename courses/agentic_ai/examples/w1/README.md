1. GET /: Returns a simple hello message
2. GET /items/{item_id}: Shows path parameters and query parameters
3. POST /users: Demonstrates POST request with body parameters

To run it:
bash
pip install fastapi uvicorn
uvicorn fastapi_example:app --reload


Then test:
bash
curl http://localhost:8000/
curl http://localhost:8000/items/5?q=test
curl -X POST "http://localhost:8000/users?name=John&age=30"


Students can also visit http://localhost:8000/docs to see the interactive API documentation.