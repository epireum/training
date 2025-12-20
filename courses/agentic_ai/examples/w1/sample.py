# Calling APIs
#%pip install requests
import json
import requests
response = requests.get("https://pokeapi.co/api/v2/pokemon/pikachu")
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data["abilities"]))
else:
    print("Error:", response.status_code)