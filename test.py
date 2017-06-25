import json,requests
from flask import Flask , abort , jsonify,request

url="http://127.0.0.1:9001/api?id=50"

r=requests.post(url)
print(r)