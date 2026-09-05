from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
@app.get("/")
def home():
    return {"status":"LIVE"}
@app.get("/health")
def health():
    return {"backend":"LIVE","tiles":400}
@app.get("/tiles")
def get_tiles():
    tiles=[{"id":i,"row":i//20,"col":i%20,"ndvi":round(random.uniform(0.2,0.9),2),"status":random.choice(["healthy","stressed","critical"])} for i in range(400)]
    return {"total_tiles":400,"tiles":tiles}
@app.get("/analyze")
def analyze():
    return get_tiles()
