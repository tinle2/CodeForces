from fastapi import FastAPI

app = FASTAPI()

@app.get("/")
async def root():
    return {"message": "Hello world"}