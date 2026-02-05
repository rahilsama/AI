import sys
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

MODEL_NAME = "distilgpt2"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)
model = model.to("cpu")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int

@app.post("/generate")
def generate_text(req: GenerateRequest):
    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to("cpu")
    output = model.generate(input_ids, max_length=req.max_length)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": response}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Add test cases
if __name__ == '__main__':
    import uvicorn
    from fastapi.testclient import TestClient

    client = TestClient(app)

    def test_health():
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_generate_text():
        response = client.post(
            "/generate",
            json={"prompt": "Once upon a time", "max_length": 50}
        )
        assert response.status_code == 200
        generated_text = response.json()["generated_text"]
        assert len(generated_text) > 0

    test_health()
    test_generate_text()
    uvicorn.run(app, host="127.0.0.1", port=8000)
