from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


app = FastAPI(title="Local AI Inference Service")


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


MODEL_NAME = "distilgpt2"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Set pad token for safe generation
tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)
model = model.to("cpu")

print("Model loaded.")


@app.post("/infer")
def infer(req: InferenceRequest):
    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}


@app.get("/health")
def health():
    return {"status": "ok"}


