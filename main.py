
# Local AI Inference API
# - FastAPI
# - Hugging Face Transformers
# - CPU-based LLM serving



# Importing packages 
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()


#We will be using the phi-2 model
MODEL_NAME = "microsoft/phi-2"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
print("Model loaded.")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50



#API call for sending prompt and getting response
@app.post("/generate")
def generate_text(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}   