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

# We will be using the distilgpt2 model
MODEL_NAME = "distilgpt2"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

# Phi-3 officially uses pad_token_id = 32000, but we set it to eos for generation safety
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id   # usually 32000 anyway

print("Loading model...5")


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model = model.to("cpu")

print("Model loaded.")

class GenerateRequest(BaseModel):
    prompt: str
    # We will be only using 50 tokens for time being 
    max_tokens: int = 100

# API call for sending prompt and getting response
@app.post("/generate")
def generate_text(req: GenerateRequest):
    inputs = tokenizer(
        req.prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            pad_token_id=tokenizer.pad_token_id,      # already good
            eos_token_id=tokenizer.eos_token_id,      # helps stop properly
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}