import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx


app = FastAPI(title="Local AI API Gateway")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


class GenerateResponse(BaseModel):
    response: str


# URL of the inference service inside the Kubernetes cluster
INFERENCE_URL = os.getenv(
    "INFERENCE_URL",
    "http://local-ai-inference-service:8000/infer",
)


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    """
    Public API endpoint. Forwards the request to the internal inference service.
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                INFERENCE_URL,
                json={"prompt": req.prompt, "max_tokens": req.max_tokens},
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Inference service unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Inference service error: {resp.status_code} {resp.text}",
        )

    data = resp.json()
    # Expecting {"response": "..."} from inference service
    if "response" not in data:
        raise HTTPException(status_code=502, detail="Invalid response from inference service")

    return GenerateResponse(response=data["response"])


@app.get("/health")
def health():
    return {"status": "ok"}


