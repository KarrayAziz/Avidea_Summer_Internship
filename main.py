import asyncio
import json
import os
from functools import lru_cache
from typing import Any, Literal, Optional, Union

from dotenv.main import logger

from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

import subprocess
import sys
import tempfile
import time
import logging
import re 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llava")

from models.embeddings.siglip2 import SigLIP2EmbeddingModel
from models.tasks.view_classifier import ViewClassifier
from models.utils.image import UnsupportedImageError, load_image_from_bytes

from prompt import (
    STAGE1_SYSTEM,
    STAGE1_USER_MESSAGE_FULL,
    STAGE2_SYSTEM,
    STAGE2_USER_MESSAGE_FULL,
    LOCAL_STAGE1_PROMPT,
)

from dotenv import load_dotenv

# This searches for a .env file and loads its variables into system environment variables
load_dotenv() 

#LLAMA VARIABELS

LLAVA_MODEL_PATH = "/home/aziz/Aziz/DigiCover/usingGeminiApi/llava-v1.6-mistral-7b.Q4_K_M.gguf"
LLAVA_MMPROJ_PATH = "/home/aziz/Aziz/DigiCover/usingGeminiApi/mmproj-model-f16.gguf"
LLAMA_CPP_BIN = "/home/aziz/.local/share/applications/llama.cpp/build/bin/llama-mtmd-cli"


# 1. Environment & Global Client Instantiation (Crucial for Connection Pooling)
MODEL_NAME = os.getenv("GEMINI_MODEL") # Enforce stable production flash model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

# Initialize once globally; client.aio will cleanly reuse underlying connection pools
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Vehicle Verification API")

def extract_json(text: str):

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    return match.group(0)

def run_llama(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []

    for line in process.stdout:
        print(line, end="")   # 👈 THIS prints live in your terminal
        sys.stdout.flush()
        output_lines.append(line)

    process.wait()

    return process.returncode, "".join(output_lines)


async def call_local_llava(image: UploadFile) -> dict:

    suffix = os.path.splitext(image.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        image_bytes = await image.read()
        tmp.write(image_bytes)
        temp_path = tmp.name

    try:
        command = [
            LLAMA_CPP_BIN,
            "-m", LLAVA_MODEL_PATH,
            "--mmproj", LLAVA_MMPROJ_PATH,
            "--image", temp_path,
            "-p", LOCAL_STAGE1_PROMPT,
            "-n", "5",
        ]

        start_time = time.time()

        # 🔥 STREAMING RUN (fix: real-time logs like terminal)
        def run_llama(cmd):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []

            for line in process.stdout:
                print(line, end="")  # 👈 live logs in terminal
                output_lines.append(line)

            process.wait()

            return process.returncode, "".join(output_lines)

        returncode, raw_output = await asyncio.to_thread(run_llama, command)

        end_time = time.time()

        logger.info(f"LLaVA execution time: {end_time - start_time:.2f} seconds")

        if returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=raw_output
            )

        # 🔥 JSON extraction
        json_str = extract_json(raw_output)

        if not json_str:
            raise HTTPException(
                status_code=500,
                detail=f"No valid JSON found in LLaVA output: {raw_output}"
            )

        return {
            **json.loads(json_str),
            "latency_sec": end_time - start_time
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

class LocalAuthenticityResponse(BaseModel):
    is_real: bool
    latency_sec: float

class ViewClassifierResponse(BaseModel):
    prediction: Literal["front", "rear", "left", "right", "null"]
    confidence: float
    scores: dict[str, float]

# ==========================================
# Pydantic Schemas (Kept clean as Codex designed)
# ==========================================
class Stage1ImageResult(BaseModel):
    index: int
    is_real: bool
    is_car: bool
    matches_group: bool
    mismatch_reasons: list[str]

class Stage1Response(BaseModel):
    status: Literal["PASS", "FAIL"]
    errors: list[str]
    same_vehicle: bool
    mismatch_reasons: list[str]
    images: list[Stage1ImageResult]

class MissingView(BaseModel):
    view: Literal["front", "rear", "left", "right"]
    status: Literal["missing", "incomplete"]

class Stage2ImageResult(BaseModel):
    index: int
    view: Optional[Literal["front", "rear", "left", "right"]] = None
    complete: Optional[bool] = None
    plate_number: Optional[str] = None

class Stage2Response(BaseModel):
    status: Literal["PASS", "FAIL"]
    errors: list[str]
    missing_views: list[MissingView]
    images: list[Stage2ImageResult]
    plate_number: Optional[str] = None

class CombinedResponse(BaseModel):
    status: Literal["PASS", "FAIL"]
    stage1: Stage1Response
    stage2: Stage2Response

VerifyVehicleResponse = Union[Stage1Response, CombinedResponse]

# ==========================================
# Helpers
# ==========================================
async def read_uploaded_images(images: list[UploadFile]) -> list[types.Part]:
    if len(images) != 4:
        raise HTTPException(
            status_code=400,
            detail="Exactly 4 image files are required in the 'images' form field.",
        )

    parts: list[types.Part] = []
    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{image.filename}' must be an image.",
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File '{image.filename}' is empty.",
            )

        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image.content_type))
    return parts

async def call_gemini_json(
    *,
    system_instruction: str,
    user_message: str,
    image_parts: list[types.Part],
) -> dict[str, Any]:
    try:
        # Reusing the global client's async implementation
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=[user_message, *image_parts],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API request failed: {exc}",
        ) from exc

    try:
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return json.loads(response.text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned invalid JSON: {exc}",
        ) from exc

def validate_gemini_payload(model: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    try:
        # Simplified to clean Pydantic v2 validation method
        return model.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini JSON did not match the expected schema: {exc}",
        ) from exc

@lru_cache(maxsize=1)
def get_view_classifier() -> ViewClassifier:
    embedding_model = SigLIP2EmbeddingModel(
        model_id=os.getenv("SIGLIP2_MODEL", "google/siglip2-base-patch16-224"),
        device=os.getenv("VIEW_CLASSIFIER_DEVICE") or None,
    )
    classifier = ViewClassifier(
        embedding_model=embedding_model,
        prompt_set=os.getenv("VIEW_CLASSIFIER_PROMPT_SET", "prompt_set_1"),
        null_margin=float(os.getenv("VIEW_CLASSIFIER_NULL_MARGIN", "0.5")),
    )
    classifier.warmup()
    return classifier


def classify_view_image(image_bytes: bytes) -> dict[str, Any]:
    image = load_image_from_bytes(image_bytes)
    return get_view_classifier().classify(image).to_dict()

# ==========================================
# Endpoints
# ==========================================
@app.post("/verify-vehicle", response_model=VerifyVehicleResponse)
async def verify_vehicle(images: list[UploadFile] = File(...)) -> VerifyVehicleResponse:
    # 1. Parse and validate uploaded binary files
    image_parts = await read_uploaded_images(images)

    # 2. Run and validate Stage 1
    stage1_payload = await call_gemini_json(
        system_instruction=STAGE1_SYSTEM,
        user_message=STAGE1_USER_MESSAGE_FULL,
        image_parts=image_parts,
    )   
    stage1 = validate_gemini_payload(Stage1Response, stage1_payload)

    # Short-circuit if fraud/mismatch is found
    if stage1.status == "FAIL":
        return stage1

    # 3. Run and validate Stage 2
    stage2_payload = await call_gemini_json(
        system_instruction=STAGE2_SYSTEM,
        user_message=STAGE2_USER_MESSAGE_FULL,
        image_parts=image_parts,
    )
    stage2 = validate_gemini_payload(Stage2Response, stage2_payload)

    # Return unified payload if everything passes
    return CombinedResponse(status=stage2.status, stage1=stage1, stage2=stage2)

@app.post("/test-local-llava-authenticity")
async def test_local_llava_authenticity(
    image: UploadFile = File(...)
):

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    payload = await call_local_llava(image)

    validated = validate_gemini_payload(
        LocalAuthenticityResponse,
        payload
    )

    return validated

@app.post("/view-classifier", response_model=ViewClassifierResponse)
async def view_classifier(image: UploadFile = File(...)) -> ViewClassifierResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await image.read()
    try:
        payload = await asyncio.to_thread(classify_view_image, image_bytes)
    except UnsupportedImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ViewClassifierResponse.model_validate(payload)
