import json
import os
from typing import Any, Literal, Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from prompt import (
    STAGE1_SYSTEM,
    STAGE1_USER_MESSAGE_FULL,
    STAGE2_SYSTEM,
    STAGE2_USER_MESSAGE_FULL,
)

from dotenv import load_dotenv

# This searches for a .env file and loads its variables into system environment variables
load_dotenv() 

# 1. Environment & Global Client Instantiation (Crucial for Connection Pooling)
MODEL_NAME = os.getenv("GEMINI_MODEL") # Enforce stable production flash model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

# Initialize once globally; client.aio will cleanly reuse underlying connection pools
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Vehicle Verification API")

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
