import httpx
from fastapi import APIRouter, HTTPException

from app.models import VerificationRequest, MIAResponse

router = APIRouter(prefix="/mia", tags=["Membership Inference Attack"])

MIA_EXTERNAL_URL = "https://postlabially-overinstructive-aurore.ngrok-free.dev/mia"

@router.post(
    "",
    response_model=MIAResponse,
    summary="Run Membership Inference Attack",
    description="Tunnels the request to an external ROME Model Editing REST API to evaluate if the given text is a likely training set member."
)
async def run_mia(request: VerificationRequest) -> MIAResponse:
    """
    Tunnels the MIA request to the external ROME model editing API.
    """
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "1"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            rome_payload = {
                "prompt": ".",  # ROME API requires non-blank string
                "target_text": request.secret
            }
            
            response = await client.post(
                MIA_EXTERNAL_URL,
                json=rome_payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Map MIA prediction to forgetting status
            prediction = data.get("prediction")
            if prediction == "likely_non_member":
                data["forgetting_status"] = "FORGOTTEN"
            elif prediction == "likely_member":
                data["forgetting_status"] = "NOT_FORGOTTEN"
            else:
                data["forgetting_status"] = "UNKNOWN"
                
            return MIAResponse(**data)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, 
            detail=f"Error connecting to external MIA service: {e}"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code, 
            detail=f"External MIA service returned an error: {e.response.text}"
        )
