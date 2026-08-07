from fastapi import APIRouter

from llm_client.services.healthcheck import HealthCheck

router = APIRouter()


@router.get("/")
def check_health():
    health_check = HealthCheck()
    result = health_check.check()
    return result
