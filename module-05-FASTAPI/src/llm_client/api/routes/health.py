from fastapi import APIRouter

from llm_client.services.healthcheck import HealthCheck, HealthCheckResult

router = APIRouter()


@router.get("/")
def check_health() -> HealthCheckResult:
    health_check = HealthCheck()
    result = health_check.check()
    return result
