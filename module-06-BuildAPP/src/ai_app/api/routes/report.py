from fastapi import APIRouter
from uuid import UUID

from ai_app.db.db_report import DBReport

router = APIRouter()

db_report = DBReport()


@router.get("/user")
async def user_report(user_id: UUID):
    response = await db_report.get_user_usage_report(user_id=user_id)

    return response


@router.get("/feature")
async def feature_report(feature: str):
    response = await db_report.get_feature_usage_report(feature=feature)

    return response
