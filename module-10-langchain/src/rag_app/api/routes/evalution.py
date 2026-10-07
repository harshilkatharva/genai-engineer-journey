from fastapi import APIRouter

from rag_app.evalution.golden_test import call_queries
from rag_app.evalution.report import EvalutionReport

router = APIRouter()

evalution_report = EvalutionReport()


@router.get("/report")
def genrate_report():
    return evalution_report.get_report()


@router.get("/test_queries")
def test_queries():
    return call_queries()
