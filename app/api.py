from __future__ import annotations

from fastapi import FastAPI, Query

from app.config import settings
from app.llm.candidate_reranker import LLMCandidateReranker
from app.llm.openrouter_client import OpenRouterClient
from app.llm.query_rewriter import LLMQueryRewriter
from app.query.text_processor import TextProcessor
from app.query.query_understanding import QueryUnderstandingModule
from app.ranking.reranker import Reranker
from app.retrieval.vector_retriever import VectorRetriever
from app.services.explanation_service import ExplanationService
from app.services.recommendation_service import RecommendationService
from app.services.response_builder import ResponseBuilder


settings.validate_runtime_files()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

text_processor = TextProcessor()
query_module = QueryUnderstandingModule(text_processor=text_processor)

retriever = VectorRetriever(
    db_path=str(settings.DB_PATH),
    index_path=str(settings.INDEX_PATH),
    ids_path=str(settings.IDS_PATH),
)

reranker = Reranker()
explanation_service = ExplanationService()
response_builder = ResponseBuilder(explanation_service=explanation_service)

llm_client = None
llm_query_rewriter = None
llm_candidate_reranker = None

if settings.OPENROUTER_API_KEY:
    llm_client = OpenRouterClient(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_MODEL,
        base_url=settings.OPENROUTER_BASE_URL,
        app_name=settings.APP_NAME,
    )
    llm_query_rewriter = LLMQueryRewriter(client=llm_client)
    llm_candidate_reranker = LLMCandidateReranker(client=llm_client)

recommendation_service = RecommendationService(
    query_module=query_module,
    retriever=retriever,
    reranker=reranker,
    response_builder=response_builder,
    llm_query_rewriter=llm_query_rewriter,
    llm_candidate_reranker=llm_candidate_reranker,
    enable_llm_query_rewrite=settings.ENABLE_LLM_QUERY_REWRITE,
    enable_llm_rerank=settings.ENABLE_LLM_RERANK,
)


@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": ["/health", "/search"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(
        query: str = Query(..., min_length=1, description="User query"),
        final_k: int = Query(10, ge=1, le=50),
        max_per_artist: int = Query(3, ge=1, le=20),
        include_debug: bool = Query(False, description="Include ranking and parsing debug fields"),
):
    return recommendation_service.run_query(
        query=query,
        final_k=final_k,
        max_per_artist=max_per_artist,
        include_debug=include_debug,
    )
