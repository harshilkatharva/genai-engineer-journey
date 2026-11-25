"""Synthesis and verification gate package for Parallel Research Team."""

from .engine import SynthesisEngine
from .reflection import ReflectionEngine, ReflectionOutcome
from .source_pool import AuthoritativeSourcePool
from .verifier import ResearchVerifier, SynthesisVerificationResult

__all__ = [
    "AuthoritativeSourcePool",
    "ReflectionEngine",
    "ReflectionOutcome",
    "ResearchVerifier",
    "SynthesisEngine",
    "SynthesisVerificationResult",
]
