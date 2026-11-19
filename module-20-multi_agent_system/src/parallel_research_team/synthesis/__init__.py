"""Synthesis and verification gate package for Parallel Research Team."""

from .engine import SynthesisEngine
from .source_pool import AuthoritativeSourcePool
from .verifier import ResearchVerifier, SynthesisVerificationResult

__all__ = [
    "AuthoritativeSourcePool",
    "ResearchVerifier",
    "SynthesisEngine",
    "SynthesisVerificationResult",
]
