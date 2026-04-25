"""Serviço de inferência (carregamento + geração) partilhado com a API OpenAI-compatível."""

from .runtime import ModelRuntime, get_runtime

__all__ = ["ModelRuntime", "get_runtime"]
