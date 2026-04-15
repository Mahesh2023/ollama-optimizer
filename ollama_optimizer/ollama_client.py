"""
REST API client for Ollama's local API.

Provides a typed, ergonomic interface for interacting with the Ollama server
(default http://localhost:11434).  All network errors are caught and surfaced
as clear log messages rather than unhandled exceptions.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OllamaModel:
    """Structured representation of a model returned by the Ollama API."""

    name: str
    tag: str
    size_bytes: int
    parameter_size: str       # e.g. "7B", "13B", "70B"
    quantization_level: str   # e.g. "Q4_K_M", "Q8_0", "F16"
    family: str
    format: str
    modified_at: str
    digest: str

    # ---- helpers -----------------------------------------------------------

    @property
    def full_name(self) -> str:
        """Return ``name:tag`` when a tag is present, otherwise just *name*."""
        if self.tag and self.tag not in self.name:
            return f"{self.name}:{self.tag}"
        return self.name

    @property
    def size_gb(self) -> float:
        """Model size in gigabytes (rounded to two decimals)."""
        return round(self.size_bytes / (1024 ** 3), 2)

    @property
    def is_embedding_model(self) -> bool:
        """Return True if this model is embedding-only (does not support generate)."""
        name_lower = f"{self.name}:{self.tag}".lower()
        family_lower = self.family.lower()
        # Known embedding model families and name patterns
        if any(tok in name_lower for tok in ["embed", "nomic-embed", "bge-",
                                              "e5-", "gte-", "sentence-"]):
            return True
        if any(tok in family_lower for tok in ["bert", "nomic-bert",
                                                "xlm-roberta"]):
            return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common quantization tokens that appear in Ollama tags.
_QUANT_PATTERN = re.compile(
    r"(Q[2-8]_[A-Z0-9_]+|F16|F32|FP16|FP32|INT8|INT4)",
    re.IGNORECASE,
)


def _parse_quant_from_tag(tag: str) -> str:
    """Extract a quantization level from a model tag string.

    Examples::

        "8b-q4_K_M"  -> "Q4_K_M"
        "latest"      -> ""
        "70b-instruct-q5_0" -> "Q5_0"
    """
    match = _QUANT_PATTERN.search(tag)
    return match.group(1).upper() if match else ""


def _parse_name_and_tag(model_name: str) -> tuple:
    """Split ``"llama3:8b-q4_K_M"`` into ``("llama3", "8b-q4_K_M")``."""
    if ":" in model_name:
        name, tag = model_name.split(":", 1)
    else:
        name, tag = model_name, "latest"
    return name, tag


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin wrapper around the Ollama REST API.

    Parameters
    ----------
    base_url:
        Root URL of the Ollama server (no trailing slash).
    timeout:
        Default request timeout in seconds.  Pulling and generating can take
        a long time so the default is generous.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        logger.debug("OllamaClient initialised – base_url=%s, timeout=%ss", self.base_url, self.timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Build a full URL from a relative API path."""
        return f"{self.base_url}{path}"

    def _get(self, path: str, **kwargs) -> requests.Response:
        """Issue a GET request with standard error handling."""
        url = self._url(path)
        logger.debug("GET %s", url)
        try:
            resp = self._session.get(url, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.ConnectionError:
            logger.error("Connection refused – is the Ollama server running at %s?", self.base_url)
            raise
        except requests.Timeout:
            logger.error("Request to %s timed out after %ss", url, self.timeout)
            raise
        except requests.HTTPError as exc:
            logger.error("HTTP error %s for GET %s: %s", exc.response.status_code, url, exc.response.text)
            raise

    def _post(self, path: str, payload: dict, stream: bool = False, **kwargs) -> requests.Response:
        """Issue a POST request with standard error handling."""
        url = self._url(path)
        logger.debug("POST %s – payload keys: %s, stream=%s", url, list(payload.keys()), stream)
        try:
            resp = self._session.post(
                url,
                json=payload,
                stream=stream,
                timeout=self.timeout,
                **kwargs,
            )
            resp.raise_for_status()
            return resp
        except requests.ConnectionError:
            logger.error("Connection refused – is the Ollama server running at %s?", self.base_url)
            raise
        except requests.Timeout:
            logger.error("Request to %s timed out after %ss", url, self.timeout)
            raise
        except requests.HTTPError as exc:
            logger.error("HTTP error %s for POST %s: %s", exc.response.status_code, url, exc.response.text)
            raise

    def _delete(self, path: str, payload: dict, **kwargs) -> requests.Response:
        """Issue a DELETE request with standard error handling."""
        url = self._url(path)
        logger.debug("DELETE %s", url)
        try:
            resp = self._session.delete(url, json=payload, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.ConnectionError:
            logger.error("Connection refused – is the Ollama server running at %s?", self.base_url)
            raise
        except requests.Timeout:
            logger.error("Request to %s timed out after %ss", url, self.timeout)
            raise
        except requests.HTTPError as exc:
            logger.error("HTTP error %s for DELETE %s: %s", exc.response.status_code, url, exc.response.text)
            raise

    @staticmethod
    def _iter_ndjson(response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """Iterate over newline-delimited JSON objects in a streaming response."""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line: %s", line[:120])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return ``True`` if the Ollama server is reachable."""
        try:
            resp = self._session.get(self._url("/"), timeout=5)
            running = resp.status_code == 200
            logger.info("Ollama server at %s is %s", self.base_url, "running" if running else "not responding correctly")
            return running
        except (requests.ConnectionError, requests.Timeout):
            logger.warning("Ollama server at %s is not reachable", self.base_url)
            return False

    # -- models ----------------------------------------------------------

    def list_models(self) -> List[OllamaModel]:
        """Fetch every locally-available model.

        Calls ``GET /api/tags`` and enriches each entry with detail info
        obtained from ``POST /api/show``.

        Returns
        -------
        list[OllamaModel]
            One entry per model tag found on the server.
        """
        resp = self._get("/api/tags")
        data = resp.json()
        raw_models = data.get("models", [])
        logger.info("Ollama reports %d local model(s)", len(raw_models))

        models: List[OllamaModel] = []
        for entry in raw_models:
            full_name = entry.get("name", "")
            name, tag = _parse_name_and_tag(full_name)
            size_bytes = entry.get("size", 0)
            digest = entry.get("digest", "")
            modified_at = entry.get("modified_at", "")

            # Extract details – the /api/tags response already includes a
            # `details` dict for each model in recent Ollama versions.
            details = entry.get("details", {})
            parameter_size = details.get("parameter_size", "")
            quantization_level = details.get("quantization_level", "")
            family = details.get("family", "")
            fmt = details.get("format", "")

            # Fallback: try to infer quantization from the tag
            if not quantization_level:
                quantization_level = _parse_quant_from_tag(tag)

            models.append(
                OllamaModel(
                    name=name,
                    tag=tag,
                    size_bytes=size_bytes,
                    parameter_size=parameter_size,
                    quantization_level=quantization_level,
                    family=family,
                    format=fmt,
                    modified_at=modified_at,
                    digest=digest,
                )
            )

        return models

    def show_model(self, name: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a specific model.

        Calls ``POST /api/show`` and returns the raw JSON dict which
        typically contains keys like *modelfile*, *parameters*, *template*,
        *details*, and *model_info*.
        """
        resp = self._post("/api/show", {"name": name})
        info = resp.json()
        logger.debug("show_model(%s) returned keys: %s", name, list(info.keys()))
        return info

    def pull_model(self, name: str, stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Pull (download) a model from the Ollama library.

        Parameters
        ----------
        name:
            Model identifier, e.g. ``"llama3"`` or ``"llama3:8b-q4_K_M"``.
        stream:
            If ``True`` (default) the response is streamed and this method
            yields progress dicts such as
            ``{"status": "downloading ...", "completed": 1024, "total": 4096}``.

        Yields
        ------
        dict
            Progress/status dicts emitted by the server.
        """
        logger.info("Pulling model '%s' (stream=%s)", name, stream)
        payload: Dict[str, Any] = {"name": name, "stream": stream}
        resp = self._post("/api/pull", payload, stream=stream)

        if stream:
            yield from self._iter_ndjson(resp)
        else:
            yield resp.json()

        logger.info("Pull of '%s' complete", name)

    def create_model(self, name: str, modelfile: str) -> Generator[Dict[str, Any], None, None]:
        """Create a new model from a Modelfile.

        Calls ``POST /api/create`` in streaming mode and yields status
        dicts emitted by the server.

        Parameters
        ----------
        name:
            Name for the newly created model.
        modelfile:
            The Modelfile content as a string.

        Yields
        ------
        dict
            Status updates from the server.
        """
        logger.info("Creating model '%s'", name)
        payload = {"name": name, "modelfile": modelfile, "stream": True}
        resp = self._post("/api/create", payload, stream=True)
        yield from self._iter_ndjson(resp)
        logger.info("Model '%s' created", name)

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single-shot text generation (non-streaming).

        Calls ``POST /api/generate`` with ``stream: false`` and returns the
        full response dict, which includes useful timing fields like
        ``total_duration``, ``load_duration``, ``eval_count``, etc.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        logger.info("generate() model=%s prompt_length=%d", model, len(prompt))
        start = time.monotonic()
        resp = self._post("/api/generate", payload)
        elapsed = time.monotonic() - start
        result = resp.json()
        logger.info(
            "generate() completed in %.2fs – eval_count=%s, eval_duration=%s",
            elapsed,
            result.get("eval_count"),
            result.get("eval_duration"),
        )
        return result

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a chat completion (non-streaming).

        Parameters
        ----------
        model:
            Model name, e.g. ``"llama3"``.
        messages:
            Conversation history – each dict must have ``role`` and
            ``content`` keys.
        options:
            Optional runtime parameter overrides.

        Returns
        -------
        dict
            Full response including the assistant message and timing data.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options

        logger.info("chat() model=%s messages=%d", model, len(messages))
        start = time.monotonic()
        resp = self._post("/api/chat", payload)
        elapsed = time.monotonic() - start
        result = resp.json()
        logger.info("chat() completed in %.2fs", elapsed)
        return result

    def delete_model(self, name: str) -> bool:
        """Delete a local model.

        Returns ``True`` on success, ``False`` if the request fails.
        """
        logger.info("Deleting model '%s'", name)
        try:
            self._delete("/api/delete", {"name": name})
            logger.info("Model '%s' deleted", name)
            return True
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            logger.error("Failed to delete model '%s': %s", name, exc)
            return False

    def copy_model(self, source: str, destination: str) -> bool:
        """Copy (duplicate) a model under a new name.

        Returns ``True`` on success, ``False`` otherwise.
        """
        logger.info("Copying model '%s' -> '%s'", source, destination)
        try:
            self._post("/api/copy", {"source": source, "destination": destination})
            logger.info("Model copied '%s' -> '%s'", source, destination)
            return True
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            logger.error("Failed to copy model '%s' -> '%s': %s", source, destination, exc)
            return False

    def get_running_models(self) -> List[Dict[str, Any]]:
        """List models currently loaded in memory.

        Calls ``GET /api/ps`` and returns a list of dicts, each describing
        a running model instance.
        """
        resp = self._get("/api/ps")
        data = resp.json()
        models = data.get("models", [])
        logger.info("%d model(s) currently loaded", len(models))
        return models

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self.base_url!r}, timeout={self.timeout})"
