"""Azure OpenAI Batch API client.

Provides a thin wrapper around the Azure OpenAI Batch API for submitting
large numbers of independent chat-completions requests at 50% of the
standard per-token price.

This client is intentionally separate from OpenAIModel: it targets the
asynchronous batch endpoint, which operates on a file-upload / status-check
/ download cycle rather than a live request-response loop.  Because batch
jobs can take up to 24 hours to complete, this client deliberately does
*not* include a blocking poll loop.  The expected workflow is:

1. Build JSONL content and call ``submit_batch`` for each group of prompts.
   Save the returned batch IDs together with any associated metadata to a
   manifest file, then let the script exit.
2. Check job status out-of-band (Azure portal, CLI, or
   ``check_batch_statuses``) until the jobs are complete.
3. Run a second script that calls ``download_results`` for each completed
   batch ID.

Typical usage
-------------
    client = AzureOpenAIBatchClient(api_key=..., azure_endpoint=...,
                                    api_version=..., model_name=...)

    # Submit one job per group of prompts.
    batch_id = client.submit_batch(
        client.build_jsonl_content(prompts, temperature=0.7, top_p=0.95),
        description="my-job-label",
    )

    # Later — check whether the job is done.
    statuses = client.check_batch_statuses([batch_id])
    # {"batch_abc123": "completed"}

    # Download and iterate raw result lines once complete.
    batch_obj = client._client.batches.retrieve(batch_id)
    for result_line in client.download_results(batch_obj):
        ...  # result_line["content"] | result_line["error"]
"""

import io
import json
import logging
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI


logger = logging.getLogger(__name__)


class AzureOpenAIBatchClient:
    """Manages the lifecycle of Azure OpenAI Batch API jobs.

    Responsibilities
    ----------------
    - Build a JSONL request file from a list of prompts, embedding
      sampling parameters (temperature, top_p) into each request body.
    - Upload the file and create the batch job.
    - Check the status of one or more batch jobs (non-blocking).
    - Download and parse the output JSONL for a completed job.

    This class does *not* handle JSON schema validation of the model
    responses — use ``BaseLLM.parse_and_validate_response`` for that.
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        model_name: str,
    ) -> None:
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Building requests
    # ------------------------------------------------------------------

    def build_jsonl_content(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
    ) -> str:
        """Return a JSONL string with one chat-completions request per prompt.

        Args:
            prompts:     One entry per independent request.  All requests in
                         a single JSONL file share the same sampling parameters.
                         Submit separate batches if you need different settings.
            temperature: Sampling temperature forwarded to every request body.
            top_p:       Nucleus sampling value forwarded to every request body.

        The ``custom_id`` of each line is the zero-based index of that prompt
        in ``prompts``, used to correlate results back to submission order when
        the output file is downloaded.
        """
        lines = []
        for i, prompt in enumerate(prompts):
            request = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            lines.append(json.dumps(request))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_batch(self, jsonl_content: str, description: str = "") -> str:
        """Upload a JSONL request file and create a batch job.

        Args:
            jsonl_content: JSONL string produced by ``build_jsonl_content``.
            description:   Human-readable label stored in the job's metadata,
                           visible in the Azure portal.

        Returns:
            The batch job ID (opaque string).
        """
        file_obj = self._client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_content.encode("utf-8"))),
            purpose="batch",
        )
        batch = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/chat/completions",
            completion_window="24h",
            metadata={"description": description} if description else {},
        )
        logger.info(f"Submitted batch job {batch.id!r}  ({description})")
        return batch.id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def check_batch_statuses(self, batch_ids: List[str]) -> Dict[str, Any]:
        """Retrieve the current status of one or more batch jobs (non-blocking).

        Makes one API call per batch ID and returns immediately — no sleeping
        or retrying.  Use this to verify jobs are complete before attempting
        to download results.

        Args:
            batch_ids: IDs of the batch jobs to query.

        Returns:
            Mapping of ``batch_id → batch object``.  Inspect
            ``batch_obj.status`` for the current state.  Possible values are:
            ``validating``, ``failed``, ``in_progress``, ``finalizing``,
            ``completed``, ``expired``, ``cancelling``, ``cancelled``.
        """
        return {
            batch_id: self._client.batches.retrieve(batch_id)
            for batch_id in batch_ids
        }

    # ------------------------------------------------------------------
    # Downloading results
    # ------------------------------------------------------------------

    def download_results(self, batch: Any) -> List[Dict[str, Any]]:
        """Download and parse the output JSONL for a completed batch job.

        Non-completed jobs (failed, expired, cancelled) return an empty list.

        Each element of the returned list has the following keys:

        ``custom_id``
            The zero-based request index (string) set during batch construction.
        ``content``
            Raw model text, or ``None`` if the request failed at the API level.
        ``usage``
            Dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            or ``None`` if usage data is unavailable.
        ``error``
            Error string if the request failed, or ``None`` on success.

        The list is sorted by ``custom_id`` (numeric) to restore submission order.
        """
        if batch.status != "completed" or not batch.output_file_id:
            return []

        raw_bytes = self._client.files.content(batch.output_file_id).read()

        parsed: List[Dict[str, Any]] = []
        for line in raw_bytes.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            custom_id = record.get("custom_id")
            response  = record.get("response")
            api_error = record.get("error")

            if api_error or not response or response.get("status_code") != 200:
                error_msg = (
                    api_error.get("message", "unknown API error")
                    if api_error
                    else f"HTTP {response.get('status_code', 'N/A') if response else 'N/A'}"
                )
                parsed.append({
                    "custom_id": custom_id,
                    "content": None,
                    "usage": None,
                    "error": error_msg,
                })
                continue

            body    = response.get("body", {})
            choices = body.get("choices", [])
            content = choices[0]["message"]["content"] if choices else None

            usage_raw = body.get("usage")
            usage = (
                {
                    "prompt_tokens":     usage_raw.get("prompt_tokens", 0),
                    "completion_tokens": usage_raw.get("completion_tokens", 0),
                    "total_tokens":      usage_raw.get("total_tokens", 0),
                }
                if usage_raw else None
            )

            parsed.append({
                "custom_id": custom_id,
                "content": content,
                "usage": usage,
                "error": None,
            })

        # Restore original request order.
        parsed.sort(
            key=lambda r: int(r["custom_id"]) if (r["custom_id"] or "").isdigit() else 0
        )
        return parsed
