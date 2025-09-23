"""
Failure tracking utilities for recording and analyzing LLM generation failures.

This module provides comprehensive tracking of failed generation attempts,
including schema validation failures, JSON parsing errors, and timeout issues.
The data is structured for academic research and statistical analysis.
"""

import time
from typing import Dict, List, Any, Optional, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class FailureAttempt:
    """Details of a single failed generation attempt."""
    attempt_number: int
    failure_type: str  # 'json_parse', 'schema_validation', 'custom_validation', 'timeout', 'model_error'
    error_message: str
    raw_response: Optional[str]
    prompt_hash: str  # Hash of the prompt for privacy
    timestamp: str
    response_time_ms: Optional[float] = None
    model_metadata: Optional[Dict[str, Any]] = None
    validation_rule_name: Optional[str] = None  # Name of failed validation rule (for custom_validation failures)


@dataclass
class PromptFailureRecord:
    """Complete failure record for a single prompt across all attempts."""
    prompt_hash: str
    total_attempts: int
    final_success: bool
    attempts: List[FailureAttempt]
    first_attempt_timestamp: str
    final_attempt_timestamp: str
    total_time_ms: float


@dataclass
class BatchFailureStatistics:
    """Statistical summary of failures for a batch of prompts."""
    total_prompts: int
    successful_prompts: int
    failed_prompts: int
    success_rate: float
    
    # Failure breakdown by type
    json_parse_failures: int
    schema_validation_failures: int
    custom_validation_failures: int
    timeout_failures: int
    model_error_failures: int
    
    # Attempt statistics
    total_attempts: int
    average_attempts_per_prompt: float
    max_attempts_for_single_prompt: int
    
    # Time statistics
    total_generation_time_ms: float
    average_time_per_prompt_ms: float
    average_time_per_successful_prompt_ms: float
    
    # Detailed failure patterns
    prompts_failed_on_first_attempt: int
    prompts_failed_on_second_attempt: int
    prompts_failed_on_third_attempt: int
    prompts_succeeded_after_retry: int


class GenerationFailureTracker:
    """Tracks and analyzes LLM generation failures for research purposes."""
    
    def __init__(self):
        """Initialize the failure tracker."""
        self.current_session_failures: List[PromptFailureRecord] = []
        self.current_prompt_record: Optional[PromptFailureRecord] = None
        self.session_start_time = time.time()
    
    def start_prompt_tracking(self, prompt: str) -> str:
        """Start tracking failures for a new prompt.
        
        Args:
            prompt: The prompt being sent to the LLM
            
        Returns:
            Hash of the prompt for identification
        """
        prompt_hash = self._hash_prompt(prompt)
        
        self.current_prompt_record = PromptFailureRecord(
            prompt_hash=prompt_hash,
            total_attempts=0,
            final_success=False,
            attempts=[],
            first_attempt_timestamp=datetime.now().isoformat(),
            final_attempt_timestamp="",
            total_time_ms=0.0
        )
        
        return prompt_hash
    
    def record_attempt_failure(
        self,
        attempt_number: int,
        failure_type: str,
        error_message: str,
        raw_response: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        validation_rule_name: Optional[str] = None
    ):
        """Record a failed attempt for the current prompt.
        
        Args:
            attempt_number: Which attempt this was (1, 2, 3, etc.)
            failure_type: Type of failure ('json_parse', 'schema_validation', 'custom_validation', 'timeout', 'model_error')
            error_message: Detailed error message
            raw_response: Raw response from LLM (if available)
            response_time_ms: Time taken for this attempt
            model_metadata: Metadata about the model/request
            validation_rule_name: Name of the validation rule that failed (for custom_validation failures)
        """
        if not self.current_prompt_record:
            raise ValueError("Must call start_prompt_tracking() before recording failures")
        
        # Truncate raw response for storage efficiency and privacy
        truncated_response = None
        if raw_response:
            truncated_response = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
        
        failure_attempt = FailureAttempt(
            attempt_number=attempt_number,
            failure_type=failure_type,
            error_message=str(error_message),
            raw_response=truncated_response,
            prompt_hash=self.current_prompt_record.prompt_hash,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time_ms,
            model_metadata=model_metadata,
            validation_rule_name=validation_rule_name
        )
        
        self.current_prompt_record.attempts.append(failure_attempt)
        self.current_prompt_record.total_attempts = attempt_number
        self.current_prompt_record.final_attempt_timestamp = failure_attempt.timestamp
        
        if response_time_ms:
            self.current_prompt_record.total_time_ms += response_time_ms
    
    def record_prompt_success(self, final_attempt_number: int, response_time_ms: Optional[float] = None):
        """Record that the current prompt finally succeeded.
        
        Args:
            final_attempt_number: The attempt number that succeeded
            response_time_ms: Time taken for the successful attempt
        """
        if not self.current_prompt_record:
            raise ValueError("Must call start_prompt_tracking() before recording success")
        
        self.current_prompt_record.final_success = True
        self.current_prompt_record.total_attempts = final_attempt_number
        self.current_prompt_record.final_attempt_timestamp = datetime.now().isoformat()
        
        if response_time_ms:
            self.current_prompt_record.total_time_ms += response_time_ms
        
        self._finalize_current_prompt()
    
    def record_prompt_final_failure(self):
        """Record that the current prompt failed all attempts."""
        if not self.current_prompt_record:
            raise ValueError("Must call start_prompt_tracking() before recording final failure")
        
        self.current_prompt_record.final_success = False
        self.current_prompt_record.final_attempt_timestamp = datetime.now().isoformat()
        
        self._finalize_current_prompt()
    
    def _finalize_current_prompt(self):
        """Move the current prompt record to the session failures list."""
        if self.current_prompt_record:
            self.current_session_failures.append(self.current_prompt_record)
            self.current_prompt_record = None
    
    def get_session_statistics(self) -> BatchFailureStatistics:
        """Calculate comprehensive statistics for the current session.
        
        Returns:
            Detailed statistics about failures and success rates
        """
        if not self.current_session_failures:
            return self._empty_statistics()
        
        total_prompts = len(self.current_session_failures)
        successful_prompts = sum(1 for record in self.current_session_failures if record.final_success)
        failed_prompts = total_prompts - successful_prompts
        success_rate = successful_prompts / total_prompts if total_prompts > 0 else 0.0
        
        # Count failure types
        failure_type_counts = {
            'json_parse': 0,
            'schema_validation': 0,
            'timeout': 0,
            'model_error': 0
        }
        
        total_attempts = 0
        total_time_ms = 0.0
        successful_prompt_time_ms = 0.0
        max_attempts = 0
        
        # Detailed failure pattern analysis
        failed_on_attempt = {1: 0, 2: 0, 3: 0}
        succeeded_after_retry = 0
        
        for record in self.current_session_failures:
            total_attempts += record.total_attempts
            total_time_ms += record.total_time_ms
            max_attempts = max(max_attempts, record.total_attempts)
            
            if record.final_success:
                successful_prompt_time_ms += record.total_time_ms
                if record.total_attempts > 1:
                    succeeded_after_retry += 1
            else:
                # Count where the final failure occurred
                if record.total_attempts in failed_on_attempt:
                    failed_on_attempt[record.total_attempts] += 1
            
            # Count failure types from all attempts
            for attempt in record.attempts:
                if attempt.failure_type in failure_type_counts:
                    failure_type_counts[attempt.failure_type] += 1
        
        avg_time_per_prompt = total_time_ms / total_prompts if total_prompts > 0 else 0.0
        avg_time_per_successful = successful_prompt_time_ms / successful_prompts if successful_prompts > 0 else 0.0
        avg_attempts_per_prompt = total_attempts / total_prompts if total_prompts > 0 else 0.0
        
        return BatchFailureStatistics(
            total_prompts=total_prompts,
            successful_prompts=successful_prompts,
            failed_prompts=failed_prompts,
            success_rate=success_rate,
            json_parse_failures=failure_type_counts['json_parse'],
            schema_validation_failures=failure_type_counts['schema_validation'],
            custom_validation_failures=failure_type_counts.get('custom_validation', 0),
            timeout_failures=failure_type_counts['timeout'],
            model_error_failures=failure_type_counts['model_error'],
            total_attempts=total_attempts,
            average_attempts_per_prompt=avg_attempts_per_prompt,
            max_attempts_for_single_prompt=max_attempts,
            total_generation_time_ms=total_time_ms,
            average_time_per_prompt_ms=avg_time_per_prompt,
            average_time_per_successful_prompt_ms=avg_time_per_successful,
            prompts_failed_on_first_attempt=failed_on_attempt[1],
            prompts_failed_on_second_attempt=failed_on_attempt[2],
            prompts_failed_on_third_attempt=failed_on_attempt[3],
            prompts_succeeded_after_retry=succeeded_after_retry
        )
    
    def get_detailed_failure_records(self) -> List[Dict[str, Any]]:
        """Get detailed failure records for research analysis.
        
        Returns:
            List of failure records with all details
        """
        return [asdict(record) for record in self.current_session_failures]
    
    def get_academic_summary(self) -> Dict[str, Any]:
        """Get a summary formatted for academic research.
        
        Returns:
            Dictionary with statistics and metadata suitable for research papers
        """
        stats = self.get_session_statistics()
        
        return {
            "generation_success_metrics": {
                "total_prompts": stats.total_prompts,
                "success_rate": round(stats.success_rate, 4),
                "failure_rate": round(1 - stats.success_rate, 4),
                "successful_prompts": stats.successful_prompts,
                "failed_prompts": stats.failed_prompts
            },
            "failure_type_breakdown": {
                "json_parsing_errors": stats.json_parse_failures,
                "schema_validation_errors": stats.schema_validation_failures,
                "custom_validation_errors": stats.custom_validation_failures,
                "timeout_errors": stats.timeout_failures,
                "model_errors": stats.model_error_failures,
                "total_failure_attempts": (stats.json_parse_failures + stats.schema_validation_failures + 
                                          stats.custom_validation_failures + stats.timeout_failures + stats.model_error_failures)
            },
            "retry_behavior_analysis": {
                "average_attempts_per_prompt": round(stats.average_attempts_per_prompt, 2),
                "maximum_attempts_single_prompt": stats.max_attempts_for_single_prompt,
                "prompts_requiring_retry": stats.prompts_succeeded_after_retry,
                "retry_success_rate": round(stats.prompts_succeeded_after_retry / max(1, stats.total_prompts), 4),
                "failure_distribution": {
                    "failed_after_1_attempt": stats.prompts_failed_on_first_attempt,
                    "failed_after_2_attempts": stats.prompts_failed_on_second_attempt,
                    "failed_after_3_attempts": stats.prompts_failed_on_third_attempt
                }
            },
            "performance_metrics": {
                "total_generation_time_seconds": round(stats.total_generation_time_ms / 1000, 2),
                "average_time_per_prompt_seconds": round(stats.average_time_per_prompt_ms / 1000, 2),
                "average_time_per_successful_prompt_seconds": round(stats.average_time_per_successful_prompt_ms / 1000, 2),
                "efficiency_loss_due_to_failures_percent": round(
                    ((stats.total_attempts - stats.successful_prompts) / max(1, stats.total_attempts)) * 100, 2
                )
            },
            "metadata": {
                "tracking_session_duration_seconds": round(time.time() - self.session_start_time, 2),
                "data_collection_timestamp": datetime.now().isoformat(),
                "failure_tracking_version": "1.0",
                "privacy_note": "Prompts are hashed for privacy; raw responses truncated to 500 characters"
            }
        }
    
    def reset_session(self):
        """Reset the tracking session (clear all recorded failures)."""
        self.current_session_failures.clear()
        self.current_prompt_record = None
        self.session_start_time = time.time()
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt for identification while preserving privacy."""
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def _empty_statistics(self) -> BatchFailureStatistics:
        """Return empty statistics when no data is available."""
        return BatchFailureStatistics(
            total_prompts=0,
            successful_prompts=0,
            failed_prompts=0,
            success_rate=0.0,
            json_parse_failures=0,
            schema_validation_failures=0,
            custom_validation_failures=0,
            timeout_failures=0,
            model_error_failures=0,
            total_attempts=0,
            average_attempts_per_prompt=0.0,
            max_attempts_for_single_prompt=0,
            total_generation_time_ms=0.0,
            average_time_per_prompt_ms=0.0,
            average_time_per_successful_prompt_ms=0.0,
            prompts_failed_on_first_attempt=0,
            prompts_failed_on_second_attempt=0,
            prompts_failed_on_third_attempt=0,
            prompts_succeeded_after_retry=0
        )