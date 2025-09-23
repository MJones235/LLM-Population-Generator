"""
Data export utilities for saving generated population data with comprehensive metadata.

This module provides functionality to save generated population data in a structured format
that includes:
- Generated household/population data
- Generation metadata (parameters, timestamps, etc.)
- Statistical summaries and comparisons
- Target data and source information
- Cost tracking information (if available)

The saved data is designed for future analysis and reproducibility.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import hashlib

from ..utils.statistics import StatisticsManager
from ..utils.token_analysis import TokenAnalyzer


@dataclass
class GenerationMetadata:
    """Metadata about a population generation run."""
    timestamp: str
    model_info: Dict[str, Any]
    generation_parameters: Dict[str, Any]
    data_sources: List[str]
    statistics_config: Optional[Dict[str, Any]] = None
    cost_tracking: Optional[Dict[str, Any]] = None
    failure_analysis: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None


@dataclass
class PopulationExport:
    """Complete export package for generated population data."""
    metadata: GenerationMetadata
    households: List[Dict[str, Any]]
    statistical_analysis: Optional[Dict[str, Any]] = None
    target_comparisons: Optional[Dict[str, Any]] = None
    raw_generation_log: Optional[List[Dict[str, Any]]] = None
    detailed_failure_records: Optional[List[Dict[str, Any]]] = None


class PopulationDataSaver:
    """Utility class for saving generated population data with comprehensive metadata."""
    
    def __init__(self, output_base_dir: Optional[Union[str, Path]] = None):
        """Initialize the data saver.
        
        Args:
            output_base_dir: Base directory for saving data. If None, uses current directory.
        """
        self.output_base_dir = Path(output_base_dir or ".")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_population_data(
        self,
        households: List[Dict[str, Any]],
        model_info: Dict[str, Any],
        generation_parameters: Dict[str, Any],
        output_name: str,
        data_sources: Optional[List[str]] = None,
        statistics_manager: Optional[StatisticsManager] = None,
        token_analyzer: Optional[TokenAnalyzer] = None,
        target_data_files: Optional[List[str]] = None,
        include_analysis: bool = True,
        format_type: str = "json",
        llm_model: Optional[Any] = None
    ) -> Dict[str, str]:
        """Save population data with comprehensive metadata.
        
        Args:
            households: Generated household data
            model_info: Information about the LLM used (name, version, etc.)
            generation_parameters: Parameters used for generation (n_households, batch_size, etc.)
            output_name: Base name for output files (without extension)
            data_sources: List of data source files used
            statistics_manager: StatisticsManager instance for statistical analysis
            token_analyzer: TokenAnalyzer instance for cost information
            target_data_files: List of target data files used
            include_analysis: Whether to include statistical analysis
            format_type: Output format ('json', 'json_and_csv', 'csv')
            llm_model: LLM model instance for extracting failure statistics
            
        Returns:
            Dictionary with paths to saved files
        """
        # Generate unique run ID
        run_id = self._generate_run_id(households, generation_parameters)
        
        # Create metadata
        metadata = self._create_metadata(
            model_info=model_info,
            generation_parameters=generation_parameters,
            data_sources=data_sources or [],
            statistics_manager=statistics_manager,
            token_analyzer=token_analyzer,
            target_data_files=target_data_files,
            run_id=run_id,
            llm_model=llm_model
        )
        
        # Perform statistical analysis if requested
        statistical_analysis = None
        target_comparisons = None
        if include_analysis and statistics_manager and households:
            statistical_analysis, target_comparisons = self._perform_analysis(
                households, statistics_manager
            )
        
        # Extract detailed failure records if available
        detailed_failure_records = None
        if llm_model and hasattr(llm_model, 'failure_tracker'):
            detailed_failure_records = llm_model.failure_tracker.get_detailed_failure_records()
        
        # Create export package
        export_package = PopulationExport(
            metadata=metadata,
            households=households,
            statistical_analysis=statistical_analysis,
            target_comparisons=target_comparisons,
            detailed_failure_records=detailed_failure_records
        )
        
        # Save in requested formats
        saved_files = {}
        
        if format_type in ["json", "json_and_csv"]:
            json_path = self._save_json_format(export_package, output_name)
            saved_files["json"] = str(json_path)
        
        if format_type in ["csv", "json_and_csv"]:
            csv_paths = self._save_csv_format(export_package, output_name)
            saved_files.update(csv_paths)
        
        return saved_files
    
    def _generate_run_id(self, households: List[Dict[str, Any]], 
                        generation_parameters: Dict[str, Any]) -> str:
        """Generate a unique run ID based on content and parameters."""
        content_str = json.dumps(households, sort_keys=True)
        params_str = json.dumps(generation_parameters, sort_keys=True)
        combined = f"{content_str}{params_str}{datetime.now().isoformat()}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _create_metadata(
        self,
        model_info: Dict[str, Any],
        generation_parameters: Dict[str, Any],
        data_sources: List[str],
        statistics_manager: Optional[StatisticsManager],
        token_analyzer: Optional[TokenAnalyzer],
        target_data_files: Optional[List[str]],
        run_id: str,
        llm_model: Optional[Any] = None
    ) -> GenerationMetadata:
        """Create comprehensive metadata for the generation run."""
        
        # Statistics configuration
        statistics_config = None
        if statistics_manager:
            statistics_config = {
                "registered_providers": list(statistics_manager.providers.keys()),
                "placeholder_mappings": statistics_manager.placeholder_mappings,
                "flexible_loading_enabled": hasattr(statistics_manager, 'data_manager'),
                "target_data_files": target_data_files or []
            }
        
        # Cost tracking information
        cost_tracking = None
        if token_analyzer:
            try:
                cost_summary = token_analyzer.get_session_summary()
                if "error" in cost_summary:
                    cost_tracking = cost_summary
                else:
                    total_cost = cost_summary.get("estimated_cost", {}).get("total", 0)
                    total_requests = cost_summary.get("total_requests", 0)
                    total_tokens = cost_summary.get("total_tokens", 0)
                    cost_tracking = {
                        "total_calls": total_requests,
                        "total_tokens": total_tokens,
                        "total_cost": total_cost,
                        "average_tokens_per_call": total_tokens / max(1, total_requests),
                        "cost_per_household": total_cost / max(1, generation_parameters.get("n_households", 1))
                    }
            except Exception as e:
                cost_tracking = {"error": f"Could not extract cost information: {e}"}
        
        # Failure analysis information for academic research
        failure_analysis = None
        if llm_model and hasattr(llm_model, 'failure_tracker'):
            try:
                failure_analysis = llm_model.failure_tracker.get_academic_summary()
            except Exception as e:
                failure_analysis = {"error": f"Could not extract failure statistics: {e}"}
        
        return GenerationMetadata(
            timestamp=datetime.now().isoformat(),
            model_info=model_info,
            generation_parameters=generation_parameters,
            data_sources=data_sources,
            statistics_config=statistics_config,
            cost_tracking=cost_tracking,
            failure_analysis=failure_analysis,
            run_id=run_id
        )
    
    def _perform_analysis(
        self, 
        households: List[Dict[str, Any]], 
        statistics_manager: StatisticsManager
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform statistical analysis on generated households."""
        
        # Convert households to DataFrame for analysis
        people_data = []
        for household_idx, household in enumerate(households):
            # Handle both data formats: direct arrays and {"household": [...]} objects
            if isinstance(household, dict) and 'household' in household:
                # Format: {"household": [person1, person2, ...]}
                people = household['household']
            elif isinstance(household, list):
                # Format: [person1, person2, ...]
                people = household
            else:
                # Skip invalid household format
                continue
            
            for person in people:
                if isinstance(person, dict):
                    person_record = dict(**person, household_id=household_idx)
                    people_data.append(person_record)
        
        if not people_data:
            return {}, {}
        
        synthetic_df = pd.DataFrame(people_data)
        
        # Compute all registered statistics
        statistical_analysis = {}
        target_comparisons = {}
        
        # Compute all registered statistics with fit metrics
        results = statistics_manager.compute_all_statistics(synthetic_df, relationship_col="relationship")
        
        for stat_name, result in results.items():
            statistical_analysis[stat_name] = {
                "observed_distribution": result.observed,
                "sample_size": len(synthetic_df),
                "households_analyzed": len(households)
            }
            
            # Include fit metrics if available
            if result.fit_metrics:
                statistical_analysis[stat_name]["fit_metrics"] = result.fit_metrics
            
            if result.target:
                target_comparisons[stat_name] = {
                    "target_distribution": result.target,
                    "observed_distribution": result.observed,
                    "differences": {
                        key: result.observed.get(key, 0) - result.target.get(key, 0)
                        for key in set(list(result.observed.keys()) + list(result.target.keys()))
                    }
                }
                
                # Include fit metrics in target comparisons as well
                if result.fit_metrics:
                    target_comparisons[stat_name]["fit_metrics"] = result.fit_metrics
        
        # Add overall fit summary if there are results with fit metrics
        fit_results = {name: result for name, result in results.items() if result.fit_metrics}
        if fit_results:
            fit_summary = statistics_manager.get_overall_fit_summary(fit_results)
            statistical_analysis["_overall_fit_summary"] = fit_summary
        
        return statistical_analysis, target_comparisons
    
    def _save_json_format(self, export_package: PopulationExport, output_name: str) -> Path:
        """Save data in comprehensive JSON format."""
        output_path = self.output_base_dir / f"{output_name}.json"
        
        # Convert to serializable format
        export_dict = {
            "metadata": asdict(export_package.metadata),
            "households": export_package.households,
            "statistical_analysis": export_package.statistical_analysis,
            "target_comparisons": export_package.target_comparisons,
            "format_version": "1.0",
            "description": "Generated population data with comprehensive metadata for analysis"
        }
        
        # Include detailed failure records if available
        if export_package.detailed_failure_records:
            export_dict["detailed_failure_records"] = export_package.detailed_failure_records
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_dict, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _save_csv_format(self, export_package: PopulationExport, output_name: str) -> Dict[str, str]:
        """Save data in CSV format with separate metadata file."""
        saved_files = {}
        
        # Save household data as CSV
        if export_package.households:
            # Flatten household data to individual records
            people_data = []
            for household_idx, household in enumerate(export_package.households):
                # Handle both data formats: direct arrays and {"household": [...]} objects
                if isinstance(household, dict) and 'household' in household:
                    # Format: {"household": [person1, person2, ...]}
                    people = household['household']
                elif isinstance(household, list):
                    # Format: [person1, person2, ...]
                    people = household
                else:
                    # Skip invalid household format
                    continue
                
                for person in people:
                    if isinstance(person, dict):
                        record = dict(**person, household_id=household_idx)
                        people_data.append(record)
            
            if people_data:
                df = pd.DataFrame(people_data)
                households_csv = self.output_base_dir / f"{output_name}_households.csv"
                df.to_csv(households_csv, index=False)
                saved_files["households_csv"] = str(households_csv)
        
        # Save metadata as JSON
        metadata_json = self.output_base_dir / f"{output_name}_metadata.json"
        metadata_dict = {
            "metadata": asdict(export_package.metadata),
            "statistical_analysis": export_package.statistical_analysis,
            "target_comparisons": export_package.target_comparisons,
            "format_version": "1.0"
        }
        
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        saved_files["metadata_json"] = str(metadata_json)
        
        # Save statistical analysis as CSV if available
        if export_package.statistical_analysis:
            try:
                stats_data = []
                fit_metrics_data = []
                
                for stat_name, stat_data in export_package.statistical_analysis.items():
                    # Skip the overall fit summary from regular stats
                    if stat_name == "_overall_fit_summary":
                        continue
                        
                    if "observed_distribution" in stat_data:
                        for category, value in stat_data["observed_distribution"].items():
                            stats_data.append({
                                "statistic": stat_name,
                                "category": category,
                                "observed_percentage": value,
                                "sample_size": stat_data.get("sample_size", 0)
                            })
                    
                    # Extract fit metrics for separate CSV
                    if "fit_metrics" in stat_data:
                        fit_record = {"statistic": stat_name}
                        fit_record.update(stat_data["fit_metrics"])
                        fit_record["sample_size"] = stat_data.get("sample_size", 0)
                        fit_metrics_data.append(fit_record)
                
                # Save main statistics CSV
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_csv = self.output_base_dir / f"{output_name}_statistics.csv"
                    stats_df.to_csv(stats_csv, index=False)
                    saved_files["statistics_csv"] = str(stats_csv)
                
                # Save fit metrics CSV
                if fit_metrics_data:
                    fit_df = pd.DataFrame(fit_metrics_data)
                    fit_csv = self.output_base_dir / f"{output_name}_fit_metrics.csv"
                    fit_df.to_csv(fit_csv, index=False)
                    saved_files["fit_metrics_csv"] = str(fit_csv)
                    
            except Exception as e:
                print(f"Warning: Could not save statistics as CSV: {e}")
        
        return saved_files
    
    def load_population_data(self, file_path: Union[str, Path]) -> PopulationExport:
        """Load previously saved population data.
        
        Args:
            file_path: Path to the JSON file containing population data
            
        Returns:
            PopulationExport object with loaded data
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = GenerationMetadata(**data["metadata"])
        
        return PopulationExport(
            metadata=metadata,
            households=data["households"],
            statistical_analysis=data.get("statistical_analysis"),
            target_comparisons=data.get("target_comparisons")
        )


def save_generation_results(
    households: List[Dict[str, Any]],
    model_info: Dict[str, Any],
    generation_parameters: Dict[str, Any],
    output_dir: Union[str, Path],
    output_name: str,
    **kwargs
) -> Dict[str, str]:
    """Convenience function to save population generation results.
    
    Args:
        households: Generated household data
        model_info: Information about the LLM used
        generation_parameters: Parameters used for generation
        output_dir: Directory to save files
        output_name: Base name for output files
        **kwargs: Additional arguments passed to PopulationDataSaver.save_population_data
        
    Returns:
        Dictionary with paths to saved files
    """
    saver = PopulationDataSaver(output_dir)
    return saver.save_population_data(
        households=households,
        model_info=model_info,
        generation_parameters=generation_parameters,
        output_name=output_name,
        **kwargs
    )