"""
Data export utilities for saving generated population data.

Provides normalized CSV format for individual records with household_id, 
plus comprehensive JSON metadata with cost, failure, and distribution analysis.
"""

import json
import pandas as pd
import statistics
import uuid
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import statistical analysis components
from .statistics import StatisticsManager
from .statistics.fit_metrics import DistributionalFitCalculator


@dataclass
class ComprehensiveMetadata:
    """Comprehensive metadata with improved statistical format."""
    # Basic metadata section
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical analysis with grouped distributions
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy fields for backward compatibility
    generation_timestamp: datetime = field(default_factory=datetime.now)
    total_households: int = 0
    total_people: int = 0
    total_cost: Optional[float] = None
    cost_breakdown: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    total_requests: int = 0
    failed_requests: int = 0
    failure_categories: Dict[str, int] = field(default_factory=dict)
    error_patterns: List[str] = field(default_factory=list)
    demographic_distributions: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    statistical_measures: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class PopulationDataSaver:
    """Enhanced population data saver with normalized CSV and comprehensive metadata."""
    
    def __init__(self, output_base_dir: str):
        """Initialize the data saver.
        
        Args:
            output_base_dir: Base directory for saving files
        """
        self.output_base_dir = Path(output_base_dir)
    
    def save_population_data(
        self,
        households: List[Dict[str, Any]],
        output_name: str,
        metadata: Optional[ComprehensiveMetadata] = None
    ) -> Dict[str, Path]:
        """Save population data in normalized CSV format with comprehensive JSON metadata.
        
        Args:
            households: Generated household data
            output_name: Base name for output files
            metadata: Optional metadata (will be generated if not provided)
            
        Returns:
            Dictionary with 'csv' and 'metadata' file paths
        """
        saved_files = {}
        
        # Ensure output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate metadata if not provided
        if not metadata:
            metadata = self._generate_comprehensive_metadata(households)
        
        # Save normalized CSV
        csv_path = self._save_normalized_csv(households, output_name)
        if csv_path:
            saved_files['csv'] = csv_path
        
        # Save comprehensive metadata
        metadata_path = self._save_comprehensive_metadata(metadata, output_name)
        saved_files['metadata'] = metadata_path
        
        return saved_files
    
    def _save_normalized_csv(self, households: List[Dict[str, Any]], output_name: str) -> Optional[Path]:
        """Save population data in normalized CSV format with household_id.
        
        Format: household_id,person_id,age,gender,relationship,name,...
        """
        if not households:
            return None
            
        # Flatten household data to individual records
        people_data = []
        for household_idx, household in enumerate(households, 1):
            # Handle {"household": [...]} format
            if isinstance(household, dict) and 'household' in household:
                people = household['household']
            else:
                continue
            
            for person_idx, person in enumerate(people, 1):
                if isinstance(person, dict):
                    record = {
                        'household_id': household_idx,
                        'person_id': person_idx,
                        **person  # Include all person attributes
                    }
                    people_data.append(record)
        
        if not people_data:
            return None
            
        # Create DataFrame with standard column ordering
        df = pd.DataFrame(people_data)
        standard_cols = ['household_id', 'person_id', 'age', 'gender', 'relationship', 'name']
        other_cols = [col for col in df.columns if col not in standard_cols]
        ordered_cols = [col for col in standard_cols if col in df.columns] + sorted(other_cols)
        df = df[ordered_cols]
        
        output_path = self.output_base_dir / f"{output_name}.csv"
        df.to_csv(output_path, index=False)
        return output_path
    
    def _generate_comprehensive_metadata(self, households: List[Dict[str, Any]], 
                                      classifiers: Optional[List] = None,
                                      statistics_manager: Optional[Any] = None) -> ComprehensiveMetadata:
        """Generate comprehensive metadata with improved statistical format.
        
        Args:
            households: Generated household data
            classifiers: Optional list of classifiers for statistical analysis
        """
        # Extract individual people for analysis
        people = []
        for household in households:
            if isinstance(household, dict) and 'household' in household:
                people.extend(household['household'])
        
        # Create pandas DataFrame for statistical analysis
        people_data = []
        for household_idx, household in enumerate(households, 1):
            # Handle {"household": [...]} format
            if isinstance(household, dict) and 'household' in household:
                household_people = household['household']
            else:
                continue
            
            for person_idx, person in enumerate(household_people, 1):
                if isinstance(person, dict):
                    record = {
                        'household_id': household_idx,
                        'person_id': person_idx,
                        **person  # Include all person attributes
                    }
                    people_data.append(record)
        
        df = pd.DataFrame(people_data) if people_data else pd.DataFrame()
        
        # Initialize metadata with new structure
        metadata = ComprehensiveMetadata()
        
        # METADATA SECTION (following the improved format)
        metadata.metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {},  # Will be populated later
            'generation_parameters': {},  # Will be populated later
            'data_sources': [],
            'cost_tracking': {
                'total_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'average_tokens_per_call': 0.0,
                'cost_per_household': 0.0
            },
            'failure_analysis': {
                'generation_success_metrics': {
                    'total_prompts': 0,
                    'success_rate': 1.0,
                    'failure_rate': 0.0,
                    'successful_prompts': 0,
                    'failed_prompts': 0
                },
                'failure_type_breakdown': {
                    'json_parsing_errors': 0,
                    'schema_validation_errors': 0,
                    'custom_validation_errors': 0,
                    'timeout_errors': 0,
                    'model_errors': 0,
                    'total_failure_attempts': 0
                },
                'retry_behavior_analysis': {
                    'average_attempts_per_prompt': 1.0,
                    'maximum_attempts_single_prompt': 1,
                    'prompts_requiring_retry': 0,
                    'retry_success_rate': 0.0,
                    'failure_distribution': {
                        'failed_after_1_attempt': 0,
                        'failed_after_2_attempts': 0,
                        'failed_after_3_attempts': 0
                    }
                },
                'performance_metrics': {
                    'total_generation_time_seconds': 0.0,
                    'average_time_per_prompt_seconds': 0.0,
                    'average_time_per_successful_prompt_seconds': 0.0,
                    'efficiency_loss_due_to_failures_percent': 0.0
                },
                'metadata': {
                    'tracking_session_duration_seconds': 0.0,
                    'data_collection_timestamp': datetime.now().isoformat(),
                    'failure_tracking_version': '1.0',
                    'privacy_note': 'Prompts are hashed for privacy; raw responses truncated to 500 characters'
                }
            },
            'run_id': metadata.run_id[:12]  # Shorter run ID like the example
        }
        
        # STATISTICAL ANALYSIS SECTION
        if (classifiers or statistics_manager) and df is not None and not df.empty:
            # Use the existing statistics manager if provided to preserve loaded target data
            if statistics_manager:
                stats_manager = statistics_manager
            else:
                # Fallback: create new stats manager and register classifiers
                # This path is less ideal as it loses the original placeholder mappings
                stats_manager = StatisticsManager(compute_fit_metrics=True)
                
                if classifiers:
                    for i, classifier in enumerate(classifiers):
                        # Use generic names when we don't have the original placeholder info
                        placeholder = f"CLASSIFIER_{i+1}_STATS"
                        stats_manager.register_classifier(placeholder, classifier, None)
            
            # Compute all statistics
            stats_results = stats_manager.compute_all_statistics(df)
            
            # Record which placeholders were used (for documentation purposes)
            used_placeholders = []
            placeholder_to_statistic = {}
            
            # Build reverse mapping from statistic name back to placeholder
            for placeholder, statistic_name in stats_manager.placeholder_mappings.items():
                if statistic_name in stats_results:
                    used_placeholders.append(placeholder)
                    placeholder_to_statistic[statistic_name] = placeholder
            
            # Convert to improved format with grouped distributions
            for name, result in stats_results.items():
                stat_entry = {
                    'target_distribution': result.target or {},
                    'observed_distribution': result.observed,
                    'sample_size': len(people),
                    'households_analyzed': len(households),
                    'fit_metrics': {}
                }
                
                # Add placeholder information for reference
                if name in placeholder_to_statistic:
                    stat_entry['placeholder_used'] = placeholder_to_statistic[name]
                
                # Add fit metrics (JSD but not absolute differences)
                if result.fit_metrics:
                    fit_metrics = {}
                    if 'jensen_shannon_divergence' in result.fit_metrics:
                        fit_metrics['jensen_shannon_divergence'] = result.fit_metrics['jensen_shannon_divergence']
                    if 'total_variation_distance' in result.fit_metrics:
                        fit_metrics['total_variation_distance'] = result.fit_metrics['total_variation_distance']
                    if 'chi_squared' in result.fit_metrics:
                        fit_metrics['chi_squared'] = result.fit_metrics['chi_squared']
                    stat_entry['fit_metrics'] = fit_metrics
                
                metadata.statistical_analysis[name] = stat_entry
            
            # Record the placeholders that were actually used in the generation parameters
            if hasattr(metadata, 'metadata') and 'generation_parameters' in metadata.metadata:
                metadata.metadata['generation_parameters']['classifiers_used'] = used_placeholders
            
            # Add overall fit summary
            if stats_results:
                js_values = []
                tv_values = []
                distribution_summaries = []
                
                for name, result in stats_results.items():
                    if result.fit_metrics:
                        js = result.fit_metrics.get('jensen_shannon_divergence')
                        tv = result.fit_metrics.get('total_variation_distance')
                        if js is not None:
                            js_values.append(js)
                        if tv is not None:
                            tv_values.append(tv)
                        
                        distribution_summaries.append({
                            'name': name,
                            'jensen_shannon': js,
                            'total_variation': tv
                        })
                
                overall_summary = {
                    'num_distributions': len(stats_results),
                    'distributions_with_targets': len([r for r in stats_results.values() if r.target]),
                    'avg_jensen_shannon': statistics.mean(js_values) if js_values else None,
                    'avg_total_variation': statistics.mean(tv_values) if tv_values else None,
                    'worst_jensen_shannon': max(js_values) if js_values else None,
                    'best_jensen_shannon': min(js_values) if js_values else None,
                    'distributions_summary': distribution_summaries
                }
                
                metadata.statistical_analysis['_overall_fit_summary'] = overall_summary
        
        # Set basic info for backward compatibility
        metadata.generation_timestamp = datetime.now()
        metadata.total_households = len(households)
        metadata.total_people = len(people)
        
        # Basic demographics analysis (legacy format for compatibility)
        ages = [p.get('age') for p in people if p.get('age') is not None]
        genders = [p.get('gender') for p in people if p.get('gender')]
        relationships = [p.get('relationship') for p in people if p.get('relationship')]
        
        demographic_distributions = {}
        
        # Age analysis
        if ages:
            demographic_distributions['age'] = {
                'mean': round(statistics.mean(ages), 2),
                'median': statistics.median(ages),
                'std_dev': round(statistics.stdev(ages) if len(ages) > 1 else 0, 2),
                'range': [min(ages), max(ages)],
                'count': len(ages)
            }
        
        # Gender analysis
        if genders:
            gender_counts = Counter(genders)
            total = sum(gender_counts.values())
            demographic_distributions['gender'] = {
                name: {'count': count, 'percentage': round(count / total * 100, 2)}
                for name, count in gender_counts.items()
            }
        
        # Relationship analysis
        if relationships:
            rel_counts = Counter(relationships)
            total = sum(rel_counts.values())
            demographic_distributions['relationship'] = {
                name: {'count': count, 'percentage': round(count / total * 100, 2)}
                for name, count in rel_counts.items()
            }
        
        # Household size analysis
        household_sizes = []
        for household in households:
            if isinstance(household, dict) and 'household' in household:
                household_sizes.append(len(household['household']))
        
        if household_sizes:
            size_counts = Counter(household_sizes)
            total = sum(size_counts.values())
            demographic_distributions['household_size'] = {
                f"size_{size}": {'count': count, 'percentage': round(count / total * 100, 2)}
                for size, count in size_counts.items()
            }
            
            demographic_distributions['household_size']['statistics'] = {
                'mean': round(statistics.mean(household_sizes), 2),
                'median': statistics.median(household_sizes),
                'mode': max(size_counts, key=size_counts.get),
                'range': [min(household_sizes), max(household_sizes)]
            }
        
        metadata.demographic_distributions = demographic_distributions
        
        # Quality metrics
        complete_records = sum(
            1 for p in people 
            if all(p.get(field) for field in ['age', 'gender', 'relationship'])
        )
        
        metadata.quality_metrics = {
            'total_people': len(people),
            'total_households': len(households),
            'complete_records': complete_records,
            'completeness_rate': round(complete_records / len(people) * 100, 2) if people else 0,
            'average_household_size': round(len(people) / len(households), 2) if households else 0
        }
        
        return metadata
    
    def _save_comprehensive_metadata(self, metadata: ComprehensiveMetadata, output_name: str) -> Path:
        """Save comprehensive metadata to JSON file in improved format."""
        output_path = self.output_base_dir / f"{output_name}_metadata.json"
        
        # Create the improved format structure
        if metadata.statistical_analysis:
            # New improved format with metadata + statistical_analysis structure
            output_data = {
                'metadata': metadata.metadata,
                'statistical_analysis': metadata.statistical_analysis,
                'format_version': '2.0'
            }
        else:
            # Fallback to legacy format for backward compatibility
            output_data = asdict(metadata)
            output_data['generation_timestamp'] = metadata.generation_timestamp.isoformat()
            if 'metadata' in output_data:
                del output_data['metadata']  # Remove empty metadata section
            if 'statistical_analysis' in output_data:
                del output_data['statistical_analysis']  # Remove empty statistical section
        
        with output_path.open('w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        return output_path

    def export_households_normalized(
        self,
        households: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """Export households to normalized CSV format (convenience method).
        
        Args:
            households: Generated household data
            output_path: Path for output CSV file
            
        Returns:
            Path to saved CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use the same logic as _save_normalized_csv
        people_data = []
        for household_idx, household in enumerate(households, 1):
            if isinstance(household, dict) and 'household' in household:
                people = household['household']
            else:
                continue
            
            for person_idx, person in enumerate(people, 1):
                if isinstance(person, dict):
                    record = {
                        'household_id': household_idx,
                        'person_id': person_idx,
                        **person
                    }
                    people_data.append(record)
        
        if people_data:
            df = pd.DataFrame(people_data)
            
            # Standard column ordering
            standard_cols = ['household_id', 'person_id', 'age', 'gender', 'relationship', 'name']
            other_cols = [col for col in df.columns if col not in standard_cols]
            ordered_cols = [col for col in standard_cols if col in df.columns] + sorted(other_cols)
            df = df[ordered_cols]
            
            df.to_csv(output_path, index=False)
        
        return output_path


def save_generation_results(
    households: List[Dict[str, Any]],
    model_info: Dict[str, Any],
    generation_parameters: Dict[str, Any],
    output_dir: Path,
    output_name: str,
    classifiers: Optional[List] = None,
    statistics_manager: Optional[Any] = None,
    **kwargs
) -> Dict[str, Path]:
    """Convenience function to save population generation results in improved format.
    
    Args:
        households: Generated household data
        model_info: Information about the LLM used
        generation_parameters: Parameters used for generation
        output_dir: Directory to save files
        output_name: Base name for output files
        classifiers: List of classifiers used for statistical analysis
        statistics_manager: Existing statistics manager with registered classifiers and target data
        **kwargs: Additional arguments for cost/failure tracking
        
    Returns:
        Dictionary with paths to saved files
    """
    saver = PopulationDataSaver(output_dir)
    
    # Generate comprehensive metadata with statistical analysis
    metadata = saver._generate_comprehensive_metadata(households, classifiers, statistics_manager)
    
    # Update metadata structure with improved format
    metadata.metadata.update({
        'model_info': {
            'name': model_info.get('name', 'unknown'),
            'type': model_info.get('type', 'unknown'),
            'temperature': model_info.get('temperature', 0.7),
            **{k: v for k, v in model_info.items() if k not in ['name', 'type', 'temperature']}
        },
        'generation_parameters': {
            'n_households': generation_parameters.get('n_households', len(households)),
            'batch_size': generation_parameters.get('batch_size', 10),
            'location': generation_parameters.get('location', 'unknown'),
            **{k: v for k, v in generation_parameters.items() 
               if k not in ['n_households', 'batch_size', 'location']}
        }
    })
    
    # Update cost tracking in metadata
    if any(key in kwargs for key in ['total_cost', 'cost_breakdown', 'token_usage', 'total_requests']):
        cost_tracking = metadata.metadata['cost_tracking']
        
        if 'total_cost' in kwargs:
            cost_tracking['total_cost'] = kwargs['total_cost']
            cost_tracking['cost_per_household'] = kwargs['total_cost'] / len(households) if households else 0
        
        if 'token_usage' in kwargs:
            token_info = kwargs['token_usage']
            cost_tracking['total_tokens'] = token_info.get('total_tokens', 0)
        
        # Determine total_calls from multiple sources
        total_calls = kwargs.get('total_requests', 0)
        
        # Fallback: if total_requests is 0 but we have households, estimate from batch size
        if total_calls == 0 and len(households) > 0:
            batch_size = generation_parameters.get('batch_size', 10)
            # Estimate number of calls based on households and batch size
            total_calls = len(households)  # Conservative estimate: one call per household
            # Could also use: math.ceil(len(households) / batch_size) for batch-based estimate
        
        cost_tracking['total_calls'] = total_calls
        if total_calls > 0 and 'token_usage' in kwargs:
            cost_tracking['average_tokens_per_call'] = (
                kwargs['token_usage'].get('total_tokens', 0) / total_calls
            )
    
    # Update failure analysis in metadata  
    if any(key in kwargs for key in ['failed_requests', 'total_requests']):
        failure_metrics = metadata.metadata['failure_analysis']['generation_success_metrics']
        total_requests = kwargs.get('total_requests', 0)
        failed_requests = kwargs.get('failed_requests', 0)
        
        # Use the same total_calls calculation from cost_tracking
        if total_requests == 0 and len(households) > 0:
            total_requests = len(households)  # Same fallback as cost_tracking
        
        failure_metrics.update({
            'total_prompts': total_requests,
            'successful_prompts': total_requests - failed_requests,
            'failed_prompts': failed_requests,
            'success_rate': (total_requests - failed_requests) / total_requests if total_requests > 0 else 1.0,
            'failure_rate': failed_requests / total_requests if total_requests > 0 else 0.0
        })
    
    # Set backward compatibility fields
    metadata.model_info = metadata.metadata['model_info']
    metadata.generation_parameters = metadata.metadata['generation_parameters']
    metadata.total_cost = kwargs.get('total_cost')
    metadata.cost_breakdown = kwargs.get('cost_breakdown', {})
    metadata.token_usage = kwargs.get('token_usage', {})
    metadata.total_requests = kwargs.get('total_requests', 0)
    metadata.failed_requests = kwargs.get('failed_requests', 0)
    
    result = saver.save_population_data(
        households=households,
        output_name=output_name,
        metadata=metadata
    )
    
    # Return in expected format for backward compatibility
    return {
        'population_csv': str(result.get('csv', '')),
        'metadata_json': str(result.get('metadata', ''))
    }


# Alias for cleaner naming
save_population = save_generation_results