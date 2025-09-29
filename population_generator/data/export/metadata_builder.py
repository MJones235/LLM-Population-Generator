"""Metadata builder for population export."""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional


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


class MetadataBuilder:
    """Builds comprehensive metadata for population export."""
    
    def __init__(self):
        """Initialize metadata builder."""
        pass  # No hardcoded statistics builder needed
    
    def build_comprehensive_metadata(self, 
                                   households: List[Dict[str, Any]], 
                                   classifiers: Optional[List] = None,
                                   statistics_manager: Optional[Any] = None,
                                   model_info: Optional[Dict[str, Any]] = None,
                                   generation_parameters: Optional[Dict[str, Any]] = None,
                                   cost_tracking: Optional[Dict[str, Any]] = None,
                                   failure_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build comprehensive metadata with flat structure matching legacy format.
        
        Args:
            households: Generated household data
            classifiers: Optional list of classifiers for statistical analysis
            statistics_manager: Optional statistics manager for classifier data
            model_info: Model information (name, type, temperature, etc.)
            generation_parameters: Generation parameters (n_households, batch_size, etc.)
            cost_tracking: Cost tracking information
            failure_analysis: Failure analysis data
            
        Returns:
            Dictionary with flattened metadata structure
        """
        # Extract people for basic counts
        people = []
        for household in households:
            if isinstance(household, dict) and 'household' in household:
                people.extend(household['household'])
        
        # Build flat metadata structure
        metadata = {
            'metadata': {
                'timestamp': datetime.now().strftime('%Y%m%dT%H:%M:%S'),
                'model_info': model_info or {
                    'name': 'unknown',
                    'type': 'unknown',
                    'temperature': 0.7
                },
                'generation_parameters': generation_parameters or {
                    'n_households': len(households),
                    'batch_size': 10,
                    'location': 'unknown',
                    'max_retries': 3,
                    'timeout_seconds': 180
                },
                'data_sources': [],  # Can be populated if available
                'cost_tracking': cost_tracking or {
                    'total_calls': len(households),
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'average_tokens_per_call': 0.0,
                    'cost_per_household': 0.0
                },
                'failure_analysis': failure_analysis or {
                    'generation_success_metrics': {
                        'total_prompts': len(households),
                        'success_rate': 1.0,
                        'failure_rate': 0.0,
                        'successful_prompts': len(households),
                        'failed_prompts': 0
                    }
                },
                'run_id': str(uuid.uuid4())
            },
            'statistical_analysis': {},
            'format_version': '2.0'
        }
        
        # Build statistical analysis with flat classifier structure
        if statistics_manager:
            # Create DataFrame for classifier analysis
            people_data = []
            for household_idx, household in enumerate(households, 1):
                if isinstance(household, dict) and 'household' in household:
                    household_people = household['household']
                    for person_idx, person in enumerate(household_people, 1):
                        if isinstance(person, dict):
                            record = {
                                'household_id': household_idx,
                                'person_id': person_idx,
                                **person
                            }
                            people_data.append(record)
            
            if people_data:
                import pandas as pd
                df = pd.DataFrame(people_data)
                
                # Get dynamic statistics from registered classifiers
                results = statistics_manager.compute_all_statistics(df)
                
                # Build flat classifier structure
                for name, result in results.items():
                    # Get placeholder from statistics manager if available
                    placeholder_used = "UNKNOWN_STATS"
                    if hasattr(statistics_manager, 'placeholder_mappings'):
                        for placeholder, statistic_name in statistics_manager.placeholder_mappings.items():
                            if statistic_name == name:
                                placeholder_used = placeholder
                                break
                    
                    metadata['statistical_analysis'][name] = {
                        'target_distribution': result.target or {},
                        'observed_distribution': result.observed or {},
                        'sample_size': len(people),
                        'households_analyzed': len(households),
                        'fit_metrics': result.fit_metrics or {},
                        'placeholder_used': placeholder_used
                    }
                
                # Add overall fit summary
                if results:
                    metadata['statistical_analysis']['_overall_fit_summary'] = self._build_overall_fit_summary(results)
        
        return metadata

    def _build_overall_fit_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall fit summary from classifier results."""
        if not results:
            return {}
        
        js_values = []
        tv_values = []
        distributions_summary = []
        
        for name, result in results.items():
            if result.fit_metrics:
                js = result.fit_metrics.get('jensen_shannon_divergence', 0)
                tv = result.fit_metrics.get('total_variation_distance', 0)
                
                js_values.append(js)
                tv_values.append(tv)
                distributions_summary.append({
                    'name': name,
                    'jensen_shannon': js,
                    'total_variation': tv
                })
        
        if not js_values:
            return {}
        
        return {
            'num_distributions': len(results),
            'distributions_with_targets': len([r for r in results.values() if r.target]),
            'avg_jensen_shannon': sum(js_values) / len(js_values),
            'avg_total_variation': sum(tv_values) / len(tv_values),
            'worst_jensen_shannon': max(js_values),
            'best_jensen_shannon': min(js_values),
            'distributions_summary': distributions_summary
        }
    
    def _build_basic_metadata(self, households: List[Dict[str, Any]], people: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build basic metadata section."""
        return {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'run_id': str(uuid.uuid4()),
                'total_households_generated': len(households),
                'total_people_generated': len(people),
                'average_household_size': round(len(people) / len(households), 2) if households else 0
            },
            'data_quality': {
                'complete_records': sum(
                    1 for p in people 
                    if all(p.get(field) for field in ['age', 'gender', 'relationship'])
                ),
                'completeness_rate': round(
                    sum(1 for p in people if all(p.get(field) for field in ['age', 'gender', 'relationship'])) 
                    / len(people) * 100, 2
                ) if people else 0
            },
            'cost_analysis': {
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
                }
            }
        }
    
    def _build_statistical_analysis(self, 
                                  households: List[Dict[str, Any]], 
                                  classifiers: Optional[List] = None,
                                  statistics_manager: Optional[Any] = None) -> Dict[str, Any]:
        """Build statistical analysis section."""
        analysis = {}
        
        # Use statistics_manager for dynamic analysis if available
        if statistics_manager:
            # Create DataFrame for classifier analysis
            people_data = []
            for household_idx, household in enumerate(households, 1):
                if isinstance(household, dict) and 'household' in household:
                    household_people = household['household']
                    for person_idx, person in enumerate(household_people, 1):
                        if isinstance(person, dict):
                            record = {
                                'household_id': household_idx,
                                'person_id': person_idx,
                                **person
                            }
                            people_data.append(record)
            
            if people_data:
                import pandas as pd
                df = pd.DataFrame(people_data)
                
                # Get dynamic statistics from registered classifiers
                results = statistics_manager.compute_all_statistics(df)
                
                demographic_distributions = {}
                for name, result in results.items():
                    demographic_distributions[name] = {
                        'observed': result.observed,
                        'target': result.target,
                        'fit_metrics': result.fit_metrics
                    }
                
                analysis['demographic_distributions'] = demographic_distributions
                analysis['statistical_measures'] = {
                    'total_people': len(df),
                    'total_households': len(households),
                    'average_household_size': round(len(df) / len(households), 2) if households else 0
                }
            else:
                analysis['demographic_distributions'] = {}
                analysis['statistical_measures'] = {}
        else:
            # Fallback to basic analysis when no statistics_manager available
            analysis['demographic_distributions'] = self._build_basic_distributions(households, [])
            analysis['statistical_measures'] = self._build_basic_measures(households, [])
        
        # Add classifier analysis if available
        if classifiers and statistics_manager:
            analysis['classifier_analysis'] = self._build_classifier_analysis(
                households, classifiers, statistics_manager
            )
        
        return analysis
    
    def _build_classifier_analysis(self, 
                                 households: List[Dict[str, Any]], 
                                 classifiers: List,
                                 statistics_manager: Any) -> Dict[str, Any]:
        """Build classifier-specific analysis."""
        # Import here to avoid circular imports
        import pandas as pd
        
        # Convert to DataFrame for classifier analysis
        people_data = []
        for household_idx, household in enumerate(households, 1):
            if isinstance(household, dict) and 'household' in household:
                household_people = household['household']
                for person_idx, person in enumerate(household_people, 1):
                    if isinstance(person, dict):
                        record = {
                            'household_id': household_idx,
                            'person_id': person_idx,
                            **person
                        }
                        people_data.append(record)
        
        df = pd.DataFrame(people_data) if people_data else pd.DataFrame()
        
        classifier_data = {}
        placeholders_used = []
        
        # Analyze each classifier
        for placeholder, statistic_name in statistics_manager.placeholder_mappings.items():
            if statistic_name in statistics_manager.providers:
                provider = statistics_manager.providers[statistic_name]
                placeholders_used.append(placeholder)
                
                # Get classifier name
                classifier_name = getattr(provider.classifier, '__class__', type(provider.classifier)).__name__
                
                # Compute current distribution
                try:
                    if hasattr(provider.classifier, 'classify_batch') and not df.empty:
                        current_dist = provider.classifier.classify_batch(df)
                    else:
                        current_dist = {}
                except Exception as e:
                    current_dist = {'error': str(e)}
                
                classifier_data[statistic_name] = {
                    'classifier_name': classifier_name,
                    'placeholder_used': placeholder,
                    'target_distribution': getattr(provider, 'target_data', {}),
                    'current_distribution': current_dist,
                    'sample_size': len(df)
                }
        
        return {
            'classifiers_used': classifier_data,
            'placeholders_used': placeholders_used,
            'total_classifiers': len(classifier_data)
        }
    
    def _build_quality_metrics(self, households: List[Dict[str, Any]], people: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build data quality metrics."""
        complete_records = sum(
            1 for p in people 
            if all(p.get(field) for field in ['age', 'gender', 'relationship'])
        )
        
        return {
            'total_people': len(people),
            'total_households': len(households),
            'complete_records': complete_records,
            'completeness_rate': round(complete_records / len(people) * 100, 2) if people else 0,
            'average_household_size': round(len(people) / len(households), 2) if households else 0
        }
    
    def _build_basic_distributions(self, households: List[Dict[str, Any]], people: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build basic distributions when no statistics_manager available."""
        # Extract people if not provided
        if not people:
            people = []
            for household in households:
                if isinstance(household, dict) and 'household' in household:
                    people.extend(household['household'])
        
        distributions = {}
        
        # Basic gender distribution
        genders = [person.get('gender') for person in people if person.get('gender')]
        if genders:
            from collections import Counter
            gender_counts = Counter(genders)
            total = sum(gender_counts.values())
            distributions['gender'] = {
                value: {'count': count, 'percentage': round(count / total * 100, 2)}
                for value, count in gender_counts.items()
            }
        
        # Basic household size distribution
        household_sizes = []
        for household in households:
            if isinstance(household, dict) and 'household' in household:
                household_sizes.append(len(household['household']))
        
        if household_sizes:
            from collections import Counter
            size_counts = Counter(household_sizes)
            total = sum(size_counts.values())
            distributions['household_size'] = {
                f"size_{size}": {'count': count, 'percentage': round(count / total * 100, 2)}
                for size, count in size_counts.items()
            }
        
        return distributions
    
    def _build_basic_measures(self, households: List[Dict[str, Any]], people: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build basic statistical measures when no statistics_manager available."""
        # Extract people if not provided
        if not people:
            people = []
            for household in households:
                if isinstance(household, dict) and 'household' in household:
                    people.extend(household['household'])
        
        measures = {
            'population_statistics': {
                'total_people': len(people),
                'total_households': len(households),
                'average_household_size': round(len(people) / len(households), 2) if households else 0
            }
        }
        
        # Basic age statistics if available
        ages = [person.get('age') for person in people if person.get('age') is not None]
        if ages:
            import statistics
            measures['age_statistics'] = {
                'mean': round(statistics.mean(ages), 2),
                'median': statistics.median(ages),
                'range': [min(ages), max(ages)],
                'count': len(ages)
            }
        
        return measures
    
    def save_metadata(self, metadata: ComprehensiveMetadata, output_path: Path) -> Path:
        """Save comprehensive metadata to JSON file.
        
        Args:
            metadata: Metadata object to save
            output_path: Path where to save the JSON file
            
        Returns:
            Path to saved file
        """
        # Create the improved format structure
        if metadata.statistical_analysis:
            # New improved format with metadata + statistical_analysis structure
            output_data = {
                'metadata': metadata.metadata,
                'statistical_analysis': metadata.statistical_analysis,
            }
        else:
            # Legacy format for backward compatibility
            output_data = asdict(metadata)
            # Convert datetime to string for JSON serialization
            if 'generation_timestamp' in output_data:
                output_data['generation_timestamp'] = output_data['generation_timestamp'].isoformat()
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path