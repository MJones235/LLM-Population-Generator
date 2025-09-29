"""Main population data exporter orchestrator."""

from pathlib import Path
from typing import Dict, List, Any, Optional

from .csv_exporter import CSVExporter
from .metadata_builder import MetadataBuilder, ComprehensiveMetadata


class PopulationDataSaver:
    """Enhanced population data saver with modular architecture."""
    
    def __init__(self, output_base_dir: str):
        """Initialize population data saver.
        
        Args:
            output_base_dir: Base directory for all output files
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.csv_exporter = CSVExporter(self.output_base_dir)
        self.metadata_builder = MetadataBuilder()
    
    def save_population_data(self,
                           households: List[Dict[str, Any]],
                           output_name: str,
                           cost_tracker=None,
                           failure_tracker=None,
                           classifiers: Optional[List] = None,
                           statistics_manager: Optional[Any] = None,
                           model_info: Optional[Dict[str, Any]] = None,
                           generation_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """Save complete population data with CSV and metadata.
        
        Args:
            households: Generated household data
            output_name: Base name for output files
            cost_tracker: Optional cost tracking object
            failure_tracker: Optional failure tracking object
            classifiers: Optional list of classifiers used
            statistics_manager: Optional statistics manager
            model_info: Optional model information
            generation_parameters: Optional generation parameters
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        # Save CSV data
        csv_path = self.csv_exporter.export_normalized_csv(households, output_name)
        if csv_path:
            saved_files['csv'] = csv_path
        
        # Build and save metadata
        metadata_dict = self.metadata_builder.build_comprehensive_metadata(
            households=households, 
            classifiers=classifiers, 
            statistics_manager=statistics_manager,
            model_info=model_info,
            generation_parameters=generation_parameters,
            cost_tracking=self._build_cost_tracking(cost_tracker),
            failure_analysis=self._build_failure_analysis(failure_tracker)
        )
        
        # Save metadata
        metadata_path = self.output_base_dir / f"{output_name}_metadata.json"
        
        # Save directly as dictionary
        import json
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False, default=str)
        
        saved_files['metadata'] = metadata_path
        
        return saved_files
    
    def export_households_normalized(self,
                                   households: List[Dict[str, Any]],
                                   output_name: str) -> Optional[Path]:
        """Export households in normalized CSV format only.
        
        Args:
            households: List of household dictionaries
            output_name: Base name for output file
            
        Returns:
            Path to saved CSV file
        """
        return self.csv_exporter.export_normalized_csv(households, output_name)
    
    def _build_cost_tracking(self, cost_tracker=None) -> Dict[str, Any]:
        """Build cost tracking data dictionary.
        
        Args:
            cost_tracker: Cost tracking object
            
        Returns:
            Dictionary with cost tracking data
        """
        if not cost_tracker or not hasattr(cost_tracker, 'get_cost_metadata'):
            return {
                'total_cost': 0.0,
                'token_usage': {'total_tokens': 0},
                'cost_breakdown': {}
            }
        
        cost_data = cost_tracker.get_cost_metadata()
        return cost_data if cost_data else {
            'total_cost': 0.0,
            'token_usage': {'total_tokens': 0},
            'cost_breakdown': {}
        }
    
    def _build_failure_analysis(self, failure_tracker=None) -> Dict[str, Any]:
        """Build failure analysis data dictionary.
        
        Args:
            failure_tracker: Failure tracking object
            
        Returns:
            Dictionary with failure analysis data
        """
        if not failure_tracker:
            return {
                'total_attempts': 0,
                'total_failures': 0,
                'failure_breakdown': {}
            }
        
        # Try different method names that might exist
        failure_data = None
        if hasattr(failure_tracker, 'get_academic_summary'):
            academic_summary = failure_tracker.get_academic_summary()
            if academic_summary:
                # Extract the relevant data from academic summary
                success_metrics = academic_summary.get('generation_success_metrics', {})
                failure_breakdown = academic_summary.get('failure_type_breakdown', {})
                
                failure_data = {
                    'total_attempts': success_metrics.get('total_prompts', 0),
                    'total_failures': success_metrics.get('failed_prompts', 0),
                    'failure_breakdown': failure_breakdown
                }
        elif hasattr(failure_tracker, 'get_summary'):
            failure_data = failure_tracker.get_summary()
        
        return failure_data if failure_data else {
            'total_attempts': 0,
            'total_failures': 0,
            'failure_breakdown': {}
        }

    def _update_metadata_with_trackers(self, 
                                     metadata: ComprehensiveMetadata,
                                     cost_tracker=None,
                                     failure_tracker=None):
        """Update metadata with cost and failure tracking data.
        
        Args:
            metadata: Metadata object to update
            cost_tracker: Cost tracking object
            failure_tracker: Failure tracking object
        """
        # Update cost information
        if cost_tracker and hasattr(cost_tracker, 'get_cost_metadata'):
            cost_data = cost_tracker.get_cost_metadata()
            if cost_data:
                metadata.total_cost = cost_data.get('total_cost', 0)
                metadata.cost_breakdown = cost_data.get('cost_breakdown', {})
                metadata.token_usage = cost_data.get('token_usage', {})
                
                # Update metadata section
                if 'cost_analysis' in metadata.metadata:
                    metadata.metadata['cost_analysis'].update({
                        'total_tokens': cost_data.get('token_usage', {}).get('total_tokens', 0),
                        'total_cost': cost_data.get('total_cost', 0.0),
                        'average_tokens_per_call': cost_data.get('token_usage', {}).get('total_tokens', 0) / max(1, metadata.total_households),
                        'cost_per_household': cost_data.get('total_cost', 0.0) / max(1, metadata.total_households)
                    })
        
        # Update failure information
        if failure_tracker and hasattr(failure_tracker, 'get_summary'):
            failure_data = failure_tracker.get_summary()
            if failure_data:
                metadata.total_requests = failure_data.get('total_attempts', 0)
                metadata.failed_requests = failure_data.get('total_failures', 0)
                metadata.failure_categories = failure_data.get('failure_breakdown', {})
                
                # Update metadata section
                if 'failure_analysis' in metadata.metadata:
                    success_rate = (metadata.total_requests - metadata.failed_requests) / max(1, metadata.total_requests)
                    metadata.metadata['failure_analysis']['generation_success_metrics'].update({
                        'total_prompts': metadata.total_requests,
                        'success_rate': round(success_rate, 3),
                        'failure_rate': round(1 - success_rate, 3),
                        'successful_prompts': metadata.total_requests - metadata.failed_requests,
                        'failed_prompts': metadata.failed_requests
                    })
                    
                    metadata.metadata['failure_analysis']['failure_type_breakdown'].update(
                        failure_data.get('failure_breakdown', {})
                    )