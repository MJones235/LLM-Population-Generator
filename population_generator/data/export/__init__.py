"""
Data export package for population generation.

Provides modular export functionality with CSV export, metadata building,
and statistical analysis components.
"""

from .main import PopulationDataSaver
from .csv_exporter import CSVExporter
from .metadata_builder import MetadataBuilder, ComprehensiveMetadata

# Legacy compatibility function
def save_generation_results(
    households,
    output_dir,
    output_name,
    cost_tracker=None,
    failure_tracker=None,
    classifiers=None,
    statistics_manager=None,
    model_info=None,
    generation_parameters=None,
    **kwargs  # Accept any additional keyword arguments for backward compatibility
):
    """Legacy function for backward compatibility."""
    saver = PopulationDataSaver(output_dir)
    return saver.save_population_data(
        households=households,
        output_name=output_name,
        cost_tracker=cost_tracker,
        failure_tracker=failure_tracker,
        classifiers=classifiers,
        statistics_manager=statistics_manager,
        model_info=model_info,
        generation_parameters=generation_parameters
    )

__all__ = [
    'PopulationDataSaver',
    'CSVExporter', 
    'MetadataBuilder',
    'ComprehensiveMetadata',
    'save_generation_results'
]