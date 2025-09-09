"""Demo of the flexible data loading system.

This example shows how the system can automatically handle multiple data formats:
1. Raw UK Census CSV files with codes and observations
2. Preprocessed CSV files with category/percentage columns
3. JSON files with metadata and structured data
"""

from population_generator.utils.data_loading import FlexibleDataManager
from population_generator.utils.statistics import StatisticsManager
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier
from pathlib import Path


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_data_loading():
    """Demonstrate flexible data loading capabilities."""
    
    print_separator("Flexible Data Loading System Demo")
    print("This demo shows automatic format detection and loading for:")
    print("• Raw UK Census CSV files")
    print("• Preprocessed CSV files") 
    print("• JSON files with metadata")
    
    # Initialize data manager
    data_manager = FlexibleDataManager()
    
    print(f"\nSupported formats:")
    for fmt in data_manager.list_supported_formats():
        print(f"  ✅ {fmt}")
    
    # Test data files
    formats_data_dir = Path("examples/data/formats")
    test_files = [
        "raw_uk_census.csv",
        "preprocessed.csv", 
        "json_with_metadata.json"
    ]
    
    print_separator("Loading Different Data Formats")
    
    for filename in test_files:
        file_path = formats_data_dir / filename
        
        if not file_path.exists():
            print(f"❌ Skipping {filename} - file not found")
            continue
            
        print(f"\n📁 Loading: {filename}")
        
        try:
            # Get metadata first
            metadata = data_manager.get_metadata(file_path)
            if metadata:
                print(f"   📋 Format: {metadata.source}")
                if metadata.region:
                    print(f"   🌍 Region: {metadata.region}")
                if metadata.year:
                    print(f"   📅 Year: {metadata.year}")
                if metadata.description:
                    print(f"   📝 Description: {metadata.description}")
            
            # Load the data
            data = data_manager.load_target_data(file_path)
            
            print(f"   📊 Categories loaded: {len(data)}")
            print("   📈 Distribution:")
            
            # Show data (limit to first few entries for readability)
            items = list(data.items())[:6]
            for category, percentage in items:
                print(f"      {category}: {percentage}%")
            
            if len(data) > 6:
                print(f"      ... and {len(data) - 6} more categories")
                
            print(f"   ✅ Successfully loaded {filename}")
            
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    print_separator("Integration with Statistics System")
    
    # Test integration with statistics system
    stats_manager = StatisticsManager(data_dir="examples/sample_data")
    
    # Register classifier with different data sources
    classifier = UKHouseholdSizeClassifier()
    
    print("\\n🔧 Testing target data loading in StatisticsManager:")
    
    for filename in test_files:
        file_path = formats_data_dir / filename
        if file_path.exists():
            print(f"\\n   📁 Testing with {filename}:")
            try:
                # This will use our flexible data loading system
                target_data = stats_manager._load_target_data(filename)
                if target_data:
                    print(f"      ✅ Loaded {len(target_data)} categories")
                    # Show first few categories
                    sample_items = list(target_data.items())[:3]
                    for cat, pct in sample_items:
                        print(f"         {cat}: {pct}%")
                else:
                    print("      ❌ No data loaded")
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    print_separator("Demo Complete")
    print("🎉 Flexible data loading system successfully demonstrated!")
    print("\\n💡 Key Benefits:")
    print("   ✅ Automatic format detection and conversion")
    print("   ✅ Support for multiple data structures")
    print("   ✅ Metadata extraction and logging")
    print("   ✅ Seamless integration with existing statistics system")
    print("   ✅ Extensible with custom loaders")
    print("   ✅ Caching for performance")
    
    print("\\n🚀 Users can now provide target data in any supported format!")


if __name__ == "__main__":
    demo_data_loading()
