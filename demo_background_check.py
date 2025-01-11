import matplotlib.pyplot as plt #type: ignore
from uniformbackground import UnifiedBackgroundAnalyzer

def visualize_analysis(image_path, method_type='all'):
    """
    Visualize the analysis results for specified method type.
    
    Parameters:
    - image_path: path to image file
    - method_type: 'basic', 'advanced', or 'all'
    """
    # Initialize analyzer
    analyzer = UnifiedBackgroundAnalyzer()
    
    # Get results
    results = analyzer.analyze_image(image_path, method_type=method_type)
    
    # Read and display image
    img = plt.imread(image_path)
    
    plt.figure(figsize=(15, 8))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display results
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # Overall result
    text = f"Overall Result ({method_type} methods):\n"
    text += f"Is Uniform: {results['is_uniform']}\n"
    text += f"Confidence: {results['confidence']:.2f}\n\n"
    
    # Individual results
    text += "Individual Methods:\n"
    for method, result in results['individual_results'].items():
        text += f"\n{method}:\n"
        text += f"  Uniform: {result['is_uniform']}\n"
        text += f"  Confidence: {result['confidence']:.2f}"
        if 'avg_difference' in result:
            text += f"\n  Avg Difference: {result['avg_difference']:.2f}"
        if 'unique_colors' in result:
            text += f"\n  Unique Colors: {result['unique_colors']}"
    
    plt.text(0.1, 0.95, text, fontsize=10, va='top')
    plt.title('Analysis Results')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Test with sample images
    test_image = "path/to/test_image.jpg"
    
    print("Testing basic methods...")
    visualize_analysis(test_image, method_type='basic')
    
    print("\nTesting advanced methods...")
    visualize_analysis(test_image, method_type='advanced')
    
    print("\nTesting all methods...")
    visualize_analysis(test_image, method_type='all')