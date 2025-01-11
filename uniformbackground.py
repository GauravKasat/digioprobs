import numpy as np
import cv2 #type: ignore
from collections import Counter 
from sklearn.cluster import KMeans #type: ignore
import torch #type: ignore
import torchvision.transforms as transforms #type: ignore
from torchvision.models.segmentation import deeplabv3_resnet101 #type: ignore
import torch.nn as nn #type: ignore
from transformers import AutoModelForImageSegmentation #type:ignore

class CompleteBackgroundAnalyzer:
    def __init__(self, use_gpu=None):
        # Initialize device
        if use_gpu is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        else:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'mps')
        
        # Initialize models and thresholds
        self.setup_dl_models()
        self.setup_thresholds()
    
    def setup_dl_models(self):
        """Initialize deep learning models and transforms."""
        # Use DeepLabV3+ with ResNet101 backbone for better segmentation
        from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights #type: ignore
        
        self.segmentation_model = deeplabv3_resnet101(
                weights=DeepLabV3_ResNet101_Weights.DEFAULT
            )
        self.segmentation_model.eval()
        self.segmentation_model.to(self.device)
        
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        
        self.foreground_classes = {
            15: 'person',
            13: 'bench',
            14: 'bird',
            16: 'cat',
            17: 'dog',
            18: 'horse',
            19: 'sheep',
            20: 'cow',
            7: 'car',
            8: 'motorcycle',
            9: 'airplane',
            10: 'bus',
            11: 'train',
            12: 'truck',
        }
    
    def setup_thresholds(self):
        """Set thresholds for all methods."""
        self.thresholds = {
            'unique_colors': 3000,      # needs tuning
            'color_range': 80,          # needs tuning
            'dominant_ratio': 0.4,      # needs tuning
            'line_variance': 100,       # needs tuning
            'avg_difference': 40,       # needs tuning
            'edge_density': 0.15,       # needs tuning
            'cluster_size': 0.6,        # needs tuning
            'histogram_peak': 0.03      # needs tuning
        }
    
    def get_background_mask(self, image):
        """Extract background mask using improved segmentation."""
        height, width = image.shape[:2]
        max_dim = 1024
        
        # Resize while maintaining aspect ratio
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size)
        
        input_tensor = self.transforms(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.segmentation_model(input_batch)['out'][0]
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=0)
            
            # Initialize background mask
            background_mask = torch.ones_like(probabilities[0])
            
            # Exclude all foreground classes with high confidence
            for class_idx in self.foreground_classes:
                class_prob = probabilities[class_idx]
                background_mask[class_prob > 0.5] = 0  # Confidence threshold of 0.5
            
            background_mask = background_mask.cpu().numpy()
        
        # Enhanced mask cleanup
        kernel_close = np.ones((7,7), np.uint8)  # Larger kernel for closing
        kernel_open = np.ones((3,3), np.uint8)   # Smaller kernel for opening
        
        # Fill holes and remove small objects
        background_mask = cv2.morphologyEx(background_mask.astype(np.uint8), 
                                         cv2.MORPH_CLOSE, kernel_close)
        background_mask = cv2.morphologyEx(background_mask, 
                                         cv2.MORPH_OPEN, kernel_open)
        
        # Remove small isolated regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            background_mask.astype(np.uint8), connectivity=8
        )
        
        # Keep only the largest background component
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            background_mask = (labels == largest_label).astype(np.uint8)
        
        # If mask is too small, likely something went wrong
        if np.mean(background_mask) < 0.1:
            # Fallback to simpler background estimation
            background_mask = self._fallback_background_detection(image)
        
        return background_mask.astype(bool)
    
    def _fallback_background_detection(self, image):
        """Fallback method for background detection when segmentation fails."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use adaptive thresholding
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find the largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels != largest_label)
        else:
            mask = np.ones_like(gray, dtype=bool)
        
        return mask
    
    def get_background_pixels(self, image):
        """Get background pixels using segmentation mask."""
        # Convert to HSV for better color analysis
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Get mask where True is foreground (objects,people)
        mask = self.get_background_mask(image)
        
        # Resize mask to match image dimensions
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
        mask = mask.astype(bool)
        
        # Create background mask (True for background pixels)
        background_mask = ~mask  # Invert mask to get background
        
        # Get background pixels in both RGB and HSV
        background_pixels_rgb = image[background_mask]
        background_pixels_hsv = hsv_image[background_mask]
        
        return background_pixels_rgb, background_pixels_hsv, background_mask
    
    # Basic Methods
    
    def simple_color_counting(self, image):
        """Count unique colors in background with improved logic."""
        background_pixels, _, _ = self.get_background_pixels(image)
        
            # Quantize colors to reduce noise
        background_pixels = (background_pixels // 10) * 10
        pixel_tuples = [tuple(pixel) for pixel in background_pixels]
        unique_colors = len(set(pixel_tuples))
        

        unique_ratio = unique_colors / len(pixel_tuples)
        
        return {
            'is_uniform': unique_ratio < 0.2,  # needs tunning
            'confidence': max(0, 1 - unique_ratio),
            'method': 'color_counting',
            'unique_colors': unique_colors,
            'unique_ratio': unique_ratio
        }
    
    def color_range_check(self, image):
        """Check color range in background with HSV analysis."""
        _, background_pixels_hsv, _ = self.get_background_pixels(image)
        
        # Analyze hue and saturation separately
        hue = background_pixels_hsv[:, 0]
        saturation = background_pixels_hsv[:, 1]
        value = background_pixels_hsv[:, 2]
        
        # Calculate ranges
        hue_range = np.percentile(hue, 95) - np.percentile(hue, 5)
        sat_range = np.percentile(saturation, 95) - np.percentile(saturation, 5)
        val_range = np.percentile(value, 95) - np.percentile(value, 5)
        
        # Normalize ranges
        hue_score = 1 - (hue_range / 180)  # Hue is 0-180 in OpenCV
        sat_score = 1 - (sat_range / 255)
        val_score = 1 - (val_range / 255)
        
        # Weighted average of scores
        avg_score = (hue_score * 0.4 + sat_score * 0.3 + val_score * 0.3)
        
        return {
            'is_uniform': avg_score > 0.7, #needs tuning
            'confidence': avg_score,
            'method': 'color_range',
            'hue_range': hue_range,
            'sat_range': sat_range,
            'val_range': val_range
        }
    
    def dominant_color_ratio(self, image):
        """Calculate dominant color ratio using color quantization."""
        background_pixels, _, _ = self.get_background_pixels(image)
        
        # Quantize colors to reduce noise
        quantized = (background_pixels // 20) * 20
        pixel_tuples = [tuple(pixel) for pixel in quantized]
        
        # Get color frequencies
        color_counts = Counter(pixel_tuples)
        total_pixels = len(pixel_tuples)
        
        # Get top 3 colors and their ratios
        top_colors = color_counts.most_common(3)
        top_ratio = top_colors[0][1] / total_pixels if top_colors else 0
        
        # Calculate ratio of top 3 colors combined
        top_3_ratio = sum(count for _, count in top_colors) / total_pixels
        
        return {
            'is_uniform': top_ratio > 0.6 or top_3_ratio > 0.8, ###needs tunning
            'confidence': max(top_ratio, top_3_ratio - 0.2),
            'method': 'dominant_color',
            'dominant_ratio': top_ratio,
            'top_3_ratio': top_3_ratio
        }
    
    def row_column_scanning(self, image):
        """Scan background rows and columns for variance."""
        _, _, mask = self.get_background_pixels(image)
        height, width = image.shape[:2]

        ### division     
        row_positions = [height//4, height//2, 3*height//4]
        col_positions = [width//4, width//2, 3*width//4]
        
        row_variances = []
        col_variances = []
        
        for row in row_positions:
            row_mask = mask[row, :]
            if np.any(row_mask):
                row_pixels = image[row, :][row_mask]
                row_variances.append(np.var(row_pixels))
        
        for col in col_positions:
            col_mask = mask[:, col]
            if np.any(col_mask):
                col_pixels = image[:, col][col_mask]
                col_variances.append(np.var(col_pixels))
        
        all_variances = row_variances + col_variances
        if not all_variances:
            return {
                'is_uniform': False,
                'confidence': 0,
                'method': 'line_scanning',
                'avg_variance': float('inf')
            }
        
        avg_variance = np.mean(all_variances)
        
        return {
            'is_uniform': avg_variance < self.thresholds['line_variance'],
            'confidence': 1 - (avg_variance / self.thresholds['line_variance']),
            'method': 'line_scanning',
            'avg_variance': avg_variance
        }
    
    def background_averaging(self, image):
        """Compare background pixels to average with improved metrics."""
        background_pixels, background_pixels_hsv, _ = self.get_background_pixels(image)
        
        # RGB analysis
        avg_color = np.mean(background_pixels, axis=0)
        std_dev = np.std(background_pixels, axis=0)
        
        # HSV analysis
        avg_hsv = np.mean(background_pixels_hsv, axis=0)
        std_dev_hsv = np.std(background_pixels_hsv, axis=0)
        
        # Calculate normalized scores
        rgb_score = 1 - (np.mean(std_dev) / 255)
        hsv_score = 1 - (np.mean(std_dev_hsv) / 255)
        
        # Combined score with more weight to HSV
        combined_score = (rgb_score * 0.4 + hsv_score * 0.6)
        
        return {
            'is_uniform': combined_score > 0.7,   ##need tunning
            'confidence': combined_score,
            'method': 'background_averaging',
            'rgb_std_dev': np.mean(std_dev),
            'hsv_std_dev': np.mean(std_dev_hsv)
        }
    
    # Intermediate methods Methods
    
    def edge_based_detection(self, image):
        """Edge detection on background."""
        _, _, mask = self.get_background_pixels(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        background_edges = edges[mask]
        edge_density = np.sum(background_edges > 0) / background_edges.size
        
        return {
            'is_uniform': edge_density < self.thresholds['edge_density'], ##needs tuning
            'confidence': 1 - edge_density,
            'method': 'edge_detection',
            'edge_density': edge_density
        }
    
    def color_clustering(self, image):
        """Color clustering on background."""
        background_pixels, _, _ = self.get_background_pixels(image)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(background_pixels)
        largest_cluster = np.max(np.bincount(kmeans.labels_)) / len(kmeans.labels_)
        
        return {
            'is_uniform': largest_cluster > self.thresholds['cluster_size'],  ###neds tunning
            'confidence': largest_cluster,
            'method': 'color_clustering',
            'largest_cluster': largest_cluster
        }
    
    def histogram_analysis(self, image):
        """Histogram analysis of background."""
        _, _, mask = self.get_background_pixels(image)
        peak_ratios = []
        
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], mask.astype(np.uint8), 
                              [256], [0, 256])
            max_peak = np.max(hist)
            total_pixels = np.sum(hist)
            peak_ratios.append(max_peak / total_pixels)
        
        avg_peak_ratio = np.mean(peak_ratios)
        
        return {
            'is_uniform': avg_peak_ratio > self.thresholds['histogram_peak'], ###needs tunning
            'confidence': avg_peak_ratio,
            'method': 'histogram_analysis',
            'peak_ratio': avg_peak_ratio
        }
    
    def color_variance_analysis(self, image):
        """Advanced color variance analysis."""
        background_pixels, _, _ = self.get_background_pixels(image)
        std_devs = np.std(background_pixels, axis=0)
        avg_std = np.mean(std_devs)
        
        return {
            'is_uniform': avg_std < self.thresholds['color_variance'], ###needs tunning
            'confidence': 1 - (avg_std / 255),
            'method': 'color_variance',
            'std_dev': avg_std
        }
    
    def analyze_image(self, image_path, method_type='all'):
        """Analyze image using improved voting logic."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define methods with updated weights
        methods = {
            'color_counting': (self.simple_color_counting, 2),
            'edge_detection': (self.edge_based_detection, 2),
            'background_averaging': (self.background_averaging, 2),
            'color_range': (self.color_range_check, 1),
            'color_clustering': (self.color_clustering, 1),
            'dominant_color': (self.dominant_color_ratio, 1),
            'histogram_analysis': (self.histogram_analysis, 1)
        }
        
        # Run methods and collect votes
        results = {}
        total_weight = 0
        weighted_uniform_votes = 0  # Count weighted uniform votes
        uniform_votes = 0  # Count total uniform votes
        non_uniform_votes = 0  # Count total non-uniform votes

        for method_name, (method_func, weight) in methods.items():
            result = method_func(image)
            results[method_name] = result
            
            # Debugging output
            print(f"{method_name}: {result}")  # Print each method's result
            
            # Count votes
            if result['is_uniform']:
                uniform_votes += 1
                weighted_uniform_votes += weight  # Count weighted uniform votes
            else:
                non_uniform_votes += 1
            
            total_weight += weight  # Total weight for normalization

        # Calculate vote ratio
        
        
        # Decision logic: If weighted uniform votes exceed half of total weight, classify as uniform
        #### needd tunning
        is_uniform = weighted_uniform_votes / total_weight >=4

        return {
            'is_uniform': is_uniform,
            'confidence': weighted_uniform_votes / total_weight if total_weight > 0 else 0,
            'total_votes': total_weight,
            'total_methods': len(methods),
            'uniform_votes': uniform_votes,
            'non_uniform_votes': non_uniform_votes,
            'method': 'weighted_voting_ensemble',
            'individual_results': results
        }

def visualize_results(image_path, method_type='all'):
    """Visualize analysis results with voting details."""
    import matplotlib.pyplot as plt
    
    analyzer = CompleteBackgroundAnalyzer()
    results = analyzer.analyze_image(image_path, method_type)

    # Read and process image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(25, 12))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    # Results text
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    text = "=== Overall Result ===\n"
    text += f"Is Uniform: {results['is_uniform']}\n"
    text += f"Confidence: {results['confidence']:.3f}\n"

    
    text += "=== Detailed Method Calculations ===\n\n"
    
    # Add method details
    for method, result in results['individual_results'].items():
        text += f"{method}:\n"
        for key, value in result.items():
            if key not in ['method']:
                if isinstance(value, float):
                    text += f"  {key}: {value:.3f}\n"
                else:
                    text += f"  {key}: {value}\n"
        text += "\n"
    
    plt.text(0.02, 0.98, text, fontsize=12, va='top', fontfamily='monospace',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    test_image = "/Users/gauravkasat/Desktop/Screenshot 2025-01-11 at 10.41.19 AM.png"
    visualize_results(test_image, method_type='all')