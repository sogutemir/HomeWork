import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Try to import rasterio for geospatial data
try:
    import rasterio
    has_rasterio = True
except ImportError:
    has_rasterio = False
    print("Rasterio not found. Using numpy for image processing.")
    print("To install rasterio: pip install rasterio")

try:
    import cv2
    has_cv2 = True
except ImportError:
    has_cv2 = False

# Function to read a multispectral image with flexible format support
def read_multispectral_image(file_path):
    # Try different methods to read the image based on available libraries and file extension
    
    if has_rasterio and file_path.lower().endswith(('.tif', '.tiff')):
        print("Reading TIFF with rasterio...")
        with rasterio.open(file_path) as src:
            # Read all bands
            image = src.read()
            return image
    
    elif has_cv2:
        print("Reading image with OpenCV...")
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # Check if image was read correctly
        if image is None:
            raise ValueError(f"Could not read image at {file_path} with OpenCV.")
        
        # Check dimensions and rearrange if needed
        if len(image.shape) == 3:
            # If the shape is (height, width, channels), transpose to (channels, height, width)
            image = np.transpose(image, (2, 0, 1))
        elif len(image.shape) == 2:
            # If the image is grayscale, add extra dimensions to make it (1, height, width)
            image = np.expand_dims(image, axis=0)
        
        return image
    
    else:
        # Try to use numpy directly
        print("Reading image with numpy...")
        try:
            # For numpy binary files
            if file_path.lower().endswith('.npy'):
                image = np.load(file_path)
                return image
            else:
                raise ValueError(f"Unsupported file format. Please install rasterio or OpenCV.")
        except:
            raise ValueError(f"Could not read image at {file_path}. Please install rasterio or OpenCV.")

# Function to perform k-means clustering
def perform_kmeans_clustering(image, n_clusters=5):
    # Reshape image for clustering
    # Original shape: (bands, height, width)
    # New shape: (height*width, bands)
    h, w = image.shape[1], image.shape[2]
    reshaped_image = image.reshape(image.shape[0], -1).T
    
    # Normalize the data
    scaler = StandardScaler()
    scaled_image = scaler.fit_transform(reshaped_image)
    
    # Apply K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_image)
    
    # Reshape labels back to image dimensions
    clustered_image = labels.reshape(h, w)
    
    return clustered_image, kmeans

# Function to calculate NDVI (Normalized Difference Vegetation Index)
def calculate_ndvi(red_band, nir_band):
    print("Calculating NDVI...")
    # Avoid division by zero
    denominator = red_band + nir_band
    ndvi = np.zeros_like(red_band, dtype=np.float32)
    valid_mask = denominator > 0
    ndvi[valid_mask] = (nir_band[valid_mask] - red_band[valid_mask]) / denominator[valid_mask]
    
    return ndvi

# Function to calculate correlation matrix between bands
def calculate_correlation_matrix(image):
    print("Calculating correlation matrix...")
    # Reshape image for correlation calculation
    reshaped_image = image.reshape(image.shape[0], -1)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(reshaped_image)
    
    return corr_matrix

# Function to visualize results
def visualize_results(image, clustered_image, ndvi, corr_matrix):
    print("Creating visualizations...")
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Normalize individual bands for better visualization
    def normalize_band(band):
        band_min = band.min()
        band_max = band.max()
        if band_max > band_min:
            return (band - band_min) / (band_max - band_min)
        return np.zeros_like(band)
    
    # Individual bands
    axes[0, 0].imshow(normalize_band(image[0]), cmap='Reds')
    axes[0, 0].set_title('Red Band')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(normalize_band(image[1]), cmap='Greens')
    axes[0, 1].set_title('Green Band')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(normalize_band(image[2]), cmap='Blues')
    axes[0, 2].set_title('Blue Band')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(normalize_band(image[3]), cmap='gray')
    axes[1, 0].set_title('NIR Band')
    axes[1, 0].axis('off')
    
    # RGB composite (using Red, Green, Blue bands)
    rgb = np.stack((normalize_band(image[0]), 
                    normalize_band(image[1]), 
                    normalize_band(image[2])), axis=2)
    
    axes[1, 1].imshow(rgb)
    axes[1, 1].set_title('RGB Composite')
    axes[1, 1].axis('off')
    
    # K-means clustering result
    axes[1, 2].imshow(clustered_image, cmap='viridis')
    axes[1, 2].set_title('K-Means Clustering (k=5)')
    axes[1, 2].axis('off')
    
    # NDVI visualization
    # Typically NDVI ranges from -1 to 1, so we use a diverging colormap
    axes[2, 0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2, 0].set_title('NDVI')
    axes[2, 0].axis('off')
    
    # Correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                xticklabels=['Red', 'Green', 'Blue', 'NIR'],
                yticklabels=['Red', 'Green', 'Blue', 'NIR'],
                ax=axes[2, 1])
    axes[2, 1].set_title('Correlation Matrix')
    
    # Empty subplot or add a legend
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('land_cover_classification_results.png', dpi=300)
    print("Results saved to 'land_cover_classification_results.png'")
    plt.show()

# Main function
def main():
    # File path - CHANGE THIS TO YOUR IMAGE PATH
    file_path = 'multispectral.tif'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please update the file_path variable in the script with the correct path to your multispectral image.")
        return
    
    # Read image
    print(f"Reading image from {file_path}...")
    try:
        image = read_multispectral_image(file_path)
    except Exception as e:
        print(f"Error reading image: {e}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Check if image has enough bands
    if image.shape[0] < 4:
        print(f"Warning: Image has only {image.shape[0]} bands, but 4 are required (R, G, B, NIR).")
        print("Attempting to continue with available bands...")
        
        # Pad with zeros if needed
        if image.shape[0] < 4:
            padded_image = np.zeros((4, image.shape[1], image.shape[2]), dtype=image.dtype)
            padded_image[:image.shape[0]] = image
            image = padded_image
            print(f"Padded image shape: {image.shape}")
    
    # Ensure we only use the first 4 bands (R, G, B, NIR)
    image = image[:4]
    
    # Perform k-means clustering with 5 clusters
    try:
        clustered_image, kmeans = perform_kmeans_clustering(image, n_clusters=5)
    except Exception as e:
        print(f"Error during clustering: {e}")
        return
    
    # Calculate NDVI (using red and NIR bands)
    try:
        ndvi = calculate_ndvi(image[0], image[3])
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        ndvi = np.zeros_like(image[0])
    
    # Calculate correlation matrix
    try:
        corr_matrix = calculate_correlation_matrix(image)
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        corr_matrix = np.eye(4)
    
    # Visualize results
    try:
        visualize_results(image, clustered_image, ndvi, corr_matrix)
    except Exception as e:
        print(f"Error during visualization: {e}")
        return
    
    # Print cluster centers
    print("\nCluster centers (Red, Green, Blue, NIR):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i+1}: {center}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()