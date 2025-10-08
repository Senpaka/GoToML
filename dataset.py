import kagglehub

# Download latest version
path = kagglehub.dataset_download("anelim288/housing-prices")

print("Path to dataset files:", path)