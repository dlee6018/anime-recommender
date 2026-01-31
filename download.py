import kagglehub

# Download latest version
path = kagglehub.dataset_download("dbdmobile/myanimelist-dataset")

print("Path to dataset files:", path)