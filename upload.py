from huggingface_hub import HfApi 
api = HfApi() 
api.upload_folder(folder_path=".", repo_id="Frewiin/sepsis-openenv", repo_type="space", ignore_patterns=["venv/*", "__pycache__/*", "*.pyc"]) 
print("Upload complete!") 
