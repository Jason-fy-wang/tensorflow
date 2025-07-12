from PIL import Image
import os

folder = "C:\\Users\\54409\\OneDrive\\图片\\wukong"

all_files = [os.path.join(folder, f) for f in os.listdir(folder)]
print(f"files: {all_files}")

for file in all_files:
    print(f"converting file: {file.removesuffix('.webp')}")
    img = Image.open(file, mode="r").convert("RGB")
    img.save(f"{file.removesuffix('.webp')}.jpg","JPEG")
    
