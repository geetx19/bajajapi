import os
import requests

input_folder = os.getenv("INPUT_FOLDER","./input")
print(input_folder)
def download_pdf_file(url :str,filename : str):
    os.makedirs(input_folder, exist_ok=True)  # Ensure folder exists
    file_path = os.path.join(input_folder,filename)

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ File saved to {file_path}")
        return file_path
    else:
        raise Exception(f"❌ Failed to download. Status code: {response.status_code}")

