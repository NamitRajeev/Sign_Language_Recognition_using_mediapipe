import os
import shutil

# Set your dataset path (make sure it's correct)
data_path = r"C:\Users\namit\OneDrive\Desktop\RSET\Other_projects\ISL_2\ISL_custom_data"

# Only keep these folders
keep_folders = {'A', 'B', 'C', 'F', 'I', 'M', 'O', 'R', 'S', 'V', 'W', 'X', 'Y', 'Z'}

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path) and folder.upper() not in keep_folders:
        print(f"🗑️ Removing: {folder}")
        shutil.rmtree(folder_path)

print("✅ Cleaned dataset. Remaining folders:", sorted(os.listdir(data_path)))
