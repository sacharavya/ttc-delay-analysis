import os

gtfs_path = os.path.expanduser("~/Desktop/ttc-delay-analysis/data/raw/gtfs")

for folder in os.listdir(gtfs_path):
    folder_path = os.path.join(gtfs_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                old = os.path.join(folder_path, file)
                new = os.path.join(folder_path, file.replace(".txt", ".csv"))
                os.rename(old, new)
                print(f"Renamed: {old} -> {new}")

print("Done!")