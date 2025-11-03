import os
import sys
import csv

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory_path> <tag_name>")
        sys.exit(1)

    directory = sys.argv[1]
    tag = sys.argv[2]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    # Create outputs folder next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "drum_cats.csv")

    # Load existing entries into a dict {file_path: tag}
    existing_entries = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline='', encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                existing_entries[row["File Path"]] = row["Tag"]

    # Walk through directory and nested folders, add/update entries
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                abs_path = os.path.abspath(os.path.join(root, file))
                existing_entries[abs_path] = tag  # update or add

    # Write updated entries back to CSV
    with open(csv_path, mode="w", newline='', encoding="utf-8") as csv_file:
        fieldnames = ["File Path", "Tag"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for path, tag_value in existing_entries.items():
            writer.writerow({"File Path": path, "Tag": tag_value})

    print(f"✅ Updated drum_cats.csv at: {csv_path}")

if __name__ == "__main__":
    main()
