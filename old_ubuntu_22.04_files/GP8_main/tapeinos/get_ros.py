import os

# -------- CONFIG --------
MAX_FILE_SIZE_MB = 10   # skip files larger than this (set None to disable)
OUTPUT_FILE = "out.txt"


def is_text_file(file_path):
    """
    Check if a file is likely a text file.
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
        if b'\x00' in chunk:
            return False
        return True
    except:
        return False


def dump_files_to_txt(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)

                out.write(f"FILE: {file_path}\n")
                out.write("-" * 60 + "\n")

                # ---- File size check ----
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if MAX_FILE_SIZE_MB and size_mb > MAX_FILE_SIZE_MB:
                        out.write(f"[Skipped: File too large ({size_mb:.2f} MB)]\n")
                        out.write("\n" + "=" * 80 + "\n\n")
                        continue
                except:
                    out.write("[Error getting file size]\n")

                # ---- Read file ----
                if is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            out.write(content + "\n")
                    except Exception as e:
                        out.write(f"[Error reading file: {e}]\n")
                else:
                    out.write("[Binary file - content not displayed]\n")

                out.write("\n" + "=" * 80 + "\n\n")


if __name__ == "__main__":
    root_directory = input("Enter folder path: ").strip()

    if not os.path.exists(root_directory):
        print("Invalid path!")
    else:
        dump_files_to_txt(root_directory, OUTPUT_FILE)
        print(f"Done! Output written to {OUTPUT_FILE}")