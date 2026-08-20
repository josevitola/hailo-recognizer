import urllib.request
from pathlib import Path
from sys import stdout

MODELS = {
    "scrfd_2.5g_h8.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/scrfd_2.5g.hef",
    "arcface_mobilefacenet.hef": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8/arcface_mobilefacenet.hef",
}

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def progress_bar(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        stdout.write(
            f"\r    Progress: {percent:5.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
        )
        stdout.flush()


def download_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {MODELS_DIR.resolve()}\n")

    for filename, url in MODELS.items():
        destination = MODELS_DIR / filename
        if destination.exists():
            print(f"[-] {filename} already exists. Skipping.")
            continue

        print(f"[+] Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, destination, reporthook=progress_bar)
            print(f"\n    Saved to {destination}\n")
        except Exception as e:
            print(f"\n    [!] Failed to download {filename}: {e}\n")
            if destination.exists():
                destination.unlink()

def main() -> None:
    download_models()

if __name__ == "__main__":
    download_models()