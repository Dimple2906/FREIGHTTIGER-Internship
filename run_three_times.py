from pathlib import Path
import hashlib
import shutil
import pandas as pd

from src.pipeline import build_results, write_output


def run_reproducibility_test():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    output_dir = root / "output"
    template_path = data_dir / "sample_output_format_v2.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    hashes = []
    file_paths = []

    print("=== Reproducibility Test: Running 3 consecutive passes ===")

    for i in range(1, 4):
        run_file = output_dir / f"reproducibility_run_{i}.csv"
        file_paths.append(run_file)

        print(f"Executing pass {i}/3...")
        result = build_results(data_dir, threshold=0.20)
        write_output(result, template_path, run_file)

        digest = hashlib.sha256(run_file.read_bytes()).hexdigest()
        hashes.append(digest)
        print(f"Pass {i} SHA-256: {digest}")

    # Verify identical hashes
    if len(set(hashes)) == 1:
        print("\nSUCCESS: All 3 runs produced 100% byte-for-byte identical outputs.")
        print(f"Common SHA-256: {hashes[0]}")
    else:
        print("\nFAILURE: Outputs differed between runs:")
        for idx, h in enumerate(hashes, 1):
            print(f"Run {idx}: {h}")
        raise AssertionError("Reproducibility check failed: hash mismatch detected across runs.")

    # Cleanup temporary test files
    for p in file_paths:
        if p.exists():
            p.unlink()

    print("Reproducibility check completed successfully.")


if __name__ == "__main__":
    run_reproducibility_test()
