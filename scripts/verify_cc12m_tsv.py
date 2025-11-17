import os
import sys

def verify_tsv(tsv_path):
    """Verify CC12M TSV file"""
    if not os.path.exists(tsv_path):
        print(f"ERROR: TSV file not found: {tsv_path}")
        return False
    
    file_size = os.path.getsize(tsv_path)
    print(f"TSV file size: {file_size / (1024**2):.2f} MB")
    
    if file_size < 1024 * 1024:  # Less than 1 MB
        print("WARNING: TSV file seems too small!")
    
    print("\nReading first 10 lines...")
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10:
                print(f"Line {i+1}: {line.strip()[:100]}...")
            if i == 9:
                break
    
    print("\nCounting total lines...")
    with open(tsv_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total samples in TSV: {total_lines:,}")
    
    if total_lines < 100:
        print("\nWARNING: Very few samples! Expected ~12 million for full CC12M dataset.")
        print("You may need to re-download the TSV file.")
        print("\nDownload command:")
        print("wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv")
        print("or")
        print("curl -o cc12m.tsv https://storage.googleapis.com/conceptual_12m/cc12m.tsv")
    
    return True

if __name__ == '__main__':
    tsv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/cc12m/cc12m.tsv'
    verify_tsv(tsv_path)
