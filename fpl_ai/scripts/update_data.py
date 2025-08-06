"""
CLI wrapper: fetch raw JSON then process to parquet.
"""
from fpl_ai.data import fetch, process

def main():
    fetch.download_static()
    fetch.download_fixtures()
    process.main()

if __name__ == "__main__":
    main()
