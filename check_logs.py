
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from delia.paths import get_data_dir, get_settings_file

def main():
    print(f"Data Dir: {get_data_dir()}")
    print(f"Settings File: {get_settings_file()}")
    
    log_file = get_data_dir() / "cache" / "live_logs.json"
    print(f"Log File: {log_file}")
    
    if log_file.exists():
        print("\nLast 10 lines of logs:")
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.strip())
    else:
        print("\nLog file does not exist.")

if __name__ == "__main__":
    main()

