#!/usr/bin/env python3
import os
import glob
import re

LICENSE_TEXT = """Copyright (C) 2024 Delia Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

def get_files():
    patterns = [
        "src/**/*.py",
        "dashboard/src/**/*.ts",
        "dashboard/src/**/*.tsx",
        "tests/**/*.py"
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files

def format_license(style):
    lines = LICENSE_TEXT.split('\n')
    formatted = []
    if style == "python":
        for line in lines:
            if line.strip():
                formatted.append(f"# {line}")
            else:
                formatted.append("#")
        return "\n".join(formatted) + "\n"
    elif style == "ts":
        formatted.append("/**")
        for line in lines:
            formatted.append(f" * {line}")
        formatted.append(" */")
        return "\n".join(formatted) + "\n"
    return ""

def detect_style(filepath):
    if filepath.endswith(".py"):
        return "python"
    if filepath.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
        return "ts"
    return None

def is_license_content(text):
    keywords = [
        "Copyright (C)",
        "This program is free software",
        "GNU General Public License",
        "SPDX-License-Identifier",
        "WITHOUT ANY WARRANTY",
        "gnu.org/licenses",
        "Free Software Foundation",
        "later version",
        "distributed in the hope",
        "MERCHANTABILITY or FITNESS"
    ]
    return any(k in text for k in keywords)

def update_file(filepath):
    style = detect_style(filepath)
    if not style:
        return

    new_header = format_license(style)
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines(keepends=True)
    
    # 1. Extract Shebang if present
    shebang = ""
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
        lines = lines[1:]

    # 2. Cleanup existing headers (both correct style and potential wrong style from previous runs)
    
    idx = 0
    in_block = False
    
    # We scan specifically for comment blocks that contain license text.
    # We will simply skip ANY top-level comment block that looks like a license.
    
    while idx < len(lines) and idx < 100:
        line = lines[idx]
        stripped = line.strip()
        
        # Check if it looks like a comment line (py or js style)
        is_comment_line = (
            stripped.startswith("#") or 
            stripped.startswith("//") or 
            stripped.startswith("/*") or 
            stripped.startswith("*/") or
            stripped.startswith("*")
        )
        
        if not is_comment_line and stripped:
            # Non-comment content found -> stop skipping
            break
            
        if not stripped:
            # Empty line -> safe to skip if we are cleaning up header area
            idx += 1
            continue

        # It is a comment line. Does it contain license text?
        if is_license_content(line):
            # It's definitely a license line. Skip it.
            in_block = True
            idx += 1
            continue
        
        # It's a comment line, but not explicitly matching keywords.
        # If we are inside a detected license block (in_block=True), assume this comment is part of it.
        if in_block:
             idx += 1
             continue
        
        # If we haven't detected a license block yet, but it's just an empty comment frame?
        # e.g. " * " or "#"
        # We should be careful. 
        # But since we are replacing the license, we probably want to remove ANY license-like block at the top.
        # If we encounter a comment that is NOT license text and we aren't in a block, 
        # it might be a file docstring or ruff config.
        # However, for the cleanup of "garbage" left by previous run, that garbage definitely looks like comments.
        # Let's be aggressive if it starts with # in a TS file, or if it matches keywords.
        
        if style == "ts" and stripped.startswith("#"):
             # Aggressively remove python-style comments in TS files at the top
             idx += 1
             continue
             
        # Otherwise, if it's a generic comment line, we only skip if we are already in a block.
        # If not in block, we stop?
        # Wait, the previous header might start with `/**` (no keyword on that line).
        # We need to peek ahead?
        # Simpler: If it starts with `/**` or `/*`, let's assume it starts a block that MIGHT be license.
        # We need to verify inside.
        
        if stripped.startswith("/*"):
             # Check next few lines for license keywords
             lookahead_found = False
             for k in range(1, 10):
                 if idx + k < len(lines):
                     if is_license_content(lines[idx+k]):
                         lookahead_found = True
                         break
             if lookahead_found:
                 in_block = True
                 idx += 1
                 continue
                 
        # If we are here, it's a comment line that doesn't match license keywords, 
        # and we aren't in a confirmed block.
        # Stop skipping to preserve other comments.
        break

    clean_lines = lines[idx:]
    
    # Assemble
    final_content = []
    if shebang:
        final_content.append(shebang)
    
    final_content.append(new_header)
    final_content.append("\n")
    final_content.extend(clean_lines)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(final_content)
    print(f"Updated {filepath}")

def main():
    files = get_files()
    for filepath in files:
        if os.path.isdir(filepath):
            continue
        update_file(filepath)

if __name__ == "__main__":
    main()
