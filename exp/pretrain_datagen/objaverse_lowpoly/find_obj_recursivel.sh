#!/bin/bash
# Usage: ./find_obj_recursivel.sh <path> <out_file>
# Recursively finds all .obj files in <path> and writes their paths to <out_file>

if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: $0 <path> <out_file>"
    exit 1
fi

SEARCH_PATH="$1"
OUTPUT_FILE="$2"

if [ ! -d "$SEARCH_PATH" ]; then
    echo "Error: Search path '$SEARCH_PATH' does not exist or is not a directory"
    exit 1
fi

# Find all .obj files recursively and write to output file
find "$SEARCH_PATH" -type f -name "*.obj" > "$OUTPUT_FILE"

# Count the number of files found
FILE_COUNT=$(wc -l < "$OUTPUT_FILE")

echo "Found $FILE_COUNT .obj files"
echo "Results written to: $OUTPUT_FILE"