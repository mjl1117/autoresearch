#!/bin/bash

# Usage: bash scripts/prepare_hmms.sh data/pfam/Pfam-A.hmm

SOURCE_HMM=$1
OUTPUT_HMM="data/pfam_tcs.hmm"

# 1. Check if source file exists
if [ -z "$SOURCE_HMM" ] || [ ! -f "$SOURCE_HMM" ]; then
    echo "Error: Pfam-A.hmm not found at '$SOURCE_HMM'"
    echo "Usage: bash scripts/prepare_hmms.sh path/to/Pfam-A.hmm"
    exit 1
fi

# 2. Index the source Pfam file (required for fetching)
if [ ! -f "${SOURCE_HMM}.ssi" ]; then
    echo "Indexing $SOURCE_HMM..."
    hmmfetch --index "$SOURCE_HMM"
fi

# 3. Extract the specific TCS domains
# Note: REC is often an alias for Response_reg, so we fetch both but ignore errors for REC
echo "Extracting domains: HisKA, HATPase_c, Response_reg..."
> "$OUTPUT_HMM" # Clear the file if it exists

for domain in HisKA HATPase_c Response_reg REC; do
    echo "  Fetching $domain..."
    # Append the HMM to our output file. Redirecting stderr to /dev/null 
    # suppresses the error message if 'REC' isn't found.
    hmmfetch "$SOURCE_HMM" "$domain" >> "$OUTPUT_HMM" 2>/dev/null
done

# 4. Press the new smaller HMM file for faster searching
echo "Pressing the new TCS HMM library..."
hmmpress -f "$OUTPUT_HMM"

echo "Success! TCS HMM library created at $OUTPUT_HMM"
