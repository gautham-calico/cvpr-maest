#!/bin/bash
#SBATCH --job-name=tseg_amos
#SBATCH --partition=gpu
#SBATCH --nodelist=t02
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch5/gautham/cvpr/logs/totalseg_amos_%j.log

# TotalSegmentator on 1,463 AMOS downstream images
# Output: /scratch5/gautham/cvpr/totalseg_maps_amos/<basename>.nii

export PYTHONUNBUFFERED=1
source activate voco

INPUT_DIR="/scratch5/gautham/cvpr/CVPR26-3DCTFMCompetition/AMOS-clf-tr-val/images"
OUTPUT_DIR="/scratch5/gautham/cvpr/totalseg_maps_amos"
mkdir -p "$OUTPUT_DIR"

TOTAL=$(ls "$INPUT_DIR"/*.nii.gz | wc -l)
echo "Total AMOS images: $TOTAL"
echo "Output dir: $OUTPUT_DIR"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

COUNT=0
SKIPPED=0
FAILED=0

for filepath in "$INPUT_DIR"/*.nii.gz; do
    fname=$(basename "$filepath" .nii.gz)
    outpath="${OUTPUT_DIR}/${fname}.nii"

    if [ -f "$outpath" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    COUNT=$((COUNT + 1))
    if [ $((COUNT % 50)) -eq 0 ]; then
        echo "Progress: $COUNT processed (skipped=$SKIPPED, failed=$FAILED)"
    fi

    TotalSegmentator -i "$filepath" -o "$outpath" --ml -q 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "FAILED: $fname"
        FAILED=$((FAILED + 1))
    fi
done

echo "Done. Processed=$COUNT, Skipped=$SKIPPED, Failed=$FAILED"
DONE=$(ls "$OUTPUT_DIR"/*.nii 2>/dev/null | wc -l)
echo "Total organ maps: $DONE / $TOTAL"
