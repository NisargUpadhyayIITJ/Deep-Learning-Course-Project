#!/bin/bash
# Script to merge all training datasets into merge_train folder
# Uses symbolic links to avoid duplicating data
# Dynamically detects ALL subdirectories

set -e

BASE_DIR="preset/datasets/train_datasets"
MERGE_DIR="$BASE_DIR/merge_train"

# Source datasets to merge
DATASETS=("DIV2K" "DIV8K" "FFHQ" "Flickr2K" "Flickr8K" "NKUSR8K")

echo "=========================================="
echo "Merging Training Datasets"
echo "=========================================="

# Create merge directory
mkdir -p "$MERGE_DIR"

# First, discover all unique subdirectories across all datasets
echo ""
echo "Discovering subdirectories..."
declare -A ALL_SUBDIRS

for dataset in "${DATASETS[@]}"; do
    dataset_dir="$BASE_DIR/$dataset"
    if [ -d "$dataset_dir" ]; then
        for subdir in "$dataset_dir"/*/; do
            if [ -d "$subdir" ]; then
                subdir_name=$(basename "$subdir")
                ALL_SUBDIRS["$subdir_name"]=1
            fi
        done
    fi
done

# Print discovered subdirectories
echo "Found subdirectories:"
for subdir in "${!ALL_SUBDIRS[@]}"; do
    echo "  - $subdir"
done

# Create all subdirectories in merge folder
echo ""
echo "Creating directory structure..."
for subdir in "${!ALL_SUBDIRS[@]}"; do
    mkdir -p "$MERGE_DIR/$subdir"
done

# Function to create symlinks for all files in a directory
merge_files() {
    local src_dir="$1"
    local dst_dir="$2"
    local dataset_name="$3"
    
    if [ ! -d "$src_dir" ]; then
        return 0
    fi
    
    # Count files being linked
    local count=0
    for file in "$src_dir"/*; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            
            # Check if it's a NULL embedding file (don't prefix these)
            if [[ "$filename" == NULL_* ]]; then
                # Only copy if it doesn't exist yet
                if [ ! -f "$dst_dir/$filename" ]; then
                    cp "$file" "$dst_dir/$filename"
                fi
            else
                # Add dataset prefix to avoid name collisions
                local newname="${dataset_name}_${filename}"
                
                # Copy the file
                cp "$file" "$dst_dir/$newname"
                count=$((count + 1))
            fi
        fi
    done
    
    if [ $count -gt 0 ]; then
        echo "    $subdir_name: $count files"
    fi
}

# Merge each dataset
echo ""
echo "Merging datasets..."
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Processing $dataset..."
    
    dataset_dir="$BASE_DIR/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo "  Warning: $dataset_dir does not exist, skipping"
        continue
    fi
    
    for subdir_name in "${!ALL_SUBDIRS[@]}"; do
        merge_files "$dataset_dir/$subdir_name" "$MERGE_DIR/$subdir_name" "$dataset"
    done
done

# Count total files in each subdirectory
echo ""
echo "=========================================="
echo "Merge Summary:"
echo "=========================================="
total_count=0
for subdir in "${!ALL_SUBDIRS[@]}"; do
    count=$(find "$MERGE_DIR/$subdir" -maxdepth 1 \( -type l -o -type f \) 2>/dev/null | wc -l)
    echo "  $subdir: $count files"
    total_count=$((total_count + count))
done

echo ""
echo "Total files: $total_count"
echo ""
echo "Done! Merged datasets are in: $MERGE_DIR"
