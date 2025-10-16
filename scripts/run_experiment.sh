#!/bin/bash
# Helper script for running experiments with proper labeling
# Usage: ./scripts/run_experiment.sh "Experiment Name" "Short Code" "Description"

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 \"Experiment Name\" \"Short Code\" \"Description\""
    echo "Example: $0 \"Experiment 4a: Learning Rate Optimization\" \"lr_opt_04a\" \"LR sweep for effective batch size\""
    exit 1
fi

EXP_NAME="$1"
EXP_CODE="$2"
DESCRIPTION="$3"

echo "Running experiment: $EXP_NAME"
echo "Code: $EXP_CODE"
echo "Description: $DESCRIPTION"
echo ""

# Run training
echo "Starting training..."
task train

# Get the latest run directory
LATEST_RUN=$(ls -t mainrun/runs/ | head -n1)
echo "Latest run: $LATEST_RUN"

# Get the best validation loss
BEST_VAL=$(cat "mainrun/runs/$LATEST_RUN/result.json" | grep -o '"best_val_loss": [0-9.]*' | cut -d' ' -f2)
echo "Best validation loss: $BEST_VAL"

# Export figures with proper naming
echo "Exporting figures..."
task snapshot \
    RUN_DIR="mainrun/runs/$LATEST_RUN" \
    EXP="$EXP_CODE" \
    EXP_NAME="$EXP_NAME" \
    TITLE="$EXP_NAME" \
    CHANGE="$DESCRIPTION" \
    WHY="See experiment description" \
    SETTINGS="Current hyperparameters" \
    BEST="$BEST_VAL"

echo "Experiment completed! Check docs/figures/ for results."
echo "Update mainrun/report.md with analysis and regenerate PDF with: task report"
