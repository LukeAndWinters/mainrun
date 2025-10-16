#!/bin/bash
echo "🔍 Monitoring LR Sweep Progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    if ! pgrep -f 'run_lr_sweep.py' > /dev/null; then
        echo ""
        echo "🎉 LR SWEEP COMPLETED!"
        echo "================================"
        echo "Final Results:"
        echo ""
        if [ -f "logs/lr_sweep.jsonl" ]; then
            echo "Base LR    Eta Min    Accum    Final Val    Best Val    Skipped"
            echo "--------   --------   ------   ----------   ---------   -------"
            while IFS= read -r line; do
                base_lr=$(echo "$line" | jq -r '.base_lr')
                eta_min=$(echo "$line" | jq -r '.eta_min')
                accum=$(echo "$line" | jq -r '.accum')
                final_val=$(echo "$line" | jq -r '.final_val')
                best_val=$(echo "$line" | jq -r '.best_val')
                skipped=$(echo "$line" | jq -r '.skipped_updates')
                printf "%-10.2e %-10.2e %-8d %-12.6f %-11.6f %-8d\n" "$base_lr" "$eta_min" "$accum" "$final_val" "$best_val" "$skipped"
            done < logs/lr_sweep.jsonl
        else
            echo "No results found in logs/lr_sweep.jsonl"
        fi
        echo ""
        echo "✅ Sweep monitoring complete!"
        break
    fi
    echo "⏳ Sweep still running... (checking every 30s)"
    sleep 30
done
