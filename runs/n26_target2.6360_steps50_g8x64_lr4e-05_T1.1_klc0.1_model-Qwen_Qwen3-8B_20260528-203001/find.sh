#!/bin/sh
# Extract the top-3 rollouts per step (by reward) into a results/ directory.
#
# Run from inside your run dir, e.g.:
#   cd runs/n26_target2.6360_..._20260528-103045/
#   bash top3_per_step.sh
#
# Output layout:
#   results/
#     step00/
#       rank1_reward2.6225_group07_rollout011.txt
#       rank1_reward2.6225_group07_rollout011.meta.json
#       rank2_reward2.6053_group03_rollout044.txt
#       ...
#       summary.txt
#     step01/
#       ...
#     all_top3.csv     # flat list of all top-3 entries across all steps

set -e

OUT_DIR="${1:-results}"
mkdir -p "$OUT_DIR"

# Header for the global CSV
echo "step,rank,reward,group,rollout,valid,msg,file" > "$OUT_DIR/all_top3.csv"

# Find all step indices present in the directory
STEPS=$(ls step*.meta.json 2>/dev/null \
        | sed -E 's/^step([0-9]+)_.*/\1/' \
        | sort -u)

if [ -z "$STEPS" ]; then
    echo "No step*.meta.json files found. Run this from inside a run directory."
    exit 1
fi

TOTAL_STEPS=$(echo "$STEPS" | wc -l)
echo "Found $TOTAL_STEPS step(s). Extracting top-3 per step ..."

for s in $STEPS; do
    step_dir="$OUT_DIR/step$s"
    mkdir -p "$step_dir"

    # Build a JSON array of {reward, group, rollout, valid, msg, file} for this step,
    # filter for non-null reward, sort descending, take top 3.
    TOP3=$(jq -nr --arg s "$s" '
        [inputs
         | select(.reward != null)
         | {
             reward: .reward,
             group: .group,
             rollout: .rollout,
             valid: .valid,
             msg: .msg,
             file: input_filename
         }]
        | sort_by(-.reward)
        | .[0:3]
        | to_entries
        | .[]
        | "\(.key+1)|\(.value.reward)|\(.value.group)|\(.value.rollout)|\(.value.valid)|\(.value.msg)|\(.value.file)"
    ' step"$s"_*.meta.json)

    if [ -z "$TOP3" ]; then
        echo "  step $s: no valid rollouts"
        echo "no rollouts with non-null reward" > "$step_dir/summary.txt"
        continue
    fi

    echo "  step $s:"
    {
        echo "Top 3 rollouts for step $s"
        echo "-----------------------------"
    } > "$step_dir/summary.txt"

    printf '%s\n' "$TOP3" | while IFS='|' read -r rank reward group rollout valid msg meta_file; do
        # Skip empty lines
        [ -z "$rank" ] && continue

        txt_file="${meta_file%.meta.json}.txt"
        # Use zero-padded reward in filename so files sort by reward when listed
        # Round reward to 4 decimal places for filename readability
        reward_fmt=$(printf "%.4f" "$reward")
        group_fmt=$(printf "%02d" "$group")
        rollout_fmt=$(printf "%03d" "$rollout")

        dest_base="rank${rank}_reward${reward_fmt}_group${group_fmt}_rollout${rollout_fmt}"

        cp "$txt_file"        "$step_dir/${dest_base}.txt"
        cp "$meta_file"       "$step_dir/${dest_base}.meta.json"

        echo "    rank $rank: reward=$reward_fmt  group=$group_fmt  rollout=$rollout_fmt  valid=$valid"
        echo "rank $rank  reward=$reward  group=$group  rollout=$rollout  valid=$valid  msg=$msg" \
            >> "$step_dir/summary.txt"

        echo "$s,$rank,$reward,$group,$rollout,$valid,\"$msg\",$txt_file" >> "$OUT_DIR/all_top3.csv"
    done
done

echo ""
echo "Done. Output written to: $OUT_DIR/"
echo "Global summary CSV:      $OUT_DIR/all_top3.csv"cd