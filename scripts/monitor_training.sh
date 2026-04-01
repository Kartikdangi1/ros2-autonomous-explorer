#!/usr/bin/env bash
# monitor_training.sh — live training health monitor for PPO_* runs
# Usage: ./scripts/monitor_training.sh [tb_log_dir]
#   tb_log_dir defaults to ./tb_logs/

set -euo pipefail

TB_DIR="${1:-./tb_logs}"
REFRESH_SECS=15

# ANSI colours
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── helpers ──────────────────────────────────────────────────────────────────

latest_tfevents() {
    find "$TB_DIR" -name 'events.out.tfevents.*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}'
}

parse_metrics() {
    local path="$1"
    python3 - "$path" << 'PYEOF'
import struct, sys, collections

def parse(path):
    metrics = collections.OrderedDict()
    try:
        with open(path, 'rb') as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR reading file: {e}"); return metrics

    pos = 0
    current_tag = None
    while pos < len(content) - 12:
        try:
            length = struct.unpack_from('<Q', content, pos)[0]
            if length > 50000 or length < 1:
                pos += 1; continue
            pos += 12
            data = content[pos:pos+length]
            pos += length + 4

            # Extract readable ASCII tags
            tags = []
            i = 0
            while i < len(data):
                if 32 <= data[i] <= 126:
                    start = i
                    while i < len(data) and 32 <= data[i] <= 126:
                        i += 1
                    s = data[start:i].decode('ascii', errors='ignore')
                    if len(s) >= 4:
                        tags.append(s)
                i += 1

            # Extract float32 values
            floats = []
            for j in range(0, len(data) - 3):
                try:
                    v = struct.unpack_from('<f', data, j)[0]
                    if 0.0001 < abs(v) < 1e6 and v == v:
                        floats.append(round(v, 5))
                except Exception:
                    pass

            tag = next((t for t in tags if '/' in t), None)
            if tag and floats:
                # Last float in record tends to be the metric value
                metrics[tag] = floats[-1]
        except Exception:
            pos += 1

    return metrics

m = parse(sys.argv[1])
for k, v in m.items():
    print(f"{k}={v}")
PYEOF
}

pid_of_training() {
    pgrep -f 'train_ppo.py' | head -1
}

cpu_of_pid() {
    local pid="$1"
    ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' '
}

mem_mb_of_pid() {
    local pid="$1"
    ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.0f", $1/1024}'
}

overall_cpu_idle() {
    # one-shot top sample
    top -b -n 1 2>/dev/null | awk '/^%Cpu/{gsub(/,/,""); for(i=1;i<=NF;i++) if($i=="id,") print $(i-1); exit}'
}

elapsed_since() {
    local start_epoch="$1"
    local now; now=$(date +%s)
    local secs=$(( now - start_epoch ))
    printf '%dh %02dm %02ds' $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

# ── main loop ────────────────────────────────────────────────────────────────

echo -e "${BOLD}PPO Training Monitor${RESET} — refreshes every ${REFRESH_SECS}s  (Ctrl-C to quit)"
echo ""

prev_file_size=0
prev_step_time=$(date +%s)

while true; do
    clear
    now_str=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BOLD}=== PPO Training Monitor  ${CYAN}${now_str}${RESET}${BOLD} ===${RESET}"
    echo ""

    # ── process status ──────────────────────────────────────────────────────
    PID=$(pid_of_training)
    if [[ -z "$PID" ]]; then
        echo -e "${RED}[PROCESS]  train_ppo.py not running!${RESET}"
    else
        start_epoch=$(stat -c %Y /proc/"$PID"/exe 2>/dev/null || date +%s)
        # Better: use process start time
        start_epoch=$(ps -p "$PID" -o lstart= 2>/dev/null | xargs -I{} date -d '{}' +%s 2>/dev/null || date +%s)
        cpu=$(cpu_of_pid "$PID")
        mem=$(mem_mb_of_pid "$PID")
        elapsed=$(elapsed_since "$start_epoch")
        echo -e "${GREEN}[PROCESS]${RESET}  PID $PID  |  CPU: ${cpu}%  |  RAM: ${mem} MB  |  Running: ${elapsed}"
    fi

    # ── system load ─────────────────────────────────────────────────────────
    idle=$(overall_cpu_idle)
    used=$(echo "$idle" | awk '{printf "%.1f", 100-$1}')
    if (( $(echo "$used > 85" | bc -l) )); then
        colour=$RED
    elif (( $(echo "$used > 70" | bc -l) )); then
        colour=$YELLOW
    else
        colour=$GREEN
    fi
    echo -e "${colour}[SYSTEM ]${RESET}  CPU used: ${used}%  |  idle: ${idle}%"
    echo ""

    # ── TB events file ───────────────────────────────────────────────────────
    tf_file=$(latest_tfevents)
    if [[ -z "$tf_file" ]]; then
        echo -e "${RED}No tfevents file found in ${TB_DIR}${RESET}"
    else
        run_name=$(basename "$(dirname "$tf_file")")
        file_size=$(stat -c %s "$tf_file" 2>/dev/null || echo 0)
        size_delta=$(( file_size - prev_file_size ))
        if [[ $size_delta -gt 0 ]]; then
            growth="${GREEN}+${size_delta} bytes since last refresh${RESET}"
            prev_step_time=$(date +%s)
        elif [[ $prev_file_size -eq 0 ]]; then
            growth="${YELLOW}(first read)${RESET}"
        else
            stale=$(( $(date +%s) - prev_step_time ))
            if [[ $stale -gt 120 ]]; then
                growth="${RED}NO GROWTH for ${stale}s — check if stuck!${RESET}"
            else
                growth="${YELLOW}no change (+${stale}s — rollout in progress?)${RESET}"
            fi
        fi
        prev_file_size=$file_size
        echo -e "${BOLD}[RUN    ]${RESET}  ${run_name}  |  ${file_size} bytes  |  ${growth}"
        echo ""

        # ── parse metrics ────────────────────────────────────────────────────
        declare -A M=()
        while IFS='=' read -r key val; do
            M["$key"]="$val"
        done < <(parse_metrics "$tf_file")

        fps="${M[time/fps]:-N/A}"
        goal_dist="${M[episode/goal_distance]:-N/A}"
        goal_reached="${M[episode/goal_reached]:-N/A}"
        collision="${M[episode/collision]:-N/A}"
        lidar="${M[safety/min_lidar_range]:-N/A}"
        stage="${M[curriculum/stage]:-N/A}"
        ev="${M[train/explained_variance]:-N/A}"
        kl="${M[train/approx_kl]:-N/A}"
        clip="${M[train/clip_fraction]:-N/A}"
        loss="${M[train/loss]:-N/A}"
        std="${M[train/std]:-N/A}"

        # Colour-code metrics
        fps_col=$GREEN
        if [[ "$fps" != "N/A" ]] && (( $(echo "$fps < 4" | bc -l 2>/dev/null || echo 0) )); then fps_col=$RED
        elif [[ "$fps" != "N/A" ]] && (( $(echo "$fps < 6" | bc -l 2>/dev/null || echo 0) )); then fps_col=$YELLOW; fi

        lidar_col=$GREEN
        if [[ "$lidar" != "N/A" ]] && (( $(echo "$lidar < 0.5" | bc -l 2>/dev/null || echo 0) )); then lidar_col=$RED
        elif [[ "$lidar" != "N/A" ]] && (( $(echo "$lidar < 1.0" | bc -l 2>/dev/null || echo 0) )); then lidar_col=$YELLOW; fi

        collision_col=$GREEN
        if [[ "$collision" != "N/A" ]] && (( $(echo "$collision > 0" | bc -l 2>/dev/null || echo 0) )); then collision_col=$RED; fi

        echo -e "${BOLD}── Episode ──────────────────────────────────${RESET}"
        printf "  %-28s %s\n" "goal_distance (m):"     "$goal_dist"
        printf "  %-28s " "goal_reached:"
        if [[ "$goal_reached" != "N/A" ]] && (( $(echo "$goal_reached > 0" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "${GREEN}${goal_reached}  ← goals being reached!${RESET}"
        else
            echo "$goal_reached"
        fi
        printf "  %-28s " "collision:"
        echo -e "${collision_col}${collision}${RESET}"
        printf "  %-28s " "min_lidar_range (m):"
        echo -e "${lidar_col}${lidar}${RESET}"
        printf "  %-28s %s\n" "curriculum stage:"      "$stage"
        echo ""
        echo -e "${BOLD}── Training ─────────────────────────────────${RESET}"
        printf "  %-28s " "fps:"
        echo -e "${fps_col}${fps}${RESET}"
        printf "  %-28s %s\n" "explained_variance:"   "$ev"
        printf "  %-28s %s\n" "approx_kl:"             "$kl"
        printf "  %-28s %s\n" "clip_fraction:"         "$clip"
        printf "  %-28s %s\n" "loss:"                  "$loss"
        printf "  %-28s %s\n" "policy std:"            "$std"

        # ── ETA estimate ─────────────────────────────────────────────────────
        echo ""
        echo -e "${BOLD}── Progress ─────────────────────────────────${RESET}"
        if [[ "$fps" != "N/A" ]] && [[ -n "$PID" ]]; then
            start_e=$(ps -p "$PID" -o lstart= 2>/dev/null | xargs -I{} date -d '{}' +%s 2>/dev/null || echo 0)
            if [[ $start_e -gt 0 ]]; then
                elapsed_secs=$(( $(date +%s) - start_e ))
                steps_done=$(echo "$fps * $elapsed_secs" | bc -l 2>/dev/null | xargs printf "%.0f")
                total_steps=500000
                pct=$(echo "scale=1; $steps_done * 100 / $total_steps" | bc -l 2>/dev/null || echo "?")
                remaining_secs=$(echo "($total_steps - $steps_done) / $fps" | bc -l 2>/dev/null | xargs printf "%.0f")
                eta_h=$(( remaining_secs / 3600 ))
                eta_m=$(( (remaining_secs % 3600) / 60 ))
                printf "  %-28s ~%s / 500k (%.1s%%)\n" "steps estimated:" "$steps_done" "$pct"
                printf "  %-28s ~%dh %02dm remaining\n" "ETA at ${fps} fps:" "$eta_h" "$eta_m"
            fi
        fi

        # ── warnings ─────────────────────────────────────────────────────────
        echo ""
        warn=0
        if [[ "$lidar" != "N/A" ]] && (( $(echo "$lidar < 0.5" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "${RED}[WARN] min_lidar_range ${lidar}m — robot very close to obstacle!${RESET}"; warn=1
        fi
        if [[ "$collision" != "N/A" ]] && (( $(echo "$collision > 0" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "${RED}[WARN] collision detected (${collision}) — check reward shaping${RESET}"; warn=1
        fi
        if [[ "$kl" != "N/A" ]] && (( $(echo "$kl > 0.05" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "${YELLOW}[WARN] approx_kl=${kl} — policy update may be too large${RESET}"; warn=1
        fi
        if [[ "$clip" != "N/A" ]] && (( $(echo "$clip > 0.15" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "${YELLOW}[WARN] clip_fraction=${clip} — consider reducing learning rate${RESET}"; warn=1
        fi
        if [[ $warn -eq 0 ]]; then
            echo -e "${GREEN}[OK] All metrics within normal range${RESET}"
        fi
    fi

    echo ""
    echo -e "Next refresh in ${REFRESH_SECS}s...  (Ctrl-C to quit)"
    sleep "$REFRESH_SECS"
done
