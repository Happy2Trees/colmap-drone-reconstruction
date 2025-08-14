#!/usr/bin/env bash
set -e

# Ultra-simple runner: set variables, run python.
# No argument parsing. Edit paths and params below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Experiment name (1st arg or $EXPERIMENT_NAME) → outputs/<name>
EXP_NAME="${EXPERIMENT_NAME:-${1:-3d_reg_run}}"
EXP_DIR="outputs/${EXP_NAME}"
mkdir -p "$EXP_DIR"

# ===== Edit here: Inputs/Paths =====
# COLMAP model directory (contains cameras.bin, images.bin)
MODEL_DIR="/hdd2/0321_block_drone_video/colmap/outputs/workspaces/section2_7x_window_sift_48_12_30fps_better"
# Detection JSON folder for tracking (leave empty if skipping or reusing tracks)
JSON_DIR="outputs/yolo_pose/sec2_7x_20fps_predict5/json"
# Output prefix for triangulation results (under experiment dir)
OUT_PREFIX="$EXP_DIR/triangulated"

# Measurement inputs (defaults point to repo data/)
XYZ_NPY="data/measurement_xyz.npy"
CANDIDATE_TXT="data/candidate_list.txt"
MEAS_PLY_OUT="$EXP_DIR/measurement_candidates.ply"

# ===== Edit here: Step control =====
# Uncomment the flags you want to apply
# SKIP_TRACKING="--skip_tracking"
## Reuse paths (helpful for reruns inside the same exp dir)
# REUSE_TRACKS_CSV="--reuse_tracks_csv $EXP_DIR/tracks.csv"
# SKIP_TRIANGULATION="--skip_triangulation"
# REUSE_TRIANGULATED_PLY="--reuse_triangulated_ply $EXP_DIR/triangulated_points.ply"
# SKIP_MEASURE_PLY="--skip_measure_ply"
# SKIP_REGISTER="--skip_register"

# ===== Edit here: Tracking params =====
PATTERN="*.json"
CONF_MIN="0.8"
KP_CONF_MIN=""        # leave empty to use python default
TAKE="all"            # 'all' or 'first'
MAX_MATCH_DIST="80.0"
MAX_MISSED="2"
# For boolean toggles, set one of the following vars to include the flag:
# FILL_MISSING_FLAG="--fill_missing"    # default behavior is already True in python
# FILL_MISSING_FLAG="--no_fill_missing"
# PREFER_HUNGARIAN_FLAG="--prefer_hungarian"   # default True in python
# PREFER_HUNGARIAN_FLAG="--no_prefer_hungarian"
# DEDUP_MEASURE_FLAG="--dedup_measure"          # default True in python
# DEDUP_MEASURE_FLAG="--no_dedup_measure"

# ===== Edit here: Triangulation params =====
MIN_VIEWS="2"
RANSAC_THRESH="3.0"
RANSAC_ITERS="400"
MIN_INLIERS="3"
POS_DEPTH_RATIO="0.5"

# ===== Edit here: Registration params =====
VOXEL_SIZE="0.05"
RANSAC_N="4"
DISTANCE_THRESH_RATIO="1.5"
ICP_MAX_ITER="50"
REG_DEBUG_DIR="$EXP_DIR/register_debug"

# ===== Edit here: XYZ sampling params =====
# Number of points to sample from xyz_npy for visualization (leave empty or 0 to disable)
XYZ_SAMPLE_COUNT=""

# Extra passthrough (optional string of flags)
EXTRA_ARGS=""

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Running 3D registration"
echo "  EXP_NAME=$EXP_NAME"
echo "  EXP_DIR=$EXP_DIR"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  JSON_DIR=${JSON_DIR:-<empty>}"
echo "  OUT_PREFIX=$OUT_PREFIX"
echo "  XYZ_NPY=$XYZ_NPY"
echo "  CANDIDATE_TXT=$CANDIDATE_TXT"
echo "  MEAS_PLY_OUT=$MEAS_PLY_OUT"

python3 -m src.3d_registration.main \
  --exp_dir "$EXP_DIR" \
  ${JSON_DIR:+--json_dir "$JSON_DIR"} \
  --model_dir "$MODEL_DIR" \
  --out_prefix "$OUT_PREFIX" \
  ${XYZ_NPY:+--xyz_npy "$XYZ_NPY"} \
  ${CANDIDATE_TXT:+--candidate_txt "$CANDIDATE_TXT"} \
  ${MEAS_PLY_OUT:+--measurement_ply_out "$MEAS_PLY_OUT"} \
  ${SKIP_TRACKING:-} ${REUSE_TRACKS_CSV:-} ${SKIP_TRIANGULATION:-} ${REUSE_TRIANGULATED_PLY:-} ${SKIP_MEASURE_PLY:-} ${SKIP_REGISTER:-} \
  ${PATTERN:+--pattern "$PATTERN"} \
  ${CONF_MIN:+--conf_min "$CONF_MIN"} \
  ${KP_CONF_MIN:+--kp_conf_min "$KP_CONF_MIN"} \
  ${TAKE:+--take "$TAKE"} \
  ${MAX_MATCH_DIST:+--max_match_dist "$MAX_MATCH_DIST"} \
  ${MAX_MISSED:+--max_missed "$MAX_MISSED"} \
  ${FILL_MISSING_FLAG:-} ${PREFER_HUNGARIAN_FLAG:-} \
  ${DEDUP_MEASURE_FLAG:-} \
  ${MIN_VIEWS:+--min_views "$MIN_VIEWS"} \
  ${RANSAC_THRESH:+--ransac_thresh "$RANSAC_THRESH"} \
  ${RANSAC_ITERS:+--ransac_iters "$RANSAC_ITERS"} \
  ${MIN_INLIERS:+--min_inliers "$MIN_INLIERS"} \
  ${POS_DEPTH_RATIO:+--pos_depth_ratio "$POS_DEPTH_RATIO"} \
  ${VOXEL_SIZE:+--voxel_size "$VOXEL_SIZE"} \
  ${RANSAC_N:+--ransac_n "$RANSAC_N"} \
  ${DISTANCE_THRESH_RATIO:+--distance_thresh_ratio "$DISTANCE_THRESH_RATIO"} \
  ${ICP_MAX_ITER:+--icp_max_iter "$ICP_MAX_ITER"} \
  ${REG_DEBUG_DIR:+--reg_debug_dir "$REG_DEBUG_DIR"} \
  ${XYZ_SAMPLE_COUNT:+--xyz_sample_count "$XYZ_SAMPLE_COUNT"} \
  ${EXTRA_ARGS}
