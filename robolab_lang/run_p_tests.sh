#!/usr/bin/env bash
# RoboLab P-tests (LIBERO-Spatial freeze §6, P1/P3/P4/P5/P7) with π0.5-DROID, in the original shared scenes.
# Three groups; each runs run_option_a.sh once per instruction key (same run records as Option A):
#   mustard  two_bin, LangMustardIn{Left,Right}BinTask       9 keys x 2 tasks x 20 = 360 episodes
#            p3_neg_not p5_contrast p5_len_ctrl p4_syn_common p4_syn_rare p4_functional p4_zh p7_robot_frame p7_image_frame
#   cube     rubiks_cube_banana_bowl, LangRubiksCubeOrBananaTask   6 keys x 20 = 120 episodes
#            p1_ask_cube p1_ask_banana p3_not_banana p3_not_cube p5_cube_not_banana p5_banana_not_cube
#   size     butter_raisin_box_grey_bin, Lang{Larger,Smaller}Object*Task   3 keys x 2 tasks x 20 = 120 episodes
#            p3_neg p5_contrast p5_len_ctrl
# Success is RoboLab's own check for the task class; which object went where comes from the outcome logger.
#
# Usage:  GROUP=mustard GPU=0 PORT=8765 HOST_TAG=aws-h200 bash robolab_lang/run_p_tests.sh
#         GROUP=all runs the three groups one after another. Other variables pass through to run_option_a.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GROUP="${GROUP:-all}"
export RUN_PREFIX="${RUN_PREFIX:-$(date -u +%Y%m%d-%H%M)}"
export PURPOSE="RoboLab P-tests (freeze §6 P1/P3/P4/P5/P7), original shared scenes, pi0.5-DROID"

run_group() {
    case "$1" in
        mustard) STAGE=p-mustard TASKS="LangMustardInLeftBinTask LangMustardInRightBinTask" \
                 KEYS="p3_neg_not p5_contrast p5_len_ctrl p4_syn_common p4_syn_rare p4_functional p4_zh p7_robot_frame p7_image_frame" ;;
        cube)    STAGE=p-cube TASKS="LangRubiksCubeOrBananaTask" \
                 KEYS="p1_ask_cube p1_ask_banana p3_not_banana p3_not_cube p5_cube_not_banana p5_banana_not_cube" ;;
        size)    STAGE=p-size TASKS="LangLargerObjectRaisinBoxInBinTask LangSmallerObjectButterInBinTask" \
                 KEYS="p3_neg p5_contrast p5_len_ctrl" ;;
        *) echo "unknown GROUP $1 (mustard|cube|size|all)"; exit 1 ;;
    esac
    STAGE="$STAGE" TASKS="$TASKS" KEYS="${ONLY_KEYS:-$KEYS}" bash "$HERE/run_option_a.sh"
}

if [[ "$GROUP" == all ]]; then
    for g in mustard cube size; do run_group "$g"; done
else
    run_group "$GROUP"
fi
