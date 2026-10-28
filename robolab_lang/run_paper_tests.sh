#!/usr/bin/env bash
# RoboLab extension of the ARR draft's tests (negation, switch, paraphrase, irrelevant content, distractor
# mention, spatial wording) with π0.5-DROID, in the original shared scenes. Three groups; each runs
# run_option_a.sh once per instruction key (same run records as Option A):
#   mustard  two_bin, LangMustardIn{Left,Right}BinTask                         9 keys x 2 tasks x 20 = 360
#            negation distractor_mention irrelevant_content paraphrase_common paraphrase_rare
#            paraphrase_functional paraphrase_zh spatial_robot spatial_image
#   cube     rubiks_cube_banana_bowl, LangRubiksCubeOrBananaTask                6 keys x 20 = 120
#            switch_cube switch_banana negation_not_banana negation_not_cube distractor_cube distractor_banana
#   size     butter_raisin_box_grey_bin, Lang{Larger,Smaller}Object*Task        3 keys x 2 tasks x 20 = 120
#            negation distractor_mention irrelevant_content
# summary.csv adds requested_delivered / other_delivered (make_run_record.py); for the cube group they
# replace RoboLab's success, which counts either object.
#
# Usage:  GROUP=mustard GPU=0 PORT=8765 HOST_TAG=aws-h200 bash robolab_lang/run_paper_tests.sh
#         GROUP=all runs the three groups one after another. ONLY_KEYS="k1 k2" limits a group to some keys.
#         VARIANT=rand10 uses the *Rand10Task files: RoboLab's official ±10 cm initial-pose randomization
#         (without it every episode starts from the same poses; gap G1 in docs/robolab/run_paper_tests.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GROUP="${GROUP:-all}"
export RUN_PREFIX="${RUN_PREFIX:-$(date -u +%Y%m%d-%H%M)}"
export PURPOSE="RoboLab extension of the ARR draft's tests, original shared scenes, pi0.5-DROID"

run_group() {
    case "$1" in
        mustard) STAGE=ext-mustard TASKS="LangMustardInLeftBinTask LangMustardInRightBinTask"
                 KEYS="negation distractor_mention irrelevant_content paraphrase_common paraphrase_rare paraphrase_functional paraphrase_zh spatial_robot spatial_image" ;;
        cube)    STAGE=ext-cube TASKS="LangRubiksCubeOrBananaTask"
                 KEYS="switch_cube switch_banana negation_not_banana negation_not_cube distractor_cube distractor_banana" ;;
        size)    STAGE=ext-size TASKS="LangLargerObjectRaisinBoxInBinTask LangSmallerObjectButterInBinTask"
                 KEYS="negation distractor_mention irrelevant_content" ;;
        *) echo "unknown GROUP $1 (mustard|cube|size|all)"; exit 1 ;;
    esac
    if [[ "${VARIANT:-}" == rand10 ]]; then
        STAGE="$STAGE-rand10"; TASKS="${TASKS//Task/Rand10Task}"
    fi
    STAGE="$STAGE" TASKS="$TASKS" KEYS="${ONLY_KEYS:-$KEYS}" bash "$HERE/run_option_a.sh"
}

if [[ "$GROUP" == all ]]; then
    for g in mustard cube size; do run_group "$g"; done
else
    run_group "$GROUP"
fi
