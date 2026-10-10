#!/bin/bash
# Run from GNNCERT root:   bash jobs/malnet_ldp.sh            (train + test, all 3 divisions)
#                          bash jobs/malnet_ldp.sh train feature
#                          bash jobs/malnet_ldp.sh test  all
# args: [train|test|both] [feature|structure|all|everything]
cd GraphGuard

STAGE=${1:-both}
WHICH=${2:-everything}

DATASET='MALNET_LDP'
SEED=42
FEATURES_METHOD='id'
DEGREE_AS_TAG=0
HASH_METHOD='md5'
GPU='0'
export EVAL_EVERY=${EVAL_EVERY:-10}      # evaluate every N epochs (original = 2; too slow for 500-node graphs)

# per-division settings:  "<division> <epochs> <num_group>"
# 'all' makes NUM_GROUP^2 subgraphs per graph (900 at 30!) -> use fewer groups / epochs for MalNet.
CONFIGS=(
  "feature   1000 30"
  "structure 1000 30"
  "all       500  10"
)

for CFG in "${CONFIGS[@]}"; do
  read -r DIV EPOCHS NG <<< "$CFG"
  if [ "$WHICH" != "everything" ] && [ "$WHICH" != "$DIV" ]; then continue; fi
  NAME="GG_DS_${DATASET}_EPOCH_${EPOCHS}_SEED_${SEED}_FOLDN_3_FM_${FEATURES_METHOD}_DAT_${DEGREE_AS_TAG}_HM_${HASH_METHOD}_NG_${NG}"
  mkdir -p "./saved_model/${NAME}" "./new_results/${NAME}"
  MODEL_PATH="./saved_model/${NAME}/GG_idx0_${DIV}.pth"
  LOG="./new_results/${NAME}/GG_idx0_${DIV}.csv"

  if [ "$STAGE" = "train" ] || [ "$STAGE" = "both" ]; then
    if [ -e "${MODEL_PATH}" ]; then
      echo "Skipping training for ${DIV}: ${MODEL_PATH} exists (delete it to retrain)"
    else
      echo "=== TRAIN ${DATASET} division=${DIV} epochs=${EPOCHS} groups=${NG} ==="
      python3 main_graphguard.py --device "${GPU}" --dataset "${DATASET}" --seed="${SEED}" \
        --fold_n=3 --fold_idx=0 --division-method="${DIV}" --degree_as_tag "${DEGREE_AS_TAG}" \
        --features-method "${FEATURES_METHOD}" --hash-method "${HASH_METHOD}" \
        --model_weight "${MODEL_PATH}" --filename "${LOG}" --epochs "${EPOCHS}" --num_group "${NG}"
    fi
  fi

  if [ "$STAGE" = "test" ] || [ "$STAGE" = "both" ]; then
    if [ ! -e "${MODEL_PATH}" ]; then
      echo "No model for ${DIV}, skipping test"
    else
      echo "=== TEST  ${DATASET} division=${DIV} ==="
      python3 -u test.py --device "${GPU}" --dataset "${DATASET}" --seed="${SEED}" \
        --fold_n=3 --fold_idx=0 --degree_as_tag "${DEGREE_AS_TAG}" --features-method "${FEATURES_METHOD}" \
        --division-method "${DIV}" --hash-method "${HASH_METHOD}" --defense_method "GG" \
        --model_weight "${MODEL_PATH}" --num_group "${NG}" --filename "${LOG}"
    fi
  fi
done
