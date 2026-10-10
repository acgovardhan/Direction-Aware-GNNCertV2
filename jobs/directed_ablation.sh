#!/bin/bash
# Direction-aware GNNCert vs original GNNCert on MalNet-Tiny + LDP.
# Run from the GNNCERT root (same place you ran jobs/malnet_ldp.sh):
#
#   bash jobs/directed_ablation.sh                       # train + certify + compare, variants A B C, fold 0
#   bash jobs/directed_ablation.sh run                   # only train+certify
#   bash jobs/directed_ablation.sh compare               # only (re)build tables/plots from finished runs
#
# Variants (tag -> model / features):
#   A  undirected_ldp5   original GIN, symmetrised graph, LDP-5          <- BASELINE (same code path, same protocol)
#   B  directed_ldp5     Dir-GIN + ordered-edge hashing, LDP-5           <- effect of the MODEL only
#   C  directed_dldp     Dir-GIN + ordered-edge hashing, directed LDP-10 <- full direction-aware pipeline
#   D  undirected_dldp   original GIN, directed LDP-10                   <- effect of the FEATURES only (optional)
#
# Everything is overridable through environment variables, e.g.
#   VARIANTS="A C" FOLDS="0 1 2" bash jobs/directed_ablation.sh
#   DIVISIONS="structure" EPOCHS_STRUCTURE=500 bash jobs/directed_ablation.sh run
#   MASK_VALUE=-1 bash jobs/directed_ablation.sh           # if training collapses with the -99 mask
STAGE=${1:-both}
cd GraphGuard || exit 1

DATASET=${DATASET:-MALNET_DIR}
VARIANTS=${VARIANTS:-"A B C"}
DIVISIONS=${DIVISIONS:-"feature structure all"}
FOLDS=${FOLDS:-"0"}                    # use "0 1 2" for 3-fold results with std / pooled significance tests
SEEDS=${SEEDS:-"42"}
EPOCHS_FEATURE=${EPOCHS_FEATURE:-1000};   NG_FEATURE=${NG_FEATURE:-30}
EPOCHS_STRUCTURE=${EPOCHS_STRUCTURE:-1000}; NG_STRUCTURE=${NG_STRUCTURE:-30}
EPOCHS_ALL=${EPOCHS_ALL:-1000};           NG_ALL=${NG_ALL:-10}      # 'all' = NG^2 subgraphs per graph
ITERS=${ITERS:-50}                     # iterations per epoch (original: 50)
BATCH=${BATCH:-32}
EVAL_EVERY=${EVAL_EVERY:-10}
MASK_VALUE=${MASK_VALUE:--99}          # original value
CHECKPOINT=${CHECKPOINT:-final}        # final = original behaviour | best_val = pick best epoch on a held-out part of TRAIN
GPU=${GPU:-0}
RESULTS=${RESULTS:-dir_results}
EXTRA=${EXTRA:-}

variant_args () {   # echoes "<direction> <feature_set>"
  case "$1" in
    A) echo "undirected ldp5";;
    B) echo "directed ldp5";;
    C) echo "directed dldp";;
    D) echo "undirected dldp";;
    *) echo "unknown variant $1" >&2; exit 1;;
  esac
}

if [ "$STAGE" = "run" ] || [ "$STAGE" = "both" ]; then
  if [ ! -f "dataset/${DATASET}/${DATASET}.txt" ]; then
    echo "dataset/${DATASET}/${DATASET}.txt missing -> run:  python prepare_malnet_directed.py" >&2; exit 1
  fi
  for DIV in $DIVISIONS; do
    case "$DIV" in
      feature)   EP=$EPOCHS_FEATURE;   NG=$NG_FEATURE;;
      structure) EP=$EPOCHS_STRUCTURE; NG=$NG_STRUCTURE;;
      all)       EP=$EPOCHS_ALL;       NG=$NG_ALL;;
    esac
    for SEED in $SEEDS; do for FOLD in $FOLDS; do for V in $VARIANTS; do
      read -r DIRN FEAT <<< "$(variant_args $V)"
      echo "=== variant ${V} (${DIRN}, ${FEAT}) | division=${DIV} NG=${NG} epochs=${EP} | seed=${SEED} fold=${FOLD} ==="
      python3 dir_gnncert.py --dataset "$DATASET" --direction "$DIRN" --feature_set "$FEAT" \
        --division-method "$DIV" --num_group "$NG" --epochs "$EP" --iters_per_epoch "$ITERS" --batch_size "$BATCH" \
        --eval_every "$EVAL_EVERY" --mask_value "$MASK_VALUE" --checkpoint "$CHECKPOINT" \
        --seed "$SEED" --fold_idx "$FOLD" --device "$GPU" --results "$RESULTS" $EXTRA
    done; done; done
  done
fi

if [ "$STAGE" = "compare" ] || [ "$STAGE" = "both" ]; then
  # optional: draw your ORIGINAL pipeline run (GG_idx0_*.csv) into the same plots
  ORIG_ARGS=""
  for DIV in feature structure; do
    F=$(ls new_results/GG_DS_MALNET_LDP_*/GG_idx0_${DIV}.csv 2>/dev/null | head -1)
    [ -n "$F" ] && ORIG_ARGS="$ORIG_ARGS --orig ${DIV}=${F}"
  done
  python3 compare_results.py --results "$RESULTS" --dataset "$DATASET" --baseline undirected_ldp5 $ORIG_ARGS
fi
