#!/usr/bin/env bash
# Run DUSTER (gold) and DUSTER_A (pred) on the temporal splits, 5 seeds each.
# Edit DATA_ROOT to the folder that contains the datasets (e.g. /content/drive/MyDrive).
set -e
DATA_ROOT=${DATA_ROOT:-/content/drive/MyDrive}
declare -A PICKLE=(
  [clothing]=$DATA_ROOT/Amazon_Datasets/ClothingShoesAndJewelry
  [movies]=$DATA_ROOT/Amazon_Datasets/MoviesAndTV
  [yelp]=$DATA_ROOT/Yelp_Dataset
  [tripadvisor]=$DATA_ROOT/TripAdvisor_Dataset
)
for ds in clothing movies tripadvisor yelp; do
  dir=${PICKLE[$ds]}
  for mode in gold pred; do
    for seed in 1 2 3 4 5; do
      python main.py --data_path $dir/reviews_time.pickle --index_dir $dir/temporal \
        --aspect_mode $mode --seed $seed --cuda \
        --checkpoint runs/${ds}_${mode}_s${seed}/ | tee runs_${ds}_${mode}_s${seed}.log
    done
    python aggregate_results.py "runs/${ds}_${mode}_s*" | tee runs/${ds}_${mode}_summary.txt
  done
done
