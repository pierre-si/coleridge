# i=0
# for x in ../input/coleridgeinitiative-show-us-the-data/train/*; do
#   if [ "$i" = 400 ]; then break; fi
#   if (( i % 4 )); then
#     cp -- "$x" ../input/smaller/train/
#   else
#     cp -- "$x" ../input/smaller/val/
#   fi

#   i=$((i+1))
# done

i=0
for x in ../input/coleridgeinitiative-show-us-the-data/train/*; do
  if [ "$i" = 1200 ]; then break; fi
  cp -- "$x" ../input/subset_dataset-split/publications/
  i=$((i+1))
done
