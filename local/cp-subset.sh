i=0
for x in ../input/coleridgeinitiative-show-us-the-data/train/*; do
  if [ "$i" = 400 ]; then break; fi
  if (( i % 4 )); then
    cp -- "$x" ../input/smaller/train/
  else
    cp -- "$x" ../input/smaller/val/
  fi

  i=$((i+1))
done
