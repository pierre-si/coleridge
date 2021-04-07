i=0
for x in ../challenge-data/train/*; do
  if [ "$i" = 700 ]; then break; fi
  cp -- "$x" ../input/small/train/ 
  i=$((i+1))
done
