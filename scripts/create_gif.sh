#!/bin/bash

TEMP_NAME="recon_learning.gif"
OUT_NAME="recon_learning_optimized.gif"

N_IMAGES=$(ls -hail | grep png | wc -l)

DELAY=10

echo "compiling $N_IMAGES images into $TEMP_NAME..."
convert -delay $DELAY -loop 0 *.png $TEMP_NAME

echo "optimizing..."
gifsicle -O3 --colors 256 $TEMP_NAME -o $OUT_NAME

echo "uploading to imgur"
cat $OUT_NAME | imgur-uploader
rm $TEMP_NAME

DETAILS=$(ls -hail | grep $OUT_NAME)
echo ""
echo "$DETAILS"
