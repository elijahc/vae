#!/bin/bash

TEMP_NAME="recon_learning.gif"
OUT_NAME="recon_learning_optimized.gif"

CALLING_DIR=$(pwd)

if [ -n "$1" ]; then
    IMG_DIR=$1
    echo "changing directory to $IMG_DIR"
    cd $IMG_DIR
    else
        IMG_DIR=$CALLING_DIR
fi

N_IMAGES=$(ls -hail | grep png | wc -l)

CUT=500

DELAY=10

echo "compiling $N_IMAGES images into $TEMP_NAME..."
convert -delay $DELAY -loop 0 *.png $TEMP_NAME

echo "optimizing..."
if [ "$N_IMAGES" >= "$CUT" ]; then
    gifsicle -U $TEMP_NAME `seq -f "#%g" 0 2 $N_IMAGES` -O3 --colors 256 -o $OUT_NAME
else
    gifsicle -O3 --colors 256 $TEMP_NAME -o $OUT_NAME
fi

echo "uploading to imgur"
URL=$(cat $OUT_NAME | imgur-uploader)
echo "$URL"

rm $TEMP_NAME

DETAILS=$(ls -hail | grep $OUT_NAME)
echo ""
echo "$DETAILS"

if [ "$IMG_DIR" != "$CALLING_DIR" ]; then
    cd $CALLING_DIR
fi
