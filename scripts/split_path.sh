#!/bin/bash

IN=$(pwd)
arrIN=(${IN//// })

for d in $arrIN; do
    echo "$d"
done
