#!/bin/sh

$NOW=`date -d "now" +%s`

python train.py --checkpoint=$NOW
python test.py --checkpoint=$NOW