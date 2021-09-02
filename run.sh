#!/bin/sh


python ./train --checkpoint=$(date -d "now" +%s)
python ./test --checkpoint=$(date -d "now" +%s)