#!/bin/bash

# split_csv [path/to/data] [path/to/target]
#
# This script loops through train/test and each class dir. It then finds every TMS file and splits
# it based on the headers so that each file contains a single run. It makes a folder with the name
# of the file, and outputs all the runs in to that folder.
#
# The expected directory structure:
#       /path/to/data/train/class_one/this_is_data
#       /path/to/data/train/class_one/this_is_data_also
#       /path/to/data/train/class_two/this_is_data
#                           ...
#       /path/to/data/test/class_one/this_is_test_data
#       /path/to/data/test/class_one/this_is_test_data_also
#       /path/to/data/test/class_two/this_is_data

for filename in $1*/*; do
    echo $filename

    # Create a directory for the file
    rel_path="$(realpath --relative-to=$1 $filename)"
    dirpath=$2/${rel_path%.*}
    echo $dirpath

    if ! [ -d  $dirpath ]; then
        mkdir -p $dirpath
    else
        echo "Output dir already exists, skipping: " $dirpath
        continue
    fi

    # Remove all lines with ==== and pipe to new folder
    tmp_path=$dirpath/tmp.csv
    grep -vwE "====" $filename > $tmp_path

    # Split the file on the header, remove empty output, repeat for entire file, and move output
    # into dirpath
    csplit $tmp_path -z -f $dirpath$"/" -b "%04d.csv" \
        '/CNT,CH0,CH1,CH2,CH3,CH4,CH5,CH6,CH7,CH8,CH9,CH10,CH11,CH12/' '{*}' > /dev/null 2>&1
    rm $tmp_path

    # For each new file, extract the timestamp and use it to rename the file
    for new_filename in $dirpath/*.csv; do
        ts=$(grep TIME $new_filename)
        grep -vwE "TIME" $new_filename > $dirpath/${ts:5:6}.csv
        rm $new_filename
    done

done
