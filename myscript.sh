#!/bin/bash
for file in $(ls .)
do 
    oldfile=$file
    newfile=${file//;/.}
    if [ $oldfile==$newfile ]
    then
	echo $newfile
    else
	mv $oldfile $newfile
    fi
done