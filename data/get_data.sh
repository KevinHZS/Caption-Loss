#!/bin/bash

for((i=0;i<=20;i++));do
    for((k=0;k<=99;k++));do
        file_in=`printf "/gemini/space/gjx/FG-CLIP/data/FineHARD/coyo_image_%d/%05d.parquet" ${i} ${k}`
        pre_dir=`printf "/gemini/space/gjx/FG-CLIP/data/down-grit-12m/coyo_image_%d" ${i}`
        dir_save=`printf "/gemini/space/gjx/FG-CLIP/data/down-grit-12m/coyo_image_%d/%05d" ${i} ${k}`
        echo $file_in
        echo $dir_save
        mkdir -p ${pre_dir}
        mkdir -p ${dir_save}
        img2dataset ${file_in} --resize_mode 'no' --input_format parquet --output_folder=${dir_save} --processes_count=16 --thread_count=64
    done
done
