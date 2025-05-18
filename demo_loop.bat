@echo off
setlocal enabledelayedexpansion

:: set folders
set image_folder=..\data\obstacle_dataset\images
set mask_folder=..\data\obstacle_dataset\masks
set log_path=demo_obstacle

:: loop over all images
for %%I in (%image_folder%\*.jpg) do (
    set "image=%%I"
    set "filename=%%~nI"
    set "mask=%mask_folder%\!filename!.png"

    echo Running on !image! with mask !mask!
    python inference_one_sample.py ^
        --image "!image!" ^
        --mask "!mask!" ^
        --log_path %log_path%
)

pause
