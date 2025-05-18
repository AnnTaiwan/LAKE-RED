@echo off
python inference_one_sample.py ^
    --image create_mask/test_data/000132.jpg ^
    --mask create_mask/src/mask/000132.png ^
    --log_path demo_res ^
    --batchsize 4 ^
    --Steps 20 
pause
