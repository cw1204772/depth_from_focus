###### DEFAULT PARAMS #######

LoG_gaussian_ksize=5
LoG_laplacian_ksize=5
AIF_sharpness_sigma=7
AIF_depth_ksize=5
graphcut_sharpness_sigma=2
graphcut_unary_scale=100
graphcut_pair_scale=1

reverse_input_order=0

#############################

method='homography'    # 'homography', 'flow', or 'RAFT'
img_path=./data/keyboard/
# flow_path=../RAFT/result/dffmp/keyboard/
base_path=./results/keyboard_${method}/

# method='homography'
# img_path=../Depth_from_Focus/dataset/07/
# base_path=./results/07_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure1/keyboard/
# flow_path=../RAFT/result/dffmp/keyboard/
# base_path=./results/dffmp_Figure1_keyboard_${method}_dffmp/

# method="homography"
# img_path=../depth_from_focus_data2/Figure6/zeromotion/
# # flow_path=../RAFT/result/dffmp/largemotion/
# base_path=./results/dffmp_Figure6_zeromotion_${method}/
# # # graphcut_unary_scale=100
# graphcut_pair_scale=5
# reverse_input_order=1

# method="RAFT"
# img_path=../depth_from_focus_data2/Figure6/largemotion_reverse/
# flow_path=../RAFT/result/dffmp/largemotion_reverse/
# base_path=./results/dffmp_Figure6_largemotion_${method}/
# # # graphcut_unary_scale=100
# graphcut_pair_scale=5
# # reverse_input_order=0

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure3/kitchen_reverse/
# flow_path=../RAFT/result/dffmp/kitchen_reverse/
# base_path=./results/dffmp_Figure3_kitchen_${method}/
# # reverse_input_order=1

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure3/bucket/
# flow_path=../RAFT/result/dffmp/bucket/
# base_path=./results/dffmp_Figure3_bucket_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure5/bottles_r/
# flow_path=../RAFT/result/dffmp/bottles_r/
# base_path=./results/dffmp_Figure5_bottles_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure5/fruits_r/
# flow_path=../RAFT/result/dffmp/fruits_r/
# base_path=./results/dffmp_Figure5_fruits_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure5/metals_r/
# flow_path=../RAFT/result/dffmp/metals_r/
# base_path=./results/dffmp_Figure5_metals_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure5/plants_r/
# flow_path=../RAFT/result/dffmp/plants_r/
# base_path=./results/dffmp_Figure5_plants_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure5/telephone_rr/
# flow_path=../RAFT/result/dffmp/telephone_rr/
# base_path=./results/dffmp_Figure5_telephone_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure5/window_r/
# flow_path=../RAFT/result/dffmp/window_r/
# base_path=./results/dffmp_Figure5_window_${method}/

# method='RAFT'
# img_path=../depth_from_focus_data2/Figure7/balls/
# flow_path=../RAFT/result/dffmp/balls/
# base_path=./results/dffmp_Figure7_balls_${method}/

# method='homography'
# img_path=./mydata_phone/keyboard2/
# base_path=./results/mydata_keyboard2_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

# method='flow'
# img_path=./mydata_phone/keyboard2_largemotion/
# base_path=./results/mydata_keyboard2_largemotion_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

# method='flow'
# img_path=./mydata_phone/cocoa1/
# base_path=./results/mydata_cocoa1_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

# method='RAFT'
# img_path=./mydata_phone/cocoa1_largemotion_reverse/
# flow_path=../RAFT/result/mydata_phone/cocoa1_largemotion_reverse/
# base_path=./results/mydata_cocoa1_largemotion_${method}/
# graphcut_sharpness_sigma=2
# # graphcut_unary_scale=100
# graphcut_pair_scale=2
# # reverse_input_order=1

# method='homography'
# img_path=./mydata_dslr/DSC_0187_dog1_smallmotion/
# # flow_path=../RAFT/result/mydata_dslr/DSC_0189_dog1_largemotion_reverse/
# base_path=./results/mydata_dog1_smallmotion_${method}/
# graphcut_unary_scale=100
# graphcut_pair_scale=1
# reverse_input_order=1

# method='RAFT'
# img_path=./mydata_dslr/DSC_0189_dog1_largemotion_reverse/
# flow_path=../RAFT/result/mydata_dslr/DSC_0189_dog1_largemotion_reverse/
# base_path=./results/mydata_dog1_largemotion_${method}/
# # AIF_sharpness_sigma=2
# # AIF_depth_ksize=0
# graphcut_unary_scale=100
# graphcut_pair_scale=1


############################################

save_path=${base_path}/align/
match_save_path=${base_path}/matches/

if [ $method == "RAFT" ]; then
    python src/image_alignment.py ${img_path} ${save_path} ${match_save_path} --method ${method} --flow_path ${flow_path}
else
    python src/image_alignment.py ${img_path} ${save_path} ${match_save_path} --method ${method}
fi

# python src/depth_from_focus.py ${base_path}
if [ $reverse_input_order == 1 ]; then
    python src/depth_from_focus_dffmp.py ${base_path} \
        --LoG_gaussian_ksize ${LoG_gaussian_ksize} \
        --LoG_laplacian_ksize ${LoG_laplacian_ksize} \
        --AIF_sharpness_sigma ${AIF_sharpness_sigma} \
        --AIF_depth_ksize ${AIF_depth_ksize} \
        --graphcut_sharpness_sigma ${graphcut_sharpness_sigma} \
        --graphcut_unary_scale ${graphcut_unary_scale} \
        --graphcut_pair_scale ${graphcut_pair_scale} \
        --reverse_input_order
else
    python src/depth_from_focus_dffmp.py ${base_path} \
        --LoG_gaussian_ksize ${LoG_gaussian_ksize} \
        --LoG_laplacian_ksize ${LoG_laplacian_ksize} \
        --AIF_sharpness_sigma ${AIF_sharpness_sigma} \
        --AIF_depth_ksize ${AIF_depth_ksize} \
        --graphcut_sharpness_sigma ${graphcut_sharpness_sigma} \
        --graphcut_unary_scale ${graphcut_unary_scale} \
        --graphcut_pair_scale ${graphcut_pair_scale}
fi
# python src/depth_refinement.py ${base_path}

# python src/focus_measure.py ${img_path} ${base_path}