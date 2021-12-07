# method='homography'
# img_path=../Depth_from_Focus/dataset/07/
# base_path=./results/07_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

method='homography'
img_path=../depth_from_focus_data2/Figure1/keyboard/
base_path=./results/dffmp_Figure1_keyboard_${method}/
save_path=${base_path}/align/
match_save_path=${base_path}/matches/

# method='homography'
# img_path=../depth_from_focus_data2/Figure6/largemotion/
# base_path=./results/dffmp_Figure6_largemotion_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

# method='flow'
# img_path=../depth_from_focus_data2/Figure3/kitchen/
# base_path=./results/dffmp_Figure3_kitchen_${method}/
# save_path=${base_path}/align/
# match_save_path=${base_path}/matches/

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




# python src/image_alignment.py ${img_path} ${save_path} ${match_save_path} --method ${method}
python src/depth_from_focus.py ${base_path}
# python src/depth_refinement.py ${base_path}

# python src/depth_from_focus_test.py ${img_path} ${base_path}