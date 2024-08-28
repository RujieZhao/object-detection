#!/usr/bin/env bash

#One gpu core training for stage 1,2 and final inference
#CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS  detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl MODEL.SELECTION.STAGEFROZEN True OUTPUT_DIR /mnt/ssd1/rujie/pytorch/C++/selecout/dissertation/ #/mnt/ssd1/rujie/pytorch/C++/selecout/output_3k/v4/dissertation/   # --resume output_test output_300minbatch result/300/alpha3/stage1/ /lvlossissue/ /layersloadissue/ /mnt/ssd1/rujie/pytorch/C++/selecout/output_test/test_stage1/alpha3_v4/300/trial2_resize/

#Evaluation
CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --eval-only OUTPUT_DIR /mnt/ssd1/rujie/pytorch/C++/selecout/dissertation/
#/mnt/ssd1/rujie/pytorch/C++/selecout/output_3k/v4/dissertation/patch3
#/mnt/ssd1/rujie/pytorch/C++/selecout/dissertation/
#/mnt/ssd1/rujie/pytorch/C++/selecout/output_test/test_stage1/alpha3_v4/300/mnt/ssd1/rujie/pytorch/C++/selecout/output_300minbatch/v4/ #./output_300minbatch/stage2/ ./output_test/lvlossissue/ result/300/alpha3/stage2/

#CUDA_VISIBLE_DEVICES=2,3 ./train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --num-gpus 2 SOLVER.IMS_PER_BATCH 2 MODEL.WEIGHTS  detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl MODEL.SELECTION.STAGEFROZEN True OUTPUT_DIR /mnt/ssd1/rujie/pytorch/C++/selecout/dissertation/ # --resume 300minbatch ./output_test/test_stage1/alpha5 ./output_3k/ /mnt/ssd1/rujie/pytorch/C++/selecout/output_test/test_stage1/alpha3_v4/300/trial2_resize/ /selecout/output_3k/v4/

#CUDA_VISIBLE_DEVICES=2 ./train_net.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --resume --num-gpus 1 SOLVER.IMS_PER_BATCH 1 MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output # --resume _300minbatch

 # --resume MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl OUTPUT_DIR ./output_300minbatch

#Something wrong with this file.
#python3 test_targetgenerator.py --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml

#python3 test_IStarget_visualizer.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/test/ --show #annotation
#/mnt/ssd1/rujie/pytorch/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml

#python3 test_visualizer.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --output-dir /mnt/ssd2/rujie/selec_minbatch/300/batch1 --show #dataloader annotation --show

#set test to true in selecvisualizer.py and targetgenerator.py
#python3 post_IS_testvisualize.py --source dataloader --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/test/ --show #dataloader

#python3 preparedata.py --source annotation --config-file /mnt/ssd1/rujie/pytorch/C++/selection/config/selection_gpu.yaml --output-dir /mnt/ssd2/rujie/predataset/coco/ --show #annotation --show dataloader INPUT.PREDATASET.DIR
# /mnt/ssd2/rujie/predataset/coco/coco_2017_trainpre_21_110/ test



