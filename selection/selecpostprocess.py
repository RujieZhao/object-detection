import torch
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.structures import Boxes,Instances

#Refer to Detectron2
# TODO: In the future, mask prediction will be added below.
def sele_posprocess(predbox:Boxes,predcls:torch.Tensor,out_height,out_width,cur_size:list,scores,nms_thre=0.85,mask_threshould:float=0.5):
    """
    This function is for getting the results ready for evaluation.
    It contains 3 steps:
    1. find the remaining index with predbox and classes;
    2. rescale results to original image size and encapsulate mask to bitmask;
    3. Wrap all of them into an Instance.
    """
    processed_results = []
    # fakescore = torch.ones(predbox.tensor.shape[0])
    # fakescore = fakescore.to(predbox.tensor)
    keep = batched_nms(predbox.tensor,scores,predcls,nms_thre)
    # print("finalkeep:",keep)
    # predbox.tensor,predcls = predbox.tensor[keep],predcls[keep]
    if isinstance(out_height,torch.Tensor):
        orig_size = torch.stack([out_height,out_width]).to(torch.float32)
    else:
        orig_size = (float(out_height),float(out_width))
    # print("orig_size:",orig_size)
    scale_x,scale_y = (orig_size[0]/cur_size[0],orig_size[1]/cur_size[1])
    print(scale_x,scale_y)
    results = Instances(orig_size)
    # print("predbox_grad:",predbox.tensor.requires_grad) #False
    finalbox = predbox.tensor.clone()
    finalbox = Boxes(finalbox[keep])
    finalbox.scale(scale_x,scale_y)
    finalbox.clip(results.image_size)#(426.0, 640.0)
    results.pred_boxes = finalbox
    print("finalbox:",finalbox,finalbox.tensor.shape)
    results.scores = scores[keep]
    # print("finalscoreshape:",result.scores.shape,result.scores)
    results.pred_classes = predcls[keep]
    print("finalcls:",results.pred_classes)

    # newpredcls = torch.tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0, 37, 37, 37], device='cuda:0')
    # results.pred_classes = newpredcls
    # newpredboxes = torch.tensor([[422.5928, 130.7058, 435.3512, 153.7444],
    #     [201.0394, 133.9625, 222.2777, 164.6647],
    #     [364.8801, 125.4439, 380.2850, 156.1233],
    #     [443.4171, 132.4251, 463.1530, 154.5921],
    #     [523.1766, 128.8505, 546.7471, 155.3386],
    #     [329.8399, 115.4952, 391.7134, 189.3378],
    #     [  0.0000, 135.1347,  29.0402, 148.7225],
    #     [219.2866, 140.5048, 247.5896, 152.0074],
    #     [318.0615, 138.3279, 334.4716, 149.6160],
    #     [342.9138, 194.9319, 386.7456, 208.4117],
    #     [223.9907, 155.6573, 246.5222, 164.2101],
    #     [456.8582, 142.9690, 470.3895, 155.8285],],
    #     device=results.pred_boxes.tensor.device)
    # results.pred_boxes = Boxes(newpredboxes)
    # print("finalresults:",results)
    processed_results.append({"instances":results})
    return processed_results






















