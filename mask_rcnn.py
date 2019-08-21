import tensorflow as tf
from backbones import get_pyramids
from rpn import build_rpn

def model(config):
    inputs = get_inputs(config)
    anchors = generate_anchors(config)
    pyramids = get_pyramids(inputs.image, config.model.backbone, 
                                config.model.pyramid_names)
    boxlist, class_loss, reg_loss = build_rpn(pyramids, anchors, inputs.gt_boxes, config)
    
    
                            