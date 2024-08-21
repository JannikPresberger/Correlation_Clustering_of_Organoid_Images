
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
from stardist.models import StarDist2D

import matplotlib
import matplotlib.pyplot as plt
import time
import numpy
from PIL import Image

StarDist2D.from_pretrained()
    # creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

def detect_nuclei_in_organoid_image(organoid_img,input_nms_thresh):

    labels, label_dict = model.predict_instances(normalize(organoid_img),nms_thresh=input_nms_thresh)

    return_tuple = (labels,label_dict["points"])

    return return_tuple

def detect_organoids_in_microscopy_image(organoid_img,input_nms_thresh):

    labels, label_dict = model.predict_instances(normalize(organoid_img),nms_thresh=input_nms_thresh)

    num_differnet_labels = label_dict["points"].shape[0]

    return_tuple = (labels,num_differnet_labels)

    return return_tuple
    
