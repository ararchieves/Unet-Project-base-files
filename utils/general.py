import cv2
import torch 
import numpy as np
from shapely.geometry import Polygon



def split_classes(t):
    return t[:, 0:1, ...], t[:, 1:2, ...]

def find_contours_batch(masks, threshold=0.5):
    bs = masks.shape[0]
    contour_list = []

    for mask in masks:
        cnt = find_contorus(mask, threshold)
        contour_list.append(cnt)

    return contour_list

def find_contorus(mask, threshold=0.5):
    image_size = mask.shape[-1]

    mask_np = mask.permute(1,2,0).detach().cpu().numpy()
    mask_bin = np.where(mask_np > threshold, 1, 0).astype(np.uint8)

    polygons = []
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        poly = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
        poly = poly.squeeze(1)
        
        polygons.append(poly)

    return polygons

def iou_polygon(poly1, poly2):
    if len(poly1) < 3 or len(poly2) < 3: return 0
    poly1_shapely = Polygon(poly1)
    poly2_shapely = Polygon(poly2)

    intersection_area = poly1_shapely.intersection(poly2_shapely).area
    union_area = poly1_shapely.union(poly2_shapely).area

    iou = intersection_area / union_area

    return iou

def area_of_polygon(polygon):
    polygon = np.vstack((polygon, polygon[0]))

    shoelace = 0.5 * np.sum(np.cross(polygon[:-1], polygon[1:]))
    return torch.tensor(abs(shoelace))  