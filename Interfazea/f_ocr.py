import random
import numpy as np
import matplotlib.pyplot as plt
import math

from collections import defaultdict


# https://shegocodes.medium.com/extract-text-from-image-left-to-right-and-top-to-bottom-with-keras-ocr-b56f098a6efe
def get_distance(predictions):
    """
    Function returns dictionary with (key,value):
        * text : detected text in image
        * idx  : index in predictions
        * center_x : center of bounding box ()
        * center_y : center of bounding box (y)
        * distance_from_origin : hypotenuse
        * distance_y : distance between y and origin (0,0)
    """

    # Point of origin
    x0, y0 = 0, 0
    detections = []
    orden = 0
    for group in predictions:
        # Get center point of bounding box
        top_left_x, top_left_y = group[1][0]
        bottom_right_x, bottom_right_y = group[1][1]
        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2
        # Use the Pythagorean Theorem to solve for distance from origin
        # Calculate difference between y and origin to get unique rows
        distance_from_origin = math.dist([x0,y0], [center_x, center_y])
        distance_x = center_x - x0 # Append all results
        distance_y = center_y - y0    # Append all results
        detections.append({'text':group[0], 'idx': orden, 
        				'center_x':center_x, 'center_y':center_y,
                        'distance_from_origin':distance_from_origin, 
                        'distance_x':distance_x, 'distance_y':distance_y
        })
        orden = orden + 1
    return detections

def distinguish_rows(lst, thresh=15):
    """Function to help distinguish unique rows"""

    sublists = []
    for i in range(0, len(lst)-1):
        if lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh:
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            yield sublists
            sublists = [lst[i+1]]
    yield sublists

def segmentaLineas(predictions, listado, threshold=15):
    A = get_distance(predictions)
    B = list(distinguish_rows(A, threshold))    # Remove all empty rows
    C = list(filter(lambda x:x!=[], B))    		# Order text detections in human readable format

    texto = defaultdict(list)
    num_linea = 0
    ordered_preds = []
    for pr in C:
        row = sorted(pr, key=lambda x:x['distance_from_origin'])
        num_palabra = 0
        palabras = []
        for each in row:
            ordered_preds.append(each['text'])
            idx = each['idx']
            palabras.append(listado.loc[each['idx']].text)
            num_palabra += 1
        texto[num_linea].append(palabras)
        num_linea += 1
    return texto, ordered_preds