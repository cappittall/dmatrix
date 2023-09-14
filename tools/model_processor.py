## proceed two type of model 
#  .tflite and _edge.tflite 

import cv2
import numpy as np
from typing import Tuple
import os

from collections import namedtuple

from tflite_runtime.interpreter import load_delegate, Interpreter
# import tensorflow as tf  

from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import run_inference

DetectedObject = namedtuple('DetectedObject', ['bbox', 'class_id', 'score'])

class ModelProcessor:
    def __init__(self, model_path: str, threshold: float):
        self.model_path = model_path
        self.edge = 'edge' in model_path or 'int8_qat' in model_path
        self.interpreter = self._initialize_interpreter()
        self.interpreter.allocate_tensors()
        self.input_size = self._get_input_size()
        self.threshold = threshold

    def update_model(self, model_path: str):
        self.model_path = model_path
        self.interpreter = self._initialize_interpreter()
        self.interpreter.allocate_tensors()

    def update_threshold(self, threshold: float):
        self.threshold = threshold

    def _initialize_interpreter(self) -> Interpreter:
        if self.edge:
            return Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
        else:
            return Interpreter(model_path=str(self.model_path))

    def _get_input_size(self) -> Tuple[int, int]:
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_size = input_shape[1:3]  
        return input_size

    def preprocess_input(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, self.input_size)
        
        normalized_image = resized_image.astype(np.uint8)
        # normalized_image = resized_image.astype(np.float32)
        input_data = np.expand_dims(normalized_image, axis=0)
        return input_data
    
    def check_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])

        intersection_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0

        return intersection_area / union_area
    # 
    def extract_detected_objects(self, interpreter, min_score_thresh=0.5, iou_threshold=0.2):
        output_details = interpreter.get_output_details()
        scores = interpreter.get_tensor(output_details[0]['index'])
        boxes = interpreter.get_tensor(output_details[1]['index'])
        num_detections = interpreter.get_tensor(output_details[2]['index'])
        classes = interpreter.get_tensor(output_details[3]['index'])
        detected_objects_list = []
        
        print('Min score threshold', min_score_thresh )
        for i in range(int(num_detections[0])):
            if scores[0][i] >= min_score_thresh:
                bbox = boxes[0][i].tolist()
                normalized_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
                detected_obj = DetectedObject(
                    bbox=normalized_bbox,
                    class_id=int(classes[0][i]),
                    score=scores[0][i]
                )
                detected_objects_list.append(detected_obj)
                
        filtered_objects_list = []
        for i, obj_a in enumerate(detected_objects_list):
            suppress = False
            for j, obj_b in enumerate(detected_objects_list):
                if j <= i:
                    continue
                iou_val = self.check_iou(obj_a.bbox, obj_b.bbox)
                if  iou_val >= iou_threshold:
                    suppress = True
                    break
            if not suppress:
                filtered_objects_list.append(obj_a)

        return filtered_objects_list

    def run_prediction_tflite(self, input_data, min_score_thresh=0.5):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        detected_objects = self.extract_detected_objects(self.interpreter, min_score_thresh=min_score_thresh)
        return detected_objects

    def run_prediction_edgetpu(self, frame, min_score_thresh=0.5, iou_threshold = 0.2):
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inference_size = self.input_size

        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        scale = (frame.shape[1] / inference_size[0], frame.shape[0] / inference_size[1])
        run_inference(self.interpreter, cv2_im_rgb.tobytes())

        objs = get_objects(self.interpreter, min_score_thresh)


        # Normalize bounding boxes
        normalized_objs = []
        for obj in objs:
            # No need score filter it is included get_object above
            # if obj.score < min_score_thresh: continue
            normalized_bbox = [obj.bbox.xmin / inference_size[0], obj.bbox.ymin / inference_size[1],
                            obj.bbox.xmax / inference_size[0], obj.bbox.ymax / inference_size[1]]
            normalized_obj = DetectedObject(bbox=normalized_bbox, class_id=obj.id, score=obj.score)
            normalized_objs.append(normalized_obj)
            
        ## Applying iou threshold
        filtered_objects_list = []
        for i, obj_a in enumerate(normalized_objs):
            suppress = False
            for j, obj_b in enumerate(normalized_objs):
                if j <= i:
                    continue
                iou_val = self.check_iou(obj_a.bbox, obj_b.bbox)
                if  iou_val >= iou_threshold:
                    suppress = True
                    break
            if not suppress:
                filtered_objects_list.append(obj_a)

        return filtered_objects_list
    
    def run_prediction(self, frame ):
        
        if self.edge:
            print('Edge ', self.model_path)
            detected_objects = self.run_prediction_edgetpu(frame, min_score_thresh=self.threshold)
        else:
            print('Non-Edge ', self.model_path)
            preprocessed_input = self.preprocess_input(frame)
            detected_objects = self.run_prediction_tflite(preprocessed_input, min_score_thresh=self.threshold)
        return detected_objects

