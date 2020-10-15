import cv2
import numpy as np
import time
import sys
import os
import time
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from tool.utils import *
from tracker.mot import (tracking_table_init_with_id,
                         tracking_table_init,
                         do_pairing,
                         remove_low_confidence,
                         none_type_checking)
from counter import (get_line_parameter,
                     update_side,
                     in_out_sum)


CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
vc = cv2.VideoCapture("../crowd.mp4")

# TRT
engine_path = './yolov4_tiny_608.engine'
TRT_LOGGER = trt.Logger()


# ---------------------------
# for saving video
# ---------------------------
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))


def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        print("A serialized engine already exist.")
        return runtime.deserialize_cuda_engine(f.read())


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def detect(context, buffers, image_src, image_size, num_classes):
    IN_IMAGE_H, IN_IMAGE_W = image_size

    ta = time.time()
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    # print("Shape of the network input: ", img_in.shape)
    # print(img_in)

    inputs, outputs, bindings, stream = buffers
    print('Length of inputs: ', len(inputs))
    inputs[0].host = img_in

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print('Len of outputs: ', len(trt_outputs))

    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)

    tb = time.time()

    print('-----------------------------------')
    print('    TRT inference time: %f' % (tb - ta))
    print('-----------------------------------')

    boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)

    return boxes

p_in = 0
p_out = 0
frame_count = 0
frame0_flag = 0  # stands for status of the first frame
with get_engine(engine_path) as engine, engine.create_execution_context() as context:
    buffers = allocate_buffers(engine, 1)
    # image_size = (416, 416)
    image_size = (608, 608)
    IN_IMAGE_H, IN_IMAGE_W = image_size
    context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))


    while cv2.waitKey(1) < 1:
        (grabbed, frame) = vc.read()
        if not grabbed:
            break
        height, width = frame.shape[:2]
    
        # ----------------------------
        # inference - obj detection
        # ----------------------------
        start = time.time()
        boxes = detect(context, buffers, frame, image_size, 1)
        end = time.time()
        print("There are {} bounding boxes." .format(len(boxes[0])))
    
        # Draw a line
	point_1_y = 250
	point_2_y = 400
        cv2.line(frame, (0, point_1_y), (frame.shape[1], point_2_y), (0, 180, 0), 2)
        line_para = get_line_parameter((0, point_1_y), (point_1_y, point_2_y))

        bboxes = []
        for box in boxes[0]:
            x = int(box[0] * width)
            y = int(box[1] * height)
            w = int(width * abs(box[2] - box[0]))
            h = int(height * abs(box[3] - box[1]))
            c = box[4]
            bbox = [x, y, w, h, c]
            bboxes.append(bbox)
        # print(bboxes)
        
        # ----------------------------
        # obj tracking
        # ----------------------------
        if frame0_flag == 0:
            # INITIALIZATION
            new = tracking_table_init_with_id(bboxes)
            frame0_flag = 1
        else:
            # INITIALIZATION
            old = new
            new = tracking_table_init(bboxes)

            # TRACKING
            do_pairing(new, old)  # pairing
            remove_low_confidence(new)  # removing
            none_type_checking(new)  # checking

        update_side(new, line_para, 1)
        (in_sum, out_sum) = in_out_sum(new)
        print("\nframe {}\n\t In : {}\n\tOut : {}" .format(frame_count, in_sum, out_sum))
        p_in += in_sum
        p_out += out_sum
        for item in new:
            print(item['q'])

        start_drawing = time.time()
        # annotate bounding box
        for box in bboxes:
            x1 = box[0]
            y1 = box[1]
            x2 = x1 + box[2]
            y2 = y1 + box[3]
            label = "person-{}%" .format(int(100*box[4]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # annotate ID
        for item in new:
            # Draw tracking IDs
            text = "ID={}, C={}" .format(item['id'], round(item['confidence'], 1))
            color = (0, 0, 255)
            cv2.putText(frame, text, (item['pos'][0] - 25, item['pos'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()

        fps_label = "FPS: %.2f" % (1 / (end - start))
        # cv2.putText(frame, fps_label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, fps_label, (5, LINE_Y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        in_label = "IN: {}" .format(p_in)
        cv2.putText(frame, in_label, (5, LINE_Y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out_label = "OUT: {}" .format(p_out)
        cv2.putText(frame, out_label, (5, LINE_Y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("detections", frame)
	
        # ---------------------------
        # for saving video
        # ---------------------------
        # out.write(frame)

