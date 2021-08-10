"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from skimage import io
import cv2

import glob
import time
import argparse
# from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)

  xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h

  bx1 = np.minimum(bb_test[...,0], bb_gt[..., 0])
  by1 = np.minimum(bb_test[..., 1], bb_gt[..., 1])
  bx2 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
  by2 = np.maximum(bb_test[..., 3], bb_gt[..., 3])
  cvx_hull_area = np.maximum(0., bx2 - bx1) * np.maximum(0., by2 - by1)

  union = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
          + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

  iou = wh / union
  giou = iou - ((cvx_hull_area - union) / cvx_hull_area)
  # print(iou)
  # print(giou)
  return(giou)


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  # if(score==None):
  return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  # else:
  #   return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def convert_bbox_to_wh(box):
    x = int(box[0])
    y = int(box[1])
    w = int(box[2] - box[0])
    h = int(box[3] - box[1])
    return (x, y, w, h)


def convert_wh_to_bbox(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2] + box[0]
    y2 = box[3] + box[1]
    return [x1, y1, x2, y2]

class CSRTTracker(object):
    count = 0
    def __init__(self, bbox, frame, min_hits, max_age):
        """
        Initializes a tracker using initial bounding box
        """
        print("DETECTION")
        # bounding box
        self.csrt = cv2.TrackerCSRT_create()
        # convert the bbox to csrt bbox format
        # print(bbox[:4])
        self.csrt.init(frame, convert_bbox_to_wh(bbox[:4]))
        self.bbox = bbox
        self.time_since_update = 0
        self.id = CSRTTracker.count
        CSRTTracker.count += 1
        self.history = []
        self.min_hits = min_hits
        self.max_age = max_age
        self.hits = 0
        self.hit_streak = 1
        self.age = 0

        self.detclass = bbox[5]
        self.confidence = bbox[4]

    # update with given detection
    def update(self, bbox, frame):
        print("DETECTION")
        # when updating we need to reinitialize the tracker with the bounding box
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox
        self.csrt.init(frame, convert_bbox_to_wh(bbox[:4]))
        self.confidence = bbox[4]
    # update with prediction
    def update_unmatched(self, bbox):
        self.bbox = bbox
        bbox[4] = self.confidence

    def predict(self, frame):
        ok, bbox = self.csrt.update(frame)
        # convert
        bbox = convert_wh_to_bbox(bbox)
        if ok:
            self.age += 1
            # # ensures that we need 3 consecutive detections before initializing
            # if(self.time_since_update > 0 and self.hit_streak < self.min_hits):
            #     self.hit_streak = 0
            # # resetting tracker when we go through 5 frames with no existing detection
            # if(self.time_since_update > self.max_age):
            #   self.hit_streak = 0
            self.time_since_update += 1
            self.history.append(bbox)
            return self.history[-1]
        else:
            # print("CSRT didn't work")
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    def get_state(self):
        # print(self.detclass)
        # print(self.confidence)
        if(self.time_since_update > 0 and self.hit_streak < self.min_hits):
            self.hit_streak = 0
        if(self.time_since_update > self.max_age):
            self.hit_streak = 0
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        arr_confidence = np.expand_dims(np.array([self.confidence]), 0)

        # arr_u_dot = np.expand_dims(self.bbox[4],0)
        # arr_v_dot = np.expand_dims(self.bbox[5],0)
        # arr_s_dot = np.expand_dims(self.bbox[6],0)
        state = self.bbox
        np.append(state, [self.detclass, self.confidence])
        return state

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0


  def update(self, frame, dets=np.empty((0, 6))):
    # case 1: Detection fed -> update matched trackers with detection value
    # case 2: No detection fed -> lonely trackers, update them with a predicted value

    # feed in an array of detections now wtf is this score thing?? -> what class??
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # print("DETECTION: ", dets.astype(int))
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 6))
    to_del = []
    ret = []
    # for each tracker make a prediction / if smth is null then delete the tracker
    for t, trk in enumerate(trks):
      # this is where the prediction is made! #
      pos = self.trackers[t].predict(frame)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    # print("PREDICTION: ", trks.astype(int))
    for t in reversed(to_del):
      self.trackers.pop(t)

    # use the predictions & detections and like associate which ones match and stuff w/ a iou threshold, simple iou approach
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
    # rn everything is in [x1, y1, x2, y2] format

    # update matched trackers with assigned detections
    # for the stuff that matched make an update
    for m in matched:
      # print(m)
      self.trackers[m[1]].update(dets[m[0], :], frame) # update tracker state
      # print("UPDATE: ", self.trackers[m[1]].get_state().astype(int)) # for matched stuff updating is simple

    # create and initialise new trackers for unmatched detections
    # make a new csrt tracker
    for i in unmatched_dets:
        # print(dets[i,:])
        # print(type(dets[i,:]))
        trk = CSRTTracker(dets[i,:], frame, self.min_hits, self.max_age)
        # trk = CSRTtracker
        self.trackers.append(trk) # also easy to make a new tracker with just the bounding box we feed

    # look at unmatched_trackers
    for t in unmatched_trks:
        trk = self.trackers[t]
        if(trk.time_since_update <= self.max_age):
            # print("SUS_UPDATE: ", trks[t])
            self.trackers[t].update_unmatched(trks[t])
            # update the john
        # otherwise it will be killed in a later step

    i = len(self.trackers)

    for trk in reversed(self.trackers): # we will have some unmatched trackers
        d = trk.get_state()
        print("Hit Streak: ", trk.hit_streak)
        print("TIME SINCE UPDATE: ", trk.time_since_update)
        if (trk.time_since_update == 0 or (trk.time_since_update <= self.max_age and trk.hit_streak >= self.min_hits)):
            ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          # print("REMOVING!")
          self.trackers.pop(i)
    if(len(ret)>0):
      # print("RETURN: ", ret)
      return np.concatenate(ret)
    return np.empty((0,6))
