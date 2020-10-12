import pandas as pd
import numpy as np
from collections import deque


Q_LEN = 8
# 設定frame與frame之間合理的移動距離上限
DISTANCE_TH = 13.0
# 設定信心度下降率
CONFIDENCE_DROP_RATE = 0.9
# 設定信心度TH
CONFIDENCE_THRESHOLD = 0.5


def get_center(box):
    '''
    取得bounding box中心點
    '''
    cx = box[0] + box[2] // 2
    cy = box[1] + box[3] // 2
    return [cx, cy]


def get_distance(position_1, position_2):
    '''
    取得兩點間歐式距離
    '''
    vector1 = np.array(position_1)
    vector2 = np.array(position_2)
    dist = np.linalg.norm(vector1-vector2)
    return dist


def tracking_table_init_with_id(boxes):
    '''
    Tracking table initialization with ID
        只會用在第一個frame

        build TABLE as
        [ {ID::int, POSITION::[x, y], CONFIDENCE::float} ]
    '''
    tracking_table = []
    for i, box in enumerate(boxes):
        # item = {'id': i, 'pos': get_center(box), 'confidence': 1.0}
        item = {'id': i, 'pos': get_center(box), 'confidence': 1.0, 'q': deque([],Q_LEN)}
        tracking_table.append(item)
    return tracking_table


def tracking_table_init(boxes):
    '''
    Tracking table initialization without ID
        只填入中心點座標

        build TABLE as
        [ {ID::int, POSITION::[x, y], CONFIDENCE::float} ]
    '''
    tracking_table = []
    for i, box in enumerate(boxes):
        # item = {'id': None, 'pos': get_center(box), 'confidence': None}
        item = {'id': None, 'pos': get_center(box), 'confidence': None, 'q': deque([],Q_LEN)}
        tracking_table.append(item)
    return tracking_table


def build_dist_table_boxes(current_table, last_table):
    '''
    建立兩個frame每個box之間的距離表
    輸入為:: TABLE
    '''
    dist_table = []
    dist = []

    for box_n in current_table:
        for box_o in last_table:  # get one row
            dist.append(round(get_distance(box_o['pos'], box_n['pos']), 2))
        dist_table.append(dist)
        dist = []

    new_names = ['curr_{}' .format(i) for i in range(len(current_table))]
    old_names = ['last_{}' .format(i) for i in range(len(last_table))]
    df = pd.DataFrame(dist_table, index=new_names, columns=old_names)
    # print("Dimension of DF=", df.shape)
    return df


# ------------------------------------
# Multiple Object Tracker 使用流程
#
# STEP 1 - Pairing:
#          >> do_pairing(new, old)
# STEP 2 - Remove [Confidence < 0.5]:
#          >> remove_low_confidence(table)
# STEP 3 - Table checking:
#          >> none_type_checking(table)
# ------------------------------------
def do_pairing(new, old):
    '''
    * 1-1 Generate dist_df
    * 1-2 Loop for each column(old items)
        * find min
        * pairing
        * Remove Column
        * Remove Row(if success)
    * 1-3 Row remain in dist_table: give new ID & C=1
    '''
    # 1-1
    dist_df = build_dist_table_boxes(new, old)
    col_names = list(dist_df.columns)
    # row_names = list(dist_df.index)

    # 1-2
    for col_name in col_names:
        # Finding Min && 在THRESHOLD內
        min = dist_df[col_name].min()
        try:
            idx = dist_df[col_name].idxmin()
        except ValueError:  # len(old) > len(new)
            # print("len(old) > len(new) ... End at {}" .format(col_name))
            # 將剩餘所有的old item加入new(作法與配對失敗相同)
            # (1) 取得剩餘的
            rest_of_cols = list(dist_df.columns)
            for col in rest_of_cols:
                idx_from_last = int(col.split('_')[-1])
                old[idx_from_last]['confidence'] *= CONFIDENCE_DROP_RATE
                new.append(old[idx_from_last])
            break

        # Pairing
        if min < DISTANCE_TH:
            idx_from_last = int(col_name.split('_')[-1])
            idx_target_current = int(idx.split('_')[-1])
            # 配對成功的處理
            # 更新 new table: (1)繼承ID (2)Confidence = 1
            new[idx_target_current]['id'] = old[idx_from_last]['id']
            new[idx_target_current]['q'] = old[idx_from_last]['q']
            new[idx_target_current]['confidence'] = 1
            # print("(Pairing) new[{}] <-- old[{}] ID = {} ...... Distance = {}"
            #       .format(idx_target_current, idx_from_last, old[idx_from_last]['id'], min))
        else:  # 沒配對成功 --> 處理舊的物件
            # 1. Ｃ * CONFIDENCE_DROP_RATE
            idx_from_last = int(col_name.split('_')[-1])
            old[idx_from_last]['confidence'] *= CONFIDENCE_DROP_RATE
            # 2. 加入(繼承)至current_table ...... 使用append實作
            new.append(old[idx_from_last])
            # print("(pairing fail) new[{}] <--- old[{}] with C = {}"
            #       .format(idx_from_last, idx_from_last, old[idx_from_last]['confidence']*CONFIDENCE_DROP_RATE))

        # 移除 Column
        dist_df = dist_df.drop(col_name, axis=1)
        # 移除 Row
        dist_df = dist_df.drop(idx)

    # 1-3 將current frame中未配對到的box指派新ID
    for i, item in enumerate(new):
        if item['id'] is None or item['confidence'] is None:
            # print("(None type ID) new[{}]: {}" .format(i, item))
            # ids: list出目前的所有id
            ids = [new[i]['id'] for i in range(len(new))]

            # 找一個在範圍內未使用的id來使用
            for j in range(len(new)):
                if j not in ids:
                    new[i]['id'] = j
                    new[i]['confidence'] = 1
                    break
            # print("(new id) new[{}] = {}" .format(i, new[i]['id']))


def remove_low_confidence(table):
    '''
    刪除信心度低於Threchold的bounding box
    '''
    # 找出要刪除的index
    l_to_remove = []
    for i in range(len(table)):
        if table[i]['confidence'] < CONFIDENCE_THRESHOLD:
            l_to_remove.append(i)

    # 要從尾往前刪
    for i in l_to_remove[::-1]:
        # print("(delete) new[{}]" .format(i))
        del table[i]


def none_type_checking(table):
    '''
    檢查有無未被賦值的id或confidence
    '''
    for i, item in enumerate(table):
        if item['id'] is None or item['confidence'] is None:
            pass
            # print("Got None type field in item\nnew[{}]: {}" .format(i, item))
