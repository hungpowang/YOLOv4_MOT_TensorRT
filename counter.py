from collections import deque
import numpy as np


Q_LEN = 8
IN_STATUS = deque([0, 0, 0, 0, 1, 1, 1, 1], Q_LEN)
OUT_STATUS = deque([1, 1, 1, 1, 0, 0, 0, 0], Q_LEN)


# 輸入兩個點的座標，回傳直線參數(a, b)
def get_line_parameter(coordinate1, coordinate2):
    a = np.array([[coordinate1[0], 1], [coordinate2[0], 1]])
    b = np.array([coordinate1[1], coordinate2[1]])
    (a, b) = np.linalg.solve(a, b)
    # print("y = {}*x + {}" .format(a, b))
    return (a, b)


# 根據一組中心點(cx, xy), 直線參數(a,b)回傳above/under
def side_classifier(pos, line_para, side=1):
    '''
    default: side=1 (image view!!!)
        y > ax + b .... under ... return 0
        y <= ax + b ... above ... return 1
    '''
    cx = pos[0]
    cy = pos[1]
    a = line_para[0]
    b = line_para[1]

    if cy > a*cx + b:
        # print("y > ax + b ... Under")
        return 1 if side == 1 else 0
    else:
        # print("y <= ax + b ... Above")
        return 0 if side == 1 else 1


# 更新整張table每個bbox的side狀態
def update_side(tracking_table, line_para, side=1):
    for item in tracking_table:
        item['q'].append(side_classifier(item['pos'], line_para, side))


# 計算一張table的IN/OUT個數以 [#IN, #OUT] 回傳
def in_out_sum(tracking_table):
    in_sum = 0
    out_sum = 0
    for item in tracking_table:
        if item['q'] == IN_STATUS:
            in_sum += 1
        elif item['q'] == OUT_STATUS:
            out_sum += 1
        else:
            pass
    return [in_sum, out_sum]
