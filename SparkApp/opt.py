from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext, functions as F, types, DataFrame
from graphframes import *
from graphframes.lib import AggregateMessages as AM
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import typing
import math
import time
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from sortedcontainers import SortedList
#import mobopt as mo
import numpy as np
import json

import networkx as nx
import matplotlib.pyplot as plt
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


def load_from_list(data):
  layout = dict()
  
  for v in data:
    id_ = v['id']
    layout[id_] = np.array(v['pos'])
  
  return layout

def draw_layout(layout, edgelist_path, node_size=10, with_labels=False, linewidths=0., save_path=None):
  plt.figure(figsize=(10, 10), dpi=80)
  G = nx.read_edgelist(edgelist_path)
  nx.draw_networkx(G, layout, node_size=node_size, with_labels=with_labels, linewidths=linewidths)
  plt.axis('off')
  plt.tight_layout()
  if save_path is not None:
    plt.savefig(save_path)


sc = SparkContext('local')
spark = SparkSession(sc)

spark.sparkContext.setLogLevel('WARN')

print(sc.getConf().getAll())

sc.setCheckpointDir(os.path.join(dir_path, 'CheckPoint'))

edge_list_txt = sc.textFile(os.path.join(dir_path, 'Datasets/ego-Facebook_500.txt'))
edge_list = edge_list_txt.map(lambda k: k.split())
edge_list_df = edge_list.toDF()

vertices_df = edge_list_df.select('_1').union(
    edge_list_df.select('_2')).distinct()

vertices_df = vertices_df.withColumnRenamed('_1', 'id')
edge_list_df = edge_list_df.withColumnRenamed(
    '_1', 'src').withColumnRenamed('_2', 'dst')

vertices_df.show()
edge_list_df.show()


def layout(vertices_df: DataFrame, edge_list_df: DataFrame, width=100, height=100, t=100., C=10., MAX_ITR=100):

    # att = d^2/k
    # rep = -k^2/d
    # k = C * sqrt(area / numberofvertices)

    area = width * height
    k = C * math.sqrt(area / vertices_df.count())

    vertices_with_xy = vertices_df.\
        withColumn('pos', F.array(F.rand(seed=42), F.rand(seed=41)))

    vertices_pair_df = vertices_df.\
        withColumnRenamed('id', 'src').\
        crossJoin(vertices_df).\
        withColumnRenamed('id', 'dst').distinct().filter(
            F.col('src') != F.col('dst'))

    cached_vertices_pair_df = AM.getCachedDataFrame(vertices_pair_df)
    cached_edge_list_df = AM.getCachedDataFrame(edge_list_df)

    # vertices_with_xy.write.json('Juan_init.json')

    start_t = time.time()

    ITR = 10#  MAX_ITR

    pbar = tqdm(range(ITR), desc='##LayoutGen##')

    for itr in pbar:
        print(f'itr')

        cached_vertices_with_xy = AM.getCachedDataFrame(vertices_with_xy)

        rep_graph = GraphFrame(cached_vertices_with_xy,
                               cached_vertices_pair_df)
        rep_df = rep_graph.aggregateMessages(
            F.array(F.sum(AM.msg[1] / AM.msg[0] * (k**2 / AM.msg[0])),
                    F.sum(AM.msg[2] / AM.msg[0] * (k**2 / AM.msg[0]))).alias('f'),
            sendToDst=F.array(F.sqrt((AM.dst['pos'][0] - AM.src['pos'][0])**2 +
                                     (AM.dst['pos'][1] - AM.src['pos'][1])**2), AM.dst['pos'][0] - AM.src['pos'][0], AM.dst['pos'][1] - AM.src['pos'][1]),
            sendToSrc=F.array(F.sqrt((AM.dst['pos'][0] - AM.src['pos'][0])**2 +
                                     (AM.dst['pos'][1] - AM.src['pos'][1])**2), AM.src['pos'][0] - AM.dst['pos'][0], AM.src['pos'][1] - AM.dst['pos'][1]),
        ).withColumnRenamed('id', 'v')

        att_graph = GraphFrame(cached_vertices_with_xy, cached_edge_list_df)
        att_df = att_graph.aggregateMessages(
            F.array(F.sum(-AM.msg[1] / AM.msg[0] * (AM.msg[0]**2/k)),
                    F.sum(-AM.msg[2] / AM.msg[0] * (AM.msg[0]**2/k))).alias('f'),
            sendToDst=F.array(F.sqrt((AM.dst['pos'][0] - AM.src['pos'][0])**2 +
                                     (AM.dst['pos'][1] - AM.src['pos'][1])**2), AM.dst['pos'][0] - AM.src['pos'][0], AM.dst['pos'][1] - AM.src['pos'][1]),
            sendToSrc=F.array(F.sqrt((AM.dst['pos'][0] - AM.src['pos'][0])**2 +
                                     (AM.dst['pos'][1] - AM.src['pos'][1])**2), AM.src['pos'][0] - AM.dst['pos'][0], AM.src['pos'][1] - AM.dst['pos'][1]),
        ).withColumnRenamed('id', 'v')

        forces_df = rep_df.union(att_df).groupBy('v').agg(
            F.array(F.sum(F.col('f')[0]), F.sum(F.col('f')[1])).alias('f'),
        )

        vertices_with_xy = vertices_with_xy.join(forces_df.withColumnRenamed('v', 'id'), 'id', 'left').withColumn(
            'pos', F.array(
                F.col('pos')[0] + F.col('f')[0] / F.sqrt(F.col('f')[0]**2 + F.col('f')
                                                         [1]**2) * F.when(F.sqrt(F.col('f')[0]**2 + F.col('f')[1]**2) < t, F.sqrt(F.col('f')[0]**2 + F.col('f')[1]**2)).otherwise(t),
                F.col('pos')[1] + F.col('f')[1] / F.sqrt(F.col('f')[0]**2 + F.col('f')
                                                         [1]**2) * F.when(F.sqrt(F.col('f')[0]**2 + F.col('f')[1]**2) < t, F.sqrt(F.col('f')[0]**2 + F.col('f')[1]**2)).otherwise(t)
            )).drop('f')

        vertices_with_xy = vertices_with_xy.checkpoint()

    # vertices_with_xy.show()

    print(time.time() - start_t)

    # vertices_with_xy.write.json('Juan.json')

    return vertices_with_xy


class FenwickTree:
    '''Reference: https://en.wikipedia.org/wiki/Fenwick_tree'''

    def __init__(self, n: int = 0) -> None:
        self._n = n
        self.data = [0] * n

    def add(self, p: int, x: typing.Any) -> None:
        assert 0 <= p < self._n

        p += 1
        while p <= self._n:
            self.data[p - 1] += x
            p += p & -p

    def sum(self, left: int, right: int) -> typing.Any:
        assert 0 <= left <= right <= self._n

        return self._sum(right) - self._sum(left)

    def _sum(self, r: int) -> typing.Any:
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r

        return s


class Node1:

    def __init__(self):
        self.cnt = 0
        self.val = 0
        self.lc, self.rc, self.pos = -1, -1, -1


class Node2:

    def __init__(self):
        self.val, self.lc, self.rc = -1, -1, -1


class Sparse2DSegTree:

    def __init__(self, R, C) -> typing.Any:
        self.R = R
        self.C = C

        self.nodes1 = []
        self.nodes2 = []

        self.root = self.newNode2()

    def newNode1(self):
        self.nodes1.append(Node1())
        return len(self.nodes1) - 1

    def newNode2(self):
        self.nodes2.append(Node2())
        return len(self.nodes2) - 1

    def update1(self, node: int, tl: int, tr: int, x: int, val, cnt):
        if tl == tr:
            self.nodes1[node].val = val
            self.nodes1[node].cnt = cnt
            return

        mid = (tl + tr) >> 1
        if self.nodes1[node].pos != -1:
            t = self.newNode1()
            if self.nodes1[node].pos <= mid:
                self.nodes1[node].lc = t

                self.nodes1[self.nodes1[node].lc].pos = self.nodes1[node].pos
                self.nodes1[self.nodes1[node].lc].val = self.nodes1[node].val
                self.nodes1[self.nodes1[node].lc].cnt = self.nodes1[node].cnt
            else:
                self.nodes1[node].rc = t

                self.nodes1[self.nodes1[node].rc].pos = self.nodes1[node].pos
                self.nodes1[self.nodes1[node].rc].val = self.nodes1[node].val
                self.nodes1[self.nodes1[node].rc].cnt = self.nodes1[node].cnt
            self.nodes1[node].pos = -1

        if x <= mid:
            if self.nodes1[node].lc == -1:
                t = self.newNode1()
                self.nodes1[node].lc = t
                self.nodes1[self.nodes1[node].lc].pos = x
                self.nodes1[self.nodes1[node].lc].val = val
                self.nodes1[self.nodes1[node].lc].cnt = cnt
            else:
                self.update1(self.nodes1[node].lc, tl, mid, x, val, cnt)
        else:
            if self.nodes1[node].rc == -1:
                t = self.newNode1()
                self.nodes1[node].rc = t
                self.nodes1[self.nodes1[node].rc].pos = x
                self.nodes1[self.nodes1[node].rc].val = val
                self.nodes1[self.nodes1[node].rc].cnt = cnt
            else:
                self.update1(self.nodes1[node].rc, mid + 1, tr, x, val, cnt)

        t = 0
        if self.nodes1[node].lc != -1:
            t += self.nodes1[self.nodes1[node].lc].val
        if self.nodes1[node].rc != -1:
            t += self.nodes1[self.nodes1[node].rc].val
        self.nodes1[node].val = t

        t = 0
        if self.nodes1[node].lc != -1:
            t += self.nodes1[self.nodes1[node].lc].cnt
        if self.nodes1[node].rc != -1:
            t += self.nodes1[self.nodes1[node].rc].cnt
        self.nodes1[node].cnt = t

    def query1(self, node: int, tl: int, tr: int, xl: int, xr: int):
        if xr < tl or tr < xl:
            return 0, 0
        if xl <= tl and tr <= xr:
            return self.nodes1[node].val, self.nodes1[node].cnt
        if self.nodes1[node].pos != -1:
            if xl <= self.nodes1[node].pos and self.nodes1[node].pos <= xr:
                return self.nodes1[node].val, self.nodes1[node].cnt
            else:
                return 0, 0
        mid = (tl + tr) >> 1
        ret, cnt = 0, 0
        if self.nodes1[node].lc != -1:
            ret_, cnt_ = self.query1(self.nodes1[node].lc, tl, mid, xl, xr)
            ret += ret_
            cnt += cnt_
        if self.nodes1[node].rc != -1:
            ret_, cnt_ = self.query1(self.nodes1[node].rc, mid + 1, tr, xl, xr)
            ret += ret_
            cnt += cnt_
        return ret, cnt

    def update2(self, node: int, tl: int, tr: int, y: int, x: int, val, cnt):
        if self.nodes2[node].val == -1:
            t = self.newNode1()
            self.nodes2[node].val = t
        if tl == tr:
            self.update1(self.nodes2[node].val, 1, self.C, x, val, cnt)
            return

        mid = (tl + tr) >> 1
        if y <= mid:
            if self.nodes2[node].lc == -1:
                t = self.newNode2()
                self.nodes2[node].lc = t
            self.update2(self.nodes2[node].lc, tl, mid, y, x, val, cnt)
        else:
            if self.nodes2[node].rc == -1:
                t = self.newNode2()
                self.nodes2[node].rc = t
            self.update2(self.nodes2[node].rc, mid + 1, tr, y, x, val, cnt)
        t, t2 = 0, 0
        if self.nodes2[node].lc != -1:
            t_, cnt_ = self.query1(self.nodes2[self.nodes2[node].lc].val,
                                   1, self.C, x, x)
            t += t_
            t2 += cnt_
        if self.nodes2[node].rc != -1:
            t_, cnt_ = self.query1(self.nodes2[self.nodes2[node].rc].val,
                                   1, self.C, x, x)
            t += t_
            t2 += cnt_
        self.update1(self.nodes2[node].val, 1, self.C, x, t, t2)

    def query2(self, node: int, tl: int, tr: int, yl: int, yr: int, xl: int, xr: int):
        if yr <= tl or tr < yl:
            return 0, 0
        if yl <= tl and tr <= yr:
            if self.nodes2[node].val != -1:
                return self.query1(self.nodes2[node].val, 1, self.C, xl, xr)
            else:
                return 0, 0

        mid = (tl + tr) >> 1
        ret, cnt = 0, 0
        if self.nodes2[node].lc != -1:
            ret_, cnt_ = self.query2(
                self.nodes2[node].lc, tl, mid, yl, yr, xl, xr)
            ret += ret_
            cnt += cnt_
        if self.nodes2[node].rc != -1:
            ret_, cnt_ = self.query2(self.nodes2[node].rc,
                                     mid + 1, tr, yl, yr, xl, xr)
            ret += ret_
            cnt += cnt_
        return ret, cnt

    def update(self, P: int, Q: int, K, C):

        self.update2(self.root, 1, self.R, P + 1, Q + 1, K, C)

    def sum(self, P: int, Q: int, U: int, V: int):

        return self.query2(self.root, 1, self.R, P + 1, U + 1, Q + 1, V + 1)


def edge_crossing3(vertices_with_xy: DataFrame, edge_list_df: DataFrame, grid_size=0.1) -> int:

    edges_with_xy_df = edge_list_df.\
        join(vertices_with_xy, edge_list_df.src == vertices_with_xy.id, 'left').\
        drop('id').\
        withColumnRenamed('pos', 'src_pos').\
        join(vertices_with_xy, edge_list_df.dst == vertices_with_xy.id, 'left').\
        drop('id').\
        withColumnRenamed('pos', 'dst_pos').\
        withColumn('id', F.monotonically_increasing_id())

    # edges_with_xy_df.rdd.getNumPartitions()
    # edges_with_xy_df = edges_with_xy_df.repartition(1000)

    @pandas_udf(types.ArrayType(types.ArrayType(types.ArrayType(types.FloatType()))))
    def locate_segments(src_pos: pd.Series, dst_pos: pd.Series) -> pd.Series:
        ret = []
        for p1, p2 in zip(src_pos, dst_pos):
            x1, y1 = map(float, p1)
            x2, y2 = map(float, p2)
            boundary_points = []

            a = y1 - y2
            b = x2 - x1
            c = (x2 - x1) * y1 + (y1 - y2) * x1

            if b != 0:
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                x0 = int(x1 / grid_size)
                x = x0 * grid_size
                while x + grid_size <= x2:
                    x += grid_size
                    y = (c - a * x) / b
                    boundary_points.append((x, y))
            '''
            if a != 0:
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                y0 = int(y1 / grid_size)
                y = y0 * grid_size
                while y + grid_size <= y2:
                    y += grid_size
                    x = (c - b * y) / a
                    boundary_points.append((x, y))
            '''

            # + [(x1, y1), (x2, y2)]
            boundary_points = list(set(boundary_points))
            boundary_points.sort()

            segments = []

            for i in range(len(boundary_points) - 1):

                p1, p2 = boundary_points[i], boundary_points[i + 1]

                # cx = (p1[0] + p2[0]) / 2
                # cy = (p1[1] + p2[1]) / 2
                # part = (int(cx / grid_size), int(cy / grid_size))

                segments.append((p1, p2))

            ret.append(segments)

        return pd.Series(ret)

    @pandas_udf(types.LongType())
    def count_intersection(id: pd.Series, segment: pd.Series) -> pd.Series:
        import faulthandler
        faulthandler.enable()

        def ccw(a, b, c):
            op = a[0] * b[1] + b[0] * c[1] + c[0] * a[1]
            op -= (a[1] * b[0] + b[1] * c[0] + c[1] * a[0])
            if op > 0:
                return 1
            elif op == 0:
                return 0
            return -1

        def check_intersect(a, b, c, d):
            ab = ccw(a, b, c) * ccw(a, b, d)
            cd = ccw(c, d, a) * ccw(c, d, b)
            if ab == 0 and cd == 0:
                if a > b:
                    a, b = b, a
                if c > d:
                    c, d = d, c
                return c <= b and a <= d
            return ab <= 0 and cd <= 0

        ret = []
        for id_, segment_ in zip(id, segment):

            cnt = 0
            # cnt += len(id_) * (len(id_) - 1) // 2
            # cnt = len(id_)

            left_col = [(min(tuple(p1), tuple(p2))[1], max(tuple(p1), tuple(p2))[1])
                        for p1, p2 in segment_]
            right_col = SortedList()

            left_col.sort()

            update_buffer = []
            prev_ly = None

            for ly, ry in left_col:

                if prev_ly is not None and prev_ly < ly:

                    while len(update_buffer):
                        right_col.add(update_buffer.pop())

                idx = right_col.bisect_right(ry)
                prev_ly = ly

                update_buffer.append(ry)

                cnt += len(right_col) - idx

            '''
            for i in range(len(id_)):
                for j in range(i + 1, len(id_)):
                    try:
                        line1_src = list(map(float, segment_[0][i]))
                        line1_dst = list(map(float, segment_[1][i]))

                        line2_src = list(map(float, segment_[0][j]))
                        line2_dst = list(map(float, segment_[1][j]))

                        if check_intersect(line1_src, line1_dst, line2_src, line2_dst):
                            # ret_.append([min(id_[i], id_[j]), max(id_[i], id_[j])])
                            # ret_.append(
                            #    hash(','.join([min(id_[i], id_[j]), max(id_[i], id_[j])])))
                            cnt += 1
                    except Exception as e:
                        print(e)
            '''

            ret.append(cnt)

        return pd.Series(ret)

    with_parts = edges_with_xy_df.withColumn(
        'segment',
        F.explode(
            locate_segments(F.col('src_pos'), F.col('dst_pos'))
        )
    ).drop(
        'src_pos', 'dst_pos'
    ).withColumn(
        'part',
        F.floor((F.col('segment')[0][0] +
                 F.col('segment')[1][0]) / 2 / grid_size)
    )

    # with_parts.show()

    # 중복 counting안되도록 locate_segments 개선 필요
    # with_parts.groupBy('part').agg(
    #     F.count(F.col('part')).alias('count')
    # ).show(200)
    df = with_parts.groupBy('part').agg(
        count_intersection(F.collect_list(F.col('id')),
                           F.collect_list(F.col('segment'))).alias('count')
    )
    # df.show()
    num_crossing = df.agg(F.sum('count')).collect()[
        0][0]  # .drop('part').distinct().count()

    return num_crossing


def edge_crossing_angle3(vertices_with_xy: DataFrame, edge_list_df: DataFrame, collinear=True, ideal_angle=70, grid_size=1.0) -> int:

    ideal_radian = ideal_angle * math.pi / 180

    edges_with_xy_df = edge_list_df.\
        join(vertices_with_xy, edge_list_df.src == vertices_with_xy.id, 'left').\
        drop('id').\
        withColumnRenamed('pos', 'src_pos').\
        join(vertices_with_xy, edge_list_df.dst == vertices_with_xy.id, 'left').\
        drop('id').\
        withColumnRenamed('pos', 'dst_pos').\
        withColumn('id', F.monotonically_increasing_id())

    # edges_with_xy_df.rdd.getNumPartitions()
    # edges_with_xy_df = edges_with_xy_df.repartition(1000)

    @pandas_udf(types.ArrayType(types.ArrayType(types.ArrayType(types.FloatType()))))
    def locate_segments(src_pos: pd.Series, dst_pos: pd.Series) -> pd.Series:
        ret = []
        for p1, p2 in zip(src_pos, dst_pos):
            x1, y1 = map(float, p1)
            x2, y2 = map(float, p2)
            boundary_points = []

            a = y1 - y2
            b = x2 - x1
            c = (x2 - x1) * y1 + (y1 - y2) * x1

            if b != 0:
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                x0 = int(x1 / grid_size)
                x = x0 * grid_size
                while x + grid_size <= x2:
                    x += grid_size
                    y = (c - a * x) / b
                    boundary_points.append((x, y))

            '''
            if a != 0:
                if y1 > y2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                y0 = int(y1 / grid_size)
                y = y0 * grid_size
                while y + grid_size <= y2:
                    y += grid_size
                    x = (c - b * y) / a
                    boundary_points.append((x, y))
            '''

            # + [(x1, y1), (x2, y2)]
            boundary_points = list(set(boundary_points))
            boundary_points.sort()

            segments = []

            for i in range(len(boundary_points) - 1):

                p1, p2 = boundary_points[i], boundary_points[i + 1]

                # cx = (p1[0] + p2[0]) / 2
                # cy = (p1[1] + p2[1]) / 2
                # part = (int(cx / grid_size), int(cy / grid_size))

                segments.append((p1, p2))

            ret.append(segments)

        return pd.Series(ret)

    @pandas_udf(types.ArrayType(types.DoubleType()))
    def compute_angles(id: pd.Series, segment: pd.Series) -> pd.Series:

        ret = []
        for id_, segment_ in zip(id, segment):

            cnt = 0
            tot_angle = 0
            # cnt += len(id_) * (len(id_) - 1) // 2
            # cnt = len(id_)

            angles = [math.atan2(p1[1] - p2[1], p1[0] - p2[0])
                      for p1, p2 in segment_]
            angles = [(angle, idx) if angle >= 0 else (math.pi + angle, idx)
                      for idx, angle in enumerate(angles)]
            angles.sort()
            sidx2aidx = [0 for _ in range(len(segment_))]
            for aidx, (angle, sidx) in enumerate(angles):
                sidx2aidx[sidx] = aidx
            angles = SortedList([angle for angle, _ in angles])

            right_col = [(max(tuple(p1), tuple(p2))[1], idx)
                         for idx, (p1, p2) in enumerate(segment_)]
            right_col.sort()
            sidx2ridx = [0 for _ in range(len(segment_))]
            for ridx, (ry, sidx) in enumerate(right_col):
                sidx2ridx[sidx] = ridx

            # fwtree = FenwickTree(len(segment_))
            # fwtree_cnt = FenwickTree(len(segment_))

            sgtree = Sparse2DSegTree(len(segment_) + 1, len(segment_) + 1)
            #sgtree_cnt = Sparse2DSegTree(len(segment_) + 1, len(segment_) + 1)

            left_col = [(min(tuple(p1), tuple(p2))[1], max(tuple(p1), tuple(p2))[1], idx)
                        for idx, (p1, p2) in enumerate(segment_)]
            left_col.sort()

            update_buffer = []
            prev_ly = None

            for ly, ry, sidx in left_col:

                if prev_ly is not None and prev_ly < ly:

                    while len(update_buffer):
                        sgtree.update(*update_buffer.pop())
                prev_ly = ly

                aidx = sidx2aidx[sidx]
                ridx = sidx2ridx[sidx]

                angle = angles[aidx]

                left_outer_less_idx = angles.bisect_left(angle - math.pi)
                left_outer_greater_idx = angles.bisect_left(
                    angle - math.pi + ideal_radian)
                left_inner_greater_idx = angles.bisect_left(
                    angle - math.pi / 2)
                left_inner_less_idx = angles.bisect_left(angle - ideal_radian)

                right_inner_less_idx = angles.bisect_left(angle)
                right_inner_greater_idx = angles.bisect_left(
                    angle + ideal_radian)
                right_outer_greater_idx = angles.bisect_left(
                    angle + math.pi / 2)
                right_outer_less_idx = angles.bisect_left(
                    angle + math.pi - ideal_radian)

                left_outer_less_sum, left_outer_less_cnt = sgtree.sum(
                    left_outer_less_idx, ridx, left_outer_greater_idx, len(right_col))
                left_outer_greater_sum, left_outer_greater_cnt = sgtree.sum(
                    left_outer_greater_idx, ridx, left_inner_greater_idx, len(right_col))
                left_inner_greater_sum, left_inner_greater_cnt = sgtree.sum(
                    left_inner_greater_idx, ridx, left_inner_less_idx, len(right_col))
                left_inner_less_sum, left_inner_less_cnt = sgtree.sum(
                    left_inner_less_idx, ridx, right_inner_less_idx, len(right_col))

                right_inner_less_sum, right_inner_less_cnt = sgtree.sum(
                    right_inner_less_idx, ridx, right_inner_greater_idx, len(right_col))
                right_inner_greater_sum, right_inner_greater_cnt = sgtree.sum(
                    right_inner_greater_idx, ridx, right_outer_greater_idx, len(right_col))
                right_outer_greater_sum, right_outer_greater_cnt = sgtree.sum(
                    right_outer_greater_idx, ridx, right_outer_less_idx, len(right_col))
                right_outer_less_sum, right_outer_less_cnt = sgtree.sum(
                    right_outer_less_idx, ridx, len(angles), len(right_col))

                tot_angle += ideal_radian * left_inner_less_cnt - \
                    (left_inner_less_sum - angle * left_inner_less_cnt)
                tot_angle += (left_inner_greater_sum - angle *
                              left_inner_greater_cnt) - ideal_radian * left_inner_greater_cnt
                tot_angle += (angle * left_outer_greater_cnt - (left_outer_greater_sum -
                              math.pi * left_outer_greater_cnt)) - ideal_radian * left_outer_greater_cnt
                tot_angle += ideal_radian * left_outer_less_cnt - \
                    (angle * left_outer_less_cnt -
                     (left_outer_less_sum - math.pi * left_outer_less_cnt))

                tot_angle += ideal_radian * right_inner_less_cnt - \
                    (angle * right_inner_less_cnt - right_inner_less_sum)
                tot_angle += (angle * right_inner_greater_cnt -
                              right_inner_greater_sum) - ideal_radian * right_inner_greater_cnt
                tot_angle += (right_outer_greater_sum + math.pi * right_outer_greater_cnt -
                              angle * right_outer_greater_cnt) - ideal_radian * right_outer_greater_cnt
                tot_angle += ideal_radian * right_outer_less_cnt - \
                    (right_outer_less_sum + math.pi *
                     right_outer_less_cnt - angle * right_outer_less_cnt)

                cnt += sgtree.sum(0, ridx, len(angles), len(right_col))[1]

                update_buffer.append((aidx + 1, ridx + 1, angle, 1))

                #sgtree.update(aidx + 1, ridx + 1, angle, 1)
                #sgtree_cnt.update(aidx + 1, ridx + 1, 1)

            ret.append((cnt, tot_angle))

        return pd.Series(ret)

    with_parts = edges_with_xy_df.withColumn(
        'segment',
        F.explode(
            locate_segments(F.col('src_pos'), F.col('dst_pos'))
        )
    ).drop(
        'src_pos', 'dst_pos'
    ).withColumn(
        'part',
        F.floor((F.col('segment')[0][0] +
                 F.col('segment')[1][0]) / 2 / grid_size)
    )

    # 중복 counting안되도록 locate_segments 개선 필요
    # with_parts.groupBy('part').agg(
    #     F.count(F.col('part')).alias('count')
    # ).show(200)
    df = with_parts.groupBy('part').agg(
        compute_angles(F.collect_list(F.col('id')),
                       F.collect_list(F.col('segment'))).alias('angles')
    )
    # df.show()
    crossing_angle = df.agg(F.sum(F.col('angles')[1]) / (F.sum(F.col('angles')[0]) * ideal_radian)).collect()[
        0][0]  # .drop('part').distinct().count()

    return crossing_angle


def node_occlusion2(vertices_with_xy: DataFrame, r=0.01) -> int:

    grid_size = r * 2

    with_parts = vertices_with_xy.withColumn(
        'part', F.explode(
            F.array(
                F.hash(F.array(F.floor(F.col('pos')[0] / grid_size),
                               F.floor(F.col('pos')[1] / grid_size))),
                F.when(
                    F.floor((F.col('pos')[0] -
                             r) / grid_size) != F.floor(F.col('pos')[0] / grid_size),
                    F.hash(
                        F.array(F.floor(F.col('pos')[0] / grid_size) - 1, F.floor(F.col('pos')[1] / grid_size)))
                ),
                F.when(
                    F.floor((F.col('pos')[0] + r) / grid_size) !=
                    F.floor(F.col('pos')[0] / grid_size),
                    F.hash(
                        F.array(F.floor(F.col('pos')[0] / grid_size) + 1, F.floor(F.col('pos')[1] / grid_size)))
                ),
                F.when(
                    F.floor((F.col('pos')[1] - r) / grid_size) !=
                    F.floor(F.col('pos')[1] / grid_size),
                    F.hash(F.array(F.floor(F.col('pos')[0] / grid_size),
                                   F.floor(F.col('pos')[1] / grid_size) - 1))
                ),
                F.when(
                    F.floor((F.col('pos')[1] + r) / grid_size) !=
                    F.floor(F.col('pos')[1] / grid_size),
                    F.hash(F.array(F.floor(F.col('pos')[0] / grid_size),
                                   F.floor(F.col('pos')[1] / grid_size) + 1))
                ),
                F.when(
                    (F.floor(F.col('pos')[0] / grid_size) * grid_size - F.col('pos')[0])**2 +
                    (F.floor(F.col('pos')[1] / grid_size) *
                     grid_size - F.col('pos')[1])**2 <= r**2,
                    F.hash(
                        F.array(F.floor(F.col('pos')[0] / grid_size) - 1, F.floor(F.col('pos')[1] / grid_size) - 1))
                ),
                F.when(
                    ((F.floor(F.col('pos')[0] / grid_size) + 1) * grid_size - F.col('pos')[0]) **
                    2 + (F.floor(F.col('pos')[1] / grid_size) *
                         grid_size - F.col('pos')[1])**2 <= r**2,
                    F.hash(
                        F.array(F.floor(F.col('pos')[0] / grid_size) + 1, F.floor(F.col('pos')[1] / grid_size) - 1))
                ),
                F.when(
                    ((F.floor(F.col('pos')[0] / grid_size) + 1) * grid_size - F.col('pos')[0])**2 +
                    ((F.floor(F.col('pos')[1] / grid_size) + 1) *
                     grid_size - F.col('pos')[1])**2 <= r**2,
                    F.hash(
                        F.array(F.floor(F.col('pos')[0] / grid_size) + 1, F.floor(F.col('pos')[1] / grid_size) + 1))
                ),
                F.when(
                    (F.floor(F.col('pos')[0] / grid_size) * grid_size - F.col('pos')[0])**2 +
                    ((F.floor(F.col('pos')[1] / grid_size) + 1) *
                     grid_size - F.col('pos')[1])**2 <= r**2,
                    F.hash(
                        F.array(F.floor(F.col('pos')[0] / grid_size) - 1, F.floor(F.col('pos')[1] / grid_size) + 1))
                ),
            )
        )
    ).na.drop()

    @pandas_udf(types.ArrayType(types.ArrayType(types.StringType())))
    def count_occlusion(id: pd.Series, pos: pd.Series) -> pd.Series:
        ret = []
        for id_, pos_ in zip(id, pos):
            ret_ = []
            for i in range(len(pos_)):
                for j in range(i + 1, len(pos_)):
                    x1, y1 = map(float, pos_[i])
                    x2, y2 = map(float, pos_[j])
                    if (x1 - x2)**2 + (y1 - y2)**2 < (2 * r)**2:
                        ret_.append([min(id_[i], id_[j]), max(id_[i], id_[j])])
            ret.append(ret_)
        return pd.Series(ret)

    '''
    @pandas_udf(types.IntegerType())
    def count_occlusion(pos: pd.Series) -> int:
        cnt = 0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                x1, y1 = map(float, pos[i])
                x2, y2 = map(float, pos[j])
                if (x1 - x2)**2 + (y1 - y2)**2 < (2 * r)**2:
                    cnt += 1
        return cnt
    # .agg(F.approx_count_distinct('pair')).collect()[0][0]
    # .approxQuantile('pair', [0.5], 0.25)
    nc = with_parts.groupBy('part').agg(
        count_occlusion(F.col('pos')).alias('cnt')
    ).agg(F.sum(F.col('cnt'))).collect()[0][0]
    '''

    nc = with_parts.groupBy('part').agg(
        F.explode(count_occlusion(F.collect_list(F.col('id')),
                  F.collect_list(F.col('pos')))).alias('pair')
    ).drop('part').distinct().count()

    return nc


def node_occlusion(vertices_with_xy: DataFrame, r=0.01, approx=False) -> int:

    vertices_with_xy1 = vertices_with_xy.\
        withColumnRenamed('id', 'v').\
        withColumnRenamed('pos', 'pos1')

    vertices_with_xy2 = vertices_with_xy.\
        withColumnRenamed('id', 'u').\
        withColumnRenamed('pos', 'pos2')

    if approx:
        nc = vertices_with_xy1.join(
            vertices_with_xy2,
            F.when(
                vertices_with_xy1.v >= vertices_with_xy2.u,
                F.lit(0)
            ).otherwise(
                F.when(
                    (F.col('pos1')[0] - F.col('pos2')[0])**2 +
                    (F.col('pos1')[1] - F.col('pos2')[1])**2 < (2 * r) ** 2,
                    F.lit(1)
                ).otherwise(F.lit(0))
            ) == F.lit(1), 'inner'
        ).rdd.countApprox(timeout=1000, confidence=0.90)
    else:
        nc = vertices_with_xy1.join(
            vertices_with_xy2,
            F.when(
                vertices_with_xy1.v >= vertices_with_xy2.u,
                F.lit(0)
            ).otherwise(
                F.when(
                    (F.col('pos1')[0] - F.col('pos2')[0])**2 +
                    (F.col('pos1')[1] - F.col('pos2')[1])**2 < (2 * r) ** 2,
                    F.lit(1)
                ).otherwise(F.lit(0))
            ) == F.lit(1), 'inner'
        ).count()

    return nc


def minimum_angle(vertices_with_xy: DataFrame, edge_list_df: DataFrame) -> float:

    cached_vertices_with_xy = AM.getCachedDataFrame(vertices_with_xy)
    cached_edge_list_df = AM.getCachedDataFrame(edge_list_df)
    graph = GraphFrame(cached_vertices_with_xy, cached_edge_list_df)

    angles_df = graph.aggregateMessages(
        F.collect_list(F.when(AM.msg >= 0, AM.msg).otherwise(
            AM.msg + 2*math.pi)).alias('angles'),
        sendToDst=F.atan2(AM.src['pos'][1] - AM.dst['pos']
                          [1], AM.src['pos'][0] - AM.dst['pos'][0]),
        sendToSrc=F.atan2(AM.dst['pos'][1] - AM.src['pos']
                          [1], AM.dst['pos'][0] - AM.src['pos'][0]),
    ).withColumnRenamed('id', 'v')

    @ pandas_udf(types.FloatType())
    def get_min_angle(angles: pd.Series) -> pd.Series:
        ret = []
        for angles_ in angles:
            delta = [angles_[i] - angles_[i - 1] for i in range(len(angles_))]
            delta[0] += 2*math.pi
            ret.append(min(delta))
        return pd.Series(ret)

    ma = angles_df.withColumn(
        'angles', F.array_sort(F.col('angles'))
    ).withColumn(
        'mina',
        (2*math.pi / F.size(F.col('angles')) - get_min_angle(
            F.col('angles'))) / (2*math.pi / F.size(F.col('angles')))
    ).agg((F.lit(1.) - F.mean(F.col('mina'))).alias('minimum_angle')).collect()[0][0]

    return ma


def edge_length_variation(vertices_with_xy: DataFrame, edge_list_df: DataFrame, approx=False) -> float:

    cached_vertices_with_xy = AM.getCachedDataFrame(vertices_with_xy)
    cached_edge_list_df = AM.getCachedDataFrame(edge_list_df)
    graph = GraphFrame(cached_vertices_with_xy, cached_edge_list_df)

    l_df: DataFrame = graph.aggregateMessages(
        F.collect_list(AM.msg).alias('l'),
        sendToDst=F.sqrt((AM.src['pos'][0] - AM.dst['pos']
                          [0])**2 + (AM.src['pos'][1] - AM.dst['pos'][1])**2),
    ).select(F.explode(F.col('l')).alias('l'))

    num_edges = l_df.count()

    lm = l_df.agg(F.mean(F.col('l'))).collect()[0][0]

    la = l_df.agg(F.sqrt(F.sum((F.col('l') - lm)**2) /
                         (num_edges * lm**2))).collect()[0][0]

    elv = la / math.sqrt(num_edges - 1)

    return elv


trials = 0
history = []


vertices_with_xy = vertices_df.\
    withColumn('pos', F.array(F.rand(seed=42) * 100., F.rand(seed=41) * 100.))
random_nc = node_occlusion(vertices_with_xy, r=1.)
random_ec = edge_crossing3(vertices_with_xy, edge_list_df, grid_size=0.1)


def opt_f(C, t):

    global trials, history

    print(f'##NewIteration:{trials}##')

    vertices_with_xy = layout(vertices_df, edge_list_df, C=C, t=t, MAX_ITR=100)

    vertices_with_xy.write.json(f'opt_{trials:02d}_{C}_{t}.json')

    trials += 1

    '''
    start_t = time.time()
    nc = node_occlusion2(vertices_with_xy, r=1.)
    nc_t = time.time() - start_t
    print('[grid]node_occlusion:', nc)
    print('took', nc_t, 'sec')
    ret['[grid]node_occlusion'] = (nc, nc_t)

    start_t = time.time()
    ec = edge_crossing3(vertices_with_xy, edge_list_df, grid_size=0.1)
    ec_t = time.time() - start_t
    print('edge_crossing:', ec)
    print('took', ec_t, 'sec')
    ret['edge_crossing_fast'] = (ec, ec_t)

    start_t = time.time()
    eca = edge_crossing_angle3(vertices_with_xy, edge_list_df, grid_size=1.0)
    eca_t = time.time() - start_t
    print('edge_crossing_angle:', eca)
    print('took', eca_t, 'sec')
    ret['edge_crossing_angle_fast'] = (eca, eca_t)

    start_t = time.time()
    ma = minimum_angle(vertices_with_xy, edge_list_df)
    print('minimum_angle took', time.time() - start_t, 'sec')

    start_t = time.time()
    elv = edge_length_variation(vertices_with_xy, edge_list_df)
    print('edge_length_variation took', time.time() - start_t, 'sec')

    start_t = time.time()
    ec = edge_crossing(vertices_with_xy, edge_list_df)
    print('edge_crossing took', time.time() - start_t, 'sec')
    '''

    print('##ComputeNodeOcclusion##')
    start_t = time.time()
    nc = node_occlusion(vertices_with_xy, r=1.)
    nc_t = time.time() - start_t
    print('node_occlusion took', nc_t, 'sec')

    print('##ComputeEdgeCrossing##')
    start_t = time.time()
    ec = edge_crossing3(vertices_with_xy, edge_list_df, grid_size=0.1)
    ec_t = time.time() - start_t
    # print('edge_crossing:', ec)
    print('edge_crossing took', ec_t, 'sec')

    # start_t = time.time()
    # eca = edge_crossing_angle3(vertices_with_xy, edge_list_df, grid_size=1.0)
    # eca_t = time.time() - start_t

    score = nc / random_nc + ec / random_ec # min(random_nc, nc) / random_nc + min(random_ec, ec) / random_ec

    history.append({
        'trial': trials - 1,
        'C': C,
        't': t,
        'nc': nc,
        'nc_t': nc_t,
        'ec': ec,
        'ec_t': ec_t,
        'score': score,
    })

    vertices_with_xy_list = map(lambda row: row.asDict(), vertices_with_xy.collect())

    print('##Visualizing##')
    draw_layout(
        load_from_list(vertices_with_xy_list), 
        os.path.join(dir_path, 'Datasets/ego-Facebook_500.txt'),
        save_path=os.path.join(dir_path, 'visualization/opt_{trials:02d}.png')
    )
    
    with open('./opt_history.json', 'w', encoding='utf8') as f:
        json.dump(history, f)

    return -score


print('partitions (vertices_df)', vertices_df.rdd.getNumPartitions())
print('partitions (edge_list_df)', edge_list_df.rdd.getNumPartitions())
vertices_df = vertices_df.repartition(5)
edge_list_df = edge_list_df.repartition(5)
print('repartitioned (vertices_df)', vertices_df.rdd.getNumPartitions())
print('repartitioned (edge_list_df)', edge_list_df.rdd.getNumPartitions())

pbounds = {'C': (1, 20), 't': (0.1, 100)}

optimizer = BayesianOptimization(
    f=opt_f,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=10,
    n_iter=40,
)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print(optimizer.max)
