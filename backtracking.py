# coding:utf-8
# 一些经典题目的GitHub参考https://github.com/youngyangyang04/leetcode-master
from collections import deque
from heapq import heappop
from json import dump
import re
from typing import List, Optional
import pymongo
from sshtunnel import SSHTunnelForwarder
import atexit
import pandas as pd

from pandas.core.reshape.reshape import stack


class Solution:
    # 回溯
    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击(任何两个皇后都不能处于同一条横行、纵行或斜线上。)。
        给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
        每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

        输入：n = 4
        输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
        解释：如上图所示，4 皇后问题存在两个不同的解法。
        """
        def change_result_lis(lis_one):
            lis_need=[]
            n=len(lis_one)
            for index in lis_one:
                str_need='.'*index+'Q'+'.'*(n-index-1)
                lis_need.append(str_need)
            return lis_need

        lis_results=[] 
        def dfs(lis_queenIncolumn_index, raw, columns, diagonal1, diagonal2):
            '''
            args:
                lis_queenIncolumn_index:list, 存放一种情况，元素为每一行放置皇后的列索引，每种情况的每一行肯定只有一个皇后
                raw:int，行数，以行为单位回溯
                columns:list，存放已经放置皇后的列
                diagonal1:list，存放已经放置皇后的正对角线
                diagonal2:list，存放已经放置皇后的反对角线
            '''
            if raw == n:
                return lis_results.append(change_result_lis(lis_queenIncolumn_index))
            for i in range(n):
                if i in columns or raw-i in diagonal1 or raw+i in diagonal2:
                    continue
                dfs(lis_queenIncolumn_index+[i], raw+1, columns+[i], diagonal1+[raw-i], diagonal2+[raw+i])
            
        dfs([],0, [],[],[])
        return lis_results

    ################################################################################################################
    # 贪心算法（感觉不成体系，就是一种牵强的感觉理解）
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        '''
        56.合并区间
        以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，
        该数组需恰好覆盖输入中的所有区间。
        输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
        输出：[[1,6],[8,10],[15,18]]
        解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
        '''
        #（所谓的贪心算法）看GitHub后自己写的。
        # 根据左边界排序，当 原数组intervals[i][0]<=结果数组lis_r[-1][1]时存在交叉，则替换lis_r[-1]，否则直接将intervals[i]添加到结果数组lis_r中
        if len(intervals)<=1:
            return intervals
        intervals=sorted(intervals, key=lambda x:x[0])
        lis_r=[intervals[0]]
        for i in range(1,len(intervals)):
            last=lis_r[-1]
            if intervals[i][0]<=last[1]:
                lis_r[-1]=([last[0],max(last[1],intervals[i][1])])
            else:
                lis_r.append(intervals[i])
        return lis_r
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        '''
        57 插入区间
        给你一个 无重叠的 ，按照区间起始端点排序的区间列表。
        在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

        示例 1:
        输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
        输出：[[1,5],[6,9]]
        '''
        # 结合merge自己写的，先按照左边界顺序插入到原列表intervals中，然后如果执行插入了的话，按照merge的方法合并区间，
        # 否则判断是与最后一个区间合并还是直接插入到最后
        if not intervals and newInterval: # 两个列表为空的情况要单独判断
            intervals.append(newInterval)
            return intervals
        elif not intervals and not newInterval:
            return intervals
        
        has_insert=False
        for i in range(len(intervals)):
            if newInterval[0]<=intervals[i][0]: # 按照左边界的顺序插入新区间
                intervals.insert(i, newInterval)
                has_insert=True
                break
        
        print(intervals)
        if has_insert: # 如果执行了插入，则按照merge的方法合并区间
            lis_r=[intervals[0]]
            for i in range(1, len(intervals)):
                last=lis_r[-1]
                if intervals[i][0]<=last[1]:
                    lis_r[-1]=[last[0],max(last[1],intervals[i][1])]
                else:
                    lis_r.append(intervals[i])
            return lis_r

        else: # 没有执行插入则与最后一个区间判断是合并区间还是直接插入到最后
            if newInterval[0]<=intervals[-1][1]:
                intervals[-1]=[intervals[-1][0],max(intervals[i][1],newInterval[1])]
            else:
                intervals.append(newInterval)
        return intervals
    
    def merge2(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        '''
        88、合并两个有序数组
        给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
        请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。
        注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
        '''
        # 自己写的，参考 string_my.py 里replaceSpace()，将空格替换为"%20"
        # 使用三个指针，i指向nums1的遍历，t指向nums2的遍历，j指向nums1扩充后的遍历，比较nums1[i]、nums2[t]大小，把大的放在nums1[j]处
        i=m-1
        j=m+n-1
        t=n-1
        while i>=0 and t>=0:
            if nums1[i]>=nums2[t]:
                nums1[j]=nums1[i]
                i-=1
            else:
                nums1[j]=nums2[t]
                t-=1
            j-=1
        if t>=0:
            nums1[:t+1]=nums2[:t+1]
        return nums1

    def jump(self, nums: List[int]) -> int:
        '''
        45 跳跃游戏2
        给你一个非负整数数组 nums ，你最初位于数组的第一个位置。
        数组中的每个元素代表你在该位置可以跳跃的最大长度。
        你的目标是使用最少的跳跃次数到达数组的最后一个位置。
        假设你总是可以到达数组的最后一个位置。
        '''
        # 典型的贪心算法
        # 该跳完成后（start==end），在 该跳区间内找到能到达的最远位置 作为下一跳的终点（end）
        if not nums:
            return 0
        steps=0
        max_pos=0
        start,end=0,0
        while start<len(nums)-1:
            max_pos=max(max_pos, start+nums[start]) # max_pos，选择该跳区间内最远位置，一定得是start+nums[start]
            if start==end:
                end=max_pos
                steps+=1
            start+=1
        return steps

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        '''47、全排列
            给定一个可以包含重复数字的序列nums，按任意顺序返回所有不重复的全排列
        '''
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path.copy())
                return
            for i in range(size):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                        continue
                    used[i] = True
                    path.append(nums[i])
                    dfs(nums, size, depth + 1, path, used, res)
                    used[i] = False
                    path.pop()

        size = len(nums)
        if size == 0:
            return []

        nums.sort()

        used = [False] * len(nums)
        res = []
        dfs(nums, size, 0, [], used, res)
        return res
    
    def combine(self, n: int, k: int) -> List[List[int]]:
        '''
        77组合
        给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。你可以按 任何顺序 返回答案。
        示例 1：
        输入：n = 4, k = 2
        输出：
        [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        '''
        # 回溯：树的深度优先遍历
        # 与全排列非常类似，每次在[i，n+1]中选一个数添加到path中，下一层再在[i+1, n+1]中选一个数，一层层递归下去，直到path长度等于k
        # 再开始下一个分支的深度遍历
        res=[]
        path=[]
        def dfs(starIndex):
            if len(path)==k:
                res.append(path.copy()) 
                return
            for i in range(starIndex, n+1):
                path.append(i)
                dfs(i+1)
                path.pop()
        dfs(1)
        return res
    def subsets(self, nums: List[int]) -> List[List[int]]:
        '''
        78、子集
        给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
        解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
        示例 1：

        输入：nums = [1,2,3]
        输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
        '''
        # 自己写的，做完“77 组合”后，直接在其基础上，外面使用循环，赋予 回溯函数 不同的长度k
        res=[]
        path=[]
        def dfs(k, startIndex):
            if len(path)==k:
                res.append(path.copy())
                return
            for i in range(startIndex, len(nums)):
                path.append(nums[i])
                dfs(k, i+1)
                path.pop()
        for k in range(len(nums)+1):
            dfs(k, 0)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        '''
        79、单词搜索
        给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
        单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
        '''
        # 定义一个循环，使board中的每个元素作为回溯的第一个字符，
        # 回溯中判断以网格的 (i, j)位置出发，能否搜索到单词 word[k:]，k的初始值是0，每次回溯将 (i, j)位置上/下/左/右移动，k+1
        direction=[(0,1),(0,-1),(1,0),(-1,0)]# 获取当前位置(i, j)的前后， 下上的位置
        def dfs(i, j, k):  # 判断以网格的 (i, j)位置出发，能否搜索到单词 word[k:]，k的初始值是0。对每个当前位置的字符(i, j)回溯判断是否能找到满足条件的结果
            if board[i][j]!=word[k]:
                return False
            if k==len(word)-1:
                return True
            vitvied.add((i,j))
            result=False
            for di,dj in direction:
                newi=i+di
                newj=j+dj
                if 0<=newi<len(board) and 0<=newj<len(board[0]):
                    if (newi, newj) not in vitvied:
                        if dfs(newi, newj, k+1):
                            result=True
                            break
            vitvied.remove((i,j)) # 判断完当前位置的结果后，将vitvied集合清空
            return result
        vitvied=set() # 在每个回溯中用于标识每个位置是否被访问过
        
        for i in range(len(board)):# 循环board中的每个元素，作为回溯的第一个字符
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False
        
if __name__ == "__main__":
    p=Solution()
    r=p.merge2([1], 1, [], 0)
    print(r)