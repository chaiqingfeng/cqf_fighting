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
    # 哈希表(python中指字典??元组、列表)
    def isAnagram(self, s: str, t: str) -> bool:
        '''
        242 有效的字母异位词
        给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
        示例 1: 输入: s = "anagram", t = "nagaram" 输出: true
        示例 2: 输入: s = "rat", t = "car" 输出: false
        '''
        # 自己的偷懒法
        # if not s or not t or len(s)!=len(t):
        #     return False
        # if sorted(s)==sorted(t):
        #     return True
        # else:
        #     return False

        # 使用哈希表
        if not s or not t or len(s)!=len(t):
            return False
        dic_count={}
        for s_sub in s:
            if s_sub in dic_count:
                dic_count[s_sub]+=1
            else:
                dic_count[s_sub]=1
        for t_sub in t:
            if t_sub not in dic_count:
                return False
            else:
                dic_count[t_sub]-=1
        for ct in dic_count.values():
            if ct!=0:
                return False
        return True
    def commonChars(self, words: List[str]) -> List[str]:
        '''
        给定仅有小写字母组成的字符串数组 A，返回列表中的每个字符串中都显示的全部字符（包括重复字符）组成的列表。例如，如果一个字符在每个字符串中出现 3 次，
        但不是 4 次，则需要在最终答案中包含该字符 3 次。

        你可以按任意顺序返回答案。
        输入：["bella","label","roller"]   ["cool","lock","cook"]
        输出：["e","l","l"]
        '''
        # 自己写的：遍历第一个字符串中的字符，并记录出现在每个字符串中的字符w及其在所有字符串中出现的最小次数min_w_count，基于(w, min_w_count)生成最终结果。
        lis_r_end=[]
        lis_r=[]
        len_words=len(words)
        for w in words[0]:
            j=0
            min_w_count=words[0].count(w)
            for i in range(len_words):
                if w in words[i]:
                    j+=1
                    min_w_count=min(words[i].count(w),min_w_count)
            if j==len_words and (w,min_w_count) not in lis_r:
                lis_r.append((w,min_w_count))
        print(lis_r)
        for w,c_min in lis_r:
            lis_r_end.extend([w]*c_min)
        return lis_r_end
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        '''
        给定两个数组，编写一个函数来计算它们的交集。
        示例 1：
        输入：nums1 = [1,2,2,1], nums2 = [2,2]
        输出：[2]
        '''
        # 自己写的1：用集合偷懒
        # return list(set(nums1)&set(nums2))

        # 自己写的2
        lis_r=[]
        for n1 in nums1:
            if n1 in nums2 and n1 not in lis_r:
                lis_r.append(n1)
        return lis_r
    def ishappyNum(self, n: int):
        '''
        快乐数
        编写一个算法来判断一个数 n 是不是快乐数。
        「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
        如果 可以变为  1，那么这个数就是快乐数。
        如果 n 是快乐数就返回 True ；不是，则返回 False 。
        示例：
        输入：19
        输出：true
        解释：
        1^2 + 9^2 = 82
        8^2 + 2^2 = 68
        6^2 + 8^2 = 100
        1^2 + 0^2 + 0^2 = 1
        '''
        # 自己写的：陷入无限循环是在“平方和”出现重复的情况（参考别人的）
        def generateNumList(num):
            lis_int=[]
            for num_s in str(num):
                lis_int.append(int(num_s))
            return lis_int
        def generateNumList1(num): # 利用数学法
            lis_int=[]
            len_num=len(str(num))
            for i in range(len_num-1,-1,-1):
                singer=num//10**i
                lis_int.append(singer)
                num=num-singer*10**i
            return lis_int
        sum_r=0
        lis_duplicate=[]
        lis_intN=generateNumList1(n)
        while sum_r!=1:
            sum_r=0
            for int_n in lis_intN:
                sum_r+=int_n**2
            if sum_r in lis_duplicate:
                print(lis_duplicate)
                return False
            lis_intN=generateNumList1(sum_r)
            lis_duplicate.append(sum_r)
        return True
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        '''
        1 两数之和
        给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
        你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
        示例:
        给定 nums = [2, 7, 11, 15], target = 9
        因为 nums[0] + nums[1] = 2 + 7 = 9
        所以返回 [0, 1]
        '''
        # 第三次自己写的：“数组中同一个元素不能使用两遍”，使用字典对已经遍历过的进行记录
        dic={}
        for i in range(len(nums)):
            n=nums[i]
            if target-n in dic:
                return [i, dic[target-n]]
            else:
                dic[n]=i
        return []
    def ThreeSum(self, nums: List[int]):
        '''
        15 三数之和
        给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
        注意： 答案中不可以包含重复的三元组。
        示例：
        给定数组 nums = [-1, 0, 1, 2, -1, -4]，
        满足要求的三元组集合为： [ [-1, 0, 1], [-1, -1, 2] ]
        '''
        # 第3次刷，自己写的（双指针法）：“不重复的三元组”，排序后相同的数一定相邻
        if len(nums)<3:
            return []
        lis_r=[]
        nums.sort()
        for i in range(len(nums)-2):
            if i>0 and nums[i]==nums[i-1]:
                continue
            a=nums[i]
            j=i+1
            k=len(nums)-1
            while j<k:
                b=nums[j]
                c=nums[k]
                if j>i+1 and b==nums[j-1]:
                    j+=1
                    continue
                if k<len(nums)-1 and c==nums[k]:
                    k-=1
                    continue
                if a+b+c==0:
                    lis_r.append([a,b,c])
                    j+=1
                    k-=1
                elif a+b+c>0:
                    k-=1
                else:
                    j+=1
        return lis_r
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        '''
        18 四数之和
        题意：给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等,
        找出所有满足条件且不重复的四元组。
        注意：
        答案中不可以包含重复的四元组。
        示例： 给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。 满足要求的四元组集合为： [ [-1, 0, 0, 1], [-2, -1, 1, 2], [-2, 0, 0, 2] ]
        '''
        # 第二次刷，自己写的（双指针法），跟“三数之和”一样
        if len(nums)<4:
            return []
        lis_r=[]
        nums.sort()
        for i in range(len(nums)-3):
            a=nums[i]
            if i>0 and a==nums[i-1]:
                continue
            for j in range(i+1, len(nums)-2):
                b=nums[j]
                if j>i+1 and b==nums[j-1]:
                    continue
                m=j+1
                k=len(nums)-1
                while m<k:
                    c=nums[m]
                    d=nums[k]
                    if m>j+1 and c==nums[m-1]:
                        m+=1
                        continue
                    if k<len(nums)-1 and d==nums[k+1]:
                        k-=1 
                        continue
                    if a+b+c+d==target:
                        lis_r.append([a,b,c,d])
                        m+=1
                        k-=1
                    elif a+b+c+d>target:
                        k-=1
                    else:
                        m+=1
        return lis_r
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        '''
        给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，使得 A[i] + B[j] + C[k] + D[l] = 0。
        为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。所有整数的范围在 -2^28 到 2^28 - 1 之间，最终结果不会超过 2^31 - 1 。
        例如:
        输入: A = [ 1, 2] B = [-2,-1] C = [-1, 2] D = [ 0, 2] 输出: 2 解释: 两个元组如下:
        (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
        (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
        '''
        # 自己写的：双指针法 (不对)
        # r=0
        # nums3.sort()
        # nums4.sort()
        # for a in nums1:
        #     for b in nums2:
        #         i=0
        #         j=len(nums4)-1
        #         while i<len(nums3) and j>=0:
        #             c=nums3[i]
        #             d=nums4[j]
        #             if a+b+c+d==0:
        #                 r+=1
        #                 i+=1
        #                 j-=1
        #             elif a+b+c+d>0:
        #                 j-=1
        #             else:
        #                 i+=1
        # return r
        # 参考GitHub的思路自己写的，有点类似于“两数之和”（两个两层循环效率高于一个四层循环）
        count_r=0
        dic_12_sumCount={}
        for a in nums1:
            for b in nums2:
                if a+b in dic_12_sumCount:
                    dic_12_sumCount[a+b]+=1
                else:
                    dic_12_sumCount[a+b]=1
        for c in nums3:
            for d in nums4:
                if 0-(c+d) in dic_12_sumCount:
                    count_r+=dic_12_sumCount[0-(c+d)]
        return count_r
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        '''
        383 赎金信
        给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串 ransom 能不能由第二个字符串 magazines 里面的字符构成。如果可以构成，返回 true ；否则返回 false。
        (题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。杂志字符串中的每个字符只能在赎金信字符串中使用一次。)
        注意：
        你可以假设两个字符串均只含有小写字母。
        canConstruct("a", "b") -> false
        canConstruct("aa", "ab") -> false
        canConstruct("aa", "aab") -> true
        '''
        # 自己写的，思路类似于 242 有效的字母异位词
        dic_ransomCount={}
        for s in ransomNote:
            if s in dic_ransomCount:
                dic_ransomCount[s]+=1
            else:
                dic_ransomCount[s]=1
        for s in magazine:
            if s in dic_ransomCount:
                dic_ransomCount[s]-=1
        for v in dic_ransomCount.values():
            if v>0:
                return False
        return True

    def singleNumber(self, nums: List[int]) -> int:
        '''
        136、只出现一次的数字
        给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
        说明：
        你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
        '''
        nums.sort()
        
        # 方法1：自己写的，借助哈希表，不存在则append，存在则pop
        # temp=[]
        # for i in range(len(nums)):
        #     if nums[i] not in temp:
        #         temp.append(nums[i])
        #     else:
        #         temp.pop()
        # return temp[0]
        
        # 方法2：遍历索引1~-2的元素，不等于其前面数且不等于其后面数，则返回该元素；特殊情况：只有一个元素则直接返回该元素，第一个为只出现一次的元素/最后一个为只出现一次的元素分别单拎出来比较判断
        if len(nums)==1:
            return nums[0]
        if nums[0]!=nums[1]:
            return nums[0]
        if nums[-1]!=nums[-2]:
            return nums[-1]
        else:
            for i in range(1, len(nums)-1):
                if nums[i]!=nums[i-1] and nums[i]!=nums[i+1]:
                    return nums[i]