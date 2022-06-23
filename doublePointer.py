# likou双指针法题目
# 字符串、栈（队列）、哈希表中也包含双指针解法

from typing import List

class solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        给定一个含有 n 个正整数的数组和一个正整数 target 。
        找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
        示例 1：
        输入：target = 7, nums = [2,3,1,2,4,3]
        输出：2
        解释：子数组 [4,3] 是该条件下的长度最小的子数组。
        '''
        # 参考最长回文子串那道题的思路……超时
        # if sum(nums)<target:
        #     return 0
        # for Len in range(1,len(nums)+1):
        #     for i in range(len(nums)):
        #         j=i+Len
        #         if j>len(nums):
        #             break
        #         if sum(nums[i:j])>=target:
        #             return Len
        # return 0

        # 滑动窗口（可变长度）：尾指针end为当前遍历位置，头指针start初始位置是0，每当当前窗口中元素之和大于等于target时，my_sum-头指针的数值，同时头指针向后移动
        if sum(nums)<target:
            return 0
        start=0
        min_len=float("inf")
        for end in range(len(nums)):
            my_sum=sum(nums[start:end+1])
            while my_sum>=target: # 注意一定要是while循环！！！
                min_len=min(end-start+1,min_len)
                my_sum-=nums[start]
                start+=1
                print(start,end)
        return min_len
    
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''75、颜色分类
        给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，
        并按照红色、白色、蓝色顺序排列。
        我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
        必须在不使用库的sort函数的情况下解决这个问题。
        示例 1：
        输入：nums = [2,0,2,1,1,0]
        输出：[0,0,1,1,2,2]
        '''
        # 自己写的冒泡排序
        # n=len(nums)
        # for i in range(n):
        #     for j in range(i+1, n):
        #         if nums[i]>nums[j]:
        #             nums[i],nums[j]=nums[j],nums[i]
        # return nums

        # 单指针，看官方解题1，使用两个循环依次将0元素移动到列表最前面，然后将1元素移动到列表[pre:]最前面
        # n=len(nums)
        # pre=0
        # for i in range(n):
        #     if nums[i]==0:
        #         nums[pre],nums[i]=nums[i],nums[pre]
        #         pre+=1
        # for j in range(pre,n):
        #     if nums[j]==1:
        #         nums[pre], nums[j]=nums[j],nums[pre]
        #         pre+=1
        # return nums
        
        # 使用双指针：类似于单指针，使用p0、p1分别记录存放0、1的位置，注意当前数值是0时，p0、p1都要加1
        p0=0
        p1=0
        for i in range(len(nums)):
            if nums[i]==1:
                nums[p1], nums[i] = nums[i], nums[p1]
                p1+=1
            elif nums[i]==0:
                nums[p0], nums[i] = nums[i], nums[p0]
                
                '''因为连续的 0之后是连续的 1，因此如果我们将 0与num[p0] 进行交换，那么我们可能会把一个1交换出去。当 p0 < p1时，
                我们已经将一些1连续地放在头部，此时一定会把一个1交换出去.
                   因此这个时候要把交换到 i 处的 1 与num[p1]交换，将这个 1 放到连续 1 的尾部，并p1+=1'''
                if p0<p1:
                    nums[p1], nums[i]=nums[i], nums[p1]
                p0+=1
                p1+=1 # 注意，p1也要加1，防止交换1时把已经处理好的0覆盖了
        return nums

    def minWindow(self, s: str, t: str) -> str:
        '''
        76、最小覆盖子串
        给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
        注意：
        对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
        如果 s 中存在这样的子串，我们保证它是唯一的答案。
        输入：s = "ADOBECODEBANC", t = "ABC"
        输出："BANC"
        '''
        # 看官方解题后自己写的。使用双指针（ 类似于minSubArrayLen），使用哈希表帮助判断当前子序列是否覆盖t中所有字符
        def count(stri):
            dic_count={}
            for e in stri:
                if e in dic_count:
                    dic_count[e]+=1
                else:
                    dic_count[e]=1
            return dic_count
        def check(dic1,dic2):
            if not dic1: # 注意因为下面的循环遍历的是dic1，注意之前判断dic1是否为空，为空也是返回False
                return False
            for k in dic1:
                if k not in dic2 or dic1[k]>dic2[k]:
                    return False
            return True
        dic_t_count=count(t)
        print(dic_t_count)
        min_len=len(s)
        min_s=''
        left=0
        for right in range(len(s)):
            dic_s_count=count(s[left: right+1])
            while check(dic_t_count, dic_s_count) and left <= right: # 循环条件注意加入 left <= right
                if min_len >= right+1-left: # 注意条件是 >=
                    min_s=s[left: right+1]
                    min_len=right+1-left
                left+=1
                dic_s_count=count(s[left: right+1])
        return min_s

# 查找、排序算法
class Solution1:
    # 基本算法（自己）：如查找、排序
    # 查找
    def search_2tree(self, nums, target):
        # 二分查找
        if not nums:
            return -1
        L,R=0,len(nums)-1
        while L<=R:
            i=(L+R)//2
            if nums[i]==target:
                return i
            elif nums[i]>target:
                R=i-1
            else:
                L=i+1
        return -1
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        '''
        74、搜索二维矩阵
        编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

        每行中的整数从左到右按升序排列。
        每行的第一个整数大于前一行的最后一个整数。
        '''
        # 看官方题解写的
        # 对矩阵的第一列的元素二分查找，找到 最后一个 不大于 目标值的元素，然后在该元素所在行中二分查找目标值是否存在。
        def searchFirstCol(col):
            low=-1 # 注意low初始值为-1
            high=len(col)-1
            while low<high: # 注意条件不是 <=
                mid=(high-low+1)//2+low # 注意，因为low初始值为-1
                if target>=col[mid]:
                    low=mid
                else:
                    high=mid-1
            return low # 注意返回的是low，不是mid
        # 典型的二分查找
        def searchTargetRow(row):
            low=0
            high=len(row)-1
            while low<=high: 
                mid=(high-low)//2+low
                if target==row[mid]:
                    return True
                elif target>row[mid]:
                    low=mid+1
                else:
                    high=mid-1
            return False
       
        targetRow_index=searchFirstCol([matrix[i][0] for i in range(len(matrix))])
        if targetRow_index<0:
            return False
        return searchTargetRow(matrix[targetRow_index])
    # 排序：冒泡、快排、堆。。。
    def maopao_sort(self,lis:List):
        # 冒泡排序，结果是大->小
        n=len(lis)
        for i in range(n):
            for j in range(i+1, n):
                if lis[i]<lis[j]:
                    lis[i],lis[j]=lis[j],lis[i]
        print(lis)
        return lis
    def quick_sort(self, lis:list):
        # 有问题！！！！！！！
        def quickSort_BP(start, end):
            if start>end:
                return
            mid=lis[start]
            i=start
            j=end
            while i<j:
                # 在参照数mid左边找一个比自己大的，右边找一个比自己小的，进行交换
                while lis[i]<=mid and i<j: # 必须包含等号，否则有相同数字致无限循环
                    i+=1
                while lis[j]>=mid and j>i:
                    j-=1
                if i<j:
                    lis[i],lis[j]=lis[j],lis[i]
            
            lis[start]=lis[i]
            lis[i]=mid
            quickSort_BP(start, i-1)
            quickSort_BP(i+1, end)
            
        quickSort_BP(0, len(lis)-1)
        print(lis)

if __name__=='__main__':
    p=solution()
    r=p.sortColors([1,2,0])
    print(r)