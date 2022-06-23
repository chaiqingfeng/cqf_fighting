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
    # 栈和队列
    class MyQueue:
        # 232.用栈实现队列
        # 看了GitHub上的自己写的
        def __init__(self):
            # in 执行push，out执行pop（out为空的时候将in中所有的数据导入）
            self.stack_in=[]
            self.stack_out=[]
        def push(self, x: int) -> None:
            self.stack_in.append(x)
        def pop(self) -> int:
            if self.empty():
                return None
            if self.stack_out:
                return self.stack_out.pop()
            else:
                for _ in range(len(self.stack_in)):
                    self.stack_out.append(self.stack_in.pop())
                return self.stack_out.pop()
        def peek(self) -> int:
            r=self.pop()
            self.stack_out.append(r) # 获取队列中的第一个元素，执行了pop，要再添加进去 ！！！？？？
            return r
        def empty(self) -> bool:
            return not (self.stack_in or self.stack_out)
    # Your MyQueue object will be instantiated and called as such:
    # obj = MyQueue()
    # obj.push(x)
    # param_2 = obj.pop()
    # param_3 = obj.peek()
    # param_4 = obj.empty()
    def removeDuplicates(self, nums: List[int]) -> int:
        ''' 
        26 删除有序数组中的重复项
        给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
        不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
        '''
        # 看了官方“另类双指针”的讲解后，自己写出代码，深深觉得自己是个SZ
        # 不在原来的list上删除，也不移动重复的数字，只是依次循环判断相邻的数字（因为排序后的，所以若重复必相邻）是否重复，重复则直接顺序通过慢指针覆盖原来的list前面遍历过的元素。
        fast,slow=1,1 # 初始值均为1
        n=len(nums)
        while fast<=n-1:
            if nums[fast]!=nums[fast-1]:
                nums[slow]=nums[fast]
                slow+=1
            fast+=1
        return slow
    def aa(self,s:str):
        '''
        1047. 删除字符串中的所有相邻重复项
        给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
        在 S 上反复执行重复项删除操作，直到无法继续删除。
        在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。
        示例：
        输入："abbaca"
        输出："ca"
        '''
        # 自己写的，结果用list存储，新的元素与栈顶元素不同则入栈，否则pop出栈顶元素
        # n=len(s)
        # if n<=1:
        #     return s
        # lis_r=[s[0]]
        # i=1
        # while i<n:
        #     if not lis_r or s[i]!=lis_r[-1]: # 当lis_r为空的时候要进行判断
        #         lis_r.append(s[i])
        #     else:
        #         lis_r.pop()
        #     i+=1
        # return ''.join(lis_r)

        # 参照以前的一种方法（26 删除有序数组中的重复项），使用快慢双指针法
        if len(s)<=1:
            return s
        fast=1 # 注意初始值设置,均为1
        slow=1
        lis_s=[s_sub for s_sub in s]
        while fast<len(s):

            if slow==0 or (slow>0 and lis_s[fast]!=lis_s[slow-1]): # 当slow变为0的时候需要进行判断
                lis_s[slow]=lis_s[fast]
                slow+=1
            else:
                slow-=1
            fast+=1
        return ''.join(lis_s[:slow])    
    def removeDuplicates2(self, nums: List[int]) -> int:
        '''
        80、删除有序数组中的重复项2
        给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。
        不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
        输入：nums = [0,0,1,1,1,1,2,3,3]
        输出：7, nums = [0,0,1,1,2,3,3]
        '''
        # 自己写的，总体思想类似于“26 删除有序数组中的重复项”， 不过额外定义一个count变量，记录当前数已保存的数量
        # j=1
        # count=1
        # for i in range(1,len(nums)):
        #     if nums[i]!=nums[j-1]:
        #         nums[j]=nums[i]
        #         j+=1
        #         count=1
        #     else:
        #         if count<2:
        #             nums[j]=nums[i]
        #             count+=1
        #             j+=1
                
        # return j, nums[:j]

        # 方法2,跟“26 删除有序数组中的重复项”基本一样，只不过是比较nums[i]!=nums[j-2]
        if len(nums)<=2:
            return nums
        j=2
        for i in range(2, len(nums)):
            if nums[i]!=nums[j-2]:
                nums[j]=nums[i]
                j+=1
        return j, nums[:j]
    def findRepeatNumber(self, nums: List[int]) -> int:
        '''
        剑指 Offer 03. 数组中重复的数字
        找出数组中重复的数字。
        在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
        '''
        # 方法1.自己写的，但是没有用到条件 ——“所有数字都在 0～n-1 的范围内”
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i]==nums[i-1]:
                return nums[i]
        return None
        
        # 方法2.使用哈希表(与方法1的内存消耗一样，但是用时7652ms，是方法1的127倍)
        # lis_has=[]
        # for ele in nums:
        #     if ele not in lis_has:
        #         lis_has.append(ele)
        #     else:
        #         return ele
        # return None
    def evalRPN(self, tokens: List[str]) -> int:
        '''
        150. 逆波兰表达式求值
        根据 逆波兰表示法，求表达式的值。
        有效的运算符包括 + ,  - ,  * ,  / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
        说明：
        整数除法只保留整数部分。 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
        '''    
        # 自己写的：使用栈，遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。
        if len(tokens)==1: # 考虑只有一个数字时，返回该值
            return int(tokens[0])
        stack_num=[]  
        for s in tokens:
            if s not in ['+','-','*','/']:
                stack_num.append(s)
            else:
                if s=='+':
                    stack_num.append(int(stack_num.pop(-2))+int(stack_num.pop()))
                elif s=='-':
                    stack_num.append(int(stack_num.pop(-2))-int(stack_num.pop()))
                elif s=='*':
                    stack_num.append(int(stack_num.pop(-2))*int(stack_num.pop()))
                else:
                    stack_num.append(int(stack_num.pop(-2))/int(stack_num.pop()))
        print(stack_num)
        return int(stack_num[-1]) # 返回值必须为整数
    def longestValidParentheses(self, s: str) -> int:
        '''
        32 最长的有效括号
        给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
        '''
        # 将有效括号替换为“0”，然后查找最长连续“0”，并返回其长度
        if not s:
            return 0
        stack_left=[]
        s=[s_sub for s_sub in s]
        # 将有效括号替换为“0”
        for i in range(len(s)):
            if s[i]=="(":
                stack_left.append(i) # 区别于“判断有效括号”将对应元素加入栈，该部分将左括号下标入栈
            else:
                if stack_left:
                    left_index=stack_left.pop()
                    s[left_index], s[i]='0','0'
        # 计算最长连续“0”的长度
        pass
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        '''
        239. 滑动窗口最大值
        给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
        返回滑动窗口中的最大值。 [1,3,-1,-3,5,3,6,7]
        '''
        # 使用堆/单调队列帮助获取窗口中的最大值。在循环元素之前就针对第一个窗口构建一个有序的列表（堆 / 单调队列），左边第一个对应的都是最大值，右边都是push的新值。
        # 使用下标判断是否在当前窗口：<=i-k

        # 自己的暴力写法超时，想办法更快的在当前窗口中找到最大值
        # lis_r=[]
        # start=0
        # end=start+k
        # while end <=len(nums):
        #     sub_nums=nums[start:end]
        #     lis_r.append(max(sub_nums))
        #     start+=1
        #     end+=1
        # return lis_r
        
        # 看官方解法1，使用堆（Python中是小顶堆），堆中存放值（取反后的）及其下标（下标用于判断是否在当前窗口（i-k）中）
        # import heapq
        # n = len(nums)
        # # 注意 Python 默认的优先队列是小根堆,所以要所有的数取反
        # q = [(-nums[i], i) for i in range(k)]
        # heapq.heapify(q)
        # lis_r=[-q[0][0]]
        # print("heapify后的q:",q)
        # for i in range(k,len(nums)):
        #     heapq.heappush(q,(-nums[i],i))
        #     print('push后的q:',q)
        #     while q[0][1]<=i-k: # 该循环是为了剔除堆中不在当前窗口中的最小值（实际是最大值的负数），那么剩下的堆中的第一个元素就是当前窗口中的最小值
        #         heapq.heappop(q)
        #     lis_r.append(-q[0][0])
        # return lis_r

        # 构建单调队列，帮助选取当前窗口中的最大值
        # 如下标i<j，且nums[i]<=nums[j]，则若nums[j]在窗口中，那么nums[i]一定不会窗口中的最大值了，所以可以将其在单调队列中彻底移除
        # q为单调队列，存放所有没有被移除的nums中元素的下标，下标是单调递增的，其对应的值是单调递减的
        import collections
        q=collections.deque()
        # 先用第一个窗口对q初始化，保证里面保存的值都是单调递减的，但是不保证里面的元素数等于k
        for i in range(k):
            while q and nums[i]>=nums[q[-1]]: # q中存放的下标对应的值是单调递减的，所以如果里面最后一个下标（肯定比i小，在q中其对应的值最小）对应的值比nums[i]还小的话，则可以将其完全移除
                q.pop()
            q.append(i)
        lis_r=[nums[q[0]]]
        for j in range(k,len(nums)):
            while q and nums[j]>=nums[q[-1]]:
                q.pop()
            q.append(j)
            while q[0]<=j-k: # 该循环剔除q中不在当前窗口中的下标，剩下的第一个元素就是当前窗口中最大值对应的下标，（因为是取每个窗口中最大的，不是数据集前k个最大的，所以每个窗口的时候弹出队列中最大的）
                q.popleft()
            lis_r.append(nums[q[0]])
        return lis_r
    def topKFrequent(self,nums:list,k):
        '''
        347.前 K 个高频元素
        给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
        示例 1:
        输入: nums = [1,1,1,2,2,3], k = 2
        输出: [1,2]
        '''
        # 自己写的，使用内置函数……
        # from collections import Counter
        # count=Counter(nums)
        # dic_count=dict(count.items())
        # dic_count=dict(sorted(dic_count.items(),key=lambda x:x[1],reverse=True))
        # return list(dic_count.keys())[:k]

        # 自己写的：使用“堆”，借鉴“239. 滑动窗口最大值”。
        # 统计完元素频率后，用前k个元素构建小根堆，在遍历第k之后的元素时，当遍历的元素大于堆里的第一个元素，则将第一个元素pop，将该元素push
        import heapq
        dic_count={}
        for num in nums:
            if num in dic_count:
                dic_count[num]+=1
            else:
                dic_count[num]=1
        lis_dicReverse=[(v,k) for k,v in dic_count.items()]
        lis=lis_dicReverse[:k]
        heapq.heapify(lis) # 不能直接将所有的转化为堆，然后获取后k个，因为堆结构不等同于排序
        print(lis)
        for item in lis_dicReverse[k:]:
            if item[0]>lis[0][0]:
                print(lis)
                heapq.heappop(lis)
                heapq.heappush(lis,item)
        return [num[1] for num in lis]

if __name__=='__main__':
    p=Solution()
    r=p.findRepeatNumber([2, 3, 1, 0, 2, 5, 3])
    print(r)