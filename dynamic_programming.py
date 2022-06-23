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
    # 动态规划
    # 找的是全局最优解
    # 将原始问题分解为子问题，用一个一维或者二维数组保存子问题的解（状态转移矩阵），找出如何利用历史值推出新的元素值（状态转移方程）。
    # 具有“无后效性”：只关心前一个状态的值，不关心前一个状态的推导过程
    # 最主要的是找出状态转移矩阵
    def climbStairs(self, n):
        '''
        70 爬楼梯
        假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
        每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
        注意：给定 n 是一个正整数。
        示例 1：
        输入： 2
        输出： 2
        解释： 有两种方法可以爬到楼顶。
        1.  1 阶 + 1 阶
        2.  2 阶
        '''
        # 动态规划
        # 方法：看别人的自己写的，其实知道逻辑也就是转移方程（先考虑最后一步，可能是迈1步或者2步（在转移矩阵dp中其实对应的就是n-1位置和n-2位置），那么dp[n]=dp[n-1]+dp[n-2]，
        # 不过忘了动态规划都一般用到转移方程和转移矩阵，这个矩阵是一个一维数组，i上记录爬到第i个台阶的方法数）
        # dp=[0]*n
        # dp[0]=1
        # if n>1: 
        #     dp[1]=2
        #     for i in range(2,n):
        #         dp[i]=dp[i-1]+dp[i-2]
        # return dp[-1]
        
        # 方法2：（参考官方解题）其实最后值需要倒数第三、倒数第二级台阶来计算倒数第一个的，所以用两个数移动记录dp[i-1]和dp[i-2]
        r=1
        p=0
        q=0
        for _ in range(n):
            p=q
            q=r
            r=p+q
        return r
    def waysToStep(self, n: int) -> int:
        '''
        三步问题。有个小孩正在上楼梯，楼梯有n阶台阶，小孩一次可以上1阶、2阶或3阶。实现一种方法，计算小孩有多少种上楼梯的方式。结果可能很大，你需要对结果模1000000007。
        '''
        # 自己写的，跟“爬楼梯”问题一模一样
        if n<1:
            return 0
        elif n==1:
            return 1
        elif n==2:
            return 2
        elif n>2:
            dp=[0]*n
            dp[0]=1
            dp[1]=2
            dp[2]=4
            for i in range(3, n):
                dp[i]=dp[i-1]+dp[i-2]+dp[i-3]
            return dp[-1]% 1000000007
    def longestPalindrome(self, s: str) -> str:
        '''
        5给你一个字符串 s，找到 s 中最长的回文子串。
        示例 1：

        输入：s = "babad"
        输出："bab"
        解释："aba" 同样是符合题意的答案。
        '''
        #  juge[i][j] 表示字符串 s 的第 i 到 j 个字母组成的串（下文表示成 s[i:j]）是否为回文串
        # 一个回文串两边加上相同的字符则一定还是回文串。如果s[i]=s[j],则juge[i][j]=juge[i+1][j-1]
        if len(s)==0:
            return False
        elif len(s)==1:
            return True
        else:
            max_len=0
            s_r=''
            n_len=len(s)
            judge=[[False]*n_len for _ in range(n_len)]
            for i in range(n_len):
                judge[i][i]=True
            # 按照回文串所有可能的长度L进行遍历，注意，当长度为2或3且两边字母相同时，则必为回文串
            for L in range(1,n_len):
                for start in range(n_len-1):
                    end=start+L
                    if s[start]==s[end]:
                        if L==2 or L==3:
                            judge[start][end]=True
                        else:
                            judge[start][end]=judge[start-1][end-1]
                    if judge[start][end] and L>max_len:
                        s_r=s[start:end]
                        max_len=L
            return s_r
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''
        139、单词拆分
        给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
        注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
        
        输入: s = "applepenapple", wordDict = ["apple", "pen"]
        输出: true
        解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
        注意，你可以重复使用字典中的单词。
        '''
        # 看官方解题自己写的：动态规划
        # f[i] 表示字符串 s 前 i 个字符组成的字符串 s[0..i-1] 是否能被空格拆分成若干个字典中出现的单词，
        # s 的前 i 个字符是否能够由 wordDict 拼接而成，取决于 f[j] 以及 s[j:i]，因此转移方程为 f[i]=f[j] and s[j:i] in wordDict
        # 注意前len(s) + 1个字符才是指的整个s
        f=[False] * (len(s)+1)
        f[0]=True # 前0个字符指空字符，故f[0]为True
        for i in range(1, len(s)+1):
            for j in range(i):
                if f[j] and s[j:i] in wordDict:
                    f[i]=True
                    break # 只要在0~i之间有一个 j 满足拆分，就可以停止对 j 的循环
        print(f)
        return f[-1]
    def uniquePaths(self, m: int, n: int) -> int:
        '''
        62 不同路径
        一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
        问总共有多少条不同的路径？
        '''
        # 动态规划，看了官方讲解自己理解写的
        f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)] # 因为第一行都只能从左边走过来，第一列都只能从上面走过来，因此都为1
        for i in range(1,m):
            for j in range(1,n):
                f[i][j]=f[i-1][j]+f[i][j-1] # 动态转移方程
        return f[m-1][n-1]
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        64最小路径和
        给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
        说明：每次只能向下或者向右移动一步。
        '''
        # 只保证每走一步选择数值最小的单元格，结果不对……
        # sum_r=grid[0][0]
        # for i in range(len(grid[0])-2):
        #     for j in range(len(grid)-2):
        #         sum_r+=min(grid[i+1][j],grid[i],[j+1])

        # 动态规划  参考“不同路径”自己写的
        # 类似于“62 不同路径”，不过转移矩阵中不保存路径数，保存之前路径最小和
        # 定义DP数组
        f=[[0]*len(grid[0]) for _ in range(len(grid))]
        
        # DP数组边界状态初始化
        # 第一行和第一列的值与“不同路径”不同
        for i in range(len(grid[0])):
            f[0][i]=sum(grid[0][:i+1])
        for j in range(1,len(grid)):
            f[j][0]=f[j-1][0]+grid[j][0]
        print(f)
        
        # DP数组其他元素填充
        for i in range(1,len(grid)):
            for j in range(1,len(grid[0])):
                f[i][j]=min(f[i-1][j],f[i][j-1])+grid[i][j]
        return f[-1][-1]
    def maxSubArray(self, nums: List[int]) -> int:
        '''
        53最大子序和 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和
        输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
        输出：6
        解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
        '''
        # 我们用 f(i)代表以第 i个数结尾的「连续子数组的最大和」，那么很显然我们要求的答案就是：max{f(i)}
        # 因此我们只需要求出每个位置的 f(i)，然后返回 f数组中的最大值即可
        sum_max=nums[0] # 注意两者的初始值，r_end初始值必须是nums中的第一个元素
        cur=0
        for i in range(len(nums)):
            # 求出以i位置结尾的「连续子数组的最大和」f(i)
            # 如果前面的最大和加上我，比我自己还小，直接丢弃前面的，从我开始从新寻找
            cur=max(cur+nums[i], nums[i])
            # 对比前面得到的连续子数组最大和sum_max和f(i)的大小，获取最大的那个作为结果
            sum_max=max(cur, sum_max)
        return sum_max
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        给定一个含有 n 个正整数的数组和一个正整数 target 。
        找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，
        返回 0 。
        示例 1：
        输入：target = 7, nums = [2,3,1,2,4,3]
        输出：2
        解释：子数组 [4,3] 是该条件下的长度最小的子数组。
        '''
        # 双指针【滑动窗口（可变长度）】：尾指针end为当前遍历位置，头指针start初始位置是0，每当当前窗口中元素之和大于等于target时，
        # my_sum-头指针的数值，同时头指针向后移动
        # 要用到“长度最小”标准，所以需要使用双指针start-end
        len_min=len(nums)+1
        start=0
        for end in range(1,len(nums)+1):
            while sum(nums[start:end])>=target: # 必须有这个内层循环遍历[start：end]的子集.比如[1,2,4,5]这种，后面数字比前面大的多
                len_min=min(len_min, end-start)
                start+=1
        if len_min<=len(nums):
            return len_min
        else:
            return 0
    def generateMatrix(self, n: int) -> List[List[int]]:
        '''
        59 螺旋矩阵2
        给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
        '''
        pass
    def trap(self, height: List[int]) -> int:
        '''
        42.接雨水
        给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水.
        '''
        # 看的官方解题思路（第二次写），使用双指针法，接水取决于min(height[L], height[R])
        L,R=0,len(height)-1
        area=0
        L_max,R_max=height[L],height[R] # 初始值0,0也行
        while L<R:
            if height[L]<=height[R]:  # 不可以比较L_max>=R_max，因为取决于min(height[L], height[R])
                if height[L]<L_max:
                    area+=L_max-height[L]
                else:
                    L_max=height[L]
                L+=1
            else:
                if height[R]<R_max:
                    area+=R_max-height[R]
                else:
                    R_max=height[R]
                R-=1
        return area
    def minDistance(self, word1: str, word2: str) -> int:
        '''
        72、编辑距离
        给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
        '''
        # 最小路径和、不同路径 类似
        # 看的官方解题，使用动态规划
        n=len(word1)
        m=len(word2)
        if n*m==0:
            return n+m
        else:
            # 定义DP数组
            dp=[[0]*(m+1) for _ in range(n+1)]
            # DP边界状态初始化
            for i in range(n+1):
                dp[i][0]=i
            for j in range(m+1):
                dp[0][j]=j
            
            # DP数组其他元素填充
            for i in range(1,n+1):
                for j in range(1,m+1):
                    left=dp[i][j-1]+1 # dp[i][j-1]为word1的前i个字符变为word2的前j-1个字符需要的最少步数（编辑距离），对于word2的第j个字符，我们在word1的末尾添加了一个相同的字符，那么D[i][j]最小可以为D[i][j-1] + 1
                    down=dp[i-1][j]+1 # dp[i-1][j]，word1的前i-1个字符和word2的前j个字符之间的编辑距离(同理)  在word2末尾添加一个相同的字符
                    left_down=dp[i-1][j-1] # dp[i-1][j-1]，word1的前i-1个字符和word2的前j-1个字符之间的编辑距离(同理，若word的第i个字符，word2的第j个字符不同，则修改word1，相同则不变)  
                    if word1[i-1] != word2[j-1]:
                        left_down+=1
                    dp[i][j]=min(left, down, left_down)
            return dp[n][m]

class my:
    def __init__(self, name):
        self.name=name
    @classmethod
    def my_print(cls, name):
        cls.content=name
        print(cls.content)

    

if __name__ == "__main__":
    import time
    time_start=time.perf_counter()
    p=Solution()
    r=p.waysToStep(61)
    print(r)
    time.sleep(5)
    time_end=time.perf_counter()
    print(time_end-time_start)

    # import requests
    # from urllib import request, parse
    # from selenium import webdriver
    # from bs4 import BeautifulSoup
    # url='https://book.douban.com/'
    # headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3880.400 QQBrowser/10.8.4554.400 '}

    # import warnings
    # warnings.simplefilter('ignore',ResourceWarning)
    # chome_options = webdriver.ChromeOptions()
    # chome_options.add_argument('--headless') # 无头浏览器模式
    # chome_options.add_argument('--disable-gpu')
    # driver=webdriver.Chrome(chrome_options=chome_options)
    # driver.get(url)
    # content=driver.page_source
    # # request=requests.get(url, headers=headers)
    # # request.encoding = 'utf-8'
    # # content=request.text
    # soup=BeautifulSoup(content, 'html.parser')
    # res=soup.find_all('p')
    # print(res[0].text)