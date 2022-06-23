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
    # 字符串
    def reverseString(self, s: List[str]) -> None:
        '''
        344.反转字符串
        编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
        不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
        你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
        示例 1：
        输入：["h","e","l","l","o"]
        输出：["o","l","l","e","h"]
        '''
        # 偷懒
        # return s[::-1]

        # 自己实现列表反转
        # lis_r=[]
        # for i in range(len(s)-1,-1,-1):
        #     lis_r.append(s[i])
        # return lis_r

        # 首尾两两交换,不占额外空间
        i=0
        j=len(s)-1
        while i<j:
            s[i],s[j]=s[j],s[i]
            i+=1
            j-=1
        return s
    def reverseStr(self, s: str, k: int) -> str:
        '''
        541. 反转字符串II
        给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。
        如果剩余字符少于 k 个，则将剩余字符全部反转。
        如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
        示例:
        输入: s = "abcdefg", k = 2
        输出: "bacdfeg"
        '''
        # 自己写的：滑动窗口，第奇数个窗口逆转，偶数个窗口保持不变，
        # 注意：最后不满足窗口大小的那部分需要判断前一个窗口是否为偶数，是则对最后一部分进行逆转
        num_k=len(s)//k
        lis_s=[ss for ss in s]
        start=0
        end_no=k
        i=1
        while i <= num_k:
            if i%2!=0:
                lis_s[start:end_no]=lis_s[start:end_no][::-1]
            start=start+k
            end_no=end_no+k
            i+=1
        # 
        if i%2!=0:
            lis_s[start:end_no]=lis_s[start:end_no][::-1]
        return ''.join(lis_s)
    def replaceSpace(self, s: str) -> str:
        '''
        请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
        示例 1： 输入：s = "We are happy."
        输出："We%20are%20happy."
        '''
        # 自己写的，返回一个新对象
        # s_new=''
        # for ss in s:
        #     if ss == ' ':
        #         s_new+='%20'
        #     else:
        #         s_new+=ss
        # return s_new
        
        # 参考GitHub自己写的，先扩充s替换后的大小，然后基于双指针法 “从后向前” 对s对应的list进行替换
        # （i指向s原始长度，j指向扩展后的长度）
        i=len(s)-1
        num_space=s.count(' ')
        s+=' '*(num_space*2)
        lis_s=list(s)
        j=len(lis_s)-1
        while i>=0:
            if lis_s[i]!=' ':
                lis_s[j]=lis_s[i]
                i-=1
                j-=1
            else:
                lis_s[j-2:j+1]='%20'
                i-=1
                j-=3
        return ''.join(lis_s)
    def reverseLeftWords(self, s: str, n: int) -> str:
        '''
        左旋转字符串
        字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，
        该函数将返回左旋转两位得到的结果"cdefgab"。
        示例 1：
        输入: s = "abcdefg", k = 2
        输出: "cdefgab"
        '''
        # 偷懒
        # return s[n:]+s[:n]

        # GitHub上的方法是： 先对前n个字符进行逆转，然后对n之后的所有字符逆转，最后对整个字符串逆转
        
        # 在原来字符串基础上进行
        for i in range(n):
            s+=s[i]
            
        return s[n:]
    def reverseWords(self, s: str) -> str:
        '''
        151.翻转字符串里的单词
        给你一个字符串 s ，逐个翻转字符串中的所有 单词 。
        单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
        请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。
        说明：
        输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
        翻转后单词间应当仅用一个空格分隔。
        翻转后的字符串中不应包含额外的空格。
        示例 1：
        输入：s = "the sky is blue"
        输出："blue is sky the"
        '''
        # 自己写的：借助list，移除多余空格，
        # 注意：else里面的判断；最后一个s_sub不为空的时候也要记得添加进去
        lis_str=[]
        s_sub=''
        for ss in s:
            if ss !=' ':
                s_sub+=ss
            else:
                if s_sub:
                    lis_str.append(s_sub)
                    s_sub=''
        if s_sub:
            lis_str.append(s_sub)

        return ' '.join(lis_str[::-1])
    def strStr(self,haystack, needle):
        '''
        28 实现strStr()
        实现 strStr() 函数。
        给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
        示例 1: 输入: haystack = "hello", needle = "ll" 输出: 2
        示例 2: 输入: haystack = "aaaaa", needle = "bba" 输出: -1
        说明: 当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。 对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。
        '''
        # 自己写的，第二次做
        if not needle:
            return 0
        start=0
        end=len(needle)
        while end<=len(haystack):
            if haystack[start:end]==needle:
                return start
            start+=1
            end+=1
        return -1
    def duplicateStr(self,s):
        '''
        459.重复的子字符串
        给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。
        示例 1:
        输入: "abab"
        输出: True
        解释: 可由子字符串 "ab" 重复两次构成。
        '''
        # 第二次写，自己写的（从长度的角度遍历）
        len_s=len(s)
        maxlen=len_s//2
        for L in range(1,maxlen+1):
            if len_s%L ==0:
                n=len_s//L
                if s[:L]*n==s:
                    return True
        return False

        # 第一次写的（从元素的角度遍历，元素逐渐增多）
        # s_sub=s[0]
        # s_len=len(s)
        # for s_s in s[1:]:
        #     n=s_len//len(s_sub)
        #     if s_sub*n==s:
        #         return True
        #     s_sub+=s_s

        # return False
    
    def unDumplicate_string(self,s):
        '''求给定字符串中最长不重复子串'''
        # 使用滑动窗口没成功……
        # start=0
        # max_lenstr=0
        # for end in range(1,len(s)):
        #     if s[end] in s[start:end+1]:
        #         start =s.index(s[end])+1 # start应该跳到s[:end+1]中与end位置元素相同的最后一个元素下一个位置
        #     max_lenstr=max(max_lenstr,end-start+1)
        #     print(start)
        #     print(max_lenstr)

        # (自己写的)开辟一个新空间，使用哈希表
        lis_r=[]
        max_len=0
        for ele in s:
            if ele in lis_r:
                lis_r=lis_r[lis_r.index(ele)+1:]
            lis_r.append(ele) # 注意:ele存在lis_r也要把ele放入里面，别忘了
            max_len=max(len(lis_r),max_len)
        return max_len

    def Dumplicate_string(self,s):
        '''求字符串最长重复子串'''
        # (自己写的)start end 滑动窗口
        start=0
        max_lenstr=''
        for end in range(1,len(s)):
            # 后一个元素与前一个元素不相同时，移动头索引到尾索引处，后再判断当前子串长度end+1-start与当前最长子串长度len(max_lenstr)大小关系，大于的话则替换为当前子串
            if s[end]!=s[end-1]:
                start=end
            if end+1-start>len(max_lenstr): # 注意:end+1-start
                max_lenstr=s[start:end+1]
        print(max_lenstr)

if __name__=='__main__':
    # p=Solution()
    # result=p.unDumplicate_string("aaaabujos")
    # print(result)
    import os
    file_path=os.path.dirname(__file__)
    print(file_path)

    print(os.path.abspath(os.path.curdir))