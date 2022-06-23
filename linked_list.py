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

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # 链表
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        '''
        203 移除链表元素
        给你一个链表的头节点 head 和一个整数 val ，请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
        '''
        if not head:
            return head
        dummy_node=ListNode(0,head)
        node=dummy_node # node和dummy_node的关系
        while node:
            if node.next and node.next.val==val:
                node.next=node.next.next
            else: # 必须有这个else，只有node的一下个节点不满足条件，不删除的时候，node才移动
                node=node.next
        return dummy_node.next
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        '''
        83、删除排序链表中的重复元素
        给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 。
        '''
        # 看评论写的，当node.next.val!=node.val时，node直接移动到她的下一个节点，
        # 相等时则将node的下一个节点略过这个重复节点，指向下一个的下一个节点
        if not head:
            return head
        node=head #因为最后返回要返回结果链表的头节点，所以head不能动，因此对node进行移动操作
        while node:
            if node.next.val!=node.val:
                node=node.next
            else:
                node.next=node.next.next
                
        return head

    def reverseList(self, head: ListNode) -> ListNode:
        # 206 链表反转
        # 定义一个pre_node指向前一个节点(初始的时候指向None)，改变每两个节点间的箭头方向，并每次循环使pre_node和head节点不断完后移动
        pre_node=None 
        while head:
            temp=head.next
            
            head.next=pre_node

            pre_node=head
            head=temp
        return pre_node
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        '''
        92、翻转链表2
        给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。
        请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
        '''
        # 难难难，晕
        def reverseList( head: ListNode) -> ListNode:
            # 206 链表反转
            # 定义一个pre_node指向前一个节点(初始的时候指向None)，改变每两个节点间的箭头方向，并每次循环使pre_node和head节点不断完后移动
            pre_node=None 
            while head:
                temp=head.next
                
                head.next=pre_node

                pre_node=head
                head=temp
        # 记录3个节点：反转之前的节点pre_node，之后的节点succ_node，开始的节点left_node，结束的节点right_node
        # 然后截取反转部分的链表，进行反转
        # 最后把反转的链表接入到原链表中
        ''' # 方法1，看官方解题写的
        dummy=ListNode(0,head)
        pre_node=dummy
        
        for _ in range(left-1): # 找到并记录前驱节点pre_node
            pre_node=pre_node.next
        
        right_node=pre_node # 找到并记录反转部分的结束节点right_node
        for _ in range(right-left+1):
            right_node=right_node.next
        
        left_node=pre_node.next # 前驱节点的下一个节点就是反转部分的开始节点 left_node
        succ_node=right_node.next # 反转节点的下一个节点就是后继节点 succ_node

        pre_node.next=None # 切断原链表，截取需反转部分的链表
        right_node.next=None
        
        reverseList(left_node) # 反转链表
        
        pre_node.next=right_node # 将截取出的链表接入原链表
        left_node.next=succ_node
        return dummy.next'''
    
        # 方法2：自己写的，与方法1的不同之处是，通过一个循环获得pre_node、left_node、right_node，
        # （succ_node使用=right_node.next获得，防止链表长度过界）
        if left==right:
            return head
        dummy=ListNode(0,head)
        node=dummy
        i=0
        while node:
            if i==left-1:
                pre_node=node
            if i==left:
                left_node=node
            if i==right:
                right_node=node
                break
            i+=1
            node=node.next
        succ_node=right_node.next

        pre_node.next=None
        right_node.next=None
        
        reverseList(left_node)
        
        pre_node.next=right_node
        left_node.next=succ_node
        return dummy.next
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        61.旋转链表
        给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
        输入：head = [1,2,3,4,5], k = 2
        输出：[4,5,1,2,3]
        '''
        # 看了别人的题解自己写的：使用双指针，fast和slow间隔k，移动双指针直到fast指向最后一个节点，执行新头结点指向slow.next，断开slow与slow.next之间连接，fast.next指向原始头结点
        # 当head为空或者只有一个节点时，返回head
        if not head or not head.next:
            return head
        
        # 统计链表中节点的数量
        n=0
        node=head
        while node:
            node=node.next
            n+=1
        
        # 考虑到k为节点数量的整数倍时，实际链表无需旋转
        k=k%n
        if not k:
            return head
        
        # 使用两个指针，fast指向k+1个节点，slow指向头结点
        fast, slow =head, head
        while k:
            fast=fast.next
            k-=1
        
        # 同时移动fast、slow指针，fast和slow之间间隔始终为k
        while fast.next:
            fast=fast.next
            slow=slow.next
        
        # 将slow.next作为新的头结点，断开slow与slow.next之间的连接，将fast.next指向原始链表的头结点，最后返回新头结点
        newHead=slow.next
        slow.next=None
        fast.next=head
        return newHead
    def swapPairs(self, head: ListNode) -> ListNode:
        '''
        24两两交换链表节点：
        给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
        你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
        '''
        # 因为第一个节点也要进行交换，所以要设置一个哑结点
        dummy=ListNode(0,head)
        node=dummy
        while node.next and node.next.next:
            node1=node.next
            node2=node.next.next
            
            node1.next=node2.next
            node2.next=node1
            node.next=node2

            node=node1
        return dummy.next
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode: 
        '''
        19删除第n个节点
        给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
        进阶：你能尝试使用一趟扫描实现吗？
        '''
        # 第二次刷，使用双指针分别指向当前节点的前一个节点pre_node和当前节点cur_node，遍历到第n个节点时，pre_node.next=cur_node.next
        if n<0:
            return head
        count=0
        node=head
        while node:
            count+=1
        n=count-n
        dummy=ListNode(0,head)
        pre_node=None
        cur_node=dummy
        for _ in range(n+1):
            pre_node=cur_node
            cur_node=cur_node.next
        pre_node.next=cur_node.next
        
        # 第一次刷题的做法 
        # j=0
        # while cur_node:
        #     if j==n:
        #         cur_node.next=cur_node.next.next
        #     cur_node=cur_node.next
        #     j+=1
        return dummy.next
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        '''
        ****
        106 链表相交
        给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 null 。
        '''
        # 双指针法，本题我有发表总结思路
        # 根据快慢法则，走的快的一定会追上走得慢的。在这道题里，有的链表短，他走完了就去走另一条链表，我们可以理解为走的快的指针。
        # 存在相交的时候，当nodeA=nodeB时返回结果是有值的，若不存在相交，连着相等时必定是最后一个节点指向的None
        if not headA or not headB:
            return None
        nodeA=headA
        nodeB=headB
        while nodeA!=nodeB:
            nodeA=nodeA.next if nodeA else headB
            nodeB=nodeB.next if nodeB else headA
        return nodeB
    def detectCycle(self, head: ListNode) -> ListNode:
        '''
        142 环形链表2
        给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
        为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 
        如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
        '''
        # 快慢指针法：fast每次移动两个节点，slow每次移动一个节点
        # 发现环：有环的话，slow进入环内的时候其实就变成fast追slow(一快一慢，肯定能追上)，fast、slow必定在环内相遇
        # 发现环入口：(公式推导得出)从头结点出发一个指针index1，从相遇节点也出发一个指针index2，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是环形入口的节点。
        if not head:
            return None
        fast=head
        slow=head
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:
                index1=head
                index2=fast

                # （自己写的）如果使用下面的循环方式，在进入循环之前，index1、index2的初始值也要进行判断（因为没有哑结点）
                # if index1==index2:
                #     return index1
                # while index1:
                #     index1=index1.next
                #     index2=index2.next
                #     if index1==index2:
                #         return index1
                
                # 参考GitHub答案
                while index1!=index2:
                    index1=index1.next
                    index2=index2.next
                return index1
        return None