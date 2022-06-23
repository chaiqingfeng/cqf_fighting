# Definition for a binary tree node.
from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        '''
        92、二叉树的中序遍历
        给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
        （二叉树遍历：先序遍历、中序遍历、后序遍历，即指父节点被访问的顺序。）
        root = [1,null,2,3] -> [1,3,2]
        '''
        # 自己写的（有看过官方解题）
        # 使用递归方法，递归函数中先遍历node左子树，然后将node添加到结果中，然后遍历node右子树。
        res=[]
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            res.append(node.val)
            inorder(node.right)
        
        inorder(root)
        return res
    
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        '''
        100、相同的树
        给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。
        如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
        [1, 1], [1, null, 1]
        '''
        # 判断中序遍历结果，不对！！
        # res1=[]
        # res2=[]
        # def inorder(node, res):
        #     if not node:
        #         res.append('null')
        #         return 
        #     inorder(node.left, res)
        #     res.append(node.val)
        #     inorder(node.right, res)
        
        # inorder(p, res1)
        # inorder(q, res2)
        # if res1==res2:
        #     return True
        # else:
        #     return False

        # 深度优先
        if not p and not q:
            return True
        if not p or not q:
            return False
        elif p.val!=q.val:
            return False
        else:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
            
