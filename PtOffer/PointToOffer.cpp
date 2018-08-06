#include "stdafx.h"
#include "PointToOffer.h"

//---------------------------树--------------------------------

//重建二叉树
TreeNode *reConstructBinaryTree(vector<int> pre, vector<int> vin) ;

//树的子结构
bool HasSubtree(TreeNode *pRoot1, TreeNode *pRoot2);
bool isSubtree(TreeNode *pRoot1, TreeNode *pRoot2);

//二叉树的镜像
void Mirror(TreeNode *pRoot) {
	if (pRoot && (pRoot->left || pRoot->right)) {
    std::swap(pRoot->left, pRoot->right);
		Mirror(pRoot->left);
    Mirror(pRoot->right);
	}
	return;
}

//从上往下打印二叉树
vector<int> PrintFromTopToBottom(TreeNode *root) {
	queue<TreeNode *> Q;
	vector<int> store;
	Q.push(root);
	while (!Q.empty()) {
		TreeNode *curr_node = Q.front();
		Q.pop();
		if (curr_node) {
			store.push_back(curr_node->val);
			Q.push(curr_node->left);
			Q.push(curr_node->right);
		}
	}
	return store;
}

//二叉搜索树的后序遍历序列
bool judgeBST(vector<int> sequence);
bool VerifySquenceOfBST(vector<int> sequence);

//二叉树中和为某一值的路径
void FindPath(vector<vector<int>> &vec_store, 
							vector<int> store,
							TreeNode *root,
              int expNumber) {
	store.push_back(root->val);
	if (!root->left && !root->right) {
		if (expNumber == root->val) vec_store.push_back(store);
		return;
	}
	if (root->left) FindPath(vec_store, store, root->left, expNumber - root->val);
	if (root->right) FindPath(vec_store, store, root->right, expNumber - root->val);
	store.pop_back();		//回溯
}
vector<vector<int>> FindPath(TreeNode *root, int expectNumber) {
  vector<vector<int>> vec_store;
	vector<int> store;
	if(root) FindPath(vec_store, store, root, expectNumber);
	return vec_store;
}

//二叉搜索树与双向链表
void recurConvert(TreeNode *root, TreeNode *&pre) {
  if (!root) return;
  recurConvert(root->left, pre);
  root->left = pre;
  if (pre) pre->right = root;
  pre = root;
  recurConvert(root->right, pre);
}
TreeNode *Convert(TreeNode *pRootOfTree) {
  if (!pRootOfTree) return pRootOfTree;
  TreeNode *pre = 0;
  recurConvert(pRootOfTree, pre);
  TreeNode *res = pRootOfTree;
  while (res->left) res = res->left;
  return res;
}

//二叉树深度
int TreeDepth(TreeNode *pRoot) {
	//递归版本
  return pRoot ? 1 + max(TreeDepth(pRoot->left), 
												 TreeDepth(pRoot->right)) : 0;
}
int TreeDepth2(TreeNode *pRoot) {
	//迭代版本
  queue<TreeNode *> Q;
	int depth = 0;
  Q.push(pRoot);
	while (!Q.empty()) {
    int len = Q.size();
		++depth;
		while (len--) {
			TreeNode *curr_node = Q.front();
			Q.pop();
			if (curr_node) {
				Q.push(curr_node->left);
				Q.push(curr_node->right);
			}
		}
	}
	return depth - 1; //将叶节点的空孩子节点也算作一层了，所以减1
}

//平衡二叉树
bool IsBalanced(TreeNode *pRoot, int &pDepth) {
	if (!pRoot) {
		pDepth = 0;
		return true;
	}
	int left_depth, right_depth;	//记录左右子树的高度
	if (IsBalanced(pRoot->left, left_depth) && 
			IsBalanced(pRoot->right, right_depth)) {
		int diff = left_depth - right_depth;
		if (diff <= 1 && diff >= -1) {
			pDepth = 1 + (left_depth > right_depth ? left_depth : right_depth);
			return true;
		}
	}
	return false;
}
bool IsBalanced_Solution(TreeNode *pRoot) {
	int pDepth = 0;
	return IsBalanced(pRoot, pDepth);
}

//二叉树的下一个结点
TreeLinkNode *GetNext(TreeLinkNode *pNode) {
	if (!pNode) return pNode;
	if (pNode->right) {		//有右孩子的情况
		pNode = pNode->right;
		while (pNode->left) {
			pNode =  pNode->left;
		}
		return pNode;
	} 
	else {		//没有右孩子的情况
		TreeLinkNode *parent = pNode->next;
		if (!parent) {
			return parent;
		} 
		else {
			if (pNode == parent->left) {	//该节点是其父节点的左孩子
				return parent;
			} 
			else {	//该节点是其父节点的右孩子，沿左侧链上升
				while (parent->next && parent == parent->next->right) {
					parent = parent->next;
				}
				return parent->next;
			}
		}
	}
}

//对称的二叉树
bool isSymmetrical(TreeNode *leftChild, TreeNode *rightChild) {
	if (!leftChild && !rightChild) {
		//左右子树同时为空
		return true;
	}
	else if (leftChild && rightChild) {
		//左右子树都不为空
		return leftChild->val == rightChild->val &&
					 isSymmetrical(leftChild->left, rightChild->right) &&
					 isSymmetrical(leftChild->right, rightChild->left);
	}
	else {
		return false;
	}
}
bool isSymmetrical(TreeNode *pRoot) {
	if (!pRoot) return true; 
	return isSymmetrical(pRoot->left, pRoot->right);
}

//把二叉树打印成多行
vector<vector<int>> Print(TreeNode *pRoot) {
	vector<vector<int>> store;
	queue<TreeNode *> Q;
	Q.push(pRoot);
	int index = 0;
	while (!Q.empty()) {
		int length = Q.size();
		store.push_back(vector<int>());
		while (length--) {
			TreeNode *curr_node = Q.front();
			Q.pop();
			if (curr_node) {
				store[index].push_back(curr_node->val);
				Q.push(curr_node->left);
				Q.push(curr_node->right);
			}
		}
		++index;
	}
	store.pop_back();
	return store;
}

//按之字形顺序打印二叉树
vector<vector<int>> Print(TreeNode *pRoot);

//二叉搜索树的第k个结点
void KthNode(TreeNode *pRoot, vector<TreeNode *> &TN);
TreeNode *KthNode(TreeNode *pRoot, int k);

//序列化二叉树
//采用先序遍历，用vector存储
void Serialize(vector<int> &sto, TreeNode *root);
char *Serialize(TreeNode *root);
TreeNode *Deserialize(int *&str);
TreeNode *Deserialize(char *str);


//---------------------------回溯-------------------------------

//矩阵中的路径
bool hasPath(char *matrix, int rows, int cols, char *str);
bool hasPath(char *matrix, int rows, int cols, char *str, bool *flag, int x,
             int y, int index);

//机器人的运动范围
bool IsVaildVal(int x, int y, int threshold) {
  int result = 0;
  while (x) {
    result += x % 10;
    x /= 10;
  }
  while (y) {
    result += y % 10;
    y /= 10;
  }
  return result <= threshold;
}
int Find(vector<int> &id, int p) {
  if (p == -1 || id[p] == -1) return -1;
  int temp = p;
  while (p != id[p]) p = id[p];
  while (p != id[temp]) {
    int q = id[temp];
    id[temp] = p;
    temp = q;
  }
  return p;
}
void Union(vector<int> &id, vector<int> &sz, int p, int q) {
  int p_root = Find(id, p);
  int q_root = Find(id, q);
  if (p_root == q_root) return;
  if (sz[p_root] < sz[q_root]) {
    sz[q_root] += sz[p_root];
    id[p_root] = id[q_root];
  } else {
    sz[p_root] += sz[q_root];
    id[q_root] = id[p_root];
	}
	return;
}
int movingCount(int threshold, int rows, int cols) {
	if(rows <= 0 || cols <= 0) return 0;
  vector<int> id(rows * cols), sz(rows * cols, 1);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      //将不满足坐标和的id值设为-1，否则设为其下标
      id[i * cols + j] = IsVaildVal(i, j, threshold) ? i * cols + j : -1;
    }
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
			int pos = i * cols + j;
      if (id[pos] != -1 && i + 1 < rows && id[pos + cols] != -1)
				Union(id, sz, pos, pos + cols);
      if (id[pos] != -1 && j + 1 < cols && id[pos + 1] != -1)
				Union(id, sz, pos, pos + 1);
		}
	}
	int count = 0;
	int base_id = Find(id, 0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int pos = i * cols + j;
      if (id[pos] != -1 && Find(id, pos) == base_id)
				++count;
		}
	}
	return count;
}

//---------------------------栈和队列-----------------------

//滑动窗口的最大值
vector<int> maxInWindows(const vector<int> &num, unsigned int size) {
  if (num.empty() || num.size() < size || size <= 0) return {};
  vector<int> re;
  deque<int> store;
  for (int i = 0; i < num.size(); ++i) {
    while (!store.empty() && num[store.back()] <= num[i])
      store.pop_back();  //保证队列首元素是当前窗口最大值下标
    while (!store.empty() && i + 1 - store.front() > size)
      store.pop_front();  //保证首元素在当前窗口范围
    store.push_back(i);
    if (i + 1 >= size) re.push_back(num[store.front()]);
  }
  return re;
}

//包含min函数的栈
void push(int value) {
  g_dataStack.push(value);
  if (g_minStack.empty() || g_minStack.top() >= value) g_minStack.push(value);
}
void pop() {
  if (g_dataStack.empty()) throw "stack is empty!";
  if (g_dataStack.top() == g_minStack.top()) g_minStack.pop();
  g_dataStack.pop();
}
int top() {
  if (g_dataStack.empty()) throw "stack is empty!";
  return g_dataStack.top();
}
int min() {
  if (g_minStack.empty()) throw "stack is empty!";
  return g_minStack.top();
}

//栈的压入、弹出序列
bool IsPopOrder(vector<int> pushV, vector<int> popV) {
  //使用栈进行模拟
  if (pushV.empty() || popV.size() != pushV.size()) return false;
  stack<int> simulate_stack;
  for (int i = 0, j = 0; i < pushV.size(); ++i) {
    simulate_stack.push(pushV[i]);
    while (!simulate_stack.empty() && popV[j] == simulate_stack.top()) {
      ++j;
      simulate_stack.pop();
    }
  }
  return simulate_stack.empty();
}

//---------------------------链表---------------------------

//从尾到头打印链表
vector<int> printListFromTailToHead(ListNode *head) {
  //非递归版本，使用栈
  vector<int> store_vec;
  stack<int> store_stack;
  while (head) {
    store_stack.push(head->val);
    head = head->next;
  }
  store_vec.reserve(store_stack.size());
  while (!store_stack.empty()) {
    store_vec.push_back(store_stack.top());
    store_stack.pop();
  }
  return store_vec;
}

//链表中倒数第k个结点
ListNode *FindKthToTail(ListNode *pListHead, unsigned int k) {
  //使用两个相距k的节点，当第二个节点到达尾部时，第一个节点即为倒数第k个结点
  ListNode *first = pListHead, *second = pListHead;
  while (k--) {
    if (!second) break;
    second = second->next;
  }
  if (!k) return nullptr;
  while (second) {
    first = first->next;
    second = second->next;
  }
  return first;
}

//反转链表
ListNode *ReverseList(ListNode *pHead) {
  //递归版本
  if (!pHead || !pHead->next) return pHead;
  ListNode *reverseHead = ReverseList(pHead->next);
  //反转过程
  pHead->next->next = pHead;
  pHead->next = nullptr;
  return reverseHead;
}
ListNode *ReverseList2(ListNode *pHead) {
  //迭代版本
  if (!pHead || !pHead->next) return pHead;
  ListNode *currNode = pHead;    //当前指针
  ListNode *reverseHead = NULL;  //新链表的头指针
  ListNode *preNode = NULL;      //当前指针的前一个结点

  while (currNode) {  //当前结点不为空时才执行
    ListNode *nextNode =
        currNode->next;  //链断开之前一定要保存断开位置后边的结点
    if (nextNode == NULL)  //当Next为空时，说明当前结点为尾节点
      reverseHead = currNode;
    currNode->next = preNode;  //指针反转
    preNode = currNode;
    currNode = nextNode;
  }
  return reverseHead;
}

//合并两个排序的链表
ListNode *Merge(ListNode *pHead1, ListNode *pHead2) {
  ListNode *result = new ListNode(0);
  ListNode *temp = result;
  while (pHead1 && pHead2) {
    if (pHead1->val < pHead2->val) {
      temp->next = pHead1;
      pHead1 = pHead1->next;
    } else {
      temp->next = pHead2;
      pHead2 = pHead2->next;
    }
    temp = temp->next;
  }
  temp->next = pHead1 ? pHead1 : pHead2;
  return result->next;
}

//复杂链表的复制
RandomListNode *Clone(RandomListNode *pHead) {
  if (!pHead) return pHead;
  RandomListNode *cloneNode = pHead;
  //复制每个节点到被复制节点的next
  while (cloneNode) {
    RandomListNode *copyNode = new RandomListNode(cloneNode->label);
    //将copyNode插入cloneNode之后
    copyNode->next = cloneNode->next;
    cloneNode->next = copyNode;
    cloneNode = copyNode->next;
  }
  cloneNode = pHead;
  while (cloneNode) {
    if (cloneNode->random)
      // random节点的复制
      //注意：random节点作为RandomListNode，在上一个while中也复制了
      cloneNode->next->random = cloneNode->random->next;
    cloneNode = cloneNode->next->next;
  }
  //进行拆分
  cloneNode = pHead->next;
  while (pHead->next) {
    RandomListNode *temp = pHead->next;
    pHead->next = temp->next;
    pHead = temp;
  }
  return cloneNode;
}

//两个链表的第一个公共结点
ListNode *FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2) {
  ListNode *p1 = pHead1, *p2 = pHead2;
  //两个链表跑一趟后互换首节点再跑一次，必然会到达相交节点
  while (p1 != p2) {
    p1 = p1 ? p1->next : pHead2;
    p2 = p2 ? p2->next : pHead1;
  }
  return p1;
}

//链表中环的入口结点
ListNode *EntryNodeOfLoop(ListNode *pHead) {
  //快慢双指针
  if (!pHead) return pHead;
  ListNode *fast_node = pHead, *slow_node = pHead;
  while (fast_node && slow_node) {
    slow_node = slow_node->next;
    if (fast_node->next) {
      fast_node = fast_node->next->next;
    } else {
      return nullptr;  //这说明无环，有环必然next不会为空
    }
    if (fast_node == slow_node) break;
  }
  fast_node = pHead;
  while (fast_node != slow_node) {
    fast_node = fast_node->next;
    slow_node = slow_node->next;
  }
  return fast_node;
}

//删除链表中重复的结点
ListNode *deleteDuplication(ListNode *pHead) {
  if (!pHead || !pHead->next) return pHead;
  ListNode *current;
  if (pHead->next && pHead->next->val == pHead->val) {
    current = pHead->next->next;
    while (current && current->val == pHead->val) {
      current = current->next;
    }
    return deleteDuplication(current);
  } else {
    current = pHead->next;
    pHead->next = deleteDuplication(current);
    return pHead;
  }
}

//---------------------------数学---------------------------

//斐波那契数列
int Fibonacci(int n) {
  if (n < 0) throw "n is negetive";
  if (n == 0) return 0;
  int front = 0, back = 1;  //记录斐波那契数列的f(n)和f(n+1)
  while (--n) {
    back += front;
    front = back - front;
  }
  return back;
}

//数值的整数次方
double Power(double base, int exponent) {
  //位运算
  if (base == 0 && exponent == 0) throw "undefine 0^0==1";
  if (base == 0) return 0;
  if (exponent == 0) return 1;
  if (exponent < 0) return Power(1 / base, -exponent);
  double re = 1;
  while (exponent) {
    if (exponent & 1) re *= base;
    base *= base;
    exponent >>= 1;
  }
  return re;
}

//孩子们的游戏(圆圈中最后剩下的数)
int LastRemaining_Solution(int n, int m) {
  //递归问题f(n,m)
  // f(n,m) = (f(n-1, m) + m) % n
  // f(1,m) = 0
  if (n <= 0 || m <= 0)
    throw "input is non-positive";  //按照牛客网的要求不应该抛异常，应该输出-1
  int re = 0;
  for (int i = 2; i <= n; ++i) {
    re = (re + m) % i;
  }
  return re;
}