#include "stdafx.h"
#include "PointToOffer.h"

//---------------------------栈和队列-----------------------

//滑动窗口的最大值
vector<int> maxInWindows(const vector<int> &num, unsigned int size) {
  if (num.empty() || num.size() < size || size <= 0) 
		return {};
	vector<int> re;
	deque<int> store;
  for (int i = 0; i < num.size(); ++i) {
    while (!store.empty() && num[store.back()] <= num[i]) 
			store.pop_back();			//保证队列首元素是当前窗口最大值下标
    while (!store.empty() && i + 1 - store.front() > size)
			store.pop_front();		//保证首元素在当前窗口范围
    store.push_back(i);
    if (i + 1 >= size) 
			re.push_back(num[store.front()]);
  }
	return re;
}

//包含min函数的栈
void push(int value) {
	g_dataStack.push(value);
	if (g_minStack.empty() || g_minStack.top() >= value)
		g_minStack.push(value);
}
void pop() {
  if (g_dataStack.empty()) 
		throw "stack is empty!";
	if (g_dataStack.top() == g_minStack.top())
		g_minStack.pop();
  g_dataStack.pop();
}
int top() {
  if (g_dataStack.empty()) 
		throw "stack is empty!";
	return g_dataStack.top();
}
int min() {
  if (g_minStack.empty())
		throw "stack is empty!";
	return g_minStack.top(); 
}

//栈的压入、弹出序列
bool IsPopOrder(vector<int> pushV, vector<int> popV) {
	//使用栈进行模拟
	if (pushV.empty() || popV.size() != pushV.size())
		return false;
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