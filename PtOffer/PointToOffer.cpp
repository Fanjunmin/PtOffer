#include "stdafx.h"
#include "PointToOffer.h"

//---------------------------ջ�Ͷ���-----------------------

//�������ڵ����ֵ
vector<int> maxInWindows(const vector<int> &num, unsigned int size) {
  if (num.empty() || num.size() < size || size <= 0) 
		return {};
	vector<int> re;
	deque<int> store;
  for (int i = 0; i < num.size(); ++i) {
    while (!store.empty() && num[store.back()] <= num[i]) 
			store.pop_back();			//��֤������Ԫ���ǵ�ǰ�������ֵ�±�
    while (!store.empty() && i + 1 - store.front() > size)
			store.pop_front();		//��֤��Ԫ���ڵ�ǰ���ڷ�Χ
    store.push_back(i);
    if (i + 1 >= size) 
			re.push_back(num[store.front()]);
  }
	return re;
}

//����min������ջ
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

//ջ��ѹ�롢��������
bool IsPopOrder(vector<int> pushV, vector<int> popV) {
	//ʹ��ջ����ģ��
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

//---------------------------����---------------------------

//��β��ͷ��ӡ����
vector<int> printListFromTailToHead(ListNode *head) {
  //�ǵݹ�汾��ʹ��ջ
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

//�����е�����k�����
ListNode *FindKthToTail(ListNode *pListHead, unsigned int k) {
  //ʹ���������k�Ľڵ㣬���ڶ����ڵ㵽��β��ʱ����һ���ڵ㼴Ϊ������k�����
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

//��ת����
ListNode *ReverseList(ListNode *pHead) {
  //�ݹ�汾
  if (!pHead || !pHead->next) return pHead;
  ListNode *reverseHead = ReverseList(pHead->next);
  //��ת����
  pHead->next->next = pHead;
  pHead->next = nullptr;
  return reverseHead;
}
ListNode *ReverseList2(ListNode *pHead) {
  //�����汾
  if (!pHead || !pHead->next) return pHead;
  ListNode *currNode = pHead;    //��ǰָ��
  ListNode *reverseHead = NULL;  //�������ͷָ��
  ListNode *preNode = NULL;      //��ǰָ���ǰһ�����

  while (currNode) {  //��ǰ��㲻Ϊ��ʱ��ִ��
    ListNode *nextNode =
        currNode->next;  //���Ͽ�֮ǰһ��Ҫ����Ͽ�λ�ú�ߵĽ��
    if (nextNode == NULL)  //��NextΪ��ʱ��˵����ǰ���Ϊβ�ڵ�
      reverseHead = currNode;
    currNode->next = preNode;  //ָ�뷴ת
    preNode = currNode;
    currNode = nextNode;
  }
  return reverseHead;
}

//�ϲ��������������
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

//��������ĸ���
RandomListNode *Clone(RandomListNode *pHead) {
  if (!pHead) return pHead;
  RandomListNode *cloneNode = pHead;
  //����ÿ���ڵ㵽�����ƽڵ��next
  while (cloneNode) {
    RandomListNode *copyNode = new RandomListNode(cloneNode->label);
    //��copyNode����cloneNode֮��
    copyNode->next = cloneNode->next;
    cloneNode->next = copyNode;
    cloneNode = copyNode->next;
  }
  cloneNode = pHead;
  while (cloneNode) {
    if (cloneNode->random)
      // random�ڵ�ĸ���
      //ע�⣺random�ڵ���ΪRandomListNode������һ��while��Ҳ������
      cloneNode->next->random = cloneNode->random->next;
    cloneNode = cloneNode->next->next;
  }
  //���в��
  cloneNode = pHead->next;
  while (pHead->next) {
    RandomListNode *temp = pHead->next;
    pHead->next = temp->next;
    pHead = temp;
  }
  return cloneNode;
}

//��������ĵ�һ���������
ListNode *FindFirstCommonNode(ListNode *pHead1, ListNode *pHead2) {
  ListNode *p1 = pHead1, *p2 = pHead2;
  //����������һ�˺󻥻��׽ڵ�����һ�Σ���Ȼ�ᵽ���ཻ�ڵ�
  while (p1 != p2) {
    p1 = p1 ? p1->next : pHead2;
    p2 = p2 ? p2->next : pHead1;
  }
  return p1;
}

//�����л�����ڽ��
ListNode *EntryNodeOfLoop(ListNode *pHead) {
  //����˫ָ��
  if (!pHead) return pHead;
  ListNode *fast_node = pHead, *slow_node = pHead;
  while (fast_node && slow_node) {
    slow_node = slow_node->next;
    if (fast_node->next) {
      fast_node = fast_node->next->next;
    } else {
      return nullptr;  //��˵���޻����л���Ȼnext����Ϊ��
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

//ɾ���������ظ��Ľ��
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

//---------------------------��ѧ---------------------------

//쳲���������
int Fibonacci(int n) {
  if (n < 0) throw "n is negetive";
  if (n == 0) return 0;
  int front = 0, back = 1;  //��¼쳲��������е�f(n)��f(n+1)
  while (--n) {
    back += front;
    front = back - front;
  }
  return back;
}

//��ֵ�������η�
double Power(double base, int exponent) {
  //λ����
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

//�����ǵ���Ϸ(ԲȦ�����ʣ�µ���)
int LastRemaining_Solution(int n, int m) {
  //�ݹ�����f(n,m)
  // f(n,m) = (f(n-1, m) + m) % n
  // f(1,m) = 0
  if (n <= 0 || m <= 0)
    throw "input is non-positive";  //����ţ������Ҫ��Ӧ�����쳣��Ӧ�����-1
  int re = 0;
  for (int i = 2; i <= n; ++i) {
    re = (re + m) % i;
  }
  return re;
}