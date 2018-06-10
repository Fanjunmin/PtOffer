#include "stdafx.h"
#include "PTHead.h"

bool Find(int target, vector<vector<int> > array)
{
	//�����½ǿ�ʼ���ң�С��target���ƣ�����target����
	int rowLen = array.size();
	int colLen = array.empty() ? 0 : array[0].size();
	int i = rowLen - 1, j = 0;
	while (i >= 0 && j < colLen)
	{
		if (array[i][j] == target)
			return true;
		else if (array[i][j] < target)
			++j;
		else
			--i;
	}
	return false;
}

int minNumberInRotateArray(vector<int> rotateArray)
{
	//���ֲ��Ұ汾O(log n)~O(n)
	if (rotateArray.empty())
		return 0;
	int left = 0, mid = 0, right = rotateArray.size() - 1;
	while (rotateArray[left] >= rotateArray[right])
	{
		//��һ��Ԫ���ϸ�С�����һ��Ԫ����˵��û�з�����ת
		if (left + 1 == right)
		{
			mid = right;
			break;
		}
		mid = (left + right) / 2;
		if (rotateArray[mid] == rotateArray[left] && rotateArray[mid] == rotateArray[right])
		{
			//��������������������ȫ���,���ʱ����Ҫ˳����ҵ�һ��С�������������
			for (int i = left + 1; i < right; ++i)
			{
				if (rotateArray[i] < rotateArray[mid])
				{
					return rotateArray[i];
				}
			}
		}
		if (rotateArray[mid] >= rotateArray[left])
		{
			left = mid;
		}
		else if (rotateArray[mid] <= rotateArray[right])
		{
			right = mid;
		}
	}
	return rotateArray[mid];
}
