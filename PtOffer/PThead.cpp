#include "stdafx.h"
#include "PTHead.h"

bool Find(int target, vector<vector<int> > array)
{
	//从左下角开始查找，小于target上移，大于target右移
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
	//二分查找版本O(log n)~O(n)
	if (rotateArray.empty())
		return 0;
	int left = 0, mid = 0, right = rotateArray.size() - 1;
	while (rotateArray[left] >= rotateArray[right])
	{
		//第一个元素严格小于最后一个元素则说明没有发生旋转
		if (left + 1 == right)
		{
			mid = right;
			break;
		}
		mid = (left + right) / 2;
		if (rotateArray[mid] == rotateArray[left] && rotateArray[mid] == rotateArray[right])
		{
			//最特殊的情况：三个数完全相等,这个时候需要顺序查找第一个小于其的数并返回
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
