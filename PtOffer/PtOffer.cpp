// PtOffer.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gtest/gtest.h"
#include "PThead.h"

TEST(Find_case, Find_test)
{
	vector<vector<int>> array1 = {
		{1, 2, 3, 4},
		{2, 4, 6, 8},
		{3, 5, 7, 9},
		{4, 10, 11, 16},
		{7, 15, 18, 20}
	};
	vector<vector<int>> array2 = {};
	EXPECT_FALSE(Find(12, array1));
	EXPECT_TRUE(Find(20, array1));
	//EXPECT_TRUE(Find(1, array2));
	EXPECT_FALSE(Find(1, array2));
}

TEST(minNumberInRotateArray_case, minNumberInRotateArray_test)
{
	vector<int> rotateArray1 = {2, 3, 4, 1}, rotateArray2 = {4, 5, 1, 2, 3};
	EXPECT_EQ(1, minNumberInRotateArray(rotateArray1));
	EXPECT_EQ(1, minNumberInRotateArray(rotateArray2));
}

int main(int argc, char** argv)
{
	//testing::GTEST_FLAG(output) = "xml:";
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	//system("pause");
    return 0;
}

