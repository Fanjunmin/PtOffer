// PtOffer.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "gtest/gtest.h"
#include "PThead.h"

TEST(Find_case, Find_test)
{
	vector<vector<int>> array = {
		{1, 2, 3, 4},
		{2, 4, 6, 8},
		{3, 5, 7, 9},
		{4, 10, 11, 16},
		{7, 15, 18, 20}
	};
	vector<vector<int>> array2 = {};
	EXPECT_FALSE(Find(12, array));
	EXPECT_TRUE(Find(20, array));
	//EXPECT_TRUE(Find(1, array2));
	EXPECT_FALSE(Find(1, array2));
}

TEST(minNumberInRotateArray_case, minNumberInRotateArray_test)
{
	vector<int> rotateArray = { 2, 3, 4, 1 };
	EXPECT_EQ(1, minNumberInRotateArray(rotateArray));
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
    return 0;
}

