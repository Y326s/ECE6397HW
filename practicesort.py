# Practice sort

class Solution:
    def quicksort(self, nums: list[int], p1, p2) -> None:
        '''
        25
        [22, 3, 22, -10, 2, 25, 100, 70, 45]
        '''

        if p1 < p2:
            pi = self.placepivot(nums, p1, p2)
            self.quicksort(nums, p1, pi-1)
            self.quicksort(nums, pi+1, p2)


    def placepivot(self, nums: list[int], p1, p2) -> int:

        pivot = nums[p1]

        while p1 < p2:
            while ((pivot <= nums[p2]) & (p1 < p2)):
                p2 -= 1
            nums[p1] = nums[p2]

            while ((pivot >= nums[p1]) & (p1 < p2)):
                p1 += 1
            nums[p2] = nums[p1]
        nums[p1] = pivot

        return p1


if __name__ == "__main__":
    test1 = [25, 3, 22, 7, 2, -10, 100, 2, 22, 45]
    test2 = [0, 1, 2, 3, 4, 5]
    test3 = [1]

    sol = Solution()
    print(sol.quicksort(test1, 0 ,len(test1)-1))
    print(sol.quicksort(test2, 0 ,len(test2)-1))
    print(sol.quicksort(test3, 0 ,len(test3)-1))
    print(test1)
    print(test2)
    print(test3)







            
