import string
from functools import reduce
from itertools import chain
from lib2to3.pgen2.tokenize import String

import numpy
import numpy as np
import collections


# def namelist(names):
#     if len(names) > 1:
#         return '{} & {}'.format(', '.join(name['name'] for name in names[:-1]),
#                                 names[-1]['name'])
#     elif names:
#         return names[0]['name']
#     else:
#         return ''


# def checkModulo(x):
#     if(x % 3 == 0 and x %5 == 0):
#         return x
#     elif(x % 3 == 0):
#         return x
#     elif (x % 5 == 0):
#         return x
#     else:
#         return 0


class counterClass:
    def __init__(self, number):
        self.number = number
        self.counter = 1

    def increment_counter(self):
        self.counter = self.counter + 1

    number = 0
    counter = 0


def find_it(seq):
    counterClasses = []
    for element in seq:
        wasAdded = False
        if (len(counterClasses) == 0):
            counterClasses.append(counterClass(element))
            continue
        for numbers in counterClasses:
            if (numbers.number == element):
                numbers.increment_counter()
                wasAdded = True
                break
        if (wasAdded != True):
            counterClasses.append(counterClass(element))
    for element in counterClasses:
        if element.counter % 2 != 0:
            return element.number
    return None


# --> better solution for find_it:
# def find_it(seq):
#     return [x for x in seq if seq.count(x) % 2][0]


def solution(number):
    l = list(filter(lambda x: (x % 3 == 0 and x % 5 == 0) or (x % 3 == 0) or (x % 5 == 0), np.arange(1, number)))
    if (len(l) != 0):
        return reduce(lambda y, x: x + y, l)
    else:
        return 0


reduce((lambda x, y: x * y), [1, 2, 3, 4])


def namelist(names):
    if len(names) == 0:
        return ""
    if len(names) == 1:
        return list(map(lambda it: it['name'], names))[-1]
    if len(names) == 2:
        return list(map(lambda it: it['name'], names))[0] + " & " + list(map(lambda it: it['name'], names))[1]
    if len(names) > 2:
        return ", ".join(list(map(lambda it: it['name'], names))[:-1]) + " & " + \
               list(map(lambda it: it['name'], names))[-1]


def digital_root(n):
    if sum(list(map(int, str(n)))) > 9:
        return digital_root(sum(list(map(int, str(n)))))
    else:
        return sum(list(map(int, str(n))))


def max_product(lst, n_largest_elements):
    # print (sorted(lst, reverse=True)[:n_largest_elements])
    return reduce((lambda x, y: x * y), sorted(lst, reverse=True)[:n_largest_elements])


def tribonacci(signature, n):
    if n == 0:
        return []
    startSum = reduce(lambda x, y: x + y, signature[0:3])
    signature.append(startSum)
    while len(signature) < n:
        signature.append(reduce(lambda x, y: x + y, signature[-3:]))

    return signature[0:n]


# better solution
# def tribonacci(signature, n):
#   res = signature[:n]
#   for i in range(n - 3): res.append(sum(res[-3:]))
#   return res
def spin_words(sentence):
    if sentence == "ot":
        return "to"
    split = sentence.split(" ")
    if len(split) == 1:
        return ''.join(e for e in reversed(split[0]))
    for c, element in enumerate(split):
        # '{} & {}'.format(', '.join(name['name'] for name in names[:-1]),
        if len(element) >= 5:
            split[c] = '{}'.format(''.join(e for e in reversed(element)))

    return '{}'.format(' '.join(s for s in split))


# better solution
# return " ".join([x[::-1] if len(x) >= 5 else x for x in sentence.split(" ")])
def delete_nth(order, max_e):
    returnarray = []

    for element in order:
        counter = 0
        if len(returnarray) != 0:
            for element1 in returnarray:
                if element1 == element:
                    counter = counter + 1
            if counter < max_e:
                returnarray.append(element)
        else:
            returnarray.append(element)
    return returnarray


# better solution
# def delete_nth(order,max_e):
#     ans = []
#     for o in order:
#         if ans.count(o) < max_e: ans.append(o)
#     return ans
def find_uniq(arr):
    hashTable = {}
    for i in range(0, len(arr)):
        if arr[i] not in hashTable:
            hashTable[arr[i]] = True
        else:
            hashTable[arr[i]] = False
    for key, value in hashTable.items():
        if (value):
            return key


# better solution
# def find_uniq(arr):
#     a, b = set(arr)
#     return a if arr.count(a) == 1 else b

def triple_double(num1, num2):
    s1 = str(num1)
    s2 = str(num2)
    l1 = list(s1)
    l2 = list(s2)
    rv1 = 0
    rv2 = 0
    number = None
    for e1 in l1:
        if l1.count(e1) >= 3:
            rv1 = 1
            number = e1
            break
    for e2 in l2:
        if number is not None:
            if l2.count(e2) == 2 and e2 == number:
                rv2 = 1
                break
    return rv1 and rv2


# better solution
# def triple_double(num1, num2):
#     return any([i * 3 in str(num1) and i * 2 in str(num2) for i in '0123456789'])

def done_or_not(board):
    if board is None:
        return "Try Again!"
    l = list(range(1, 10))
    slicecounter1 = 0
    slicecounter2 = 0
    for c in range(len(board)):
        row = board[c][:]
        column = numpy.array(board)[:, c]
        if slicecounter1 >= len(board):
            slicecounter1 = 0
        if c % 3 == 0 and c != 0:
            slicecounter2 = slicecounter2 + 3
        matrix = numpy.array(board)[slicecounter1:slicecounter1 + 3, slicecounter2:slicecounter2 + 3]
        matrix = matrix.reshape(1, 9)
        for x in l:
            if row.count(x) > 1 or row.count(x) == 0:
                return "Try Again!"
            if column.tolist().count(x) > 1 or column.tolist().count(x) == 0:
                return "Try Again!"
            if matrix[0] is not None:
                if matrix[0].tolist().count(x) > 1 or matrix[0].tolist().count(x) == 0:
                    return "Try Again!"
        slicecounter1 = slicecounter1 + 3

    return "Finished!"


# better solution
# board = np.array(aboard)
#
# rows = [board[i, :] for i in range(9)]
# cols = [board[:, j] for j in range(9)]
# sqrs = [board[i:i + 3, j:j + 3].flatten() for i in [0, 3, 6] for j in [0, 3, 6]]
#
# for view in np.vstack((rows, cols, sqrs)):
#     if len(np.unique(view)) != 9:
#         return 'Try again!'
#
# return 'Finished!'
class CustomInt(int):
    def __call__(self, v):
        return CustomInt(self + v)


def add(v):
    return CustomInt(v)
# [75],
#     [95, 64],
#     [17, 47, 82],
#     [18, 35, 87, 10],
#     [20,  4, 82, 47, 65],
#     [19,  1, 23, 75,  3, 34],
#     [88,  2, 77, 73,  7, 63, 67],
#     [99, 65,  4, 28,  6, 16, 70, 92],
#     [41, 41, 26, 56, 83, 40, 80, 70, 33],
#     [41, 48, 72, 33, 47, 32, 37, 16, 94, 29],
#     [53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14],
#     [70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57],
#     [91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48],
#     [63, 66,  4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31],
#     [ 4, 62, 98, 27, 23,  9, 70, 98, 73, 93, 38, 53, 60,  4, 23],
#     ]))

# def determineLongestSlideDown(element, level, flatten):
#     if (level == 0):
#         if (flatten[level + 1] > flatten[level + 2]):
#             return determineLongestSlideDown(element + flatten[level + 1], level + 1, flatten)
#         else:
#             return determineLongestSlideDown(element + flatten[level + 2], level + 2, flatten)
#     if (level == 1):
#         if (flatten[2*level + 1] > flatten[2*level + 2]):
#             return determineLongestSlideDown(element + flatten[2*level + 1], 2*level + 1, flatten)
#         else:
#             return determineLongestSlideDown(element + flatten[2 * level + 2], 2 * level + 2, flatten)
#     if (2 * level + 1 > len(flatten) - 1 and 2 * level > len(flatten)):
#         return element + 0
#     elif flatten[2 * level] > flatten[2 * level + 1]:
#         return determineLongestSlideDown(element + flatten[2 * level], level * 2, flatten)
#     return determineLongestSlideDown(element + flatten[2 * level + 1], level * 2 + 1, flatten)


def primeFactors(n):
    s = ""
    l = []
    isPrime = None
    sum = 1
    for i in chain([2], range(3, n//2, 2)):
        if sum == n:
            return s
        if n % i == 0:
            power = 1
            intermFactor = n // i
            while intermFactor > i and intermFactor % i == 0:
                intermFactor = intermFactor//i
                power = power + 1
            isPrime = [True if i%x == 0 else False for x in l]
            if (not any(isPrime)):
                sum = sum * pow(i, power)
                s = s +"({base}{pow})".format(base=str(i), pow="" if power == 1 else "**" + str(power))
                l.append(i)
            else:
                isPrime = None
            if len(l) == 0:
                sum = sum * pow(i, power)
                s = s + "({base}{pow})".format(base=str(i), pow="" if power == 1 else "**" + str(power))
                l.append(i)
    return "({})".format(n)

# better solution
# def primeFactors(n):
#     ret = ''
#     for i in xrange(2, n + 1):
#         num = 0
#         while(n % i == 0):
#             num += 1
#             n /= i
#         if num > 0:
#             ret += '({}{})'.format(i, '**%d' % num if num > 1 else '')
#         if n == 1:
#             return ret

# def primeFactors(n):
#     i, j, p = 2, 0, []
#     while n > 1:
#         while n % i == 0: n, j = n / i, j + 1
#         if j > 0: p.append([i,j])
#         i, j = i + 1, 0
#     return ''.join('(%d' %q[0] + ('**%d' %q[1]) * (q[1] > 1) + ')' for q in p)

highestSum=0
counter=0
pyramidArray=[]
def determineLongestSlideDown(level1, level2, currentSum, path):
    global highestSum
    global pyramidArray
    global counter
    path[level1] = pyramidArray[level1][level2]
    currentSum = currentSum + pyramidArray[level1][level2]
    if (level1 + 1 < len(pyramidArray)):
        for x in range(0, 2):
            if (level2 + x) < len(pyramidArray[level1 + 1]):
                determineLongestSlideDown(level1 + 1, level2 + x, currentSum, path)
    else:
        if currentSum > highestSum:
            highestSum = currentSum
        print ('+'.join(map(str, path)))
        counter = counter + 1
        return
    return



def longest_slide_down(pyramid):
    global pyramidArray
    global highestSum
    pyramidArray=pyramid
    determineLongestSlideDown(0,0,0,[None]*len(pyramidArray))
    return highestSum
    # return 0 + determineLongestSlideDown(flatten[0], 0, flatten)

def main():
    # print ("Hi")
    # print (max_product ({8, 10 , 9, 7}, 3))
    # print (digital_root(942))
    # print (namelist([{'name': 'Bart'}, {'name': 'Lisa'}, {'name': 'Maggie'}, {'name': 'Lisa'}, {'name': 'Maggie'}]))
    # print (tickets([25, 25, 50]));
    # print(solution(10))
    # find_it([20,1,-1,2,-2,3,3,5,5,1,2,4,20,4,-1,-2,5])
    # tribonacci([1,1,1], 10)
    # delete_nth([20,37,20,21], 1)
    # print (find_uniq([1, 1, 1, 2, 1, 1]))
    # print(spin_words("ot"))
    # triple_double(10560002, 100)
    # print(done_or_not(
    #             [[1, 3, 2, 5, 7, 9, 4, 6, 8],
    #              [4, 9, 8, 2, 6, 1, 3, 7, 5],
    #              [7, 5, 6, 3, 8, 4, 2, 1, 9],
    #              [6, 4, 3, 1, 5, 8, 7, 9, 2],
    #              [5, 2, 1, 7, 9, 3, 8, 4, 6],
    #              [9, 8, 7, 4, 2, 6, 5, 3, 1],
    #              [2, 1, 4, 9, 3, 5, 6, 8, 7],
    #              [3, 6, 5, 8, 1, 7, 9, 2, 4],
    #              [8, 7, 9, 6, 4, 2, 1, 3, 5]]))
    print(longest_slide_down([
    [75],
    [95, 64],
    [17, 47, 82],
    [18, 35, 87, 10],
    [20,  4, 82, 47, 65],
    [19,  1, 23, 75,  3, 34],
    [88,  2, 77, 73,  7, 63, 67],
    [99, 65,  4, 28,  6, 16, 70, 92],
    [41, 41, 26, 56, 83, 40, 80, 70, 33],
    [41, 48, 72, 33, 47, 32, 37, 16, 94, 29],
    [53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14],
    [70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57],
    [91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48],
    [63, 66,  4, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31],
    [ 4, 62, 98, 27, 23,  9, 70, 98, 73, 93, 38, 53, 60,  4, 23],
    ]))
    # print(longest_slide_down([[3],
    #                           [7, 4],
    #                           [2, 4, 6],
    #                           [8, 5, 9, 3]]))
    # print(primeFactors(7919))


if __name__ == '__main__':
    main()
