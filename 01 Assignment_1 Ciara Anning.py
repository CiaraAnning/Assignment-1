#!/usr/bin/env python
# coding: utf-8

# ![LL%20&%20SU.jpg](attachment:LL%20&%20SU.jpg)

# # Assignment \#1: Introduction to Python
# 
# 
# 

# This Assignments aims to test introductory Python skills, specifically content covered in the _Introduction to Python_ and _Intermediate Python_ DataCamp courses.

# ## Lists and sets
# 
# ### Write dynamic code to solve the following questions:
# 
# Display the given Python list in reverse order.

# In[15]:


a = [100, 200, 300, 400, 500]

# Insert code here
a.reverse()
print(a)


# Given a list, turn every item of the list into its square.

# In[ ]:


a = [1, 2, 3, 4, 5, 6, 7]

# Insert code here

def square(a):
    return [i ** 2 for i in a]

square(a)


# Remove empty strings from the list of strings.

# In[14]:


a = ["Mike", "", "Emma", "Kelly", "", "Brad"]

# Insert code here
def remove(a):
    for i in a:
      if i== "":
        a.remove(i)
    print(a)

remove(a)
    
   


# Given a Python list, find value 15 in the list, and if it is present, replace it with 150. Only update the first occurrence of a value.

# In[ ]:


a = [5, 10, 15, 20, 15, 50, 20]

# Insert code here
a[a.index(15)] = 150
print(a)


# Given a Python list, remove all occurrence of 15 from the list.

# In[ ]:


a = [5, 10, 15,15,15, 20, 15,15, 50, 20,15]


# Insert code here
for i in a:
  if i == 15:
    a.pop(a.index(i))
print(a)


# Add list of elements in *aList* to a given set, *aSet*.

# In[40]:


aSet = {"Dog", "Bird", "Horse"}
aList = ["Lizard", "Cat", "Rabbit"]

# Insert code here
aSet.update(aList)


# Return a set of identical items from two Python sets.

# In[24]:


set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

# Insert code here
set1.intersection(set2)


# Given two Python sets, update first set with items that exist only in the first set and not in the second set.

# In[38]:


set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

# Insert code here
set1 = set1.difference(set2)


# Remove items from *set1* that are not common to both *set1* and *set2*.

# In[42]:


set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

# Insert code here
set1 = set1.intersection(set2)


# Return a set of all elements in either *set1* or *set2*, but not both.

# In[45]:


set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

# Insert code here
set3 = set1.difference(set2).union(set2.difference(set1))


# ## Dictionaries
# Create a dictionary with the following key-value pairs: apple = 10, orange = 12, banana = 16, grape = 4.

# In[70]:


# Insert code here
my_dict = {'apple':10,'orange':12,'banana':16,'grape':4}


# Use the dictionary to retrieve the number of apples.

# In[8]:


# Insert code here
my_dict['apple']


# Use the dictionary to display the total fruit.

# In[73]:


# Insert code here
values = my_dict.values()
total=sum(values)
print(total)


# Add the following record to the dictionary: 23 pears

# In[214]:


# Insert code here
my_dict['pear']=23


# Retrieve and display the most occuring fruit from dictionary.
# 

# In[83]:


# Insert code here
from collections import Counter
total = Counter(my_dict)
print(total.most_common(1))


# Display fruit  in alphabetiacal order:
# 
# 

# In[84]:


# Insert code here
sorted(my_dict)


# ## Python Functions
# Create a function named "sign()" that returns "positive", "negative" or "zero" given an integer as input.

# In[85]:


#Insert function:
def sign(x):
  if x<0:
    return("negative")
  elif x>0:
    return("positive")
  else:
    return("zero")
# Test
for x in [-1, 0, 1]:
  print(sign(x))


# Write a function named "my_min()" that returns the minimum value given arbitrary 
# number of integers as input parameters.

# In[91]:


# Insert function 
def my_min(*args):
  return min(*args)
# Test
print(my_min(4, 5, 6, 7, 2))
print(my_min(-10, 5, 3, -4, 1, 10, 3, 6))


# Write a recursive function to display the Fibonacci sequence using recursion. 
# 
# The first two terms are 0 and 1. All other terms are obtained by adding the preceding two terms i.e. the n-th term is the sum of (n-1)-th and (n-2)-th term.

# In[102]:


def recur_fibo(n):
  # Insert code here
    if n <= 1:  
       return n  
    else:  
       return(recur_fibo(n-1) + recur_fibo(n-2))

# Test
for i in range(10):
  print(recur_fibo(i))


# ## Python Classes
# Create a constructor for class "Circle" that takes x and y coordinates (center of cirlce) and radius as arguments to initialize an instance of the class.

# In[234]:


import math

from math import pi

# Insert class here:
class Circle():
  # Insert constructor code
    def __init__(self,x,y,r):
        self.x_coord = x
        self.y_coord = y
        self.radius = r

    def str_circle(self):
    # Insert code
        print("Circle at location x =" + self.x_coord + "and y =" + self.y_coord + "with radius =" + self.radius)

    def dist(self, other):
    # Insert code
        dist = np.sum(np.square(circle1-circle2))
        print(np.sqrt(dist))

    def overlap(self, other):
    # Insert code
        if circle1.dist(circle2) <= (circle1[self.radius]+circle2[self.radius]):
            print(True)
        else:
            print(False)

    def area(self):
    # Insert code
        return self.radius + self.radius * np.pi

    def overlap_area(self, other):
    # Insert code
        


# Create the following two circles:
# *   circle 1: x = 2, y = 1, rad = 1
# 
# *   circle 2: x = -1, y = -2, rad 3 
# 

# In[213]:


# Insert code
circle1 = Circle(2, 1, 1)
circle2 = Circle(-1, -2, 3)


# Complete the function "str_circle()" to print the properties of both circles as follow:
# 
# "Circle at location x = 2 and y = 3 with radius = 1"
# 
# 

# In[ ]:


# Insert code
str_circle(circle1)
str_circle(circle2)


# Complete the function "dist()" to return the euclidean distance between two circles' centers. Print the distance between circle 1 and 2.

# In[209]:


# Test
dist = circle1.dist(circle2)
print(dist)


# Complete the function "overlap()". The function should return True if the circles overlap and False otherwise.

# In[ ]:


# Test
circle1.overlap(circle2)


# Complete fuction "area()". The function should return the area of the circle.

# In[ ]:


# Test
circle1.area()


# Complete function "overlap_area". The function should return the overlapping area between two circles. 
# 
# If the circles do not overlap, return the message: "Circles do not overlap"
# 
# The formula for calculating the overlapping area between two circles is provided below:

# \begin{equation}
# \small
# A = 
# r^2\text{cos}^{-1}(\frac{d^2+r^2-R^2}{2dr}) + R^2\text{cos}^{-1}(\frac{d^2+R^2-r^2}{2dR}) -
# \frac{1}{2}\sqrt{(-d+r+R)(d+r-R)(d-r+R)(d+r+R)}
# \end{equation}
# 

# $\;\;\;\;\;\;\;\;$A = area of overlapping <br>
# $\;\;\;\;\;\;\;\;$r = radius of circle 1 <br>
# $\;\;\;\;\;\;\;\;$R = radius of circle 2 <br>
# $\;\;\;\;\;\;\;\;$d = distance between the circles <br>

# In[ ]:


# Test
circle1.overlap_area(circle2)


# ##  Introduction to Numpy

# In[106]:


# Import numpy library
import numpy as np


# Given the matrix, write code to compute the average over all the entries in the matrix.

# In[107]:


A = [[ 1.27411064,  0.05188032, -1.27088046],
       [-0.78844599, -0.14775522, -0.28198009]]

a = np.array(A)

# Insert code
a.mean()


# Given two vectors of data, print the Euclidean distance between them.

# In[117]:



VEC1 = [-0.25560104,  0.06393334, -0.43760861,  0.35258494, -0.06174621]
VEC2 = [0.16257878, -0.88344182,  1.14405499,  0.33765161,  1.206262]

vec1 = np.array(VEC1)
vec2 = np.array(VEC2)

# Insert code
dist = np.sum(np.square(vec1-vec2))
print(np.sqrt(dist))


# Given two matrices of appropriate dimensions, print their matrix product.
# 

# In[111]:


A = [[0.6583596987271446, 1.0128241391924433],
        [0.37783705753739877, 0.42421340135829255]]

B = [[0.6583596987271446, 1.0128241391924433],
        [0.37883705753739877, 0.42421340135829255]]

a = np.array(A)
b = np.array(B)

# Insert code
print(a*b)


# In[ ]:





# In[ ]:





# Given a vector of integer data, print all the entries that are even.
# 

# In[137]:


start = 3
end = 10

# Insert code
list = (range(3,11))

for num in list:
    if num % 2 == 0:
        print(num)


# In[132]:


VEC = [-3, 5, 1, 2, 18, 2, 234, 11]

vec = np.array(VEC)

# Insert code
even = filter(lambda x: (x%2 ==0), vec)

even_list = list(even)

print(even_list)


# Given two matrices, print whether they are equal (i.e. all their corresponding entries are the same).
# 

# In[141]:


A = [[0.6583596987271446, 1.0128241391924433],
        [0.37783705753739877, 0.42421340135829255],
        [-0.6905233695318467, -0.498554227530507]]
B = [[0.6583596987271446, 1.0128241391924433],
        [0.37883705753739877, 0.42421340135829255],
        [-0.6905233695318467, -0.498554227530507]]
 
a = np.array(A)
b = np.array(B)

#Insert code

if a.all() == b.all():
    print(True)
else:
    print(False)


# Given two vectors of data, print the cosine distance between them.
# 

# In[154]:


VEC1 = [-0.25560104,  0.06393334, -0.43760861,  0.35258494, -0.06174621]
VEC2 = [ 0.16257878, -0.88344182,  1.14405499,  0.33765161,  1.206262]

vec1 = np.array(VEC1)
vec2 = np.array(VEC2)

# Insert code 
from numpy.linalg import norm
cosine = np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
print(1-cosine)


# Given a vector of data, scale (multiply) all the entries in the vector by the vector mean and print the result.

# In[113]:


VEC = [-0.25560104,  0.06393334, -0.43760861,  0.35258494, -0.06174621]

vec = np.array(VEC)

# Insert code
vec*vec.mean()


# Given a matrix, subtract the column mean from each entry. This has the effect of zero-centering the data and is often done in algorithms such as principal components analysis or when running computer vision models.

# In[179]:


A = [[0.6583596987271446, 1.0128241391924433],
       [0.37783705753739877, 0.42421340135829255],
       [-0.6905233695318467, -0.498554227530507]]

a = np.array(A)

# Insert code
c1 = a[:,0].mean()
c2 = a[:,1].mean()

m1 = a[:,0] - c1
m2 = a[:,1] - c2

print(m1,m2)


# Given a vector of data, compute the softmax scores of the vector.
# 
# 

# In[183]:


VEC = [-0.25560104,  0.06393334, -0.43760861,  0.35258494, -0.06174621]

vec = np.array(VEC)

# Insert code
def softmax(x):
    """Compute softmax value"""
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

print(softmax(vec))


# Given a matrix, print whether or not there are any NaN values in the matrix. 

# In[187]:


A = [[0.6583596987271446, 1.0128241391924433],
        [0.37783705753739877, float("nan")],
        [-0.6905233695318467, -0.498554227530507]]

a = np.array(A)

# Insert code
print(np.isnan(a))


# Given a vector of data, normalize the vector to have 0 mean and variance of 1. (Z-score normalization)]

# In[193]:


VEC = [-0.25560104,  0.06393334, -0.43760861,  0.35258494, -0.06174621]

vec = np.array(VEC)

# Insert code
import pandas as pd
import scipy.stats as stats

print(stats.zscore(vec))

