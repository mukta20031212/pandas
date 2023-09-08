# -*- coding: utf-8 -*-
"""
Created on Tue May  9 08:54:31 2023

@author: hp
"""

import pandas as pd
technologies={'course':["spark","pyspark","python","pandas"],
              'fee':[2000,35000,8700,3400],
              'duration':['30day','40days','50days','60days']
              }
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)

technologies2={'course':["spark","java","python","go"],
               'Discount':[2000,2300,1300,7600]
               }
index_labels2=['r1','r6','r3','r5']
df2=pd.DataFrame(technologies2,index=index_labels2)
#pandas join,by default it will join the table left join
df3=df1.join(df2,lsuffix="_left",rsuffix="_right")
print(df3)
#pandas inner join dataframe
df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='inner')
print(df3)
#pandas left join dataframe

df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='left')
print(df3)

df3=df1.join(df2,lsuffix="_left",rsuffix="_right",how='right')
print(df3)
#pandas join on column
df3=df1.set_index('course').join(df2.set_index('course'),how='inner')
print(df3)
#pandas merge dataframe

import pandas as pd
technologies={'course':["spark","pyspark","python","pandas"],
              'fee':[2000,35000,8700,3400],
              'duration':['30day','40days','50days','60days']
              }
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)

technologies2={'course':["spark","java","python","go"],
               'Discount':[2000,2300,1300,7600]
               }
index_labels2=['r1','r6','r3','r5']
df2=pd.DataFrame(technologies2,index=index_labels2)

#using pandas.merge()
df3=pd.merge(df1,df2)
#using Dataframe.merge()
df3=df1.merge(df2)
print(df3)

#use pandas.concat()to concat two dataframe
import pandas as pd
df=pd.DataFrame({'course':["spark","pyspark","python","panndas"],
                 'Fee':[2000,7000,6700,8700]
                 })
df1=pd.DataFrame({'course':["pandas","hadoop","hyperion","java"],
                  'Fee':[2500,7600,5600,7600]
                  })
data=[df,df1]
df2=pd.concat(data)
df2
#concatenate the multiple dataframe using pandas
df=pd.DataFrame({'course':["spark","pyspark","python","panndas"],
                 'Fee':[2000,7000,6700,8700]
                 })
df1=pd.DataFrame({'course':["pandas","hadoop","hyperion","java"],
                  'Fee':[2500,7600,5600,7600]
                  })
df2=pd.DataFrame({'Duration':['45day','56day','56day','67day'],
                  'Discount':[4000,7800,6000,7600]})
#Appending multiple dataframe
df3=pd.concat([df,df1,df2])
print(df3)
#write dataframe to csv file defalt parameter
df3.to_csv("c:/1-python/courses.csv")
#read csv
#import pandas
import pandas as pd
#read csv file into Dataframe
df=pd.read_csv('courses.csv')
print(df)
#write dataframe to excel file
df.to_excel('c:/1-python/courses.xlsx')

#series is used to model one dimentional data
#similar to list in python
#object few more bit of data
#of data,including an index and a name
import pandas as pd
songs2=pd.Series([145,142,38,13],name='counts')
#it is easy to inspect index of series

songs2.index

songs3=pd.Series([145,142,38,13],name='counts',
   index=['Paul','John','Geoge','Ringo'])
songs3.index
songs3
#the NaN values
#this value stand for Not a number
#and is usually ignored in arithmetic
#operation(similar null in sql)
#if you load data form csv file
#an empty value for an otherwise
#numeric column well NaN
import pandas as pd
f1=pd.read_csv('c:/1-python/age.csv')
f1

#None,NaN,nan,and null are synonyms
#series
import numpy as np
numpy_Ser =np.array([145,142,38,13])
songs3[1]

#they both the methods in common
songs3.mean()
#the pandas series data structure provide
#support for basic crud
#operation-create,read,update,delete
george=pd.Series([10,7,1,22])
index=['1968','1969','1970','1970']
name='george songs'
george

#interesting feature of pandas the index value
#to read or select data form series
george['1968']

george['1970']
#we can iterate over thhe series
#when iterating over a series
for item in george:
    print(item)
    #updating
george['1969']=68
george['1969']
#deletion
#del statement apper to have
#promblem with duplicate value
s=pd.series([2,3,4],index=[1,2,3])
del s[1]
#convert type
#string use.astype(str)
#numeric use pd.to_numeric
#integer use .astype(int)
#note that this is fail with NaN
#datatime use pd.to datatime
songs_66=pd.Series([3,None,11,9])
index=['George','Ringo','John','Paul']
name='counts'

pd.to_numeric(songs_66.apply(str))
#there will be error
pd.to numeric(songs_66 as type)
    
songs_66.fillna(-1)
songs_66.dropna()
#Append ,combining and joining two series
import pandas as pd
songs_69=pd.Series([7,16,21,39],
index=['Ram','Sham','ghansham','krishna'],
name='Counts')
#to cancatenate two series
songs=songs_66.append(songs)

############################

#plotting two series
import matplotlib.pyplot as plt
fig=plt.figure()
songs_69.plot()
plt.legend()
############################
fig=plt.figure()
songs_69.plot(kind='bar')
songs_66.plot(kind='bar',color='b',alpha=.5)
plt.legend()

##############################
import numpy as np
data=pd.Series(np.random.randn(500),
name='500 random')
fig=plt.figure()
ax=fig.add_subplot(111)
data.hist()
###########################
#in  the input array.the function
#numpy.remainder()also produce the same result

#perform mod function
 
import numpy as np
arr1=np.array([7,20,13])
arr2=np.array([3,5,2])
arr1
arr1.dtype

#########
import numpy as np
arr=np.array([10,20,30])
print(arr)
##############
#create a multiDimentional Array
arr=np.array([[10,20,30],[40,50,60]])
print(arr)
############
#represent the minimum dimension
#use ndmin called(minimum dimension) param to specify how many minimum
#dimension you wanted to create array
#minimum dimension
arr=np.array([10,20,30,40],ndmin=2)
print(arr)
###########
#change the datatype of array
arr=np.array([10,20,30],dtype=complex)
print(arr)
########
#get the dimension of array
arr=np.array([[1,2,3,4],[7,8,6,7],[9,10,11,12]])
print(arr.ndim)
print(arr)

#########################11April  Morging session
#Boolean array indexing
#this advanced indexing occur when the object
#object array is boolean type
#it return in comparison operation
#pick the element
#array which satisfy all condition
#Boolean array indexing
import numpy as np
arr=np.arange(12).reshape(3,4)
print(arr)

rows=np.array([False,True,True])#not zeroth row only first and second rows
wanted_rows=arr[rows,:]
print(wanted_rows)
############
#Convert one dimensional array to list
#create array

array=np.array([10,20,30,40])
print("Array:",array)
print(type(array))

lst=array.tolist()
print("list:",lst)
print(type(lst))

#convert muti dimentional array
#create array
array=np.array([[10,20,30,40],
               [50,60,70.80],
               [60,40,20,10]])
print("Array:",array)
################################
lst=[20,40,60,80]

#their are two methhod converting array-1)numpy.array 2)numpy.asarrary
#use asarray
lst=[20,40,60,80]
array=np.asarray(lst)
print("Array:",array)
print(type(array))
#output:Array:[20,40,60,80]

#####################Numpy
#shape
array=np.array([[1,2,3],[4,5,6]])
print(array.shape)
#output (2, 3)
array=np.array([[10,20,30],[40,50,60]])
array.shape=(3,2)
print(array)

############Numpy 
array=np.array([[10,20,30],[40,50,60]])
new_array=array.reshape(3,2)
print(new_array)

##########ndim
array=np.array([[1,2,3,4],[7,8,6,7],[9,10,11,12]])
print(array.ndim)

########
arr1=np.arange(16).reshape(4,4)
arr2=np.array([1,3,2,4])
add_arr=np.add(arr1,arr2)
print(f"Adding two arrays:\n{add_arr}")


sub_arr=np.subtract(arr1,arr2)
print(f"subtracting two arrays:\n{sub_arr}")

mul_arr=np.multiply(arr1,arr2)
print(f"multipying two arrays:\n{mul_arr}")
########divide
div_arr=np.divide(arr1,arr2)
print(f"dividing two arrays:\n{div_arr}")

#########################numpy.reciprocol of array
#the function return  the reciprocol of argument
#element -wise.for element with absolute value
arr1=np.array([3,10,5])
pow_arr1=np.power(arr1,3)
print(f"After applying power function to array:{pow_arr1}")

arr2=np.array([3,10,5])
print("My second array:\n",arr2)
pow_arr2=np.power(arr1,arr2)
print(f"After applying power function to array:{pow_arr2}")
##################
#to perform the mod function
#on Numpy array
import numpy as np
arr1=np.array([7,20,13])
arr2=np.array([3,5,2])
arr1
arr1.dtype
#mod()
mod_arr=np.mod(arr1,arr2)
print(f"After applying mod function to array:\n{mod_arr}")

#create an empty array
from numpy import empty
a=empty([3,3])
print(a)
#create zero array
from numpy import zeros
a=zeros([3,3])
print(a)

from numpy import ones
a=ones([5])
print(a)
#create array  with vstack

from numpy import array
from numpy import vstack
a1=array([1,2,3])
print(a1)

a2=array([4,5,6])
print(a2)

a3=vstack((a1,a2))
print(a3)
print(a3.shape)
############################
#create array with hstack
from numpy import array
from numpy import hstack
#create first array
a1=array([1,2,3])
print(a1)
#create second array
a2=array([4,5,6])
print(a2)
#create horital stack
a3=hstack((a1,a2))
print(a3)
print(a3.shape)

#######################
One dimensional array
data=[11,22,33,44,55]
#array of data
data=array(data)
print(data)
print(type(data))

###################12 April (Morning session)
#Two dimentional list of list to array
from numpy import array
data=[[11,22,33],[34,54,67],[56,87,43]]
data=array(data)
print(data)
print(type(data))

#index in one dimentional array
from numpy import array
data=array((11,22,33,44,55))
print(data[0])
print(data[4])
#index array out of bound
from numpy import array
data[6]
###############

data=array([[11,22],[33,44],[55,66]])
#index data
print(data[0,0])
######################
#index row of two dimentional array
from numpy import array
data=array([
[11,22],
[33,44],
[55,66]])
print(data[0,])#0 th row all column
######################
#slice a one dimentional array
data=array([11,22,33,44,55])
print(data[1:4])
######################
#neative slicing of one dimentional array
from numpy import array
data=array([11,22,33,44,55])
print(data[-2:])
####################
#split input and output data
from numpy import array
#define array
data=array([
    [11,22,33],
    [44,55,66],
    [77,88,99]])
#seperate data
X,y=data[:,:-1],data[:,-1]
X
y
#data{:,:-1}all rows and allcolumn
#except all rows and last column
#data[:,-1]taking all rows(:)
#but keeping the last column (-1)
###################
#broadcast scalar to one dimentional array
from numpy import array
#define array
a=array([1,2,3])
print(a)
#define scalar
b=2
print(b)
#broadcast
c=a+b
print(c)
###############
#vector addition
from numpy import array
a=array([1,2,3])
print(a)
#define second vector
b=array([1,2,3])
print(b)
#add vectors
c=a+b
print(c)
####################
#vector subtraction
from numpy import array
a=array([1,2,3])
print(a)
#define second vector
b=array([0.5,0.5,0.5])
print(b)
################
#vector L1 norm
from numpy import array
from numpy.linalg import norm
#define vector
a=array([1,2,3])
print(a)
#calculate norm
l1=norm(a,1)
print(l1)
####################
#vector l2 norm
from numpy  import array
from numpy.linalg import norm
#define vector
a=array([1,2,3])
print(a)
#calculate norm
l2=norm(a)
print(l2)
##################
#triangular matrix
from numpy import array
from numpy import tril
from numpy import triu
M=array([
[1,2,3],
[1,2,3],
[1,2,3]])
print(M)
lower=tril(M)
print(lower)
#upper triangular matrix
upper=triu(M)
print(M)
#diagonal matrix
from numpy import array
from numpy import diag
#define  square matrix
M=array([[1,2,3],
        [1,2,3],
        [1,2,3]])
print(M)
d=diag(M)
print(d)

#################12 May Afternoon
#identity matrix
from numpy import identity
I=identity(3)
print(I)


#Orthonal matrix
from numpy import array
from numpy.linalg import inv
#define orthogonal matrix
Q=array([
[1,0],
[0,-1]])
print(Q)

#inverse equivalence
V=inv(Q)
print(Q.T)
print(V)

#identity equivalence
I=Q.dot(Q.T)
print(I)

##
import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.show()

##multiline plots
import matplotlib.pyplot as plt
x=range(1,5)
plt.plot(x, [xi*1.5 for xi in x])

plt.plot(x, [xi*3.0 for xi in x])

plt.plot(x, [xi/3.0 for xi in x])
plt.show()

#Adding a grid
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.grid(True)
plt.show()

#Handling axes
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.axis()  #shows current axis limits value
plt.axis([0,5,-1,13])  # set new axes limits
#[xmin,xmax,ymin,ymin]
plt.show()

#Adding lables
import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.xlabel('This is the xaxis')
plt.ylabel('This is the yaxis')
plt.show()

#Adding title
import matplotlib.pyplot as plt
plt.plot([1,3,2,4])
plt.title('Simple plot')
plt.show()

#Adding a legend
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(1,5)
plt.plot(x,x*1.5, label='Normal')
plt.plot(x,x*3.0, label='Fast')
plt.plot(x,x/3.0, label='Slow')
plt.legend()
plt.show()

#Control colors
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1, 3)
plt.plot(y, 'y');
plt.plot(y+1, 'm');
plt.plot(y+2, 'c');
plt.show()

 #specifying styles in multiline plot
 import matplotlib.pyplot as plt
 import numpy as np
 y=np.arange(1,3)
 plt.plot(y, 'y', y+1, 'm', y+2, 'c');
 plt.show()
   
#control line styles
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3)
plt.plot(y, '--', y+1, '--', y+2, ':');
plt.show()

#symbol marker
**
import matplotlib.pyplot as plt
import numpy as np
y=np.arange(1,3,0.2)
plt.plot(y,'x',y+0.5,'o',y+1,'D',y+1.5, '^',y+2,'s');
plt.show()

################Histogram charts
import matplotlib.pyplot as plt
import numpy as np
y=np.random.randn(1000)
plt.hist(y);
plt.show()
#########count plot
import matplotlib.pyplot as plt
plt.bar([1,2,3],[3,2,5]);
plt.show()

##Bar graph is called unique analysis
import matplotlib.pyplot as plt
import numpy as np
x=np.random.randn(1000)
y=np.random.randn(1000)
plt.scatter(x,y);
plt.show()

size=50*np.random.randn(1000)
colour=np.random.rand(1000)
plt.scatter(x,y,s=size,c=colour);
plt.show()

####################
#Adding text
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-4,4,1024)
y=.25*(x+4.)*(x+1.)*(x-2.)
plt.text(-0.5,-0.25,'Brackmard minimum')
plt.plot(x,y,c='k')
plt.show()

#################
#pip install seaborn
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
#seaborn has 18 in-build dataset
#that can be found using the following command
sns.get_dataset_names()
df=sns.load_dataset('titanic')
df.head()

######################
# -*- ""coding: utf-8 -*-


#from scipy.start import kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
cars=pd.read_csv('a.csv')
cars.columns
cars.describe()

#output
Index      speed        dist
count  50.00000  50.000000   50.000000
mean   25.50000  15.400000   42.980000
std    14.57738   5.287644   25.769377
min     1.00000   4.000000    2.000000
25%    13.25000  12.000000   26.000000
50%    25.50000  15.000000   36.000000
75%    37.75000  19.000000   56.000000
max    50.00000  25.000000  120.000000

##
plt.hist(cars.speed)
sns.displot(x='speed',kde=True,bins=6,data=cars)

##box graph
cars.describe
sns.boxplot(cars.speed)

##men  or miminum speed of all cars
plt.hist(cars.speed)

##Dist.Data is right skewed
sns.boxplot(cars.dist)

##skeweness od speed
speed=cars['speed'].tolist()
speed
from scipy.stats import skew
from scipy.stats import kurtosis
print("skewness of speed",skew(speed))
dist=cars['dist'].tolist()
print("skewness of dist",skew(dist))
print(skew(dist,axis=0,bias=True))

#it signifies that distributation
print(kurtosis(dist,axis=0,bias=True))


##
from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd
import seaborn as np
import matplotlib.pyplot as plt
cars=pd.read_csv('a.csv')
cars.columns
cars.describe()

##
sns.countplot (x='speed',data=cars)
sns.countplot(x='speed',hue='',data=df,palette='set1')

sns.kdeplot(x='speed',data=cars,color='red')

sns.scatterplot(x='speed',y='dist',data=cars)

##In the plot above we can observe that on iris flower
with a sepal length < 6cm and petal length > 2cm
is most likely of type setosa.

##Although there is no distinct boundary present between
the versicolor dots and virginica dots,
an iris flower with petal length between 2cm and 5 cm
is most likely of type versicolor,
while iris flowers with petal length > 5cm
are most likely of type verginica.

#join plot
sns.jointplot(x='speed',y='dist',kind='reg')
sns.jointplot(x='speed',y='dist',kind='hist')
sns.jointplot(x='speed',y='dist',kind='kde')

#pair plot
sns.pairplot(cars)

##
corr=cars.corr()
sns.heatmap(corr)

#Data dictionary
#index
#speed of car running by speed
import pandas as pd
df=df.read_csv('Titanic-Dataset.csv')
df.column
df.describe(x='Pclass',kde=True,bins=4,data=df)



########################
#16 May Morning session
#TOKENIZATION
txt='Welcome to new year 2023'
x=txt.split()
print(x)

################Remvoing the special character
#non alphabetical
#imports
import re# function to remove special character
def remove_special_characters(text):
    #define the pattern to keep
    pat=r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat,'',text)

#call function
remove_special_characters("007 Not sure@ abount% the placement&&,,,,,!!!")

import re# function to remove special character
def remove_numbers(text):
    #define the pattern to keep
    pat=r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat,'',text)
#call function
remove_numbers("007 Not sure@ abount% the placement&&,,,,,!!!")

txt='welcome:to the,new year:2023!'
import re
x=re.split(r'(?:,',txt)
print(x)

#function remove punctuation
import string
remove_punctuation(text):
    txt='',join([c for c in text if c not in string.punctuation])
    return txt
remove_punctuation('Artical')


#################
line='asdf fjdk; afed,fed,fjek,asdf,foo'
re.split(r'(?:,!;!\s)\s*',line)
###########
pattern=r'(?:,!;!\s)\s*'
x=re.split(pattern,txt)
print(x)
############
#matching text at the start or end of string
filename='spam.txt'
filename.endswith(',txt')
###############
area_name='6 th lane west Andheri'
area_name.endswith('west Andheri')
#############
choices=('http:','ftp:')
url='http://www.python.org'
url.startswith(choices)

################
#slicing a string
#if S is a string the expression S[start:stop]
#return the portion of string from index
#at a step size step
S='ABCDEFGHI'
print(S[2:7])
#slicing with negative indices
S='ABCDEFGHI'
print(S[-7:-2])
#slicing with positive &negative indices
S='ABCDEFGHI'
print(S[2:-5])
#spcify step of the slicing
#you can specify the step of slicing using
#step parameter is optional
S='ABCDEFGHI'
print(S[2:7:2])
#return every 2nd steps
S='ABCDEFGHI'
print(S[6:1:-2])
#slicing for first three character
S='ABCDEFGHI'
print(S[:3])
S[start:len(S)] 
#slicing of last 3 character
S='ABCDEFGHI'
print(S[6:])
#reverse string by omitting both start
#specify step as -1
S='ABCDEFGHI'
print(S[::-1])
#similar operation can done with slices
filename='spam.txt'
filename[-4:]=='.txt'

#########
url='http://www.python.org'
url[:5] == 'http:'or url[:6]=='https:' or url[:4]=='ftP:'

##########
from fnmatch import fnmatch,fnmatchcase

names=['Dat1.csv','Dat2.csv','config.ini','foo.csv']
[name for name in names if fnmatch(name,'Dat*.csv')]
###################
from fnmatch import fnmatch,fnmatchcase
names=['Andheri East','Parle East','Dadar west']
[name for name in names if fnmatch(name,'*East')]

################
from fnmatch import fnmatch,fnmatchcase
addresses=['5412 N CLARK ST',
           '1060 W ADDISON ST'
           '1039 W GRANVILLE AVE',
           '2122 N CLARK ST',
           '4802 N BROADWAY']
[addr for addr in addresses if fnmatchcase(addr,'*ST')]


text='yeah, but no, but yeah, but no, but yeah'
#exact match
text=='yeah'
#match at start or end
text.startswith('yeah')

text.endswith('no')
##search for the location of first  occurence
text.find('no')
#########/d+ date to digit
text1='11/27/2012'
text2='Nov 27,2012'

import re
#simple  matching :\d+ means match one or more digit

if re.match(r'\d+/\d+/\d+',text1):
    print('yes')
else:
    print('no')
    
##########
import re
#simple  matching :\d+ means match one or more digit

if re.match(r'\d+/\d+/\d+',text2):
    print('yes')
else:
    print('no')
###########23 MAY
import re
import string
import nltk
text='UPPER PYTHON,lower python,Mixed python'
def matchcase(word):
    def replace(m):
        text=m.group()
        if text.isupper():
            return word.upper()
        elif text.islower():
            return word.lower()
        elif text[0].isupper():
            return word.capitalize()
        else:
            return word
    return replace
re.sub('python',matchcase('snake'),text,flags=re.IGNORECASE)
        





