# 吳恩達_機器學習_第1週_2_Linear Algebra Review
###### tags: `andrew` `machine learning` `coursera`
## Linear Algebra Review
[課程連結](https://www.coursera.org/learn/machine-learning/home/week/1)
### Matrices and Vectors
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/38jIT/matrices-and-vectors)
![](https://i.imgur.com/O9Pt3HQ.png)

#### Section_1
![](https://i.imgur.com/5HGsobN.png)

矩陣的維度寫法是(rows, columns)，所以上圖左是一個(4,2)的矩陣，右邊是一個(2,3)的矩陣。
#### Section_2
![](https://i.imgur.com/e6pKFRP.png)

矩陣元素的表示方式，$A_{ij}$，代表第i_row的第j_column的元素    
#### Section_3
![](https://i.imgur.com/HJtTW8a.png)

* 向量
    * 向量是一種特殊矩陣(n,1)    
    * 上圖上範例的向量y，也可以稱為4-dimension vector，代表有四個元素的向量。    
    * 元素表示方式：$y_i$表示第i個元素。

註：第一個索引的表示方式視不同程式語言而定，但本課程以1代表第一個索引值
註：普遍來說，以大寫字母表示矩陣，小寫表示向量、或數值

### Addition and Scalar Multiplication
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/R4hiJ/addition-and-scalar-multiplication)
![](https://i.imgur.com/WkaVLox.png)

#### Mtrix Addition
![](https://i.imgur.com/P7pK916.png)

矩陣的相加即是將相對應的矩陣元素做加法即可，**只有相同維度的矩陣才能相加**。
#### Scalar Multiplication
![](https://i.imgur.com/Mg0clLK.png)

矩陣乘上一個純量(scalar)(或實數)，單純的將矩陣元素逐一乘上該純量即可，也可以以相同的方式來寫除法。
#### Combination of Operands
![](https://i.imgur.com/sx6s9za.png)

結合一起來看，先處理前後兩個純量乘積，再做加減即可。
### Matrix Vector Multiplication
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/aQDta/matrix-vector-multiplication)
![](https://i.imgur.com/iyBtA1K.png)

#### Example
![](https://i.imgur.com/ADzy9qz.png)
![](https://i.imgur.com/fk5DR5r.png)

矩陣(m,n)與向量(n,1)相乘結果是(m,1)    
將矩陣每一個row的column與向量的row相對應的index逐一元素相乘之後相加即為結果
#### Example
![](https://i.imgur.com/Al2lYFB.png)

矩陣(3,4)乘向量(4,1)結果為(3,1)
#### Tricks
![](https://i.imgur.com/5dLFWgv.png)

假設你有一個hypothesis:$h_\theta(x)=-40+0.25x$    
現在有四筆資料要一次計算，可以這麼做：
1. 將資料矩陣化，column_1設置為1
2. 將參數向量化
3. 相乘即結果

盡可能的不使用迴圈逐一計算
### Matrix Matrix Multiplication
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/dpF1j/matrix-matrix-multiplication)
![](https://i.imgur.com/KbboS4c.png)

#### Example
![](https://i.imgur.com/iMXCZU4.png)
![](https://i.imgur.com/G7ZJ2JZ.png)

矩陣相乘，維度為(m,n)x(n,o)=(m,o)，上圖來看，(2,3)x(3,2)=(2,2)。    
下圖說明計算的細節，B矩陣每次取一個向量與A相乘之後得到C的一個向量結果，依序計算。

#### Example
![](https://i.imgur.com/fKjo1dB.png)
這是另一個範例，右方矩陣每次取一個向量與左方矩陣相乘之後得到結果，依序逐一計算。
#### Example
![](https://i.imgur.com/umXSVv2.png)
你有四筆資料，並且有三個hypothesis，透過矩陣計算可以做到一次得到全部的數值。
### Matrix Multiplication Properties
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/W1LNU/matrix-multiplication-properties)
![](https://i.imgur.com/tGpC7Ac.png)

#### Section_1
![](https://i.imgur.com/i4ik4m3.png)

在實數的乘法中，先後順序是可以交換的，這就是乘法的交換律，但是在矩陣中先後順序是不能隨意交換的，具體來說，矩陣A\*矩陣B不等於矩陣B\*矩陣A，即使維度相同所得的結果也不會相同。
#### Section_2
![](https://i.imgur.com/iSxFPUl.png)

實數乘法，3x5x2，可以後面先乘，即3x10，或是前面先乘，即15x2，結果是相同的，這是數學的結合律，事實證明，矩陣的乘法也符合結合律。
#### Identity Matrix
![](https://i.imgur.com/KOKXyBm.png)

單位矩陣，它的特性是對角線是1，其餘元素皆為0，任何矩陣乘上單位矩陣都還是等於自己(前提是它們的維度正確)。    
雖然之前提過矩陣不符合交換律，但是在單位矩陣中是成立的。

### Inverse and Transpose
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/FuSWY/inverse-and-transpose)
![](https://i.imgur.com/THWojhg.png)

#### Inverse
![](https://i.imgur.com/sI7wkxn.png)

實數有一個特性，就是有一個倒數，這個倒數乘上這個實數就會得到一個『單元1』(Identity 1)，舉例來說3的倒數就是$3^{-1}$，$3*\frac{1}{3}=1$，這個倒數，也就是逆。    

矩陣的逆：$A*(A^{-1})=A^{-1}*A=I$，只有在維度為(mxm的情況下才會有逆矩陣，這種矩陣又稱方陣)，當一個矩陣沒有逆的時候又稱為奇異矩陣
 
註：**非所有的實數都有倒數，0沒有。**
#### Transpose
![](https://i.imgur.com/FifjfC4.png)

矩陣的轉置會在矩陣名稱上有個『T』(如上圖範例)，取得轉置的方式如下：
1. 取矩陣的第一行(row)變為轉置矩陣的第一列(column)
2. 取第二行(row)變為轉置矩陣的第二列(column)....so on

A(mxn)，轉置矩陣B(nxm)，即元素$B_{ij}=A_{ji}$
