# 吳恩達_機器學習_第1週_1_Linear Regression with One Variable
###### tags: `andrew` `machine learning` `coursera`
## Model and Cost Function
[課程連結](https://www.coursera.org/learn/machine-learning/home/week/1)
### Model Representation
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/db3jS/model-representation)
![](https://i.imgur.com/t2norI7.png)

#### Section_1
![](https://i.imgur.com/D0gVcTu.png)

範例是一個房子尺寸與房價的數據集，而你的朋友有間房子尺寸為1250，可以賣多少?    
這是一個迴歸類的監督式學習(Supervised Learning)，因為我們『給出了正確的答案』，而這個答案是『數值』    
#### Section_2
![](https://i.imgur.com/0D1fA1g.png)

監督式學習中有一個資料集，稱為『訓練資料集』。我們的任務就是從這資料集來學習預測房屋價格。   
符號約定(Notation)：
* m：資料集數量
* x：輸入的變數/特徵
* y：輸出的變數/目標
* $x^{(i)}$：上標i代表m中的第i筆資料

#### Section_3
![](https://i.imgur.com/sGTKzJI.png)

機器學習的過程就是提供資料集由演算法學習到一個函數(h:hypothesis)，這個h再根據輸入的x來得到y。    
h就是一個由x到y的映射函數，$h_\theta(x)=\theta_0+\theta_1x$，這個模型即為『線性迴歸_Linear regression』，又因為它只有一個變數，又可以稱為『單變量線性迴歸』，『簡單線性迴歸』    
### Cost Function
[課程連結]()
![](https://i.imgur.com/uEMtyiz.png)

#### Section_1
![](https://i.imgur.com/yjz97kx.png)

hypothesis=$h_\theta(x)=\theta_0+\theta_1x$    
* $\theta$:模型參數
#### Section_2
![](https://i.imgur.com/VvpgCJz.png)

上圖說明著在各種不同參數情況下會有的線性呈現    
#### Section_3
![](https://i.imgur.com/sAcZM4E.png)

在線性迴歸中，我們會有一個資料集如左上圖分佈，而我們要做的就是得到一組參數來畫出一條線並盡可能的擬合這些數據分佈。也就是我們的hypothesis($h_\theta$)要讓我們映射出的y能更接近，即是兩者之間的差異必需是很小的。    

兩者之間的差異總合：    
$\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$    
其中$h_\theta(x^{(i)})=\theta_0+\theta_1x^{(i)}$，這代表$h_\theta(x)$會因為兩個參數的變動而變動，所以我們要的就是得到一個能夠最小化兩者差異的$\theta_0$與$\theta_1$。    

成本函數\_$J(\theta_0, \theta_1)$：    
* Cost Function
    * 此計算模型亦稱平方誤差函數
    * $J(\theta_0, \theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$   
        * $\frac{1}{2m}$主要用來減少平方之後的數值。 

目標：    
* 最小化成本函數，尋得最佳的$(\theta_0, \theta_1)$

m:dataset
### Cost Function-Intuition I
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/N09c6/cost-function-intuition-i)
![](https://i.imgur.com/EVxu4er.png)

#### Section_1
![](https://i.imgur.com/AgvsmXX.png)

我們想找一條線來擬合數據，因此有了相關數學式的定義：
* hypothesis:
    * $h_\theta(x)=\theta_0 + \theta_1x$
* parameters:
    * $\theta_0,\theta_1$
    * 透過不同的參數會有不同的擬合
* cost function:
    * $J(\theta_0, theta_1)=\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})+y^{(i)})^2$
* goal:
    * minimize$J(\theta_0, \theta_1)$

為了能夠更直觀的來說明，會假設$\theta_0$=0(註一)，即我們的hypothesis為$h_\theta(x)=\theta_1x$

註一：代表該線性迴歸會通過原點座標(0, 0)
#### Section_2
![](https://i.imgur.com/q97VlCh.png)

我們假設$\theta_1$=1，即$h_\theta(x)=1*x$    
左圖我們將x為1、2、3與經過hypothesis所映射出的y計算成本函數，會得到0，將這結果畫到右邊$\theta_1$=1與$J(\theta_1)=0$。
#### Section_3
![](https://i.imgur.com/djKmmXe.png)

接下來，假設$\theta_1$=0.5，再計算所映射出的y來計算成本函數，會得到0.58，將這結果一樣的畫到右邊的座標圖上。    
$J(1)=0,J(0.5)=0.58,J(1.5)=0.58,J(0)=2.33,J(2)=2.33,J(-0.5)=5.25,J(2.5)=5.25$，依序的求出成本函數之後會得到右圖左線曲線，不同的參數對應到不同的成本。    

回到一開始我們所提的，我們要做的就是找到一個$\theta_1$來將$J(\theta_1)$最小化，此例來看，似乎$\theta_1$=1存在著最佳解。    

下一節會帶入$\theta_0$，對成本函數會有更直觀的瞭解。

註：成本函數所計算的即為藍色線段的平方和(點($y^{(i)}$)到點($h_\theta(x^{(i)})$)的距離)
### Cost Function-Intuition II
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/nwpe2/cost-function-intuition-ii)
![](https://i.imgur.com/0kJsl6p.png)

#### Section_1
![](https://i.imgur.com/PlRxZXB.png)

這節我們會利用輪廓圖來呈現兩個參數($\theta_0,\theta_1$)下的成本函數
#### Section_2
![](https://i.imgur.com/gtWjOjG.png)
![](https://i.imgur.com/MmJyB2N.png)
假設我們的hypothesis=50+0.06x，左圖上我們就會得到黑直線走勢，但因為我們用了兩個參數，所以成本函數必需以多維度的方式來呈現，基本上還是呈現一個弓形，兩個軸分別代表$\theta_0,\theta_1$，另一軸則代表成本。
#### Section_3
![](https://i.imgur.com/c205LGM.png)

為了方便理解，我們將3D的成本函數以輪廓圖來表示(上圖右)    

兩軸分別為兩個參數($\theta_0,\theta_1$)，每一個圓圈上都是$J(\theta_0,\theta_1)$相同的集合，意思就是在相同圓圈上的點即使它的參數不同，但它的成本是相同的(如三個粉紅色的點)。    
然後我們觀察紅色點，此點即代表左邊所擬合的線，很明顯的這個hypothesis並不是很好，距離成本函數的最低(中心)還有點距離。
#### Section_4
![](https://i.imgur.com/bqrORme.png)
![](https://i.imgur.com/zM4YZxF.png)
我們找尋另一個點$\theta_0=360,\theta_1=0$，對應左邊的圖來看似乎有好一點，最後我們找到一個離中心最接近的點，它能夠最小化每一個點跟線的距離平方和(即使它也許不是最佳)。    

這個過程是尋找一個hyothsis來取得最小成本的一個直觀的說明，而我們希望能夠讓它自動找到最佳參數來取得最小成本，後面就會說到這個觀念。

## Parameter Learning
### Gradient Descent
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/8SpIM/gradient-descent)
![](https://i.imgur.com/eCS7n3c.png)

#### Section_1
![](https://i.imgur.com/gCiYAW3.png)

梯度下降(Gradient Descent)並非只能應用於線性迴歸，事實上整個機器學習都可以看的到它的影子，而成本函數的收斂也並非只能應用於兩個參數，只是課程上只會以兩個參數來說明。    
作法如下：    
1. 初始化參數
    * 通常初始化為0
2. 不停改變參數去減少成本函數直到最小化(局部或全域)
#### Section_2
![](https://i.imgur.com/HRonW60.png)

直觀的來觀察成本函數的收斂過程，我們初始化了參數，對應成本函數，也許在山頂，然後觀察四週之後往下走一步，一直到山底(也許全域最低，也許區域最低)。    
一個特點是，隨著初始化的不同，最開始的位置也不同，也有可能造成得到不同解。    
#### Section_3
![](https://i.imgur.com/4hrb4Ir.png)

梯度下降公式：    
$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1)$    
$\alpha:$稱為學習效率(learning rate)，控制每次下降的步幅    
$\frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1):$微分項，後面會談    
實現梯度下降的一個重點是，**所有的參數必需計算完之後才可以更新**。    

註：:=代表賦值
註：左下區為正確的梯度下降，右下區為錯誤
### Gradient Descent Intution
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/GFFPB/gradient-descent-intuition)
![](https://i.imgur.com/RZq8PJQ.png)

#### Section_1
![](https://i.imgur.com/Bn3mgHV.png)

為了能夠直觀的以圖面來說明，範例上會以一個參數$\theta_1$來說明。

註：$\alpha:$學習效率(learning rate)，控制參數更新的幅度
#### Section_2
![](https://i.imgur.com/7e0mB9m.png)

左上圖是$J(\theta_1)(\theta_1\in R)$的呈現，右線最高點是我們初始化$\theta_1$的地方。    
$\theta_1:=\theta_1-\alpha\frac{\partial}{\partial\theta_1}J(\theta_1))$    
求導的過程，我們取點的切線(紅線，與曲線相切)，並計算該線的斜率，其斜率為高除於寬，這邊會得到一個正斜率。    
套進公式不難理解，學習效率為正，導數為正，$\theta_1$減掉正數之後會變小，因此會往左移動接近最小值。    

左下圖，我們從另一邊取初始化參數，以相同的方式求導，這時候是負斜率，也因此$\theta_1$後面是加項，會增加，因此會往右移動接近最小值。
#### Section_3
![](https://i.imgur.com/STSAF03.png)

* 右上圖：
過小的學習效率$\alpha$會造成需要很多次的迭代才有辦法到達最低
* 右下圖：
過大的學習效率會像彈力球般彈跳，可能永遠無法達到最低
#### Section_4
![](https://i.imgur.com/TzoIVfU.png)

一個很有趣的是，如果初始化的參數剛好落在區域最佳解(local optima)的話，那它就不會再有任何的收斂，因為它的斜率已經是0了。
$\theta_1-\alpha*0=\theta_1$
#### Section_5
![](https://i.imgur.com/ZyRhpLF.png)

隨著梯度下降不斷的收斂會發現，導數(斜率)愈來愈小，也會讓整個收斂愈來愈慢，但是它終究可以收斂到一個局部極小值，這證明了即使我們固定住學習率效它還是可以有效的收斂。
### Gradient Descent For Linear Regression
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/kCvQc/gradient-descent-for-linear-regression)
![](https://i.imgur.com/CH4Zdem.png)

#### Section_1
![](https://i.imgur.com/4HO2QaB.png)

目前我們有hypothesis，有成本函數，有梯度下降，現在要將它應用在線性迴歸上，以此最小化誤差平方和。
#### Section_2
![](https://i.imgur.com/o0ciDhP.png)

$\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)=\frac{\partial}{\partial\theta_j}*\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2=\frac{\partial}{\partial\theta_j}*\frac{1}{2m}\sum_{i=1}^m(\theta_0+\theta_1x^{(i)}-y^{(i)})^2$    
$\theta_0:\frac{\partial}{\partial\theta_0}J(\theta_0, \theta_1)=\frac{1}{m}\sum^m_{i=1}(h\theta(x^{(i)}-y^{(i)}))$    
$\theta_1:\frac{\partial}{\partial\theta_1}J(\theta_0, \theta_1)=\frac{1}{m}\sum^m_{i=1}((h\theta(x^{(i)}-y^{(i)}))*x^{(i)})$    
#### Section_3
![](https://i.imgur.com/syKY926.png)

最終得到的梯度下降式子如上，加入了學習效率做迭代更新，並且要注意，梯度下降要所有的參數同時進行計算之後同時更新。
#### Section_4
![](https://i.imgur.com/0CH16Pn.png)

線性迴歸的成本函數始終呈現碗形(bowl shape)，這稱為convex function，它沒有區域最佳，只有全域最佳，這種情況下去使用梯度下降是可以有效取得全域最佳解。
#### Section_5
![](https://i.imgur.com/tRxrI05.png)

目前我們所學的梯度下降又稱為批次梯度下降(Batch Gradient Descent)，它代表每次的迭代訓練都將所有的訓練資料集考慮進去計算，後續會提到其它模式的梯度下降。