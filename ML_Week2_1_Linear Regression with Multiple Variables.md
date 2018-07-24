# 吳恩達_機器學習_第2週_1_Linear Regression with Multiple Variables
###### tags: `andrew` `machine learning` `coursera`
[課程連結](https://www.coursera.org/learn/machine-learning/home/week/2)
## Multivariate Linear Regression
### Multiple Features
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/6Nj1q/multiple-features)    
![](https://i.imgur.com/0DmN9tr.png)

#### Multiple features(variables)
![](https://i.imgur.com/pt9tgMT.png)    
![](https://i.imgur.com/aLgSVo9.png)

上章節中利用單一變量(特徵)來預測房價，本週要說明利用多變量(特徵)。    

符號約定：    
* 下標為指定特徵
    * $x_1$:size
    * $x_2$:Number of bedrooms
    * ...so on
* y:預測輸出變量
* n:特徵數量
    * 上圖例為4
* m:資料樣本數
* 上標為第i筆資料
    * $x^{(1)}$:代表第1筆資料的特徵向量
* $x^{(i)}_j$:代表第i筆資料的第j個特徵

#### Hypothesis
![](https://i.imgur.com/HBSgUPz.png)    
![](https://i.imgur.com/s02PyvR.png)

多變量線性迴歸為：$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$    

實務上，我們會加了一個$x_0=1$，所以會有n+1個特徵($x\in R^{n+1}$)。(為了向量計算)    
$h_\theta (x) = \theta_0x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n=\theta^Tx$

如下：    
$\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}$
### Gradient Descent for Multiple Variables
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/Z9DKX/gradient-descent-for-multiple-variables)    
![](https://i.imgur.com/z5tRqFa.png)

#### Hypothesis
![](https://i.imgur.com/bwVEjdK.png)

多變量線性迴歸除了Hypothesis改變之外，Cost function也有所變化，但不管如何，你以向量化的角度來看它們的時候，其實整體的數學式並沒有變化多少。    

1. Parameters(參數)的部份看成向量$\theta$，一個有n+1維的向量
2. $J(\theta_0,\theta_1,....\theta_n)=...$看成是 $J(\theta)$
    * $J(\theta) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x^{(i)}) - y^{(i)} \right)^2$
3. Gradient descent:$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$
    * 迴圈迭代更新
#### Gradient Descent 
![](https://i.imgur.com/2uaSrOo.png)

在多變量線性迴歸的梯度下降中，$\theta_0$的部份，因為$x_0$為1，故整個式子是等價於單變量線性迴歸的$\theta_0$梯度下降。


$\begin{align*} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align*}$



### Gradient Descent in Practice I - Feature Scaling
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)    
![](https://i.imgur.com/6ztVD4W.png)

#### Feature Scaling
![](https://i.imgur.com/EfWSzZr.png)

多個特徵的情況下，如果可以讓特徵的在相近的範圍內的話，那梯度下降就可以更快速的收斂。    

上例來看，兩個特徵，一個是房子大小，一個是房間數，沒有重新縮放大小的情況下，那成本函數會是一個又瘦又高的橢圓形狀(圖左)，這造成需要更多的時間來收斂。    

如果我們將特徵都除上一個最大值做一個統一的縮放(縮放至0與1之間)，那成本函數會是變為圖右，這種情況下，梯度下降會收斂的更為快速。
#### Feature Scaling
![](https://i.imgur.com/pr77WuN.png)

我們通常將特徵縮放至-1至1的範圍內，但不用太過於介意-1至1，如果特徵的範圍不是太超過的話，理論上都是可以直接使用，但如果像-100至100的這區間，那可能就需要做一個縮放了。

註：andrew認為，-3至3之間的範圍是可接受不做縮放的一個值。
#### Mean normalization
![](https://i.imgur.com/2NZDp9h.png)

部份時候也可以透過Mean normalization(均值歸一)來縮放數值，這作法可以讓均值為0，作法如下：    
$x_i=\frac{x_i-\mu}{s}$    

註：s取該特徵最大、最小的差異，或標準差    
註：特徵縮放不用太精確，只是要讓梯度下降可以更快速收斂 

### Gradient Descent in Practice II - Learning Rate
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/3iawu/gradient-descent-in-practice-ii-learning-rate)    
![](https://i.imgur.com/vHoAzhO.png)

#### Making sure gradient descent is working correctly
![](https://i.imgur.com/Z35IcxP.png)
![](https://i.imgur.com/wcW9uvi.png)

通常我們的透過迭代次數(x軸)與成本函數(y軸)的走勢圖來確認成本函數是否確實的收斂，如果梯度下降有確實的收斂的話，那每次的迭代它會應該會向下才對。

#### Summary
![](https://i.imgur.com/ul7keHM.png)

另外一種方式，就是進行收斂測試，透過設置閥值，當梯度下降小於閥值的時候就代表收斂完成，但實務上這個閥值的設置是非常困難的。

### Features and Polynomial Regression
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/Rqgfz/features-and-polynomial-regression)    
![](https://i.imgur.com/mwjqOzb.png)

#### Housing prices prediction
![](https://i.imgur.com/GxrnRKj.png)

以房價預測為例，兩個特徵，各別為房屋的寬與深，在建置模型的時候，我們並不一定只能應用這兩個特徵直接學習，可以自己『創造特徵』，此例我們可以創造一個『面積』的特徵。
#### Polynomial regression
![](https://i.imgur.com/s6xYVUk.png)

另一種作法是多項式迴歸，將特徵延伸至多項式，上例是取二次方(藍色線)，但二次方有一個缺點，就是它會往下，因而選擇三次方(綠色線)，愈高次方可能會有更大的數值區間，因此在使用多項式特徵的時候更需要使用特徵縮放。
#### Choice of features
![](https://i.imgur.com/R63hdxp.png)

二次方的多項式迴歸有著會向下的特性，除了採三次方之外，另一個方式不取二次方，改取開平方的特徵，平方根的特性是一直向上，雖然上升緩慢，但是能確保不會下降。
## Computing Parameters Analytically
### Normal Equation
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/2DKxQ/normal-equation)    
![](https://i.imgur.com/56Ta92E.png)

#### Section_1
![](https://i.imgur.com/dd9GI0w.png)
![](https://i.imgur.com/irSWAKn.png)

不同於梯度下降需透過多次迭代來取得最佳解，正規方程式(Normal Equation)可以一次求得最佳解。    
計算最佳解的方式就是導數為0，所以我們要做的就是計算出導數為0的時候的$\theta$的值了。
#### Section_2
![](https://i.imgur.com/mKcp61n.png)

舉例來說，資料集有四筆資料，四個特徵，補上$x_0=1$，以此建置一個矩陣X(包含$x_0$(m,n+1))，再設置一個向量y(實際數值m-dimension)，以下面數學式計算，即可求得最佳解    
* $\theta = (X^T*X)^{-1}X^T*y$

註：-1，代表矩陣的逆
註：T，代表矩陣轉置
#### Section_3
![](https://i.imgur.com/cvWbf0b.png)

$x^{(i)} \in R^{(n+1)}$，利用『設計矩陣(design matrix)』的概念，將資料集轉為(m, n+1)的矩陣，再以此矩陣來計算出$\theta$

#### Section_4
![](https://i.imgur.com/9VQWy5q.png)

在octave中如下處理
```
pinv(X'*X)*X'*y
>> pinv(X'*X)*X'*y
ans =

   188.40032
     0.38663
   -56.13825
   -92.96725
    -3.73782
```

註：$A=(X^T X)$，那$(X^T X)^{-1}$可稱為矩陣A的逆!
註：使用正規方程式來求解並不需要做特徵縮放
#### Section_5
![](https://i.imgur.com/nvdLNtJ.png)

老師也有在這部份給了說明應用環境
|Gradient Descent|	Normal Equation|
|-----|-----|
|Need to choose $\alpha$	|No need to choose alpha|
|Needs many iterations	|No need to iterate|
|O (kn2)	|O (n3), need to calculate inverse of XTX|
|Works well when n is large	|Slow if n is very large|
|需標準化|不需要標準化|

要注意到的是，透過正規方程式求解的時間複雜度為n的3次方(O(n^3))，這在特徵量大的時候是非常恐怖的一件事，並且一個n\*n維的矩陣在計算的時候所需的記憶體空間也是一個問題。    

註：特徵數10000是一個選擇的門檻
### Normal Equation Noninvertibility
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/zSiE6/normal-equation-noninvertibility)    
![](https://i.imgur.com/kpvWVOg.png)

#### Section_1
![](https://i.imgur.com/3Nt9X6d.png)

**選修課程_矩陣的不可逆**

通常我們稱不可逆的矩陣為奇異矩陣，或退化矩陣，普遍來說，不可能的問題很少發生。

註：octave中計算逆的是`pinv`與`inv`
#### Section_2
![](https://i.imgur.com/De65qcX.png)

可能發生不可逆的原因：
* 在特徵選擇中有著多餘的特徵
    * $x_1$是英尺、$x_2$是平方米，$x_1$與$x_2$始終有3.28的倍率關聯
        * $x_1=(3.28)^2*x_2$
        * 多元共線性?
        * 特徵間的相關系數過高
    * 適時的刪除多餘的特徵
* 資料集數量(m)<特徵數(n)的時候
    * 可透過正規化來處理

