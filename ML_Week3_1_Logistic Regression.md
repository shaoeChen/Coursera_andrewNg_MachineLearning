# 吳恩達_機器學習_第3週_1_Logistic Regression
###### tags: `andrew` `machine learning` `coursera`
[課程連結](https://www.coursera.org/learn/machine-learning/home/week/3)    

[TOC]    

## Classification and Representation
### Classification
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/wlPeP/classification)    
![](https://i.imgur.com/c4jAdfV.png)

#### Classification
![](https://i.imgur.com/MGp6KpW.png)

上面列出是幾個分類的例子，是否為垃圾郵件、是否為詐欺、良性或惡性腫瘤。    

註：通常0表示負，1表示正

#### Classification
![](https://i.imgur.com/6RwxG7x.png)

以良性或惡性腫瘤為例，類別上只會有良性(1)或惡性(0)，以我們學過的線性迴歸，也許會得到一條斜線。    
然後我們在垂直軸上設置一個閥值(0.5)，大於閥值的為1，小於為0，空間上你會得到一個點跟非常完美的切割。    
但是我們再加入一個資料集(最右邊)會發現，整個線性迴歸被影響了(變更為藍線)，以一樣的方式來判斷的時候原本是惡性腫瘤的變良性了，這並不是一個好的結果。    
也因此用線性迴歸來解分類的問題並不是一個好的方式。    

#### Classification
![](https://i.imgur.com/ThYfAdG.png)

分類問題是y={0, 1}，但線性迴歸所得的卻是大於1或小於0，這似乎有點怪，所以我們會利用邏輯斯迴歸，它的迴歸值總是介於0與1之間
### Hypothesis Representation
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/RJXfB/hypothesis-representation)    
![](https://i.imgur.com/4ZuVEEA.png)

#### Logistic Regression Model
![](https://i.imgur.com/RlI1VKw.png)

調整線性迴歸的式子如下：    
* $h_\theta (x) = g ( \theta^T x )$
    * $h_\theta (x) = \dfrac{1}{1 + e^{-\theta^Tx}}$
    * $z = \theta^T x$
    * $g(z) = \dfrac{1}{1 + e^{-z}}$

$g(z)$隨著z接近正、負無窮，會愈接近1與0，以此方式確保我們的輸出不會超過0與1。

註：Sigmoid function，義同Logistic function

#### Interpretation of Hypthesis Output
![](https://i.imgur.com/MOgnHTP.png)

這個hypothesis的意思即『當我們輸入x之後得到y=1的機率有多少』，也就是$h_\theta(x)=P(y=1|x:\theta)$。    
上圖案例為腫瘤預測，特徵是腫瘤尺寸，輸入之後得到的$h_\theta(x)=0.7$，這代表有70%的機率是惡性。    
相反的，是良性的機率即為$1-P(y=1|x:\theta)$，因為機率的總合為1。

### Decision Boundary
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/WuL1H/decision-boundary)    
![](https://i.imgur.com/4ekQlj3.png)

#### Logistic regression
![](https://i.imgur.com/8qRV5EK.png)

$h_\theta(x)=g(\theta^Tx)=P(y=1|x:\theta)$    
$g(z) = \dfrac{1}{1 + e^{-z}}$    

假設邏輯斯迴歸輸出如下：    
$\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}$    

* $y=1$
    * $z \geq 0$
        * $h_\theta(x) = g(\theta^T x) \geq 0.5$
        * 從sigmoid來看(右上圖)，當$z \geq 0$，則所得的輸入就會$\geq 0.5$，也就是輸出為$1$

* $y=0$
    * $z < 0$
        * $h_\theta(x) = g(\theta^T x) < 0.5$
        * 從sigmoid來看(右上圖)，當$z < 0$，則所得的輸入就會$< 0.5$，也就是輸出為0

註：$z=\theta^T x$
#### Decision Boundary
![](https://i.imgur.com/RcSQYRT.png)

現在談決策邊界(Decision Boundary)，左上圖是資料集的分佈，$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2)$    

上一張投影片我們談到了$z=\theta^T x \geq 0 , y=1$，此例來看，$-3+x_1+x_2 \geq 0, y=1$，即$x_1+x_2 \geq 3$，將此函數繪製出可見其線性(左上圖粉線)    

這時候平面被切割成兩部份，線的右側是$y=1\rightarrow(x_1+x_2 \geq 3$)的區域，而線的左邊是$y=0\rightarrow(x_1+x_2 < 3$)，這條將平面切割的線即為決策邊界(Decision Boundary)
#### Non-linear decision boundaries
![](https://i.imgur.com/9yWkIEO.png)

如上圖左的資料分佈，這並不是一個線性可分的資料集，這時候可以應用更複雜的模型，如之前提過的多項式(Polynomial)，就可以處理更複雜的決策邊界，而非只能處理線性可分。    

**決策邊界本身不是訓練數據集的屬性，而是模型(Hopythesis)及其參數的屬性，在確認參數的同時，決策邊界也決定了**    

數據只是數據，決定這個模型的還是在於你的參數設置，而參數透過訓練數據集來擬合模型。

註：x:正樣本;o:負樣本


## Logistic Regression Model
### Cost Function
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/1XG8G/cost-function)    
![](https://i.imgur.com/zXrWGMF.png)

#### How to choose parameters?
![](https://i.imgur.com/gyDRQmF.png)

我們已經知道了Logistic Regression的基礎，現在的問題是該如何選擇(擬合)參數$\theta$
#### Cost function
![](https://i.imgur.com/ivWSStC.png)

擬合參數從成本函數說起，在之前定義過線性迴歸的成本函數，但如果計算Logistic Regression的時候以相同的成本函數來處理會造成上圖左的狀況(non-convex)，這會有太多的區域最佳解，最佳狀況當然還是上圖右的convext。

註：造成non-convex的原因在於$h_\theta(x)$是一個non-linear的函數(sigmoid)
#### Logistic regression cost function
![](https://i.imgur.com/eHLdLIj.png)
![](https://i.imgur.com/czvwbLg.png)

Logistic Regression Cost Function如下：  
$\begin{align*}&  \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}$

log屬遞增函數，所以log(z)(z為$h_\theta(x)$)所得線性如上圖右藍線，因此-log(z)會趨勢會反過來如紅線。

* y=1
    * $-log(h_\theta(x))$
        * X軸是$h_\theta(x)$，介於0與1之間，y軸是成本函數$-log(h_\theta(x))$
        * $h_\theta(x)=1, y=1$則Cost=0
        * $h_\theta(x)=0, y=1$則$Cost\rightarrow\infty$

#### Logistic regression cost function
![](https://i.imgur.com/nVEJ176.png)

* y=0
    * $-log(1-h_\theta(x))$
        * X軸是$h_\theta(x)$，介於0與1之間，y軸是成本函數$-log(1-h_\theta(x))$
        * $h_\theta(x)=0, y=0$則Cost=0
        * $h_\theta(x)=1, y=0$則$Cost\rightarrow\infty$
### Simplified Cost Function and Gradient Descent
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/MtEaZ/simplified-cost-function-and-gradient-descent)    
![](https://i.imgur.com/HX4WEJP.png)

#### Logistic regression cost function
![](https://i.imgur.com/56mCCon.png)

上圖是Logistic regression的成本函數，在0與1的時候有不同的算式，現在就要把這兩個式子結合成一個式子    

$\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$    

* $y=1:-(1 - y) \log(1 - h_\theta(x))=0$
* $y=0:- y \log(h_\theta(x))=0$

#### Logistic regression cost function
![](https://i.imgur.com/ehiE5b3.png)

$J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$    
調整之後，整個式子可以如上，我們將負號取出，這個數學式是從統計學中的極大似然概率得來的(Maximum Likelihood)，而hypothesis的輸出就是概率值$P(y=1|x;\theta)$($x$以$\theta$為參數而$y=1$的機率)，重點是它是convex。    

有了成本函數之後，就是要去思考我們如何利用梯度下降來最小化成本，以訓練資料集來擬合參數$\theta$    
#### Gradient Descent
![](https://i.imgur.com/8LWLHfS.png)


$Cost Function$:    
$J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$    

Cost Function透過Gradient Descent不斷迭代更新學習參數如下：    

$\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*}$    

其中偏微分項的計算如下：    
$\dfrac{\partial}{\partial \theta_j}J(\theta)=\dfrac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)}))x_j^{(i)}$

將偏微分項的算式帶入Gradient Descent：    
$\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}$

仔細一看會發現，這個部份與之前學習線性迴歸的時候所得的式子是相同的，但在$h_\theta$的定義上還是有差異：    
* Logistic Regression
    * $h_\theta=\sigma(\theta^Tx)$
* Linear Regression
    * $h_\theta=\theta^Tx$


透過向量來處理的話可以避免使用迴圈計算：    
$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$

### Advanced Optimization
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/licwf/advanced-optimization)    
![](https://i.imgur.com/KHBvhHu.png)

#### Optimization algorithm
![](https://i.imgur.com/DCKsKGp.png)
![](https://i.imgur.com/HIzYv7Z.png)


並非只有梯度下降可以最佳化學習參數，另外有更高階的演算法，但這已經超過本課程的範圍，只簡單說明優缺點。    
優點：    
1. 不需手動設定學習參數
2. 收斂的速度較梯度下降來的快
缺點：    
1. 更複雜的計算

註：後半段是老師演示利用octave求解的過程，視個人需求。
## Multiclass Classification
### Multiclass Classification: One-vs-all
[課程連結](https://www.coursera.org/learn/machine-learning/lecture/68Pol/multiclass-classification-one-vs-all)    
![](https://i.imgur.com/fk0RGAE.png)

#### Multiclass classification
![](https://i.imgur.com/0EEPihE.png)

多類別即指並非預測yes/no，而是有多個預測目標，案例如上圖：    

註：多類別由0開始或1開始都不影響結果
#### Binary classification
![](https://i.imgur.com/Th2erGA.png)

上圖左為二元分類，單純兩個類別    
上圖右為多類別分類    

#### One vs all(One vs rest)
![](https://i.imgur.com/7IwSxHD.png)

作法上，是利用訓練資料集將它分成三個二元分類問題。    
* 第一個($h\theta^{(1)}(x)$)：
    * 將三角型視為一類(正)，其餘兩個視為一類(負)，訓練之後就得到一個決策邊界
* 第二個($h\theta^{(2)}(x)$)：
    * 將正方型視為一類(正)，其餘兩個視為一類(負)，訓練之後就得到一個決策邊界
* 第三個($h\theta^{(3)}(x)$)：
    * 將x視為一類(正)，其餘兩個視為一類(負)，訓練之後就得到一個決策邊界


#### One vs all
![](https://i.imgur.com/SDho7Mw.png)

再以此三個分類器來計算各自機率之後取最大機率即為該類別，如下數學表示式：    
\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}