# **Machine Learning (APA) Course Individual Problems**

This repository contains some solved problems for the 'Machine Learning' (APA) course at FIB in 2021-22 Q1. For the project of this course check this [repository](https://github.com/DavidNogales/fib-apa-project-skillcraft).

**Author:**

David Nogales Pérez

## **Assignments**

Each assignment may contain a PDF file or a `.ipynb` with the respective code or both (If necessary, the needed dataset is included as well). All the statements (with their respective grade) are described down below:

### **Assignment 1-Min Squares (grade: 10/10)**

---------------

We have a sequence of real numbers <!-- $x_1,...,x_n$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=x_1%2C...%2Cx_n">. We define the function:

<!-- $$
f(x)= \frac{1}{n} \sum_{i=1}^n (x-x_i)^2
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20(x-x_i)%5E2%0D"></div>

(a) Demonstrate that <!-- $f$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=f"> has a unique minimum, find it and interpret the result.

(b) Consider the new function:

<!-- $$
f(x)= \sum_{i=1}^n p_i (x-x_i)^2
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=f(x)%3D%20%5Csum_%7Bi%3D1%7D%5En%20p_i%20(x-x_i)%5E2%0D"></div>

Where each term is weighted by a factor <!-- $p_i > 0$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=p_i%20%3E%200">, having that <!-- $\sum_{i=1}^n p_i =1$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5En%20p_i%20%3D1">. Recalculate the solution.

(c) **[PROG]** Let <!-- $n=100$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%3D100">. Apply the obtained result to a sequence of numbers chosen by you. Use the same weights based on uniform independent numbers in <!-- $(0,1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=(0%2C1)">. Plot the function with the minimum.

See the original statement [here](imgs/a1-min_squares.png).

### **Assignment 2- [PROG] Traveling USA (grade: 9.5/10)**

---------------

The following table shows the distances (in miles) between Baltimore and other 12 cities of the EE.UU., along with their corresponding airplane's ticket price (in dollars).

| City       | Distance | Fare |
|------------|----------|------|
| Atlanta    | 576      | 178  |
| Boston     | 370      | 138  |
| Chicago    | 612      | 94   |
| Dallas     | 1216     | 278  |
| Detroit    | 409      | 158  |
| Denver     | 1502     | 258  |
| Miami      | 946      | 198  |
| NewOrleans | 998      | 188  |
| NewYork    | 189      | 98   |
| Orlando    | 787      | 179  |
| Pittsburgh | 210      | 138  |
| St.Louis   | 737      | 98   |

(a) Pose and solve numerically the problem of predicting the Fare with the Distance using the ***LinearRegression()*** model. Plot a graph using the given data and the obtained solution.

(b) Observe that some cities have abnormally low fares taking into account the distance in which are located. Design a way of reducing the influence of this cases and recalculate the solution.

See the original statement [here](imgs/a2-trav_usa.png).

### **Assignment 3- [PROG] Be Discrete (grade: 9/10)**

---------------

There are many preprocesses that can be applied to the attributes of a dataset. We are used to working with continuous data, but sometimes it is easier to cross to the discrete world for different reasons. In this problem we are going to experiment with different discretizations.

The ***appendicitis*** dataset is included in the Penn Machine Learning Benchmarks. In order to download it you will have to follow the instructions from their [web page](https://epistasislab.github.io/pmlb/index.html).

(a) We are going to start with the original dataset. Split the data into training and test (60%/40%), the partition has to be stratified (fix also the random state for reproducibility).

(b) Train a naive bayes, a logistic regression and a K-nearest neighbours exploring their hyper-parameters as you see convenient. Preprocess first the data adequately for these models. Obtain the crossvalidation error, the test error, confusion matrix and classification report.

(c) Scikit-learn has a preprocess able to discretize continuous data (***KBinsDiscretizer***). This preprocess can convert each variable to a set of discrete values in different ways. The more adequate methods are the ones that try to approximate in some way the distribution of the original data. The quantile discretization uses the quantiles of the distribution assuming gaussianity. The kmeans discretization does not assume a specific distribution and looks for a number of compact densities in the data. We are going to apply both methods to all the attributes for obtaining a discretization into 2, 3 and 4 values (use onehot -dense as encoding so each value is now an attribute of the dataset). Fit the same models as before considering that now you have binary variables and compare the different results. Has this process affected to the quality of the models? Which one would you chose and why?

(d) Considering the interpretability of the models, do you think that has advantages to work with discretized data? why?

See the original statement [here](imgs/a3-be_discrete.png).

### **Assignment 4- [PROG] MPL 3 (grade: 10/10)**

---------------

You are tasked with the identification of a continuous non-linear system **SISO**(single-input/single-output), from a number of points obtained by consecutively sampling the dynamics <!-- $u \rightarrow y$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u%20%5Crightarrow%20y">. Define:

<!-- $$
y(k)= y_1(k-1)+y_2(k-1)
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=y(k)%3D%20y_1(k-1)%2By_2(k-1)%0D"></div>

Where

<!-- $$
\begin{align}
y_1(k) = 2.5 y(k)sin(\pi e^{-u^2(k)-y^2(k)}) \\
y_2(k) = u(k)(1+u^2(k))
\end{align}
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign%7D%0D%0Ay_1(k)%20%3D%202.5%20y(k)sin(%5Cpi%20e%5E%7B-u%5E2(k)-y%5E2(k)%7D)%20%5C%5C%0D%0Ay_2(k)%20%3D%20u(k)(1%2Bu%5E2(k))%0D%0A%5Cend%7Balign%7D%0D"></div>

Where the output <!-- $y(k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y(k)"> depends on the previous input <!-- $u(k-1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u(k-1)"> and output <!-- $y(k-1)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y(k-1)">. Train a MLP network to learn the task. Generate a learning set of size <!-- $n=500$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=n%3D500"> having <!-- $y(0)=0$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=y(0)%3D0"> and aleatory exciting the system using a signal <!-- $u(k)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=u(k)"> uniformly sampled in <!-- $[-2,2]$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5B-2%2C2%5D">. You will need to estimate the best architecture, which can be done with ***cross-validation*** and regularization using your own criteria.

See the original statement [here](imgs/a4-mlp3.png).

### **Assignment 5-Polynomial Kernel (grade: 10/10)**

---------------

Consider the polynomial kernel <!-- $k(x,x')=(\langle x,x'\rangle+c)^q, q \in \mathbb{N}, c \leq 0, x,x' \in \mathbb{R}^d$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=k(x%2Cx')%3D(%5Clangle%20x%2Cx'%20%5Crangle%2Bc)%5Eq%2C%20q%20%5Cin%20%5Cmathbb%7BN%7D%2C%20c%20%5Cleq%200%2C%20x%2Cx'%20%5Cin%20%5Cmathbb%7BR%7D%5Ed">. Do the "Kernel Trick" (expression of 'k' as scalar product) for <!-- $d=2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=d%3D2"> and <!-- $q=3$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=q%3D3"> and explicitly find the function <!-- $\phi$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cphi"> in this case. What is the dimension of the feature space to which takes <!-- $\phi$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cphi">?

See the original statement [here](imgs/a5-kernel_pol.png).
