Download Link: https://assignmentchef.com/product/solved-cs433-exercise-3-least-squares-ridge-regression-and-overfitting
<br>
Least Squares, Ridge Regression, and Overfitting

Goals.       The goal of this exercise is to

<ul>

 <li>Implement and debug least-squares.</li>

 <li>Implement, debug and visualize basis function models.</li>

 <li>Understand overfitting.</li>

 <li>Implement ridge regression.</li>

</ul>

Setup, data and sample code.               Obtain the folder labs/ex03 of the course github repository

<a href="https://github.com/epfml/ML_course/tree/master/labs/ex03">github.com/epfml/ML</a> <a href="https://github.com/epfml/ML_course/tree/master/labs/ex03">course</a>

We will continue to use the dataset height weight genders.csv as well as a new dataset dataEx3.csv in this exercise. We have provided sample code that already contains useful snippets of code required for this exercise.

You will be working in the notebook ex03.ipynb for the exercises of this week, by filling in the corresponding functions in the provided template code.

<h1>1           Least Squares and Linear Basis Functions Models</h1>

1.1       Least squares

Exercise 1:

<ul>

 <li>Fill in the notebook function least squares(y, tx) which implements the solution of the normal equations as discussed in the class. This function should return the optimal weights, and the mean-squared error.</li>

 <li>To debug your code, you can use the output of the last exercise. Run gradient descent or grid search on the height-weight data from the last exercise, and make sure you get a similar resulting <em>w </em>vector using all three methods.</li>

</ul>

This is a useful method to debug your code, i.e. first implementing a simple method and then using it to check more complicated methods. If you have not finished Exercise 2, please first finish implementing the grid search method. If you are lagging behind, do not worry. You will get the opportunity to catch up later, but it is important that you eventually take time to finish previous exercises.

1.2          Least squares with a linear basis function model

We will now implement and visualize a basis function model for the data dataEx3.csv.

As explained in the class, linear regression might not be directly suitable for nonlinear data. We will use polynomial basis functions to fit nonlinear data.

<em>φ<sub>j</sub></em>(<em>x</em>) := <em>x<sup>j                                                                                                                                        </sup></em>(1)

As we have seen in the lecture notes, the technique of feature expansion by the linear basis function model does allow us to still use linear regression techniques, to fit nonlinear data (recall that in our first simple setting, we assume that each input point is just one real value). As a result, we will be able to fit the data using different degrees of polynomials, e.g. a degree two polynomial (which is a linear combination of 1, <em>x </em>and <em>x</em><sup>2</sup>), or a degree three polynomial (which is a linear combination of 1, <em>x</em>, <em>x</em><sup>2 </sup>and <em>x</em><sup>3</sup>), etc.. Higher degree polynomials are more expensive to compute and to fit, but can capture finer details in the data, which results in more expressive models. Think about the pros and cons of choosing a very high or very low degree.

To measure the fit of our model, we will use a cost function called the Root-Mean-Square-Error (RMSE). It is related to MSE as follows:

RMSE(<em>w</em>) := <sup>p</sup>2 · MSE(<em>w</em>)                                                                            (2)

The magnitude of MSE can be difficult to interpret since it involves a square, while RMSE provides a more interpretable measure on the same scale as the error of one point. There are better measures in terms of statistical properties, like <em>R</em><sup>2</sup>, but we don’t need these for now. See the book “Introduction to Statistical learning” if you’re interested in more details.

Let us now implement polynomial regression, using the technique of linear basis functions, and visualize the predictions.

Exercise 2:

The goal of this exercise is to plot the data along with predictions using polynomial regression. Your goal is to find a good <em>w </em>using polynomial regression, when using polynomials of degrees 1, 3, 7, and 12 respectively. You might want to reuse the function from the previous exercise to calculate the RMSE.

<ul>

 <li>Fill in the notebook function build poly(x, degree). The input of this function is the vector of the data examples <em>x<sub>n </sub></em>∈ R for 1 ≤ <em>n </em>≤ <em>N</em>. As an output, the function must return the extended feature matrix</li>

</ul>

<em>φ</em>(<em>x</em>1)

<em>..</em>

<sup>                                                                    </sup>2         3                      degree

<strong>Φ</strong><sub>e </sub>:= <sub></sub><em>φ</em>(<em>x<sub>n</sub></em>)<sub> </sub>where <em>φ</em>(<em>x<sub>n</sub></em>) := [1<em>, x<sub>n</sub>, x<sub>n</sub>, x<sub>n</sub>, …, x<sub>n                         </sub></em>]

 <em>.. </em> <em>φ</em>(<em>x<sub>N</sub></em>)

that is the matrix formed by applying the polynomial basis functions to all input data, for the degree of <em>j </em>= 0 up to <em>j </em>=degree.

When finished, you must COPY your implementation to the separate file build polynomial.py for the plot function to work.

<ul>

 <li>If the code runs successfully, you will see the data and the fit. You will clearly see why linear regression is not a good fit, while polynomial regression produces a better fit.</li>

 <li>Filling in the notebook function polynomial regression(), you can see that RMSE decreases as we increase the degree of the polynomial. Does it mean that the fit gets better as we increase the degree? Which fit is the best in your view?</li>

</ul>

<h1>2           Evaluating Model Prediction Performance</h1>

The answer to the last question should be clear if you followed the lecture. If not, discuss with others and clarify.

In practice, it matters that predictions are good for unseen examples, not only for training examples. To simulate the reality, we will now split our dataset into two parts: <em>training </em>and <em>testing</em>. We will fit the data using training data and compute RMSE on both test and training data.

Exercise 3:

The notebook function train test split demo() is supposed to show the train and test splits for various polynomial degrees.

<ul>

 <li>To split the data, please fill in the notebook function split data(x, y, ratio, …). Do you think that the order of samples is important when doing the split?</li>

 <li>Fill in the notebook function train test split demo(). If the code runs successfully, you will see RMSE values printed for degrees 3, 7 and 12. For each degree, there are again three RMSE values which correspond to the following three splits of the data.

  <ul>

   <li>90% training, 10% testing</li>

   <li>50% training, 50% testing</li>

   <li>10% training, 90% testing</li>

  </ul></li>

 <li>Look at the training and test RMSE for degree 3. Does this makes sense? Why? Discuss with others if you are unclear.</li>

 <li>Now look at RMSE for other two degrees. Do these make sense? Why? Discuss with others if you are unclear.</li>

 <li>Which split is better? Why? Refer to the lecture notes if unclear.</li>

 <li>The test RMSE for degree 12 is ridiculously high for the split 10%-90%. Why do you think this is the case? The answer lies in numerical inaccuracies. Make sure you understand this.</li>

 <li>BONUS: Imagine you have 5000 samples instead of 50. Which split might be better in that situation?</li>

</ul>

<h1>3           Ridge Regression</h1>

The previous exercise shows overfitting when using complex models. Let us now correct it using Ridge Regression, as discussed in the class.

<ul>

 <li>Fill in the notebook function ridge regression(). You can debug your code by setting <em>λ </em>= 0. This should essentially give the same answer as least-squares code. You can also check that for large value of lambda, RMSE should be really bad.</li>

 <li>Play with the demo ridge regression demo() by choosing a split of 50%-50% and plot train and test errors vs <em>λ </em>for polynomial degree 7. You should get a similar plot as Figure 1.</li>

</ul>

Figure 1: Ridge Regression Demo.

<h1>Theory Exercises</h1>

<ol>

 <li>Warm-Up

  <ul>

   <li>Show that the sum of two convex functions is convex.</li>

  </ul></li>

</ol>

Hint: use the definition of convexity,

<em>f </em>: <em>X </em>→ R is convex                   ⇔ ∀<em>x</em><em>,y </em>∈ <em>X,</em>∀<em>λ </em>∈ [0<em>,</em>1] : <em>f</em>(<em>λ</em><em>x </em>+ (1 − <em>λ</em>)<em>y</em>) ≤ <em>λf</em>(<em>x</em>) + (1 − <em>λ</em>)<em>f</em>(<em>y</em>)<em>.</em>

<ul>

 <li>How do you solve the linear system <strong>A</strong><em>x </em>= <em>b</em>? When is it not possible, and why?</li>

</ul>

Hint: <a href="https://en.wikipedia.org/wiki/Invertible_matrix">Invertible matrix</a>

<ul>

 <li>What is the computational complexity of

  <ul>

   <li>Grid search?</li>

   <li>(one step of) Gradient Descent for linear regression with MSE cost?</li>

   <li>(one step of) Stochastic Gradient Descent for linear regression with MSE cost?</li>

  </ul></li>

</ul>

If needed, refresh your memory of the <a href="https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra">complexity of algebraic operations</a><a href="https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra">.</a>

<ul>

 <li>Consider a problem with two input variables, <em>x </em>= (<em>x</em><sub>1</sub><em>,x</em><sub>2</sub>), and one output variable <em>y</em>. Given the two samples below, find the coefficients <em>w </em>= (<em>w</em><sub>1</sub><em>,w</em><sub>2</sub>) of the linear relationship <em>x</em><sup>&gt;</sup><em>w </em>= <em>w</em><sub>1</sub><em>x</em><sub>1</sub>+<em>w</em><sub>2</sub><em>x</em><sub>2 </sub>= <em>y</em>.</li>

</ul>

<table width="188">

 <tbody>

  <tr>

   <td width="67"> </td>

   <td width="48"><em>x</em><sub>1</sub></td>

   <td width="40"><em>x</em><sub>2</sub></td>

   <td width="32"><em>y</em></td>

  </tr>

  <tr>

   <td width="67">Sample 1</td>

   <td width="48">400</td>

   <td width="40">-201</td>

   <td width="32">200</td>

  </tr>

  <tr>

   <td width="67">Sample 2</td>

   <td width="48">-800</td>

   <td width="40">401</td>

   <td width="32">-200</td>

  </tr>

 </tbody>

</table>

Do the exercise again, but with a slight change in the inputs: <em>x</em><sub>1 </sub>for sample 1 is now 401 instead of 400

<table width="188">

 <tbody>

  <tr>

   <td width="67"> </td>

   <td width="48"><em>x</em><sub>1</sub></td>

   <td width="40"><em>x</em><sub>2</sub></td>

   <td width="32"><em>y</em></td>

  </tr>

  <tr>

   <td width="67">Sample 1</td>

   <td width="48">401</td>

   <td width="40">-201</td>

   <td width="32">200</td>

  </tr>

  <tr>

   <td width="67">Sample 2</td>

   <td width="48">-800</td>

   <td width="40">401</td>

   <td width="32">-200</td>

  </tr>

 </tbody>

</table>

Compare the resulting <em>w </em>= (<em>w</em><sub>1</sub><em>,w</em><sub>2</sub>) for both cases. Familiarize yourself with the concept of <a href="https://en.wikipedia.org/wiki/Condition_number">condition </a><a href="https://en.wikipedia.org/wiki/Condition_number">number</a> as a way to diagnose ill-conditionning. You can find condition number calculators online or use <a href="https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.cond.html">numpy.linalg.cond</a><a href="https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.cond.html">)</a>.

<ol start="2">

 <li>Cost functions</li>

</ol>

A cost function defines how you evaluate a solution, and you might have different requirements depending on the problem. Using the MSE, if a your model makes an error of 5 on a sample, you add 25/2 to the cost of your model, regardless of the target. You might want to penalize this differently if you care about the relative error; an output of 1005 when 1000 was expected might be OK, but mistaking a 6 for a 1 might not. In this case, you can use a function that takes the relative error of the target <em>y<sub>n </sub></em>into account, like this one:

<em>.</em>

Where <em>f </em>is the model and <em> </em>is a small constant to avoid divisions by zero. Note that we have defined the cost function per example here. You can imagine the total cost function being defined as.

<ul>

 <li>Try the function on some [prediction, target] pairs, or plot it, to see how it behaves (by hand or using Python or the wolfram alpha website, no need to code)</li>

 <li>Compute its gradient, assuming a standard linear regression<em>w</em></li>

 <li>How would you implement the gradient? Again, no need to code – try to find a formula using standard matrix operations, along with element-wise multiplication and summation/product over columns/rows.</li>

 <li>How sensitive is this function to outliers? Compare two cases: the target is 1, but in one case our model assigns it 10, and in the other 100. (e.g. with <em>ε </em>= 1) How does the error changes? Compare with the following cost function, for the <em>n </em>data example:</li>

</ul>

<em>.</em>

Note: The higher the error on a sample is, relative to the other samples, the more your model will try to fit this sample.