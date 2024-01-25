**Aim:** to enhance neural network performance by refining the backpropagation algorithm through techniques like cross-entropy cost function, regularization methods, improved weight initialization, and optimal hyper-parameter selection.

## The cross-entropy cost function

#### Learning Slowdown

As the gradient is back-propagated through layers, it can become increasingly small, effectively slowing down the learning process. This is particularly problematic in deep networks with many layers.
#### Addressing Learning Slowdown

The learning slowdown in neural networks can be addressed by replacing the quadratic cost with the cross-entropy cost function. This function is defined for a neuron with multiple input variables $x_1, x_2, \ldots$, weights $w_1, w_2, \ldots$, and a bias $b$. The neuron's output is $a = \sigma(z)$, where $z = \sum_j w_j x_j + b$ is the weighted sum of the inputs.

#### Cross-Entropy Cost Function

The cross-entropy cost function is given by:

$$
C = -\frac{1}{n} \sum_x [y \ln a + (1 - y) \ln(1 - a)],
$$

where $n$ is the total number of training data items, the sum is over all training inputs $x$, and $y$ is the corresponding desired output.

#### Properties of Cross-Entropy

1. **Non-Negativity**: $C > 0$. This is because the logarithmic terms are always negative as they are within the range $0$ to $1$, and there is a negative sign in front of the sum.

2. **Approaches Zero**: If the neuron's output is close to the desired output for all training inputs, the cross-entropy will be close to zero. This is under the assumption that $y$ is either $0$ or $1$.

#### Avoiding Learning Slowdown

The cross-entropy cost function avoids the learning slowdown by directly influencing the weight adjustment proportional to the error in the output. The partial derivative of the cross-entropy with respect to the weights is:

$$
\frac{\partial C}{\partial w_j} = \frac{1}{n} \sum_x x_j (\sigma(z) - y).
$$

This formula shows that the rate at which the weight learns is controlled by $\sigma(z) - y$, the error in the output. The bigger the error, the faster the learning, avoiding the slowdown seen in quadratic cost due to the $\sigma'(z)$ term.

Similarly, for the bias, the partial derivative is:

$$
\frac{\partial C}{\partial b} = \frac{1}{n} \sum_x (\sigma(z) - y).
$$

Again, this avoids the learning slowdown by eliminating the dependence on the $\sigma'(z)$ term.

#### Cross-Entropy Details

#### Origin of Cross-Entropy Concept
- **Problem Addressed:** Learning slowdown in neural networks due to $\sigma′(z)$ terms in certain equations.
- **Solution Approach:** Idea to eliminate $\sigma′(z)$ by choosing an appropriate cost function.

#### Mathematical Derivation
- **Initial Goal:** Achieve $\frac{\partial C}{\partial w_j} = \frac{\partial C}{\partial b} = x_j(a - y)$.
- **Chain Rule Application:** $\frac{\partial C}{\partial b} = \frac{\partial C}{\partial a} \sigma′(z)$.
- **Sigmoid Derivative:** $\sigma′(z) = \sigma(z)(1 - \sigma(z)) = a(1 - a)$.
- **Simplified Form:** $\frac{\partial C}{\partial b} = \frac{\partial C}{\partial a} a(1 - a)$.
- **Deriving Cross-Entropy:**
  - $\frac{\partial C}{\partial a} = \frac{a - y}{a(1 - a)}$.
  - Integrating leads to $C = -[y \ln a + (1 - y) \ln(1 - a)] + \text{constant}$.
  - Averaging over examples: $C = -\frac{1}{n} \sum_x [y \ln a + (1 - y) \ln(1 - a)] + \text{constant}$.

#### Conceptual Understanding
- **Cross-Entropy as a Measure of Surprise:** 
  - **Information Theory Perspective:** Measures the "surprise" when the true value of $y$ is learned.
  - **Neuron's Task:** Neuron estimates probability of $y$ being 1 or 0.
  - **Surprise Dynamics:** Low surprise for expected outcomes, high surprise for unexpected outcomes.
  - **Further Reading:** Wikipedia for an overview, and Cover and Thomas's book on information theory for detailed understanding.

#### Introduction to Softmax
- **Context**: Softmax layers are an alternative to address learning slowdown, primarily used in deep neural networks (Chapter 6).
- **Core Concept**: Softmax is a new type of output layer for neural networks.
  
#### How Softmax Works
- **Weighted Inputs**: Similar to sigmoid layers, it starts with weighted inputs \( z_{Lj} = \sum_k w_{Ljk} a_{L-1k} + b_{Lj} \).
- **Softmax Function**: Instead of the sigmoid function, the softmax function is applied:
  
$$
  a_{Lj} = \frac{e^{z_{Lj}}}{\sum_k e^{z_{Lk}}}
$$

  This ensures all output activations are positive and sum up to 1, forming a probability distribution.

#### Characteristics of Softmax Outputs
- **Probability Distribution**: The outputs can be interpreted as probabilities, useful in problems like MNIST classification.
- **Positive Activations**: All output activations are positive.
- **Sum Equals One**: The sum of all softmax outputs is always 1.

#### Addressing the Learning Slowdown Problem
- **Log-Likelihood Cost Function**: Defined as  $C \equiv -\ln a_{Ly}$ .
- **Behavior of Cost Function**: When network performance is good (high probability for the correct output), the cost is low, and vice versa.
- **Solving Learning Slowdown**: Softmax layers with log-likelihood cost do not encounter learning slowdown, similar to sigmoid layers with cross-entropy cost.
  
$$
  \frac{\partial C}{\partial b_{Lj}} = \frac{\partial C}{\partial w_{Ljk}} = a_{Lj} - y_j
$$
  
  This ensures a consistent learning rate without slowdown.

#### Practical Use
- **Flexibility in Choice**: Both softmax with log-likelihood and sigmoid with cross-entropy are effective, depending on the problem and desired interpretation of outputs.
- **Application in Later Chapters**: Softmax layers will be used in later networks to align with certain academic papers and when outputs as probabilities are beneficial.
## Overfitting and regularization

#### Key Points on Regularization

1. **Purpose of Regularization:** Regularization techniques reduce overfitting in neural networks with a fixed architecture and training data set. They enable the network to prefer smaller weights, balancing between minimizing the original cost function and maintaining small weights based on the regularization parameter λ.

2. **L2 Regularization (Weight Decay):**
   - Equation: 
$$
     C = -\frac{1}{n} \sum_x \left[ y_j \ln a^L_j + (1 - y_j) \ln (1 - a^L_j) \right] + \frac{\lambda}{2n} \sum_w w^2
$$
   - Here, C is the regularized cross-entropy cost function. The first term is the standard cross-entropy, and the second term is the sum of squares of all the weights in the network, scaled by $\frac{\lambda}{2n}$. This technique also applies to other cost functions, like the quadratic cost.

3. **Impact of Regularization:**
   - Regularization makes the network prefer learning small weights unless larger weights significantly improve the first part of the cost function. The balance is controlled by the regularization parameter λ.
   - The gradient descent learning rule for weights in a regularized network becomes:
$$
     w \rightarrow w - \eta \left( \frac{\partial C_0}{\partial w} + \frac{\lambda}{n} w \right) = (1 - \eta \frac{\lambda}{n})w - \eta \frac{\partial C_0}{\partial w}
$$
   - The learning rule for biases remains unchanged.

4. **Effectiveness of Regularization:**
   - Regularization has been empirically shown to suppress overfitting and improve network generalization.
   - Regularized networks tend to be less sensitive to noise in the training data and focus more on learning patterns frequent across the training set.

5. **Application of Regularization in Practice:**
   - An example with MNIST data showed that using L2 regularization with cross-entropy cost function improved classification accuracy and reduced the effects of overfitting.
   - The choice of λ is crucial, especially when changing the size of the training set. The regularization parameter needs adjustment accordingly.

6. **Philosophical and Practical Considerations:**
   - While regularization often helps in practice, there is no complete theoretical explanation for its effectiveness.
   - Regularization does not constrain biases as large biases do not make a neuron as sensitive to its inputs as large weights do.

#### Other Techniques for Regularization

#### L1 Regularization
- **Concept**: Modifies the cost function by adding the sum of absolute values of the weights.
- **Equation**:
$$
  C = C_0 + \frac{\lambda}{n} \sum |w|
$$
- **Behavior**: Penalizes large weights, tending to favor small weights, which is different from L2 regularization.
- **Gradient Descent Update Rule**:
$$
  w \rightarrow w' = w - \frac{\eta \lambda}{n} \text{sgn}(w) - \eta \frac{\partial C_0}{\partial w}
$$
- **Comparison with L2**: In L1, weights shrink by a constant amount, unlike the proportional shrinking in L2.

#### Dropout
- **Concept**: Regularization technique that does not modify the cost function but alters the network.
- **Process**: Randomly delete a subset of hidden neurons during training, and adjust the weights accordingly.
- **Effectiveness**: Helps in reducing overfitting by mimicking the training of multiple networks.

#### Artificially Increasing Training Data Size
- **Approach**: Enhance performance by expanding training data through transformations like rotations, translations, etc.
- **Example**: Rotating MNIST images slightly to create new training data.
- **Benefits**: Exposes the network to more variations, improving its generalization capabilities.

#### Exercise on MNIST Data
- **Issue**: Using small rotations is beneficial, but large rotations might lead to misclassification.
- **Insight**: The effectiveness of machine learning models can vary significantly with different sizes and types of training data.

#### General Observations
- **Impact of Data Size**: More training data can sometimes compensate for a less sophisticated algorithm.
- **Algorithm Performance**: Performance comparisons between algorithms are context-dependent and can vary based on the training data used.
- **Focus on Data and Algorithms**: Improvements in algorithms should be balanced with efforts to obtain more or better training data.

## Weight initialization

**Weight Initialization in Neural Networks:**
   Neural networks require initial choices for weights and biases. A common approach has been to initialize them using independent Gaussian random variables with mean 0 and standard deviation 1. However, this method can lead to problems like saturation in neurons, where small changes in weights result in negligible changes in the neuron's activation, thereby slowing down learning.

**Improved Initialization Technique:**
   For a neuron with $n_{\text{in}}$ input weights, a better initialization method involves setting these weights as Gaussian random variables with mean 0 and standard deviation  $\frac{1}{\sqrt{n_{\text{in}}}}$ . This approach reduces the likelihood of neuron saturation and thus avoids slow learning. Biases can still be initialized as Gaussian random variables with mean 0 and standard deviation 1.

**Impact on Learning Efficiency:**
   Using the MNIST digit classification task as an example, the improved weight initialization approach  $\frac{1}{\sqrt{n_{\text{in}}}}$  demonstrated a quicker rise in classification accuracy compared to the traditional method. While both methods eventually reached similar accuracy levels, the new initialization method allowed for faster early progress. This indicates that better weight initialization not only speeds up learning but can also potentially improve the final performance of the neural network.

## How to choose a neural network's hyper-parameters?

1. **Broad Strategy for Choosing Hyper-parameters**:
    - When tackling a new problem with neural networks, begin by simplifying the problem to achieve non-trivial learning. 
    - Example: For MNIST, start by distinguishing between just two digits (like 0s and 1s) instead of all ten. 
    - Use a simple network architecture initially and gradually increase complexity based on the results.
    - Monitor performance frequently and adjust hyper-parameters like learning rate ($\eta$) and regularization parameter ($\lambda$).

2. **Specific Recommendations for Key Hyper-parameters**:
    - **Learning Rate ($\eta$):** Identify the threshold where the training cost begins to decrease instead of oscillate or increase. Start with a value like $\eta = 0.01$ and adjust to find the threshold. Use a value slightly lower than this threshold for stable learning.
    - **Regularization Parameter ($\lambda$):** Initially, start with no regularization ($\lambda = 0.0$) and adjust $\eta$. Then, use validation data to find a good value for $\lambda$, starting with a value like $\lambda = 1.0$ and adjusting by factors of 10.
    - **Mini-batch Size:** Balance between the computational efficiency of larger batches and the frequency of updates in smaller batches. The optimal size is somewhat independent of other hyper-parameters. Use real-time performance metrics to find the best size.

3. **Use of Early Stopping and Automated Techniques**:
    - Implement early stopping based on validation accuracy to determine the number of training epochs. Adjust the patience level (like no-improvement-in-ten epochs) based on the specific problem.
    - Explore automated hyper-parameter optimization techniques such as grid search and Bayesian optimization for more systematic searches.

## Other techniques

1. **Variations on Stochastic Gradient Descent**: 
   - *Hessian Technique*: This method aims to minimize a cost function $C$  by using its second-order Taylor series approximation. The update rule is given by $\Delta w = -H^{-1}\nabla C$, where $H$  is the Hessian matrix.
   - *Momentum-based Gradient Descent*: It introduces a "velocity" concept for parameters, modifying gradient descent. The update rules are  $v' = \mu v - \eta \nabla C$  and  $w' = w + v'$ , where $\mu$  is the friction or damping factor.

2. **Other Models of Artificial Neuron**:
   - *tanh Neurons*: These neurons use the hyperbolic tangent function and can output values ranging from -1 to 1. They are closely related to sigmoid neurons.
   - *Rectified Linear Units (ReLU)*: They output the maximum of zero and the input (i.e., $\max(0, w \cdot x + b)$). ReLUs are used extensively in neural networks, especially for image recognition tasks.

3. **Heuristic Approaches in Neural Networks**:
   - The field of neural networks often relies on empirical evidence and heuristic methods due to the complexity of interactions in these systems. Heuristics serve as a useful guide but also pose challenges for deeper investigation and understanding.
