**Aim:** To understand how the back propagation algorithm works.

## Warm up: a fast matrix-based approach to computing the output from a neural network


**Weight and Activation Notation in Neural Networks:**
   - Weights: $w_{ljk}$ denotes the weight from the $k$th neuron in layer $l-1$ to the jth neuron in layer l.
   - Biases and Activations: $b_{lj}$ for the bias, and $a_{lj}$ for the activation of the jth neuron in the lth layer.

**Matrix-Based Calculation of Neuron Activations:**
   - The activation $a_{lj}$ of the jth neuron in the lth layer is given by the equation: $a_{lj} = \sigma\left(\sum_k w_{ljk} a_{l-1,k} + b_{lj}\right)$
    where $\sigma$ is the activation function, and the sum is over all neurons k in the $l-1$th layer.

**Vectorization and Efficient Computation:**
   - Vectorized Form: The equation above can be rewritten in a compact vectorized form as:
     $$
     a_l = \sigma(w_l a_{l-1} + b_l),
     $$
     where $w_l$ is the weight matrix, $b_l$ is the bias vector, and $a_l$ is the activation vector for layer $l$.
   - The concept of vectorization applies functions like $\sigma$ to every element of a vector, simplifying and speeding up computations in neural networks.

## The two assumptions we need about the cost function

1. **Goal of Backpropagation**: To compute the partial derivatives $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$ of the cost function $C$ with respect to any weight $w$ or bias $b$ in the network.

2. **Assumptions about Cost Function**:
   - **Assumption 1**: The cost function can be written as an average over individual training examples: 
     $$ C = \frac{1}{n} \sum_x C_x $$
     Here, $C_x$ is the cost for a single training example, and this form is applicable to the quadratic cost function: 
     $$ C_x = \frac{1}{2} \| y - a^L \| ^2 $$
   - **Assumption 2**: The cost can be expressed as a function of the outputs from the neural network. For instance, in the quadratic cost function: 
     $$ C = \frac{1}{2} \| y - a^L \| ^2 = \frac{1}{2} \sum_j (y_j - a^L_j)^2 $$
     Here, $C$ is a function of the output activations $a^L$, with $y$ being a fixed parameter and not a variable influenced by the network's weights and biases.

3. **Quadratic Cost Function Example**:
   - The quadratic cost function is defined as:
     $$ C = \frac{1}{2n} \sum_x \| y(x) - a^L(x) \| ^2 $$
     Where $n$ is the total number of training examples, $y(x)$ is the desired output, $a^L(x)$ is the vector of activations from the network for input $x$, and $L$ is the number of layers in the network.

## The Hadamard product
The Hadamard product, denoted as $s \odot t$, is the element wise product of two vectors of the same dimension. In this operation, each component of the resulting vector is the product of the corresponding components of the two original vectors. The equation for the j-th component of the Hadamard product is given as:

$$
   (s \odot t)_j = s_j \times t_j
$$

## The four fundamental equations behind backpropagation

These are the four fundamental equations essential for backpropagation:
    - Error in the output layer ($\delta_L$):
        $$
        \delta_L = \nabla_a C \odot \sigma'(z_L).
        $$
    - Error in terms of the next layer's error ($\delta_{l+1}$):
        $$
        \delta_l = \left( (w_{l+1})^T \delta_{l+1} \right) \odot \sigma'(z_l).
        $$
    - Rate of change of cost with respect to any bias:
        $$
        \frac{\partial C}{\partial b_{lj}} = \delta_{lj}.
        $$
    - Rate of change of cost with respect to any weight:
        $$
        \frac{\partial C}{\partial w_{ljk}} = a_{l-1,k} \delta_{lj}.
        $$
## The backpropagation algorithm

1. **Input `x`**:
    - Set activation $a^1$ for the input layer.

2. **Feedforward**:
    - For each $l=2,3,\ldots,L$, compute:
      $$ z^l = w^l a^{l-1} + b^l $$
      $$ a^l = \sigma(z^l) $$

3. **Output Error $\delta^L$**:
    - Compute:
      $$ \delta^L = \nabla_a C \odot \sigma'(z^L) $$

4. **Backpropagate Error**:
    - For each $l=L-1,L-2,\ldots,2$, compute:
      $$ \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l) $$

5. **Output**:
    - Gradient of the cost function:
      $$ \frac{\partial C}{\partial w_{jk}^l} = a^{l-1}_k \delta^l_j $$
      $$ \frac{\partial C}{\partial b_j^l} = \delta^l_j $$

#### Modifications to Backpropagation Algorithm

1. **Modified Neuron**:
    - If output from a neuron is given by $f(\sum_j w_j x_j + b)$, where $f$ is not the sigmoid function, modify the backpropagation accordingly.

2. **Linear Neurons**:
    - Replace $\sigma$ with $\sigma(z) = z$ and rewrite the backpropagation steps.

#### Combining Backpropagation with Stochastic Gradient Descent

1. **Input a Set of Training Examples**.
2. **For Each Training Example `x`**:
    - Set input activation $a_{x,1}$ and follow the steps:

3. **Feedforward**:
    - For each $l=2,3,\ldots,L$, compute:
      $$ z_{x,l} = w^l a_{x,l-1} + b^l $$
      $$ a_{x,l} = \sigma(z_{x,l}) $$

4. **Output Error $\delta_{x,L}$**:
    - Compute:
      $$ \delta_{x,L} = \nabla_a C_x \odot \sigma'(z_{x,L}) $$

5. **Backpropagate Error**:
    - For each $l=L-1,L-2,\ldots,2$, compute:
      $$ \delta_{x,l} = ((w^{l+1})^T \delta_{x,l+1}) \odot \sigma'(z_{x,l}) $$

6. **Gradient Descent**:
    - Update weights and biases:
      $$ w^l \rightarrow w^l - \frac{\eta}{m} \sum_x \delta_{x,l} (a_{x,l-1})^T $$
      $$ b^l \rightarrow b^l - \frac{\eta}{m} \sum_x \delta_{x,l} $$

## Solutions to Exercises

1. **Question:** Backpropagation with a single modified neuron Suppose we modify a single neuron in a feedforward network so that the output from the neuron is given by $f\left(\sum_j w_j x_j + b\right)$, where f is some function other than the sigmoid. How should we modify the backpropagation algorithm in this case?
	**Solution:**
	When you modify a single neuron in a feedforward network such that its output is given by $f(\sum_j w_j x_j + b)$, where $f$ is a different function than the sigmoid, you need to adjust the backpropagation algorithm to account for the derivative of this new function $f$. Specifically, during the backpropagation step, when calculating the gradient of the error with respect to the weights, you should use the derivative of $f$ instead of the derivative of the sigmoid function.

2. **Question:** Backpropagation with linear neurons Suppose we replace the usual non-linear σ function with $σ(z)=z$ throughout the network. Rewrite the backpropagation algorithm for this case.
	**Solution:**
	If you replace the non-linear sigmoid function $\sigma$ with a linear function $\sigma(z) = z$ throughout the network, the backpropagation algorithm simplifies. The derivative of $\sigma(z) = z$ is $1$, so the error derivative with respect to the weights and biases becomes simpler to compute. In the backpropagation algorithm, where you would normally multiply by the derivative of the sigmoid function, you would instead multiply by $1$.