**Aim:** To write a computer program implementing a neural network that learns to recognize handwritten digits.

The program is just 74 lines long, and uses no special neural network libraries. But this short program can recognize digits with an accuracy over 96 percent, without human intervention.

## Perceptron

**Introduction to Perceptrons:** Perceptrons were developed in the 1950s and 1960s by Frank Rosenblatt, inspired by earlier work by Warren McCulloch and Walter Pitts. These are early models of artificial neurons, preceding the more commonly used sigmoid neurons in modern neural networks.

**Functioning of Perceptrons:** A perceptron takes several binary inputs (`x_1, x_2, ...`) and produces a single binary output. It computes the output by weighing the inputs with respective weights (`w_1, w_2, ...`) and comparing the weighted sum against a threshold value. The output is `1` if the sum is greater than the threshold and `0` otherwise.

**Weights and Thresholds:** Weights represent the importance of each input, and the threshold is a parameter of the neuron, determining its activation.

**Decision Making and Evidence Weighing:** Perceptrons model decision-making processes by weighing different factors. Adjusting the weights and the threshold allows perceptrons to model various decision-making scenarios.

**Bias as an Alternative to Threshold:** Introducing 'bias' (`b = -threshold`) simplifies the perceptron's mathematical model. The perceptron's output can then be expressed as `1` if `w * x + b > 0` and `0` otherwise.

**Logical Functions with Perceptrons:** Perceptrons can compute basic logical functions like AND, OR, and NAND. An example is provided to demonstrate how a perceptron can function as a NAND gate.

**Universality of Perceptrons:** Networks of perceptrons can compute any logical function, as they can simulate NAND gates, which are universal for computation. This implies that perceptrons can theoretically simulate any computing device.

**Beyond NAND Gates:** While perceptrons are computationally equivalent to NAND gates, they offer more through learning algorithms. These algorithms adjust weights and biases in response to external stimuli, allowing neural networks to learn and solve problems without explicit programming for specific tasks.


## Sigmoid neurons

Learning algorithms in neural networks adjust weights and biases to achieve correct output. However, in a network of perceptrons, a small change in weight or bias can drastically flip the output, making controlled learning difficult.

**Introduction of Sigmoid Neurons**

Sigmoid neurons solve this problem by ensuring small changes in weights and biases lead to small changes in output. This is key for a network's learning ability.

- **Structure:** Sigmoid neurons are similar to perceptrons but with a key difference in output. The output is given by $\sigma(w \cdot x + b)$, where $\sigma$ is the sigmoid function, defined as:

  $$
  \sigma(z) \equiv \frac{1}{1+e^{-z}}.
  $$

- **Output:** For inputs $x_1,x_2,\ldots$, weights $w_1,w_2,\ldots$, and bias $b$, the output is:

  $$
  \frac{1}{1+\exp(-\sum_j w_j x_j - b)}.
  $$

**Similarity to Perceptrons**

Sigmoid neurons function like perceptrons under certain conditions. For example, when $z = w \cdot x + b$ is large and positive, $\sigma(z) \approx 1$; and when very negative, $\sigma(z) \approx 0$.

**Importance of the Sigmoid Function**

The sigmoid function's shape is crucial, allowing small changes in weights and biases to cause small changes in output. This is expressed as:

  $$
  \Delta \text{output} \approx \sum_j \frac{\partial \, \text{output}}{\partial w_j} \Delta w_j + \frac{\partial \, \text{output}}{\partial b} \Delta b,
  $$

where the partial derivatives indicate how changes in weights and biases affect the output.

**Interpreting Sigmoid Neuron Output**

Unlike perceptrons, sigmoid neurons output values between 0 and 1. This requires conventions for interpretation, such as considering outputs above 0.5 as one class and below as another.

## The architecture of neural networks
**1. Neural Network Architecture:**
   - **Input Layer:** Contains input neurons. Example: $64 \times 64$ greyscale image translates to $4,096$ input neurons.
   - **Hidden Layer:** Intermediate layer(s), neither input nor output. Multiple hidden layers can exist, e.g., a four-layer network with two hidden layers.
   - **Output Layer:** Contains output neurons. For instance, in a network determining if an image depicts a "9", an output value less than $0.5$ indicates "not a 9", and a value greater than $0.5$ indicates "is a 9".

**2. Design of Neural Networks:**
   - **Input and Output Layers:** Often straightforward in design, based on the problem being solved.
   - **Hidden Layers:** Design is more complex and lacks simple rules. Involves trade-offs in the number of layers and training time. Heuristics guide the design to achieve desired behavior.

**3. Types of Neural Networks:**
   - **Feedforward Neural Networks:** Output from one layer is input to the next, with no loops. Information flows only forward.
   - **Recurrent Neural Networks:** Allow feedback loops, with neurons firing for a limited duration. They are closer to brain functionality but are less influential due to less powerful learning algorithms. Recurrent networks might solve complex problems difficult for feedforward networks.


## A simple network to classify handwritten digits

1. **Problem Split into Two Sub-Problems**
   - **Segmentation**: Break an image containing multiple digits into separate images, each with a single digit. This is challenging for computers but easier once individual digit classification is effective.
   - **Classification**: Identify each digit in the segmented images. For example, recognizing the digit '5' from its image.

2. **Focus on Classifying Individual Digits**
   - Approach: Trial different segmentations, using a digit classifier to score each. High scores indicate confident classification in all segments.
   - Strategy: Concentrate on developing a neural network for recognizing individual handwritten digits, considering the segmentation problem more manageable with a good classifier.

3. **Three-Layer Neural Network for Digit Recognition**
   - **Input Layer**: 784 neurons (28x28 pixels), representing greyscale values (0.0 for white, 1.0 for black, and in-between values for grey shades).
   - **Hidden Layer**: Variable number of neurons, denoted as $ n $. Example given with $ n = 15 $.
   - **Output Layer**: 10 neurons, each representing a digit (0-9). The network identifies the digit by determining which neuron has the highest activation value.
   - **Rationale for 10 Output Neurons**: Empirically more effective than using 4 binary neurons. The network evaluates evidence from the hidden layer, potentially identifying shapes or patterns indicative of specific digits.
   - **Neural Network Operation Heuristic**: The hidden neurons may detect components or features of digits, influencing the final output. A heuristic approach suggests that 10 output neurons, each dedicated to a specific digit, may be more effective than a binary encoding system.


## Learning with Gradient Descent

**MNIST Dataset for Training**
   - The MNIST dataset, a collection of scanned images of handwritten digits, is used for training neural networks in digit recognition.
   - It consists of 60,000 training images and 10,000 test images, each 28x28 pixels in greyscale.
   - The training set includes handwriting samples from US Census Bureau employees and high school students, while the test set comes from a different group of people.

**Design of the Neural Network**
   - Each training input $ x $ is considered as a 784-dimensional vector (28x28 pixels).
   - The desired output $ y = y(x) $ is a 10-dimensional vector representing the digit.
   - The goal is to find weights and biases so that the network output approximates $ y(x) $ for all inputs.
   - The cost function, $ C(w, b) $, measures the performance of the network, where $ w $ and $ b $ represent weights and biases, respectively.

**Gradient Descent for Minimizing Cost Function**
   - Gradient Descent is an algorithm used to minimize the cost function $ C(w, b) $.
   - The quadratic cost function, also known as mean squared error (MSE), is used initially for simplicity.
   - This method involves calculating the gradient of the cost function and adjusting the weights and biases in the opposite direction of the gradient.
   - The learning rate $ \eta $ determines the size of the steps taken towards minimizing the cost function.

**Variations of Gradient Descent:**
- Gradient descent variations mimicking a physical ball have advantages but require computing costly second partial derivatives of the cost function, C. For instance, computing 

$$
\partial^2 C/\partial v_j \partial v_k
$$

for a million variables would need about half a trillion calculations, despite the symmetry 

$$
\partial^2 C/\partial v_j \partial v_k = \partial^2 C/\partial v_k \partial v_j.
$$

**Applying Gradient Descent in Neural Networks:**
- Gradient descent helps find the optimal weights, $ w_k $, and biases, $ b_l $, minimizing the cost function. The update rule in the context of neural networks is 

$$
w_k \rightarrow w_k' = w_k - \eta \frac{\partial C}{\partial w_k}
$$ 

and 

$$
b_l \rightarrow b_l' = b_l - \eta \frac{\partial C}{\partial b_l}.
$$

**Challenges in Gradient Descent:**
- Computing the gradient, $ \nabla C $, is time-consuming for large datasets, as it involves averaging gradients $ \nabla C_x $ for each input.

**Stochastic Gradient Descent:**
- It estimates $ \nabla C $ by averaging over a small, randomly selected sample (mini-batch) of training inputs. The update rule is 

$$
w_k \rightarrow w_k' = w_k - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial w_k}
$$ 

and 

$$
b_l \rightarrow b_l' = b_l - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial b_l},
$$ 

speeding up the learning process.

**Scaling in Cost Function and Updates:**
- Conventions vary in scaling the cost function and mini-batch updates. Omitting scaling factors like $ \frac{1}{n} $ or $ \frac{1}{m} $ is conceptually equivalent to adjusting the learning rate, $ \eta $.

**Visualization in High Dimensions:**
- Visualizing the cost function, C, in high-dimensional spaces is challenging. Effective

strategies involve algebraic and other non-visual representations, helping to conceptualize movements in multi-dimensional spaces.

## Implementing our network to classify digits

**MNIST Data and Preparation**
- **MNIST Data Acquisition**: 
  - Use `git clone https://github.com/mnielsen/neural-networks-and-deep-learning.git` for Git users or download from the provided link.
- **Data Splitting**:
  - Official MNIST: 60,000 training, 10,000 test images.
  - Custom Split: 50,000 training, 10,000 validation, 10,000 test images.

**Neural Network Implementation in Python**
- **Dependencies**: Python 2.7, Numpy library for linear algebra.
- **Network Class**:
  - `__init__`: Initializes biases and weights randomly using Numpy.
  - `sizes`: List indicating the number of neurons in each layer.
  - Example: `net = Network([2, 3, 1])` for a 3-layer network.
- **Bias and Weight Initialization**:
  - Gaussian distributions with mean $0$ and standard deviation $1$.
  - No biases for the input layer.

**Neural Network Operations**
- **Sigmoid Function**:
  - `def sigmoid(z): return 1.0/(1.0+np.exp(-z))`
- **Feedforward Method**:
  - Applies $a' = \sigma(wa + b)$ for each layer.
- **Stochastic Gradient Descent (SGD)**:
  - `def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None)`: Trains the network.
  - Randomly shuffles and partitions the training data into mini-batches.
  - Updates weights and biases using `update_mini_batch`.

**Backpropagation and Mini-batch Update**
- `update_mini_batch`: Applies gradient descent using backpropagation.
- `backprop`: Computes gradients for cost function (details in the next chapter).

**Running the Network**
- **MNIST Data Loading**: Use `mnist_loader.py`.
- **Training Example**:
  - Network with 30 hidden neurons.
  - Command: `net.SGD(training_data, 30, 10, 3.0, test_data=test_data)`
- **Performance**:
  - Varies with hyperparameters: number of epochs, mini-batch size, learning rate ($\eta$).
  - Example Results: 95.42% accuracy with 30 hidden neurons.

**Hyperparameters and Debugging**
- **Importance**: Poor choices lead to bad results.
- **Adjusting Learning Rate**: Start low ($\eta=0.001$), then increase to improve performance.
- **Debugging Challenges**: Involves tuning hyperparameters and architecture.

**Baseline Comparisons and SVM**
- **Baseline Tests**: Random guessing (10%) and average darkness method (22.25%).
- **SVM Classifier**: Achieves 94.35% accuracy with default settings in scikit-learn.
- **Tuned SVM**: Can reach over 98.5% accuracy.

**Neural Networks vs. SVM**
- **Current Records**: Neural networks outperform other techniques, including SVMs.

**Best Result (Neural Networks vs. SVM)**
- **Record Performance**: Neural networks classifying 9,979 out of 10,000 MNIST images correctly as of 2013.
- **Comparison**: Better than well-tuned SVMs, which have around 98.5% accuracy.
- **Human Comparison**: Comparable or better than human performance, given the complexity of some MNIST images.

**Key Insights**
- **Simplicity and Learning**: Neural networks often use simple algorithms, with complexity learned from training data.
- **Training Data Importance**: Good training data is crucial for effective learning.
- **Algorithmic Approach**: 
  - For certain problems: sophisticated algorithm ≤ simple learning algorithm + good training data.
## Towards Deep Learning

**Mystery of Neural Network Performance**
Neural networks, while impressive in performance, often operate without clear explanations for their decision-making processes. The weights and biases are learned automatically, raising questions about our understanding of these networks. For instance, if neural networks lead to artificial intelligence (AI), will we understand the principles behind their operation? The concern is that AI might become an opaque entity, with its inner workings, especially the learned weights and biases, remaining a mystery.

**Decomposing the Problem in Neural Networks**
A strategy to understand neural networks is to decompose a problem into sub-problems. For example, in face detection, instead of a holistic approach, one might look for specific features like eyes, nose, mouth, and hair, each as a separate sub-problem. This decomposition can be extended further into more granular questions, like the presence of an eyebrow or iris. Such an approach suggests that if sub-problems can be solved using neural networks, they can be combined to solve the larger problem. This is illustrated by considering a network architecture composed of sub-networks for each facial feature.

**Deep Neural Networks and Learning Algorithms**
Deep neural networks, characterized by multiple layers, decompose complex questions into simpler ones at the pixel level. The challenge has been in the practical implementation, especially in designing weights and biases. Early attempts using stochastic gradient descent and backpropagation in the 1980s and 1990s were not very successful with deep networks. However, since 2006, new techniques have been developed, enhancing the learning in deep neural networks. These techniques, still based on stochastic gradient descent and backpropagation, have allowed for the training of much deeper networks. Deep networks, with their ability to build complex hierarchies of concepts, have outperformed shallower networks in various tasks. This advancement in deep learning is akin to the use of modular design and abstraction in conventional programming, enabling the creation of complex systems.


## Solutions to Exercises

1. **Question:**
	Sigmoid neurons simulating perceptrons, part I 
	Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, $c>0$
	. Show that the behaviour of the network doesn't change.
	
	**Solution:**
	Multiplying all weights and biases in a perceptron network by a positive constant $c > 0$ does not change its behavior. The output of a perceptron is given by:
	$$
	 y = \text{step}\left(\sum_{i}(w_i \cdot x_i) + b\right)
	$$
	After multiplying the weights and biases by $ c $, the equation becomes:
	$$
	y' = \text{step}\left(\sum_{i}(c \cdot w_i \cdot x_i) + c \cdot b\right) = \text{step}\left(c \cdot \left( \sum_{i}(w_i \cdot x_i) + b \right)\right)
	$$
	The step function's output is based on the sign of its input. Multiplying by a positive  $c$  does not change the sign of the input. Therefore, the behavior of the perceptron remains unchanged, as the output of the step function is determined by the input's sign, not its magnitude.

2. **Question :** 
	Sigmoid neurons simulating perceptrons, part II 
	Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to the network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been fixed. Suppose the weights and biases are such that $w⋅x+b≠0$
	 for the input $x$
	 to any particular perceptron in the network. Now replace all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a positive constant$ c>0$
	. Show that in the limit as $c→∞$
	 the behaviour of this network of sigmoid neurons is exactly the same as the network of perceptrons. How can this fail when $w⋅x+b=0$
	 for one of the perceptrons?
	 
	**Solution :**
	In a network of perceptrons, the output is 1 if $w \cdot x + b > 0$ and $0$ otherwise. When these perceptrons are replaced by sigmoid neurons, the output is given by 
	$$\sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$
	
	Multiplying the weights and biases by a constant $c > 0$  in a sigmoid neuron changes its input to  $c(w \cdot x + b)$ . As  $c \rightarrow \infty$ , if  $w \cdot x + b \neq 0$ , the expression  $c(w \cdot x + b)$  tends towards positive or negative infinity, causing the sigmoid function to approach $1$ or $0$, respectively, thus mimicking a perceptron's behavior.
	
	However, if  $w \cdot x + b = 0$  for any neuron, multiplying by  $c$  doesn't change this value, and the sigmoid function yields  $\sigma(0) = 0.5$ , not matching a perceptron's output. Thus, the limit only holds when  $w \cdot x + b \neq 0$  for all neurons.


3. **Question:**
	Prove the assertion of the last paragraph. Hint: If you're not already familiar with the Cauchy-Schwarz inequality, you may find it helpful to familiarize yourself with it.
	
	**Solution:**
	The assertion is that the choice of $\Delta v = -\eta \nabla C$, where $\eta = \epsilon / \|\nabla C\|$ minimizes $\nabla C \cdot \Delta v$ under the constraint $\|\Delta v\| = \epsilon$. To prove this, we use the Cauchy-Schwarz inequality which states that for any vectors $a$ and $b$, $|a \cdot b| \leq \|a\| \|b\|$. Applying this to $\nabla C \cdot \Delta v$:
	
	$$
	   |\nabla C \cdot \Delta v| \leq \|\nabla C\| \|\Delta v\|
	$$
	
	Given that $\|\Delta v\| = \epsilon$, we have:
	
	$$
	   |\nabla C \cdot \Delta v| \leq \|\nabla C\| \epsilon
	$$
	
	Equality is achieved when $\Delta v$ is in the direction of $\nabla C$ or its opposite. The direction which minimizes $\nabla C \cdot \Delta v$ (i.e., makes it most negative) is the opposite of $\nabla C$. Therefore, $\Delta v = -\eta \nabla C$ with $\eta = \epsilon / \|\nabla C\|$ is the optimal choice.

4. **Question:**
	I explained gradient descent when $c$ is a function of two variables, and when it's a function of more than two variables. What happens when $c$ is a function of just one variable? Can you provide a geometric interpretation of what gradient descent is doing in the one-dimensional case?
	
	**Solution:**	
	When $C$ is a function of just one variable, say $v$, gradient descent simplifies significantly. The gradient $\nabla C$ becomes a single derivative $\frac{dC}{dv}$. The update rule then becomes $v \rightarrow v' = v - \eta \frac{dC}{dv}$.
	
	In the one-dimensional case, you can visualize the cost function $C(v)$ as a curve on a graph. The derivative $\frac{dC}{dv}$ represents the slope of this curve at any point. Gradient descent in this scenario involves moving along this curve in the direction that reduces the height of the curve (the cost). If the slope is positive, you move left (decreasing $v$), and if the slope is negative, you move right (increasing $v$). This process is akin to a ball rolling down the curve of the function towards its lowest point, which is the minimum of $C$.