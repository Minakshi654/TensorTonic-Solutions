# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Mina's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/mina.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Anchor Box Generation | Generate object-detection anchor boxes across a feature grid for every scale and aspect-ratio combination. | https://www.tensortonic.com/problems/anchor-box-generation |
| Compute AUC (Area Under ROC) | Calculate binary-classification ROC AUC from false-positive and true-positive rates using trapezoidal integration. | https://www.tensortonic.com/problems/auc |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Binning | Assign numeric values to ordered bins using supplied boundaries while handling values at interval edges. | https://www.tensortonic.com/problems/binning |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| Chi-Square Test | Run a chi-square independence test on a contingency table using expected counts and the chi-square statistic. | https://www.tensortonic.com/problems/chi2-independence |
| Cohen's Kappa | Calculate Cohen's kappa from two label sequences by comparing observed agreement with chance agreement. | https://www.tensortonic.com/problems/cohens-kappa |
| Advantage Computation | Compute reinforcement-learning advantages by subtracting value estimates from observed returns at each timestep. | https://www.tensortonic.com/problems/compute-advantage |
| Compute Confusion Matrix with Normalization | Build a multiclass confusion matrix and optionally normalize counts by true-class rows or predicted-class columns. | https://www.tensortonic.com/problems/confusion-matrix-norm |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Edit Distance | Compute Levenshtein edit distance between two strings using dynamic programming over insertions, deletions, and substitutions. | https://www.tensortonic.com/problems/edit-distance |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Calibration Error | Calculate expected calibration error by binning prediction confidence and weighting accuracy-confidence gaps. | https://www.tensortonic.com/problems/expected-calibration-error |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Feature Store Lookup | Combine stored offline and request-time features in input order, using defaults for unknown user IDs. | https://www.tensortonic.com/problems/feature-store-lookup |
| Geometric Probability Mass Function & Mean | Compute the geometric distribution probability mass and mean from a valid success probability. | https://www.tensortonic.com/problems/geometric-pmf-mean |
| Compute Gini Impurity for a Split | Compute weighted Gini impurity for a candidate decision-tree split from the class labels on both sides. | https://www.tensortonic.com/problems/gini-impurity |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Apply 4×4 Homogeneous Transform | Apply a 4x4 homogeneous transformation matrix to 3D points using rotation, translation, and homogeneous coordinates. | https://www.tensortonic.com/problems/homogeneous-transform |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Image Histogram | Count grayscale image pixels into intensity bins and return the histogram in ascending intensity order. | https://www.tensortonic.com/problems/image-histogram |
| Compute Information Gain for a Split | Compute information gain for a decision-tree split from parent entropy and weighted child entropies. | https://www.tensortonic.com/problems/information-gain |
| Intersection over Union (IoU) | Compute intersection over union for two axis-aligned bounding boxes from overlap and combined area. | https://www.tensortonic.com/problems/iou-bounding-box |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Log Loss (Per-Sample) | Compute binary log loss for each prediction with clipped probabilities to prevent undefined logarithms. | https://www.tensortonic.com/problems/log-loss-per-sample |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Implement Majority Class Classifier | Fit a majority-class baseline and predict the most frequent training label for every requested sample. | https://www.tensortonic.com/problems/majority-classifier |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Monte Carlo Policy Evaluation | Estimate state values from complete Monte Carlo episodes by averaging discounted returns for each visited state. | https://www.tensortonic.com/problems/mc-policy-evaluation |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Model Versioning | Select a production model by highest accuracy, then lower latency, then the most recent timestamp. | https://www.tensortonic.com/problems/model-versioning-basics |
| Implement Nesterov Momentum (NAG) | Implement a Nesterov accelerated-gradient update using lookahead momentum and the current gradient. | https://www.tensortonic.com/problems/nesterov-momentum |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Percentiles / Quantiles | Calculate requested percentiles from numeric data using the interpolation rule specified by the problem. | https://www.tensortonic.com/problems/percentiles |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Tabular Q-Learning (Single Update) | Perform one tabular Q-learning update from reward, discount, learning rate, and the best next-state value. | https://www.tensortonic.com/problems/q-learning-update |
| Implement R² Score (Coefficient of Determination) | Compute the coefficient of determination from targets and predictions with explicit constant-target handling. | https://www.tensortonic.com/problems/r2-score |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Backward (Vanilla RNN) | Backpropagate through one vanilla RNN timestep to compute input, hidden-state, weight, and bias gradients. | https://www.tensortonic.com/problems/rnn-step-backward |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| One-Step TD Value Update | Perform one temporal-difference value update from reward, discount, next-state value, and learning rate. | https://www.tensortonic.com/problems/td-value-update |
| Text Chunking | Split text into ordered chunks under the requested size and overlap rules without dropping content. | https://www.tensortonic.com/problems/text-chunking |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Compute 3D Vector Norm | Compute the Euclidean norm of a 3D vector from the square root of summed squared coordinates. | https://www.tensortonic.com/problems/vector-norm-3d |
| Implement Wasserstein Critic Loss | Compute Wasserstein critic loss as the difference between mean fake and real critic scores. | https://www.tensortonic.com/problems/wasserstein-critic-loss |
| Word Count Dictionary | Count token occurrences in text and return a dictionary mapping each distinct word to its frequency. | https://www.tensortonic.com/problems/word-count-dict |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| GAN Discriminator | Implement a GAN discriminator that maps input samples through dense layers to real-versus-fake probabilities. | https://www.tensortonic.com/research/gan/gan-discriminator |
| Complete GAN System | Assemble a complete GAN system that generates samples, scores them with the discriminator, and returns both outputs. | https://www.tensortonic.com/research/gan/gan-full-network |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| GAN Loss Functions | Compute numerically stable binary cross-entropy losses for the GAN generator and discriminator objectives. | https://www.tensortonic.com/research/gan/gan-loss |
| Mode Collapse Detection | Detect GAN mode collapse by measuring diversity across generated samples and flagging low-variance outputs. | https://www.tensortonic.com/research/gan/gan-mode-collapse |
| GAN Training Loop | Implement one GAN training iteration with separate discriminator and generator forward and update steps. | https://www.tensortonic.com/research/gan/gan-training-loop |
| Reset Gate | Implement a GRU reset gate that controls how much of the previous hidden state contributes to the candidate state. | https://www.tensortonic.com/research/gru/gru-reset-gate |
| Complete LSTM Cell | Build a complete LSTM cell with forget, input, candidate, cell-state, output, and hidden-state calculations. | https://www.tensortonic.com/research/lstm/lstm-cell |
| Cell State Update | Implement the LSTM cell-state update by combining retained memory with input-gated candidate information. | https://www.tensortonic.com/research/lstm/lstm-cell-state |
| Forget Gate | Implement an LSTM forget gate by combining the previous hidden state and current input with a sigmoid projection. | https://www.tensortonic.com/research/lstm/lstm-forget-gate |
| Input Gate | Implement the LSTM input gate and candidate activation that control new information written to the cell state. | https://www.tensortonic.com/research/lstm/lstm-input-gate |
| Output Gate | Implement the LSTM output gate and expose the current hidden state from the updated cell memory. | https://www.tensortonic.com/research/lstm/lstm-output-gate |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Backpropagation Through Time | Implement one backpropagation-through-time step using the tanh derivative and hidden-to-hidden weight gradients. | https://www.tensortonic.com/research/rnn/rnn-bptt |
| RNN Cell | Implement an Elman RNN cell that combines the current input and previous hidden state before applying tanh. | https://www.tensortonic.com/research/rnn/rnn-cell |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Complete Vanilla RNN | Assemble a vanilla RNN that processes sequences into recurrent hidden states and per-time-step output logits. | https://www.tensortonic.com/research/rnn/rnn-full-network |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| VAE Decoder | Build a variational autoencoder decoder that maps sampled latent vectors back to reconstructed input probabilities. | https://www.tensortonic.com/research/vae/vae-decoder |
| ELBO Loss Function | Implement the VAE evidence lower bound from reconstruction loss and KL divergence with configurable weighting. | https://www.tensortonic.com/research/vae/vae-elbo-loss |
| VAE Encoder | Implement a variational autoencoder encoder that maps inputs to separate latent mean and log-variance vectors. | https://www.tensortonic.com/research/vae/vae-encoder |
| Complete VAE Network | Assemble a complete variational autoencoder forward pass with encoding, reparameterized sampling, and decoding. | https://www.tensortonic.com/research/vae/vae-full-network |
| KL Divergence Loss | Compute the VAE KL-divergence term between a diagonal Gaussian posterior and the standard normal prior. | https://www.tensortonic.com/research/vae/vae-kl-divergence |
| Reparameterization Trick | Implement the VAE reparameterization trick by sampling latent vectors from the predicted mean and log variance. | https://www.tensortonic.com/research/vae/vae-reparameterization |
| VGG Classifier Head | Build the VGG classifier by flattening spatial features and applying two ReLU hidden layers plus a logits projection. | https://www.tensortonic.com/research/vgg/vgg-classifier |
| VGG Configuration | Generate the canonical convolution and max-pooling layer configuration for VGG11, VGG13, VGG16, or VGG19. | https://www.tensortonic.com/research/vgg/vgg-config |
| VGG Conv Block | Implement a VGG convolutional block as sequential channel projections with ReLU activation at every spatial position. | https://www.tensortonic.com/research/vgg/vgg-conv-block |
| VGG Feature Extractor | Implement a configuration-driven VGG feature extractor that alternates ReLU projections with 2x2 max pooling. | https://www.tensortonic.com/research/vgg/vgg-feature-extractor |
| VGG Max Pooling | Implement VGG 2x2 max pooling with stride two while preserving the input batch and channel dimensions. | https://www.tensortonic.com/research/vgg/vgg-maxpool |
| Patch Embedding | Implement Vision Transformer patch embeddings by splitting images into fixed patches and linearly projecting each patch. | https://www.tensortonic.com/research/vit/vit-patch-embedding |
| Frequent-Word Subsampling | Implement Word2Vec frequent-word subsampling by computing token retention probabilities from corpus frequencies. | https://www.tensortonic.com/research/word2vec/word2vec-subsampling |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/mina)
<!-- tensortonic:end -->
