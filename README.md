# Word2Vec from Scratch

Implementation of the skip-gram Word2Vec model in both PyTorch and NumPy, covering full softmax and negative sampling variants.

## What's implemented

### Full Softmax (`SkipGramSoftmax`, `SkipGramSoftmaxNumpy`)

- Skip-gram pair generation with a configurable context window
- Forward pass: input embedding lookup → linear projection → softmax over full vocabulary
- Backward pass: cross-entropy gradient derived by hand, implemented in NumPy using an outer product for `dW2` and a matrix-vector product for `dW1`
- PyTorch version uses `nn.CrossEntropyLoss`; NumPy version uses SGD with manual gradient updates

### Negative Sampling (`SkipGramNegSampling`, `SkipGramNegSamplingNumpy`)

- Binary classification loss: maximize $\sigma(\mathbf{u}_o^\top \mathbf{v}_c)$ for true pairs, minimize it for $k$ noise pairs
- Two samplers:
  - `UniformNegativeSampler` — draws negatives uniformly at random
  - `UnigramNegativeSampler` — draws from a unigram$^{0.75}$ distribution (the original Word2Vec approach), with rejection sampling to exclude true context words
- Trained on Jane Austen's *Emma* (NLTK Gutenberg corpus, ~6,900 vocab, ~815,000 training pairs)
- Embeddings visualized with PCA

## Bugs fixed

| # | Location | Fix |
|---|----------|-----|
| 1 | `SkipGramNegSamplingNumpy.train_one_epoch` | Replaced `W2[neg_indices] -= lr * du_n` with `np.add.at(W2, neg_indices, -lr * du_n)`. NumPy fancy-index assignment reads the original value once and drops all but the last duplicate update; `np.add.at` accumulates correctly. |
| 3 | `SkipGramNegSamplingNumpy.sigmoid` | Replaced `1 / (1 + exp(-x))` with a branch-stable form: positive `x` uses `1/(1+exp(-x))`, negative `x` uses `exp(x)/(1+exp(x))`. The original overflowed to `inf` for large negative inputs, producing `nan` in the log-loss. |

## Gradient derivations (`derivations.md`)

### Full softmax

The loss for a (center $c$, context $o$) pair is $\mathcal{L} = -\log \hat{y}_o$ where $\hat{y}$ is the softmax over all $V$ scores.

The key intermediate is the score gradient:

$$\frac{\partial \mathcal{L}}{\partial z_w} = \hat{y}_w - \mathbb{1}[w = o]$$

From this, the output and input embedding gradients follow:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_w} = (\hat{y}_w - \mathbb{1}[w=o]) \cdot \mathbf{v}_c \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = \sum_{w=1}^{V} (\hat{y}_w - \mathbb{1}[w=o]) \cdot \mathbf{u}_w$$

These map to `np.outer(dl_dz, v_c)` and `W2.T @ dl_dz` respectively. Cost is $O(V \cdot d)$ per pair.

### Negative sampling

Replaces the $V$-way softmax with a binary task over $k+1$ words. The loss is:

$$\mathcal{L} = -\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c) - \sum_{i=1}^{k} \log \sigma(-\mathbf{u}_{n_i}^\top \mathbf{v}_c)$$

Using the identities $\frac{d}{dx}[-\log \sigma(x)] = \sigma(x) - 1$ and $\frac{d}{dx}[-\log \sigma(-x)] = \sigma(x)$, the gradients are:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = (\sigma(s_o) - 1)\,\mathbf{v}_c \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{u}_{n_i}} = \sigma(s_{n_i})\,\mathbf{v}_c$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = (\sigma(s_o) - 1)\,\mathbf{u}_o + \sum_{i=1}^{k} \sigma(s_{n_i})\,\mathbf{u}_{n_i}$$

The structure mirrors full softmax but only touches $k+1$ embeddings per step, reducing cost to $O(k \cdot d)$.

Negatives are sampled from $P(w) \propto f(w)^{3/4}$, which smooths the unigram distribution to give rare words adequate negative signal.
