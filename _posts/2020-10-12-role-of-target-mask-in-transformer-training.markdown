---
mathjax: true
layout: post
title:  "Role of Target Mask in Transformer Training"
date:   2020-10-12 12:10:56 +0900
categories: deep-learning nlp transformers
url: http://127.0.0.1:4000
excerpt_separator: <!--more-->
---
<style type="text/css">
.circle:before {
  content: ' \25CF';
  font-size: 40px;
}
</style>

While transformers have been becoming increasingly popular across different domains, there are still points which aren't emphasized enough. One of those points is how the target mask given to the transformer decoder behaves in the training phase of the network. I'll try to 
explain its behavior with a visual intuition.

<!--more-->

Unlike RNNs, Transformer's self-attention is able to learn from all timesteps regardless of the current timesteps. In the training phase the decoder has all target tokens as input, but it should give its outputs based on timesteps before the current timestep for a proper learning process. This is why target masks are introduced.

## Recap: RNN & LSTM
RNNs and LSTMs don't need such masking due to their recurrence relations which make each time step dependending on only previous timesteps. Let's remember RNN formulas:

$$
\begin{aligned}
& {a_t} = b + W{h_{t-1}} + U{x_t} & (1) \\ &
{h_t} = tanh({a_t}) & (2) \\ &
{o_t} = c + V{h_t}  & (3) \\ &
{\widehat{y}_t} = softmax({o_t}) & (4)
\end{aligned}
$$

If we write output in terms of weights, bias, input and hidden states we can barely see the relation between current timestep and previous ones.

$$
\begin{aligned}
& {\widehat{y}_t} = softmax({c + V{tanh({b + W{h_{t-1}} + U{x_t}})}})
\end{aligned}
$$


## Overview: Transformers

![]({{ site.baseurl }}{% link assets/transformer-arch.png %}){: style="float:left; height: 700px"}
The diagram from the paper _Attention Is All You Need_ clearly shows that a Transformer mostly consists of Multi-Head Attentions. The unit on the left is the encoder of the Transformer and the one on the right is the decoder. We will focus on the decoder, especially on Masked Multi-Head Attention which prevents information from the future timesteps to leak with its mask.

A Multi-Head Attention is formed by multiple Scaled Dot-Product Attention as shown below. After concatenating outputs of each attention, they are passed through a linear layer. 

On the other hand, Scaled Dot-Product Attention has Key, Query and Value matrices. Multiplication of Key and Query matrices determines how much the model attends to each value.


![]({{site.baseurl}}{% link assets/attention.png %}){: style="align:center; margin:auto; padding:auto; display:block"}


## Towards Visual Understanding

It's not easy to think how the values flow through all the network and make sure the output of each timestep only depends on the previous timesteps. Since numbers, especially floating numbers, are too abstract to understand such complex interactions, we need a more intuitive representation to follow information flow across the network.

Here is a colorful idea: Using colors to show interactions between numbers. Since human brain is highly specialized in visual capabilities, it's one of the best ways to express information flow across the network. Also colors have quite useful properties letting us to *colorize* some mathematical operations. First of all, we can mix two colors to get a new color, which is similar to addition. 

Furthermore, any color can be created by mixing the colors accepted as primary colors (e.g. red, green, blue) and any primary color cannot be created by mixing the other two primary colors. These properties of the primary colors make them somehow *linearly independent*. If we think primary colors as basis of the colorspace, then we can *colorize* some of the operations easily.

### Colorizing Operations
Let's replace $$\widehat{i}, \widehat{j}, \widehat{k}$$ unit vectors with red, blue, green *unit vectors*.

$$ \widehat{i} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} $$ <span class="circle" style="color: rgba(255, 0, 0, 1.0)"></span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ \widehat{j} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} $$ <span class="circle" style="color: rgba(0, 255, 0, 1.0)"></span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$ \widehat{k} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} $$ <span class="circle" style="color: rgba(0, 0, 255, 1.0)"></span>

An interactive example for addition:

<iframe src="https://matrix-vis-app.herokuapp.com/" style="width:100%; border:none; height:700px;"></iframe>


However it's not the case that every operation can be *colororized* in this manner. Mathematical operations such as multiplication, exponentiation make trouble. For example, think about element wise multiplication of red and blue unit vectors. The zeroes will cancel out red and blue values although we want the result to carry information from both blue and red values. Another example is that exponentiating the red unit vector will give positive blue and green values. To colorize such operations meaningfully, after carrying out the operation with *normal* numbers we can distribute values to the colors involved in operands. For this to happen, we must make sure that those operations only have effect on those colors.

[//]: #  <MULTIPLICATION EXAMPLE>

## Visualizaton

For the sake of simplicity, let our transformer be 1-layered 1-headed transformer and the target sequence be 3 tokens length. The fact that we want to visualize is each timestep at output only carries information from the target tokens from the previous timesteps. 

Let assign a primary color for each token embedding. Each element of the embedding can be thought as scalar-unit vector multiplication, so each element will have a different shade of the color chosen for that token embedding. If we think of weight-embedding interactions as scalar-vector multiplications, we can expect that each output must consist of colors of the token embeddings from the previous timesteps.

The tokens embeddings look like this:

<iframe src="{{site.baseurl}}{% link assets/visualizations/embedding.html %}" style="width:100%; border:none; height:350px;"></iframe>

### Self-Attention

#### Linear Layers

To get Query, Key and Value matrices, it's needed to multiply embeddings with weight matrices.

$$ \begin{aligned}
	 XW^{Q} = Q,  \;\;\;  XW^{K} = K,  \;\;\;  XW^{V} = V
\end{aligned} $$

Since weight matrices don't carry any information directly related to the tokens, they are colorless and behave like scalar in the multiplications.

<iframe src="{{site.baseurl}}{% link assets/visualizations/attn_q.html %}" style="width:100%; border:none; height:550px;"></iframe>

<iframe src="{{site.baseurl}}{% link assets/visualizations/attn_k.html %}" style="width:100%; border:none; height:550px;"></iframe>

<iframe src="{{site.baseurl}}{% link assets/visualizations/attn_v.html %}" style="width:100%; border:none; height:550px;"></iframe>


#### Scaled Dot-Product Attention

Attention is defined as

$$ \begin{aligned}
	Attention(Q, K, V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V
	\end{aligned} $$

For the decoder's self-attention, masking is applied to $$QK^{T}$$. Let's visualize this part.

<iframe src="{{site.baseurl}}{% link assets/visualizations/attn_scores.html %}" style="width:100%; border:none; height:400px;"></iframe>

##### Masking and Softmax

Let's define the mask I:

$$ \begin{aligned} I_{ij} = \left\{
	\begin{array}{ll}
		1 & \mbox{if } i \leq j \\
		-inf & \mbox{if } i > j
	\end{array}
 \right. \end{aligned} $$ 

In this case it will be:

$$ \begin{aligned} I = \begin{bmatrix} 1 & -inf & -inf \\ 1  & 1 & -inf \\ 1 & 1 & 1 \end{bmatrix} \end{aligned} $$ 

Apply mask to attention scores with element wise multiplication:  

$$ \begin{aligned}
	QK^{T} * I =  \begin{bmatrix} 0.82 & 0.74 & 0.91 \\ 0.65  & 0.58 & 0.71 \\ 0.77 & 0.69 & 0.85 \end{bmatrix}  * \begin{bmatrix} 1 & -inf & -inf \\ 1  & 1 & -inf \\ 1 & 1 & 1 \end{bmatrix} 
	 = \begin{bmatrix} 0.82 & {-inf} & {-inf} \\ 0.65  & 0.58 & {-inf} \\ 0.77 & 0.69 & 0.85 \end{bmatrix} 
	\end{aligned} $$

Softmax is defined as 

$$ softmax(x)_{i} = \frac{e^{x_{i}}}{\sum_{j}^{}{e^{x_{j}}}} $$

Since $$ e^{-inf} = 0 $$ all $$ QK^{T}_{ij} $$ will be zero where i < j, which means dot-products of current timestep (i) with future timesteps will cancel out.

Masked $$ QK^{T} $$ will be scaled by 0.5 according to the formula since $$ d_{k} = 4 $$.


<iframe src="{{site.baseurl}}{% link assets/visualizations/softmax.html %}" style="width:100%; border:none; height:400px;"></iframe>

<iframe src="{{site.baseurl}}{% link assets/visualizations/qktv.html %}" style="width:100%; border:none; height:550px;"></iframe>

##### Linear Layer

<iframe src="{{site.baseurl}}{% link assets/visualizations/attn_last_linear.html %}" style="width:100%; border:none; height:400px;"></iframe>

### Add & Normalize

#### Add

<iframe src="{{site.baseurl}}{% link assets/visualizations/add.html %}" style="width:100%; border:none; height:350px;"></iframe>

#### LayerNorm

$$ \begin{aligned} 
	y = \frac{x-\mu_{x}}{\sqrt{\sigma^{2}_{x} + \epsilon}} * \gamma + \beta
   \end{aligned} $$

LayerNorm is not easy-to-colorize. Due to the expression in the denominator, it's impossible to keep a linear relation between colors and values of the cells. Although it's not easily colorizable, we know LayerNorm will not carry any information between timesteps because it's a normalization across features. So mean and variance calculation just involves the timesteps that have been involved so far for current timestep. Since learnable parameters $$ \gamma $$ and $$ \beta $$ are used for elementwise multiplication and addition respectively, they will not change information distribution across timesteps, so we can safely distribute new values to nonzero colors of each value for visualization purposes.

<iframe src="{{site.baseurl}}{% link assets/visualizations/layernorm.html %}" style="width:100%; border:none; height:350px;"></iframe>


### Encoder-Decoder Attention

In encoder-decoder attention Query and Value matrices are formed from output of the encoder network. Since they only carry information of the source sequence, they can be behaved as same as weight matrices for visualization purposes. At the same time the Query matrix formed from the output of the LayerNorm. By putting all of these together, encoder-decoder attention can be defined as:

$$ \begin{aligned} 
	Attention(Q, K, V)_{encdec} = softmax(\frac{Q_{enc}K^{T}}{\sqrt{d_{k}}})V_{enc} 
	\end{aligned} $$

Visualization of encoder-decoder attention is very similar to self-attention except that $$ Q_{enc} $$ and $$ V_{enc} $$ are colorless like weight matrices. Inferring from this similarity, we can safely say that information distribution across timesteps is preserved and skip an additional visualization.

### Add & Normalize

This is almost the same as previous Add & Normalize except that we add the output of the previous LayerNorm to the output of the encoder-decoder attention. In terms of visualization, color/information distribution across timesteps is the same as the previous Add & Normalize layer, so we can skip this part.

### FFN Layer

In the Transformer architecture fully-connected feed-forward networks are formulated as:

$$ \begin{aligned}
	FFN(x) = ReLU(xW_{1} + b_{1})W_{2} + b_{2} = max(0 , xW_{1} + b_{1})W_{2} + b_{2}
   \end{aligned} $$

For the sake of the simplicity, let's skip biases.

<iframe src="{{site.baseurl}}{% link assets/visualizations/ffn.html %}" style="width:100%; border:none; height:700px;"></iframe>

Since all values are nonzero, $$ xW_{1} $$ is equal to $$ ReLU(xW_{1}) $$.

### Add & Normalize

This is almost the same as previous Add & Normalize layers except that this time, we add the output of the previous LayerNorm to the output of the feed-forward network. In terms of visualization, color/information distribution across timesteps is just like previous Add & Normalize layers, so we can skip this part.

### Linear

Assuming our vocabulary size is 10, linear layer will output logits.

<iframe src="{{site.baseurl}}{% link assets/visualizations/logits.html %}" style="width:100%; border:none; height:600px;"></iframe>

### Softmax

As we have seen before, softmax will be applied for rows and will not change information/color distribution.


## Conclusion

By applying mask to self-attention scores in the decoder, all scores related to future timesteps were zeroed out for each timestep which leads timesteps to carry the information of only themselves and previous timesteps until the end of the pipeline. By hovering each value we can easily see how each timestep consists of the colors from previous timesteps.

## Notes about Color Space

There are some issues with color spaces restricting us in terms of mathematical flexibility. One of them is each of the colors (RGB) can be 1.0 at maximum. So weights and input values were tuned in order not to exceed the limit value. In the interactive additon example, if the sum of two colors exceeds 1.0, it's accepted as 1.0. 

Another issue is dealing with nonlinearities. Although color spaces are very convenient for linear operations, it's hard to show nonlinear operations. After showing that information distribution across timesteps does not change mathematically, we distribute new values to colors proportional to the amount of the colors involved. It worths mentioning that the key point of visualization with colors is showing information distribution across timesteps.


## References

1. [**Deep Learning**, _Goodfellow et al._, 2016 - RNN ](http://www.deeplearningbook.org/contents/rnn.html)
2. [**The Annotated Transformer**](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [**The Illustrated Transformer**, _Jay Alammar_](http://jalammar.github.io/illustrated-transformer/)