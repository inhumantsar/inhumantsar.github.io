---
{"dg-publish":true,"dg-path":"Algorithms/TF-IDF & Cosine Similarity.md","permalink":"/algorithms/tf-idf-and-cosine-similarity/","tags":["cs/algos/nlp"],"created":"2024-12-03T11:10:58.240-06:00","updated":"2024-12-11T18:41:59.304-06:00"}
---

TF-IDF is a method for creating a vector to represent a document. The two parts of the name stand for *term frequency* and *inverse document frequency*. It's a relatively simple concept and the math behind isn't too wild.

Let's say you have a collection of documents $D$ and you want to make targeted connections between those documents for searching or graphing or whatever. The documents could be the collected works of Shakespeare or the last 90d of log data or a hard drive full of spicy memes, it doesn't really matter.

TF-IDF helps with the search by giving *weight* to documents where the term frequency within a document is high and the document frequency for a given term is low.  

It's normal in practice to start by cleaning the data. In most situations, this means filtering out common terms (called *stop words* in natural language processing) which don't contribute to the *meaning* of a document, like *the*, *an*, etc. 

# Term Frequency

In the first part, *term frequency*, you would go through each document term-by-term (words, key/value pairs, pixel position and colour values). As you work your way through the documents, you add each term to a numbered index and make note of the number of times it appears in each document. By the end you'll have two results:

1. A *vocabulary*, the index of terms, which looks like $\{(1, \text{romeo}), (2, \text{juliet}), ... \}$, and
2. A set of *vectors* for which looks like $\{(12, 17, 3, 0, ...), ...\}$ where each vector represents a document and each value is the number of times each term in the index showed up in that document. 

These can be defined mathematically, where $F$ is the total number of terms in the index.

$$
\begin{aligned}
\text{Vocabulary}\\
E(t) &= \left\{ 
  \begin{aligned}
  &1 \text{ if }t=\text{ romeo} \\ 
  &2 \text{ if }t=\text{ juliet} \\
  &...
  \end{aligned}
\right\} \\
\\
\text{Counting Functions}\\
\text{fr}(x, t) &= \left\{ 
  \begin{aligned}
  &1 \text{ if } x = t \\ 
  &0 \text{ else } \\
  \end{aligned}
\right\}\\
\\
\text{tf}(t, d) &= \sum_{x\in d} fr(x, t) \\
\\
\text{Document Vector}\\
v_{d} &= \{tf(t_i, d_i): i = 1 ... F\}
\end{aligned}
$$

The next part, *inverse document frequency* is a little more complicated, but not much more. 

# Inverse Document Frequency

So we have a list of terms and a their frequency in each document. Does that tell us much though? If a word appears 100 times across all documents and another word appears 10 times, does that mean the first word is 10x more important than the second term? Likely not. To mitigate this, the IDF values will be used to *normalize* term frequencies on a logarithmic scale, scaling up the importance, or *weight*, of rare terms while scaling down the importance common terms.

The IDF value for each term is the $\log$ of the total number of documents divided by the number of documents where that term appears. To avoid dividing by zero, we can simply add 1 to the divisor rather than filtering the 0s out of every vector. This won't impact our results since a) we'll be dividing by 1, b) the normalization process will flatten those values out, and c) we'll be multiplying the IDF value by the term frequency, which is zero. 

$$
\begin{aligned}
\text{idf}(t) &= log\left(\frac{|D|}{1+ |\{{d} : t \in d\}|}\right) \\
\end{aligned}
$$

The result here is that terms which only appear in low numbers of documents will end up with a higher IDF value than terms which appear in many documents.

Combining TF and IDF:

$$
\text{tf-idf}(t) = \text{tf}(t, d) \times \text{idf}(t)
$$
This completes the weighting formula and illustrates how TF-IDF isolates highly relevant connections. The combined TF-IDF value will only be large if the term appears frequently in a document and infrequently across all documents. 
# Normalization

So now we have the TF-IDF values, but we still need to *normalize* them correct for their wide range. To do this, we'll calculate the [unit vector](https://en.wikipedia.org/wiki/Unit_vector) using [*Lebesgue spaces*](https://en.wikipedia.org/wiki/Lp_space) ($L^p$):

$$
\hat{v} = \frac{v}{||v||_p}
$$
*Lebesgue spaces* have a ton of applications across different fields, from probability to statistics, across finite and infinite dimensions, but anyone who remembers trigonometry will have encountered them too. 

The gist is that they're used to normalize distances in *vector spaces*. Let's say you draw two points on a sheet of graph paper, one at $(6, 2)$ and another at $(2, 5)$. Then you draw a line straight down from $(2, 5)$ toward $x=0$ and another line from $(6,2)$ toward $y=0$. 

The two lines would intersect at $(6, 2) - (2, 5) = (4, 3)$. The length of the lines between each point and the intersection would be 3 and 4 along the x and y axis respectively. The lines would also meet at a 90 degree angle. Now draw a line directly between the two points. Starting to look familiar?

If you dig into your deep past to pull out those middle school math lessons, you'll probably find this formula: $a^2 + b^2 = c^2$, or $c = \sqrt{a^2 + b^2}$. This is the Euclidean distance between two points. It's also the *L2-norm* ($L^2$).  The $p$ in $L^p$ indicates that any real value can be used in this kind of normalization, so this can be generalized to $||u||_p = (|u_1|^p + |u_2|^p, ... + |u_n|^p)^{\frac{1}{p}} = (\sum_{i=1}^n |u_i|^p)^{\frac{1}{p}}$ where $u$ indicates that this is a *unit vector* in a normed vector space.

>The *L1-norm* would be sum of the components of the vector. Going back to the graph paper, it would be the sum of the lengths of each line if you zigzagged along the paper's grid instead of drawing a straight line.

We don't really need to worry about $p > 2$ here though. While our vectors have more than 2 coordinates, *L2-norms* can be used on vectors with an arbitrary number of dimensions.

To simplify things, let's run through this process with a single vector rather than the full matrix first.

$$
\begin{aligned}
p &= 2 \\
v_d &= (12, 17, 3, 0) \\
\hat{v_d} &= \frac{v_d}{||v_d||_p} \\
\hat{v_d} &= \frac{(12, 17, 3, 0)}{(12^2 + 17^2 + 3^2 + 0^2)^{\frac{1}{2}}} \\
\hat{v_d} &= \frac{(12, 17, 3, 0)}{\sqrt{12^2 + 17^2 + 3^2 + 0^2}} \\
\hat{v_d} &= \frac{(12, 17, 3, 0)}{\sqrt{442}} \\
\hat{v_d} &= (0.57, 0.89, 0.14, 0)
\end{aligned}
$$
Since this is a normalization process, we can feed the result back in to a similar equation and expect to get $1.0$ out the other side:
$$
\begin{aligned}
p &= 2 \\
\hat{v_d} &= (0.57, 0.89, 0.14, 0) \\
1.0 &= \frac{\sum_{i=1}^{|\hat{v_d}|}\hat{v_d}} {||\sum_{i=1}^{|\hat{v_d}|}\hat{v_d}||_p} \\
1.0 &= \frac{0.57 + 0.89 + 0.14 + 0}{\sqrt{(0.57 + 0.89 + 0.14 + 0)^2}} \\
1.0 &= \frac{1.6}{1.6}
\end{aligned}
$$

# Putting it all together

Now that we have a theoretical foundation in place, we can move on applying these concepts to matrices.

We start with our separate TF and IDF data. $\text{tf}$ is a matrix sized $|D| \times F$, and $\text{idf}$ is a vector of length $F$.

$$
\begin{aligned}
M_{\text{tf}} &= \left\{
	\begin{aligned}
	&\text{tf}(t_i, d_1): i = 1 ...F \\
	&\text{tf}(t_i, d_2): i = 1 ...F \\
	&...\\
	&\text{tf}(t_i, d_{|D|}): i = 1 ...F \\
	\end{aligned}
	\right\} = \left[
		\begin{matrix}
		12 & 17 & 3 & 0 & ...\\
		8 & 32 & 1 & 18 & ...\\
		...\\
		... & 9 & 2 & 7 & 11
		\end{matrix}\right]
	\\
	\\
v_{\text{idf}} &= (\text{idf}(t_i): i = 1 ...F) = (0.57, 0.89, 0.14, 0, ..., 0.42)
\end{aligned}
$$

In order to apply the $v_{\text{idf}}$ weights to $M_{tf}$, we'll have to transform it into a square diagonal matrix with both dimensions equal to $F$ and then multiply the two matrices:

$$
\begin{aligned}
M_{\text{idf}} &= \left[
		\begin{matrix}
		0.57 & 0 & 0 & 0 & ...\\
		0 & 0.89 & 0 & 0 & ...\\
		...\\
		...&0&0&0& 0.42
		\end{matrix}\right] \\
\\
M_{\text{tf-idf}} &= M_{\text{tf}} \times M_{\text{idf}} \\
	&= \left[
		\begin{matrix}
		12 & 17 & 3 & 0 & ...\\
		8 & 32 & 1 & 18 & ...\\
		...\\
		... & 9 & 2 & 7 & 11
		\end{matrix}\right]
		\times
		\left[
		\begin{matrix}
		0.57 & 0 & 0 & 0 & ...\\
		0 & 0.89 & 0 & 0 & ...\\
		...\\
		...&0&0&0& 0.42
		\end{matrix}\right] \\
	&= \left[
		\begin{matrix}
		\\
		\text{...let's just pretend i did the math...} \\
		\\
		\end{matrix}\right]		
\end{aligned}
$$
Then finally we *L2 norm* to the matrix, row by row.
$$
M_{\text{tf-idf}} = \frac{M_{\text{tf-idf}}}{||M_{\text{tf-idf}}||_2}
$$
As before, the result can be verified by taking the L2 norm of each row and they'll all come out to 1.0. 

# Cosine Similarity

So now we have a matrix with a vector for each document containing weighted scores for each term in the vocabulary. That on its own is useful for ranking documents by individual terms, keyword extraction, clustering, and anomaly detection. Cosine similarity adds another layer of usefulness on top of that though by enabling us to search and compare documents effectively.

The process involves taking the dot product of two vectors, and then taking the cosine of the angle between them. The vectors could come from two documents or an external source (like a search query) and a document.

Dot products are fairly straight-forward conceptually. They're the sum of each element multiplied together.

$$
a.b = \sum_{i=1}^n a_ib_i = a_1b_1 + a_2b_2 + ... + a_nb_n
$$

An interesting property appears when the dot product is 0. This happens when two vectors are orthogonal to one another. This is easy to visualize in 2-dimensional space: Two vectors intersecting at a 90 degree angle, the vectors won't project onto each other to form a triangle. The neat thing is that this also holds true in higher dimensional spaces.

The closer two vectors are to orthogonal, the less similar they are. That is, the angle formed between two vectors is a measure of how closely related they are. This is useful for us since two documents where related terms are used but in very different frequencies will still have a small angle, making it easy to identify them as being related. While two documents which might share a few of the same highly weighted words but not much else will have an angle closer to orthogonal. Basic keyword matching would have marked them as similar, but cosine similarity is able to see past that. 

The formula for cosine similarity is straightforward as well, this is trigonometry after all. As a refresher:

$$
cos\theta = \frac{adjacent}{hypotenuse}
$$
The adjacent vector is simply the dot product of the other two and the hypotenuse is the same:
$$
\begin{aligned}
cos\theta &= \frac{a.b}{||a||\space||b||} \\
&= \frac{\sum_{i=1}^n a_ib_i}{\sqrt{\sum_{i=1}^na^2}\sqrt{\sum_{i=1}^nb^2}}
\end{aligned}
$$