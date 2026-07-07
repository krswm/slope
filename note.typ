= GPT-2 Architecture Note

I want to learn how actually to calculate in LLM.

I found a nice article on the architecture of GPT-2: https://jaykmody.com/blog/gpt-from-scratch/.
Thank you for the author.

Matrices are written with sans selif.
Vectors are written with bold.

I did _not_ use generative AI for this note.

#set table(
  stroke: none,
  align: left,
)
#show table: it => align(center, it)

== Hyperparameters

These numbers are for GPT-2 124M.
I'm not sure whether these numbers are also appliable for different parameter sizes (different numbers for 124M)...

#table(
  columns: 2,
  $1 <= v <= V = 50257$, [Number of tokens in the vocabulary],
  $N_"max" = 1024$, [Maximum sequence length],
  $1 <= m <= M = 768$, [Embedding dimension],  // EM-bedding
  $1 <= m' <= 3M = 2304$, [],
  $1 <= m'' <= 4M = 3072$, [],
  $1 <= h <= H = 12$, [Number of attention heads],
  $1 <= l <= L = 12$, [Number of layers],
)
Suppose I input tokens $v_1 thin v_2 thin ... thin v_n thin ... thin v_N$.
Suppose it's shorter than maximum sequence length ($N <= N_"max"$) so everything is inside the context.

== Model parameters

#let sT = $sans(T)$
#let sP = $sans(P)$
#let sX = $sans(X)$
#let bgamma = $bold(gamma)$
#let bbeta = $bold(beta)$
#let sW = $sans(W)$
#let bB = $bold(B)$
#let sw = $sans(w)$
#let bb = $bold(b)$

#table(
  columns: 3,
  $sT_(v m)$, `wte[v, m]`, [Token embedding],
  $sP_(n m)$, `wpe[n, m]`, [Positional embedding],
  $bgamma^"fin"_m$, `ln_f/b[m]`, [Layer normalization constant (final step)],
  $bbeta^"fin"_m$, `ln_f/g[m]`, [Layer normalization constant (final step)],
  $bgamma^("C"(l))_m$, [`h`$(l)$`/ln_1/g[m]`], [Layer normalization constant],
  $bbeta^("C"(l))_m$, [`h`$(l)$`/ln_1/b[m]`], [Layer normalization constant],
  $bgamma^("F"(l))_m$, [`h`$(l)$`/ln_2/g[m]`], [Layer normalization constant],
  $bbeta^("F"(l))_m$, [`h`$(l)$`/ln_2/b[m]`], [Layer normalization constant],
  $bB^("D"(l))_m''$, [`h`$(l)$`/attn/c_attn/b[m'']`], [Attention linear bias],
  $sW^("D"(l))_(m'' m)$, [`h`$(l)$`/attn/c_attn/w[m'', m]`], [Attention linear weight],
  $bb^("E"(l))_m$, [`h`$(l)$`/attn/c_attn/b[m'']`], [Attention projection linear bias],
  $sw^("E"(l))_(m m)$, [`h`$(l)$`/attn/c_attn/w[m'', m]`], [Attention projection linear weight],
)

== Embeddings

$
  [sX^(0) = mat(sT_(v_1 1), dots.c, sT_(v_1 M); dots.v, , dots.v; sT_(v_N 1), dots.c, sT_(v_N M))
  + mat(sP_(1 1), dots.c, sP_(1 M); dots.v, , dots.v; sP_(N 1), dots.c, sP_(N M))]
  wide
  sX^0_(n m) = sT_(v_n m) + sP_(n m)
$

== Decoder Stack

$ sX^l = #[`transformer_block`]^l (sX^(l - 1)) $

== Decoder Block

#let sY = $sans(Y)$

$
  #[`transformer_block`]^l (sX) = sY + #[`ffn`]^l (sY)
  wide sY = sX + #[`mha`]^l (sX)
$

== Layer Normalization


$ #[`layer_norm`] $
