# Transformer Learning Roadmap

This repository is an educational Transformer implementation path. It starts with a manual NumPy implementation, moves to a trainable PyTorch encoder-decoder model, then branches into two specialized Transformer families: BERT-style encoder-only modeling and GPT2-small-style decoder-only modeling.

The goal is to make the architecture easy to inspect, modify, quantize, sparsify, and compare against future Memory Mosaics-style experiments.

---

## Roadmap

```text
+-----------------------+      +--------------------------+      +---------------------------+
| Transformer_S1.ipynb  | ---> | Transformer_S2(1).ipynb  | ---> | Transformer_S3.ipynb       |
| NumPy / from scratch  |      | reusable functions       |      | PyTorch / real experiment  |
+-----------------------+      +--------------------------+      +---------------------------+
                                                                     |                  |
                                                                     v                  v
                                                        +--------------------+   +----------------------+
                                                        | BERT.ipynb         |   | GPT2Small.ipynb       |
                                                        | encoder-only       |   | decoder-only          |
                                                        +--------------------+   +----------------------+
```

Main idea:

```text
S1 -> understand every small operation
S2 -> organize the operations into reusable functions
S3 -> train a full encoder-decoder baseline on BabiStories
BERT -> study encoder-only masked language modeling
GPT2Small -> study decoder-only causal language modeling
```

---

## File Map

| File | Architecture Type | Main Role | Training / Output Task |
|---|---|---|---|
| `Transformer_S1.ipynb` | manual encoder-decoder | understand operations from scratch | forward pass / token prediction demo |
| `Transformer_S2(1).ipynb` | function-based encoder-decoder | organize the same logic into functions | forward, loss, accuracy |
| `Transformer_S3.ipynb` | PyTorch encoder-decoder | trainable seq2seq baseline | source story segment -> next story segment |
| `BERT.ipynb` | encoder-only Transformer | simplest BERT-style model | predict masked tokens |
| `GPT2Small.ipynb` | decoder-only Transformer | simplest GPT2-style model | predict next token |

---

## Dataset

The current dataset is BabiStories.

Expected local structure:

```text
BabiStories/data/
|-- extracted/
|-- babistories-dataset.7z
|-- babistories-dataset.7z.002
|-- babistories-dataset.7z.003
|-- babistories-dataset.7z.004
`-- babistories-dataset.7z.005
```

The notebooks read from:

```text
BabiStories/data/extracted
```

---

# 1. Transformer_S1.ipynb

## Scope

`Transformer_S1.ipynb` is the most detailed notebook. It uses NumPy-style manual operations to show the internal structure of a Transformer.

It focuses on:

```text
tokenization, one-hot, embedding, Q/K/V, softmax, attention, multi-head attention,
residual connection, LayerNorm, FFN, encoder block, decoder block, cross-attention
```

## S1 Input and Embedding Flow

S1 should show small steps because the code itself is low-level.

```text
Sentence ----> split() ----> token IDs ----> one-hot vectors ----> E * one_hot ----> X
```

Expanded view:

```text
+----------+    +------------+    +-----------+    +--------------+    +-----------------+
| sentence | -> | token list | -> | token ids | -> | one-hot vecs | -> | embeddings X    |
+----------+    +------------+    +-----------+    +--------------+    +-----------------+
```

## S1 Self-Attention Operation

Because S1 explicitly computes Q, K, V, scores, softmax, and output, the diagram should expose those operations.

```text
                         +--------------------+
X ---- W_Q * X --------> | Q                  |
                         +--------------------+
X ---- W_K * X --------> | K                  | ----+
                         +--------------------+     |
X ---- W_V * X --------> | V                  |     |
                         +--------------------+     |
                                                      v
        +---------------------------------------------------------------+
        | scores = Q.T * K / sqrt(d_head)                               |
        +---------------------------------------------------------------+
                                      |
                                      v
        +---------------------------------------------------------------+
        | attention_weights = softmax(scores)                            |
        +---------------------------------------------------------------+
                                      |
                                      v
        +---------------------------------------------------------------+
        | attention_output = V * attention_weights.T                     |
        +---------------------------------------------------------------+
```

## S1 Softmax Operation

S1 has an explicit softmax function, so the README diagram should show softmax internally.

```text
scores ----> exp(scores) ----> row-wise sum ----> exp(scores) / row-wise sum ----> probabilities
```

Expanded view:

```text
+--------+    +-------------+    +--------------+    +------------------------------+    +-------------+
| scores | -> | exp(scores) | -> | sum by row   | -> | exp(scores) / row-wise sum   | -> | softmax out |
+--------+    +-------------+    +--------------+    +------------------------------+    +-------------+
```

## S1 Encoder-Decoder Path

```text
Source sentence -> source embedding -> encoder self-attention -> encoder FFN -> encoder output
                                                                              |
                                                                              v
Target prefix -> target embedding -> masked self-attention -> cross-attention with encoder output -> decoder FFN -> vocab projection -> predicted token
```

Block view:

```text
+--------+   +-----------+   +----------------+   +---------+   +----------------+
| Source |-> | Embedding |-> | Self-Attention |-> | FFN+Norm |-> | Encoder Output |
+--------+   +-----------+   +----------------+   +---------+   +----------------+
                                                                          |
+--------+   +-----------+   +-------------------+   +----------------+   v   +-----------+   +--------+
| Target |-> | Embedding |-> | Masked Self-Attn |-> | Cross-Attn     |------> | FFN+Norm  |-> | Logits |
+--------+   +-----------+   +-------------------+   +----------------+       +-----------+   +--------+
```

## Why S1 Exists

```text
Purpose: understand every operation before using classes, autograd, batching, and datasets.
Best for: studying math and debugging shapes.
Not best for: long training or fair model comparison.
```

---

# 2. Transformer_S2(1).ipynb

## Scope

`Transformer_S2(1).ipynb` keeps the same encoder-decoder logic but turns the repeated code into reusable functions.

It focuses on:

```text
prepare_input(), multi_head_attention(), feed_forward_network(), encoder_block(),
decoder_block(), encoder_stack(), decoder_stack(), vocabulary_projection(), loss, accuracy
```

## S2 Data Preparation Flow

S2 should show named functions instead of every low-level operation.

```text
raw sentences -> prepare_dataset_context() -> vocabulary/context/E
              -> prepare_input() ----------> tokens + ids + X
```

Expanded view:

```text
+----------------+    +---------------------------+    +-----------------------------+
| sentence pairs | -> | prepare_dataset_context() | -> | vocab, token_to_id, E, ctx  |
+----------------+    +---------------------------+    +-----------------------------+
                              |
                              v
+-------------+    +----------------+    +-------------------+
| sentence    | -> | prepare_input()| -> | tokens, ids, X    |
+-------------+    +----------------+    +-------------------+
```

## S2 Multi-Head Attention Function

```text
query_input + key_value_input
             |
             v
+--------------------------------------------------------------------------------+
| multi_head_attention()                                                          |
| head 1: Q,K,V -> scores -> softmax -> output                                    |
| head 2: Q,K,V -> scores -> softmax -> output                                    |
| ...                                                                            |
| concat heads -> output projection W_O                                           |
+--------------------------------------------------------------------------------+
             |
             v
attention_output, attention_weights
```

## S2 Encoder-Decoder Function Flow

```text
encoder_X -> encoder_stack() -> encoder_output
                                  |
decoder_X -> decoder_stack(encoder_output) -> decoder_output -> vocabulary_projection() -> loss/accuracy
```

Expanded view:

```text
+-----------+     +----------------------+     +----------------+
| encoder_X | --> | encoder_block x N    | --> | encoder_output |
+-----------+     +----------------------+     +----------------+
                                                       |
+-----------+     +----------------------+             |
| decoder_X | --> | decoder_block x N    | <-----------+
+-----------+     | masked self-attn     |
                  | cross-attn           |
                  | FFN                  |
                  +----------------------+
                              |
                              v
                  +--------------------------+     +----------------+
                  | vocabulary_projection()  | --> | loss/accuracy  |
                  +--------------------------+     +----------------+
```

## Why S2 Exists

```text
Purpose: bridge from manual math to modular model flow.
Best for: checking function boundaries, shapes, and reusable encoder/decoder logic.
Not best for: large-scale training.
```

---

# 3. Transformer_S3.ipynb

## Scope

`Transformer_S3.ipynb` is the trainable PyTorch encoder-decoder Transformer. It is the main seq2seq baseline.

It focuses on:

```text
BabiStories loading, story-to-pair conversion, train/valid/test split, batching,
model classes, training loop, validation, generation, quantization hooks, sparsity hooks
```

## S3 Data Pipeline

S3 should show dataset and training-level modules, not softmax internals.

```text
BabiStories text -> make_babistory_pairs() -> train/valid/test -> build_context() -> make_batch()
```

Expanded view:

```text
+------------------+   +-------------------------+   +--------------------+   +------------------+   +----------------------+
| BabiStories text |-> | make_babistory_pairs()  |-> | train/valid/test   |-> | build_context()  |-> | make_batch()         |
+------------------+   | source = story segment  |   +--------------------+   | vocab + IDs      |   | src, dec, y, masks   |
                       | target = next segment   |                            +------------------+   +----------------------+
                       +-------------------------+
```

## S3 Encoder-Decoder Model

```text
src -> token+pos embedding -> Encoder Stack --------------------+
                                                                |
                                                                v
dec -> token+pos embedding -> Masked Decoder Stack -> Cross-Attention -> logits -> CE loss / accuracy
```

Expanded view:

```text
+------------+   +----------------------+   +-------------------------------+
| src tokens |-> | src embedding + pos  |-> | Encoder Stack                 |
+------------+   +----------------------+   | EncBlock x num_enc            |
                                            | MHA -> FFN -> residual/norm    |
                                            +-------------------------------+
                                                            |
                                                            v
+------------+   +----------------------+   +-------------------------------+   +--------+   +----------------+
| dec tokens |-> | dec embedding + pos  |-> | Decoder Stack                 |-> | logits |-> | loss/accuracy  |
+------------+   +----------------------+   | masked MHA                    |   +--------+   +----------------+
                                            | cross-attn with encoder output |
                                            | FFN -> residual/norm           |
                                            +-------------------------------+
```

## S3 Multi-Head Attention Level

At S3, it is enough to show MHA as a module.

```text
input -> MHA(num_heads) -> concat heads -> output projection -> residual + LayerNorm -> FFN -> residual + LayerNorm
```

Expanded module view:

```text
+-------+   +------------------+   +--------------+   +-------------------+   +-----+   +------+
| input |-> | MultiHeadAttention|-> | concat heads |-> | output projection |-> | Add |-> | Norm |
+-------+   +------------------+   +--------------+   +-------------------+   +-----+   +------+
                                                                                |
                                                                                v
                                                                       +------+   +-----+   +------+
                                                                       | FFN  |-> | Add |-> | Norm |
                                                                       +------+   +-----+   +------+
```

## S3 Quantization and Sparsity Hook Flow

```text
Linear layer input X + weight W
       |
       v
qa(X) + qw(W) -> F.linear() -> output
```

Detailed hook view:

```text
W -> sparsify(W) if use_ws=True -> qste(W, w_bits) if use_wq=True -> simulated low-bit W
X -> qste(X, a_bits) if use_aq=True -------------------------------> simulated low-bit X

simulated low-bit X + simulated low-bit W -> floating-point matmul in PyTorch
```

Important:

```text
w_bits=8 and a_bits=8 only define the simulated quantization bit-width.
They do not mean real INT8 hardware execution unless real quantized kernels are used.
```

## S3 Training Loop

```text
make_batch() -> model(src, dec) -> logits -> cross_entropy -> backward -> clip_grad_norm -> AdamW step -> evaluate -> save best_state
```

Expanded view:

```text
+-------------+   +---------------+   +--------+   +----------------+   +------------+   +----------+   +----------+
| make_batch  |-> | model forward |-> | logits |-> | CE loss        |-> | backward() |-> | AdamW    |-> | evaluate |
+-------------+   +---------------+   +--------+   +----------------+   +------------+   +----------+   +----------+
                                                                                                      |
                                                                                                      v
                                                                                              save best_state
```

## Why S3 Exists

```text
Purpose: train and evaluate your encoder-decoder Transformer baseline.
Best for: BabiStories seq2seq experiments, quantization simulation, sparsity simulation.
Closest comparison target: not Memory Mosaics directly, but your own encoder-decoder baseline.
```

---

# 4. BERT.ipynb

## Scope

`BERT.ipynb` is a minimal encoder-only Transformer trained from scratch. It is not official pretrained BERT.

It focuses on:

```text
BabiStories fixed-length examples, random masking, encoder-only blocks,
masked token prediction, fill-mask inference, quantization/sparsity hooks
```

## BERT Architecture

BERT should be shown horizontally as an encoder-only masked-language model.

```text
text -> encode -> randomly mask tokens -> token+pos embedding -> Encoder Stack -> vocab logits -> predict masked token
```

Expanded view:

```text
+------+   +--------+   +----------------------+   +----------------------+   +---------------+   +--------+   +----------------------+
| text |-> | encode |-> | random mask          |-> | token embedding+pos  |-> | Encoder Stack |-> | logits |-> | masked token output  |
+------+   +--------+   | Mary went to <MASK>  |   +----------------------+   +---------------+   +--------+   +----------------------+
                         +----------------------+
```

## BERT Attention Pattern

BERT uses bidirectional self-attention. It hides only padding positions, not future tokens.

```text
Token positions:  1   2   3   4   5
                  |\  |\  |\  |\  |\
                  | \ | \ | \ | \ | \
Each token  ----> can attend to all non-PAD positions
```

Matrix view:

```text
BERT attention mask, ignoring PAD:

        attends to ->  1  2  3  4  5
query 1              [1  1  1  1  1]
query 2              [1  1  1  1  1]
query 3              [1  1  1  1  1]
query 4              [1  1  1  1  1]
query 5              [1  1  1  1  1]
```

## BERT Training Objective

```text
Original:      Mary went to the park .
Masked input:  Mary went to the <MASK> .
Target:        park
```

## Why BERT Exists

```text
Purpose: study encoder-only Transformer behavior.
Best for: understanding both left and right context.
Not best for: direct left-to-right text generation.
```

---

# 5. GPT2Small.ipynb

## Scope

`GPT2Small.ipynb` is a minimal decoder-only causal Transformer trained from scratch. It is not official pretrained GPT-2.

It focuses on:

```text
BabiStories language-modeling examples, causal self-attention,
next-token prediction, validation, generation, quantization/sparsity hooks
```

## GPT2-small Architecture

GPT2-small should be shown horizontally as a decoder-only language model.

```text
tokens -> x=tokens[:-1] -> token+pos embedding -> Causal Decoder Stack -> logits -> next-token prediction
```

Expanded view:

```text
+--------+   +-------------------+   +----------------------+   +----------------------+   +--------+   +------------------+
| tokens |-> | x=tokens[:-1]     |-> | token embedding+pos  |-> | Causal Decoder Stack|-> | logits |-> | next token       |
+--------+   | y=tokens[1:]      |   +----------------------+   +----------------------+   +--------+   +------------------+
             +-------------------+
```

## GPT2-small Causal Attention Pattern

GPT-style models generate left to right. Each token can attend only to itself and previous tokens.

```text
Token positions:  1 -> 2 -> 3 -> 4 -> 5
Prediction flow:  use previous context to predict next token
```

Matrix view:

```text
GPT2 causal attention mask:

        attends to ->  1  2  3  4  5
query 1              [1  0  0  0  0]
query 2              [1  1  0  0  0]
query 3              [1  1  1  0  0]
query 4              [1  1  1  1  0]
query 5              [1  1  1  1  1]
```

## GPT2-small Training Objective

```text
Input x:   Mary went to the
Target y:  went to the park

Mary -> went
went -> to
to   -> the
the  -> park
```

## GPT2-small Generation

```text
prompt -> model -> next token -> append token -> model again -> next token -> ... -> EOS or max length
```

Expanded view:

```text
+--------+   +------------+   +--------------+   +-------------+   +----------------+
| prompt |-> | model step |-> | next token   |-> | append      |-> | repeat         |
+--------+   +------------+   +--------------+   +-------------+   +----------------+
```

## Why GPT2Small Exists

```text
Purpose: study decoder-only causal language modeling.
Best for: generation and comparison with Memory Mosaics-style language modeling baselines.
Closest to Memory Mosaics comparison direction: yes.
```

---

# Horizontal Architecture Comparison

```text
Your S3 encoder-decoder:
source -> encoder ----------------------------+
                                             |
target prefix -> masked decoder -> cross-attn +-> target prediction

BERT encoder-only:
text with <MASK> -> encoder stack -> masked token prediction

GPT2Small decoder-only:
previous tokens -> causal decoder stack -> next token prediction
```

---

# Quantization and Sparsity Meaning

```python
"w_bits": 8
"a_bits": 8
```

These fields define the bit-width used by fake/simulated quantization hooks.

```text
use_wq=False and use_aq=False:
    normal floating-point computation

use_wq=True or use_aq=True:
    simulated low-bit values through qste()
    computation still uses PyTorch floating-point matmul

real INT8 execution:
    not implemented yet
    requires real quantized kernels/operators
```

---

# Suggested Experiment Order

```text
1. Run S3 baseline without quantization or sparsity.
2. Run GPT2Small baseline on the same BabiStories source.
3. Compare train_loss, val_loss, accuracy, and generated text.
4. Enable fake weight quantization: use_wq=True.
5. Enable fake activation quantization: use_aq=True.
6. Enable weight sparsity: use_ws=True.
7. Enable attention sparsity: attn_top_k=8, 16, or 32.
8. Later add a Memory Mosaic notebook and compare it mainly against GPT2Small.
```

---

# Future Memory Mosaic Comparison Path

```text
Current baseline closest to Memory Mosaics:
GPT2Small.ipynb

Reason:
Memory Mosaics language experiments are closest to decoder-only language modeling,
not encoder-decoder seq2seq or BERT-style masked token prediction.
```

Future notebooks:

```text
MemoryMosaic_S1.ipynb       -> minimal associative memory unit
MemoryMosaic_S2.ipynb       -> GPT2-like Memory Mosaic block
Compare_GPT2_vs_Mosaic.ipynb -> same BabiStories split, same metrics, same sequence length
```
