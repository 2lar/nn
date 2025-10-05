**Overview**
- This notebook series builds a character-level name generator. It starts from raw names, encodes them into token indices, trains a multilayer perceptron (MLP) to predict the next character given a fixed window of previous characters, and samples new names token-by-token.
- The core educational goals are to expose the full pipeline end to end: data → tokenization → embeddings → deep MLP with BatchNorm → training loop → evaluation → autoregressive sampling.

**Key Learnings**
- Vocabulary and special tokens: introduce a dedicated end-of-word token `'.'` (index `0`) and build `stoi/itos` mappings for indices ↔ characters.
- Context windows: use a fixed `block_size` (e.g., `3`) and train the model to predict the next character from the last 3 characters.
- Embeddings: learn a table `C` with shape `(vocab_size, n_embd)` to transform token indices into dense vectors; flatten the context’s embeddings into one long feature vector.
- Custom layers: implement `Linear`, `BatchNorm1d`, and `Tanh` with a minimal API; stack them to form a deeper MLP.
- BatchNorm and bias: when BatchNorm follows a Linear layer, the Linear’s bias is often redundant; BN provides a learnable shift via `beta`.
- Initialization and stability: scale weights by `1/sqrt(fan_in)`; optionally reduce the final BN `gamma` to avoid overconfident initial logits.
- Training loop: minibatch sampling, forward → `cross_entropy` → backward → optimizer step; optional LR scheduler; track grad:data ratios.
- Evaluation: put BatchNorm in eval mode to use running statistics; be careful when batch size is 1.
- Autoregressive sampling: slide the context window, forward once per token, sample from softmax probabilities, stop on `'.'`.

**Data Pipeline**
- Input: `names.txt` (one name per line), loaded into `words`.
- Tokenization:
  - Build `chars` from the unique characters in `words`.
  - Create `stoi = {ch: i+1, ...}` and reserve `stoi['.'] = 0`.
  - Invert with `itos = {i: ch for ch, i in stoi.items()}`.
  - `vocab_size = len(itos)`.
- Context construction:
  - Set `block_size = 3` (i.e., condition on 3 previous tokens).
  - For each word, add start/end sentinel `'.'`, e.g., `". . . w1 w2 ... wk ."` (conceptually) and build (X, Y) pairs by sliding a window of length 3 to predict the next token.
- Splits: produce `Xtr, Ytr`, `Xdev, Ydev`, `Xte, Yte` (e.g., 80/10/10) and assert that all indices are within `[0, vocab_size)`.
- Device and seeding:
  - `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
  - Use a `torch.Generator(device=device)` with a fixed seed for reproducibility and to keep sampling on the same device as tensors.

**Model Architecture**
- Embedding table `C`: `(vocab_size, n_embd)`. For an input context of shape `(B, T)`, lookup produces `(B, T, E)`. Flatten to `(B, T*E)` for the MLP.
- Custom `Linear` layer:
  - Parameters: `weight ∈ R[fan_in, fan_out]`, optional `bias ∈ R[fan_out]`.
  - Initialization: `weight ~ N(0, 1/sqrt(fan_in))` for stable activations across depth.
  - Forward: `x @ weight (+ bias)` producing shape `(B, fan_out)`.
- Custom `BatchNorm1d`:
  - Learnable `gamma, beta ∈ R[dim]` and running `mean/var` buffers.
  - Train mode: normalize with batch stats; Eval mode: normalize with running stats.
- `Tanh` activation: simple nonlinearity with saturation monitoring (optional stats in the notebook).
- Deep MLP stack (6 layers total projecting to the vocabulary):
  - `[Linear(n_embd*block_size → n_hidden, bias=False) → BatchNorm1d(n_hidden) → Tanh] × 5`
  - `Linear(n_hidden → vocab_size, bias=False) → BatchNorm1d(vocab_size)`
  - Rationale:
    - Remove Linear biases when followed by BN; BN’s `beta` covers shift.
    - Make the last layer less confident at init by scaling its BN `gamma` down (e.g., `*= 0.1`).
  - Parameter collection: `parameters = [C] + [p for layer in layers for p in layer.parameters()]`.

**Forward Pass Shapes**
- `(B, T)` indices → `emb = C[X]` → `(B, T, E)` → `x = emb.view(B, T*E)` → pass through deep MLP → logits `(B, vocab_size)`.
- Loss: `F.cross_entropy(logits, targets)` where `targets` has shape `(B,)`.

**Training Loop**
- Batch sampling (on-device to avoid sync):
  - `ix = torch.randint(0, Xtr.shape[0], (batch_size,), device=device)` → `Xb, Yb = Xtr[ix], Ytr[ix]`.
- Forward:
  - `emb = C[Xb]` → flatten → iterate `for layer in layers: x = layer(x)` → logits.
- Loss and backward:
  - `loss = F.cross_entropy(logits, Yb)`.
  - `optimizer.zero_grad(set_to_none=True)` then `loss.backward()`.
- Update:
  - `optimizer.step()` and optionally update LR via a scheduler (e.g., `MultiStepLR`).
- Diagnostics:
  - Periodically print `loss`.
  - Track log-loss (e.g., `loss.log10()`) and grad:data ratios `((lr * p.grad).std() / (p.data.std()+1e-12)).log10()` to monitor training health.

**Evaluation**
- Switch BN to eval for consistent statistics at validation/test time: `for layer in layers: layer.training = False`.
- Compute `split_loss('train')` and `split_loss('val')` by running a forward pass over the corresponding tensors without grad tracking.
- Note on BatchNorm:
  - With very small batch sizes (especially 1), per-batch stats are noisy; use eval mode to rely on running stats. Earlier parts of the notebook demonstrate a BN “calibration” pass (computing mean/std on the full dataset) vs using running stats.

**Autoregressive Generation**
- Put BN in inference mode (eval) to avoid per-batch stats with batch size 1.
- Initialize the context as all end tokens: `context = [0] * block_size` (index 0 is '.')
- Loop until `'.'` is sampled or a step cap is reached:
  - Embed: `emb = C[context]` → flatten to `(1, T*E)`.
  - Forward through the deep MLP: `for layer in layers: x = layer(x)` → `logits`.
  - Convert to probabilities with `F.softmax(logits, dim=1)`.
  - Sample next index with `torch.multinomial(probs, num_samples=1, generator=g)`.
  - Slide the window: `context = context[1:] + [ix]` and append to output.
  - Stop when `ix == 0` (the `'.'` token).
- Decode the collected indices with `itos` to produce a name.
- Variations to explore (optional): temperature scaling on logits, top-k/top-p sampling, beam search.

**Initialization & Stability Tips**
- Weight scale: use `1/sqrt(fan_in)` to keep activations/gradients stable across depth.
- Final confidence: reduce final BN `gamma` (or final layer weight scale) to start with softer logits.
- Gradients: monitor grad:data ratios; large deviations may indicate LR or initialization issues.
- Biases with BN: disable Linear biases when followed by BN to save parameters and reduce redundancy.

**Device & Reproducibility**
- Keep generators and tensors on the same `device` (CPU/CUDA) to avoid device mismatch and sync overhead.
- Seed both CPU and CUDA RNGs for reproducibility.
- Indexing with CUDA LongTensor keeps embedding lookups on-GPU.

**Comparing Notebook Sections**
- Early prototype (2-layer MLP with manual tensors): demonstrates the mechanics of embeddings, affine transforms, BN calibration, and sampling without the modular layer classes.
- Deeper modular model (6-layer): introduces `Linear/BatchNorm1d/Tanh` classes to mirror a subset of `nn.Module` behavior, enabling a deeper stack and easier experimentation with inits and BN.

**Practical Pointers**
- Parameter count: printing `sum(p.nelement() for p in parameters)` helps verify model size (e.g., ~47k in one config shown).
- BatchNorm mode:
  - Training: use batch stats; Eval: use running stats; Sampling: set BN to eval.
- Optimizers: start with SGD; Adam often converges faster; schedulers can help with later-stage refinement.
- Stopping criteria: training length is a tradeoff; watch validation loss and qualitative samples.

**Where To Look In The Notebook**
- Data/Vocab/Context: construction of `stoi/itos`, `block_size`, and `X*/Y*`.
- Model setup: `n_embd`, `n_hidden`, creation of `C` and `layers` with `Linear/BatchNorm1d/Tanh`.
- Training loop: minibatch sampling, forward, `F.cross_entropy`, `optimizer.step()`.
- Evaluation: BN set to eval and `split_loss` calls.
- Generation: context loop using `C`, `layers`, `softmax`, and `torch.multinomial` to print names.

This README summarizes the learning-focused structure implemented in the makemore notebook, emphasizing clear shapes, stable initialization, BN behavior, and an explicit sampling procedure. Use it as a guide while reading or modifying the notebook cells.

