* overview of AI ![alt text](images/ai_overview.png)
* process of training an LLM ![alt text](images/train_llm.png)
* simplified transformer arch p.12 ![alt text](images/simplified_trans.png)
* overview of encoder decoder modules p.14 ![alt text](images/enc_dec.png)
* pretraining dataset of gpt3 p.16 ![alt text](images/gpt3_pre_ds.png)
* transformer arch and tasks ![alt text](images/trans_task.png)
* decoder only approach ![alt text](images/decoder_only.png)
* stages for building an llm p.20 ![alt text](images/build_stages_llms.png)
* embedding models ![alt text](images/embeddings.png)
* tokenization process ![alt text](images/tokenization_proc.png)
* vocabulary ![alt text](images/vocab.png)
* llms predicts next token ![alt text](images/predict_n_tok.png)
* embeddings layer perform lookup ops ![alt text](images/embedding_lookup.png)
* a note about embedding layer p.58 ![alt text](images/embedding_note.png)
* input embeddings ![alt text](images/input_embed.png)
* ![alt text](images/embedding_pipeline.png)

# about self attention
* self attention mechanism ![alt text](images/self_attention.png)
* about RNNs p.67 ![alt text](images/rnns.png)
* rnns continuation ![alt text](images/rnns_cont.png)
* note on scaled dot product attention ![alt text](images/scaled_dot_product.png)
* about q,k,w matrices ![alt text](images/about_qkv.png)
```python
# a simple implementation of the attention mechanism
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        """
        note: you should use nn.Linear
        because it give weight_initialization, and __apply__ defaults to matrix multplication
        when bias unit disabled.
        ex.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        queries = self.W_query(x)
        """

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```
* summarization of the simple self attention mechanism ![alt text](images/self_attn_summary.png)


# about different LLms arch
note: base on [article](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
* multihead vs grouped query attention ![alt text](images/mh_gqa.png)
* multihead latent attention (mla) ![alt text](images/mla.png)
* mixture of experts (MoE) ![alt text](images/mixture_of_experts.png)
* types of MoE ![alt text](images/moe.png)
* post-normalization, pre-norm, and olmo ![alt text](imags/prolmo.png)
* llama3 vs olmo2 ![alt text](images/llama_olmo.png)
* sliding attention (in the gemma models) vs normal ![alt text](images/sliding_attn.png)
* olmo vs gemma3 arch comparison ![alt text](images/olmo_vs_gemma3.png)
* gemma3 vs mistral 3.1 arch ![alt text](images/gemma_mistral.png)
* llama3.2 vs qwen3 ![alt text](images/llama_qwen.png)
* deepseekv3 vs Qwen3 ![alt text](images/deepseekv3_qwen3.png)
* smollm3 vs qwen3 ![alt text](images/smollm3_qwen3.png)
* about NoPE(no positional encodings)![alt text](images/nope.png)
* gptoss ![alt text](images/gptoss.png)

# notes
* **about embeddings**: while word embeddings are the most common form of text embedding, there are also
embeddings for sentences, paragraphs, or whole documents. Sentence or paragraph
embeddings are popular choices for retrieval-augmented generation. Retrieval-augmented
generation combines generation (like producing text) with retrieval (like searching an
external knowledge base) to pull relevant information when generating text, which is a
technique that is beyond the scope of this book