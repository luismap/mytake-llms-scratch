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

# about different LLms arch
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


# notes
* **about embeddings**: while word embeddings are the most common form of text embedding, there are also
embeddings for sentences, paragraphs, or whole documents. Sentence or paragraph
embeddings are popular choices for retrieval-augmented generation. Retrieval-augmented
generation combines generation (like producing text) with retrieval (like searching an
external knowledge base) to pull relevant information when generating text, which is a
technique that is beyond the scope of this book