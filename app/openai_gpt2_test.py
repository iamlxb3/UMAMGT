from transformers import GPT2LMHeadModel, TextGenerationPipeline
import ipdb


# python openai_gpt2_test.py

def main():
    # ------------------------------------------------------------------------------------------------------------------
    # test OPEN-AI GPT2
    # ------------------------------------------------------------------------------------------------------------------
    from transformers import GPT2Tokenizer, GPT2Model
    import torch

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs)
    # ipdb> outputs.keys()
    # odict_keys(['logits', 'past_key_values'])
    text_generator = TextGenerationPipeline(model, tokenizer)
    generate_results = text_generator("Hello, my dog is cute", max_length=50, do_sample=False)
    print(text_generator("Hello, my dog is cute", max_length=20, top_p=0.92, top_k=0, temperature=0.85))
    ipdb.set_trace()
    # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
