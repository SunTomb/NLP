#!/usr/bin/env python3
"""Patch finetune.py to support non-Llama tokenizers (e.g. Qwen2.5)"""

filepath = 'self-rag/retrieval_lm/finetune.py'

with open(filepath, 'r') as f:
    content = f.read()

old_text = """    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings"""

new_text = """    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    else:
        # Generic tokenizer (Qwen, etc.)
        if args.use_special_tokens is True:
            special_token_dict = {"additional_special_tokens": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]}
            num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
            print(f'Added {num_added_tokens} special tokens for generic tokenizer')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        context_markups = []
        for token in ["<paragraph>", "</paragraph>"]:
            context_markups.append(tokenizer.convert_tokens_to_ids(token))

    # We resize the embeddings"""

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(filepath, 'w') as f:
        f.write(content)
    print('SUCCESS: Patched finetune.py with else branch for generic tokenizer')
else:
    print('ERROR: Could not find target text. Already patched?')
    # Check if already patched
    if 'Generic tokenizer' in content:
        print('INFO: Already patched.')
    else:
        print('WARN: Unknown state. Please check manually.')
