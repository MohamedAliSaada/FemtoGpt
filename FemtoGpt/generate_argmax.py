import tensorflow as tf

# Step 1: Define Configuration
gpt_config = {
    "vocab_size": 6000,
    "max_len": 512,
    "d_model": 128,
    "dropout_rate": 0.1,
    "num_heads": 2,
    "expansion": 4,
    "num_layers": 2,
    "rich": 0
}


def generate_argmax(model, tokenizer, prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt)
    
    for _ in range(max_new_tokens):
        input_tensor = tf.convert_to_tensor([input_ids], dtype=tf.int32)
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, -1]
        probs = tf.nn.softmax(next_token_logits)
        next_token_id = tf.argmax(probs).numpy()
        input_ids.append(int(next_token_id))

    return tokenizer.decode([int(x) for x in input_ids])

