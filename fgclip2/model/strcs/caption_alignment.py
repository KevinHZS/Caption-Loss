def get_caption_text_logits(caption_logits, num_image_tokens):
    return caption_logits[:, num_image_tokens - 1 : -1, :]
