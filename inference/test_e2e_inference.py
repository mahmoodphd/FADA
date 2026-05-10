"""
Full End-to-End ONNX Inference Test for FADA Mobile Pipeline.

Simulates the exact Android inference pipeline:
1. Load all 3 ONNX models (embed_tokens, vision_encoder, decoder)
2. Preprocess a real ultrasound image
3. Run vision encoder
4. Build prompt tokens
5. Get token embeddings
6. Merge vision features
7. Chunked prefill through decoder
8. Autoregressive token generation
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import json
import os
import time
import sys

MODEL_DIR = "/media/mshz88/OS/Qwen/fetal_vlm_kd/onnx_export/fada_mobile"
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")

# Model constants (from ModelConfig)
HIDDEN_SIZE = 1024
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
ALIGN_SIZE = PATCH_SIZE * SPATIAL_MERGE_SIZE  # 32
MAX_DIM = 1280
MAX_PATCHES = 1960
PATCH_DIM = 1536  # temporal(2) * channels(3) * patch_h(16) * patch_w(16)
VOCAB_SIZE = 248320
NUM_LAYERS = 24
DELTANET_LAYERS = {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22}
FULL_ATTN_LAYERS = {3,7,11,15,19,23}

# Cache shapes
CONV_SHAPE = (1, 6144, 4)
RECURRENT_SHAPE = (1, 16, 128, 128)
NUM_KV_HEADS = 2
HEAD_DIM = 256

# Special token IDs
IM_START = 248045
IM_END = 248046
EOS = 248044
VISION_START = 248053
VISION_END = 248054
IMAGE_PAD = 248056

print("=" * 60)
print("FADA MOBILE - Full End-to-End Inference Test")
print("=" * 60)

# === STEP 1: Load all 3 models ===
print("\n[STEP 1] Loading ONNX models...")
t_start = time.time()

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.inter_op_num_threads = 4
sess_options.intra_op_num_threads = 4

print("  Loading embed_tokens_q4.onnx...")
t0 = time.time()
embed_sess = ort.InferenceSession(
    os.path.join(ONNX_DIR, "embed_tokens_q4.onnx"),
    sess_options,
    providers=['CPUExecutionProvider']
)
print(f"    Loaded in {time.time()-t0:.1f}s")

print("  Loading vision_encoder_fp16.onnx...")
t0 = time.time()
vision_sess = ort.InferenceSession(
    os.path.join(ONNX_DIR, "vision_encoder_fp16.onnx"),
    sess_options,
    providers=['CPUExecutionProvider']
)
print(f"    Loaded in {time.time()-t0:.1f}s")

print("  Loading decoder_model_merged_q4.onnx...")
t0 = time.time()
decoder_sess = ort.InferenceSession(
    os.path.join(ONNX_DIR, "decoder_model_merged_q4.onnx"),
    sess_options,
    providers=['CPUExecutionProvider']
)
print(f"    Loaded in {time.time()-t0:.1f}s")

print(f"\n  All models loaded in {time.time()-t_start:.1f}s")

# Print model I/O specs
print("\n=== EMBED TOKENS INPUTS ===")
for inp in embed_sess.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print("=== EMBED TOKENS OUTPUTS ===")
for out in embed_sess.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

print("\n=== VISION ENCODER INPUTS ===")
for inp in vision_sess.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print("=== VISION ENCODER OUTPUTS ===")
for out in vision_sess.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

print("\n=== DECODER INPUTS ===")
for inp in decoder_sess.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print("\n=== DECODER OUTPUTS ===")
for out in decoder_sess.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

# === STEP 2: Load and preprocess an image ===
print("\n" + "=" * 60)
print("[STEP 2] Loading and preprocessing image...")

image_dirs = [
    "/media/mshz88/OS/Qwen/fetal_ultrasound_interpret/images/Abdomen/",
    "/media/mshz88/OS/Qwen/fetal_ultrasound_interpret/images/Thorax/",
    "/media/mshz88/OS/Qwen/fetal_ultrasound_interpret/images/Trans-thalamic/",
]

test_image_path = None
for d in image_dirs:
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            test_image_path = os.path.join(d, files[0])
            break

if not test_image_path:
    print("  No test image found, using synthetic gray image")
    img = Image.new('RGB', (640, 480), color='gray')
else:
    print(f"  Using test image: {test_image_path}")
    img = Image.open(test_image_path).convert('RGB')

print(f"  Original image size: {img.size}")

# Preprocess: resize, align to 32px grid, normalize, create patches
w, h = img.size
scale = min(MAX_DIM / max(w, h), 1.0)
new_w = int(w * scale)
new_h = int(h * scale)

# Align to ALIGN_SIZE (32) grid
aligned_w = ((new_w + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE
aligned_h = ((new_h + ALIGN_SIZE - 1) // ALIGN_SIZE) * ALIGN_SIZE

# Check patch limit
total_patches = (aligned_h // PATCH_SIZE) * (aligned_w // PATCH_SIZE)
if total_patches > MAX_PATCHES:
    import math
    scale_down = math.sqrt(MAX_PATCHES / total_patches)
    aligned_h = int((aligned_h * scale_down) // ALIGN_SIZE) * ALIGN_SIZE
    aligned_w = int((aligned_w * scale_down) // ALIGN_SIZE) * ALIGN_SIZE
    aligned_h = max(aligned_h, ALIGN_SIZE)
    aligned_w = max(aligned_w, ALIGN_SIZE)
    total_patches = (aligned_h // PATCH_SIZE) * (aligned_w // PATCH_SIZE)

grid_h = aligned_h // PATCH_SIZE
grid_w = aligned_w // PATCH_SIZE
num_patches = grid_h * grid_w

# After spatial merge, num_features = num_patches / (SPATIAL_MERGE_SIZE^2)
merge_area = SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE
num_features = num_patches // merge_area

print(f"  Aligned size: {aligned_w}x{aligned_h}")
print(f"  Grid: {grid_h}x{grid_w} = {num_patches} patches")
print(f"  After spatial merge ({SPATIAL_MERGE_SIZE}x{SPATIAL_MERGE_SIZE}): {num_features} features")

# Resize and normalize
img_resized = img.resize((aligned_w, aligned_h), Image.BILINEAR)
img_array = np.array(img_resized).astype(np.float32) / 255.0

# Normalize with ImageNet mean/std (Qwen VL uses this)
mean = np.array([0.48145466, 0.4578275, 0.40821073])
std = np.array([0.26862954, 0.26130258, 0.27577711])
img_normalized = (img_array - mean) / std

# Create patches: [num_patches, PATCH_DIM=1536]
# Each patch: temporal(2) * channels(3) * patch_h(16) * patch_w(16)
# For single image, temporal=1 but we duplicate for temporal=2
patches = img_normalized.reshape(grid_h, PATCH_SIZE, grid_w, PATCH_SIZE, 3)
patches = patches.transpose(0, 2, 1, 3, 4)  # [grid_h, grid_w, patch_h, patch_w, 3]
patches = patches.reshape(num_patches, PATCH_SIZE, PATCH_SIZE, 3)

# For temporal dim = 2, duplicate patches
temporal_patches = np.stack([patches, patches], axis=1)  # [num_patches, 2, 16, 16, 3]
temporal_patches = temporal_patches.transpose(0, 1, 4, 2, 3)  # [num_patches, 2, 3, 16, 16]
pixel_values = temporal_patches.reshape(num_patches, -1).astype(np.float32)  # [num_patches, 1536]

print(f"  pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
print(f"  pixel_values range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

# === STEP 3: Run vision encoder ===
print("\n" + "=" * 60)
print("[STEP 3] Running vision encoder...")
image_grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int64)
print(f"  image_grid_thw: {image_grid_thw.tolist()}")
print(f"  Verify: 1 * {grid_h} * {grid_w} = {grid_h * grid_w} == {num_patches} patches")

t0 = time.time()
try:
    vision_outputs = vision_sess.run(None, {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw
    })
    vision_time = time.time() - t0
    vision_features = vision_outputs[0]
    print(f"  SUCCESS! Vision features shape: {vision_features.shape}")
    print(f"  Vision time: {vision_time:.1f}s")
    print(f"  Feature stats: min={vision_features.min():.4f}, max={vision_features.max():.4f}, "
          f"mean={vision_features.mean():.4f}, std={vision_features.std():.4f}")
    
    actual_num_features = vision_features.shape[0]
    actual_hidden = vision_features.shape[1]
    print(f"  num_features={actual_num_features} (expected {num_features}), hidden={actual_hidden} (expected {HIDDEN_SIZE})")
    
    if actual_num_features != num_features:
        print(f"  WARNING: Feature count mismatch! Using actual: {actual_num_features}")
        num_features = actual_num_features
        
except Exception as e:
    print(f"  FAILED! Error: {e}")
    sys.exit(1)

# === STEP 4: Load tokenizer and build prompt ===
print("\n" + "=" * 60)
print("[STEP 4] Building prompt tokens...")

try:
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("  Loaded HuggingFace tokenizer")
    
    # Build chat prompt
    system_text = "You are a fetal ultrasound analysis assistant."
    user_prompt = "Interpret this image"
    
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    text = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"  Chat template (first 200 chars): {text[:200]}...")
    
    token_ids = hf_tokenizer.encode(text, add_special_tokens=False)
    print(f"  Tokenized length (before IMAGE_PAD fix): {len(token_ids)}")
    
    # Count existing IMAGE_PAD tokens
    template_pad_count = token_ids.count(IMAGE_PAD)
    print(f"  IMAGE_PAD tokens in template: {template_pad_count}")
    print(f"  Needed IMAGE_PAD tokens: {num_features}")
    
    # Replace IMAGE_PAD block with correct count
    if VISION_START in token_ids and VISION_END in token_ids:
        vision_start_idx = token_ids.index(VISION_START)
        vision_end_idx = token_ids.index(VISION_END)
        
        new_tokens = token_ids[:vision_start_idx + 1]
        new_tokens.extend([IMAGE_PAD] * num_features)
        new_tokens.extend(token_ids[vision_end_idx:])
        token_ids = new_tokens
    
    print(f"  Final token sequence length: {len(token_ids)}")
    
except ImportError:
    print("  transformers not available, using manual token construction")
    # Load tokenizer.json for vocab
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    
    # Manual construction
    token_ids = [IM_START]
    system_tokens = [vocab.get('system', 9125)]
    token_ids.extend(system_tokens)
    newline_id = vocab.get('\n', 198)
    token_ids.append(newline_id)
    # Approximate system text
    for word in "You are a fetal ultrasound analysis assistant.".split():
        wid = vocab.get(word, vocab.get(word.lower(), 100))
        token_ids.append(wid)
    token_ids.append(IM_END)
    token_ids.append(newline_id)
    
    token_ids.append(IM_START)
    token_ids.append(vocab.get('user', 882))
    token_ids.append(newline_id)
    token_ids.append(VISION_START)
    token_ids.extend([IMAGE_PAD] * num_features)
    token_ids.append(VISION_END)
    token_ids.append(newline_id)
    for word in "Interpret this image".split():
        wid = vocab.get(word, vocab.get(word.lower(), 100))
        token_ids.append(wid)
    token_ids.append(IM_END)
    token_ids.append(newline_id)
    
    token_ids.append(IM_START)
    token_ids.append(vocab.get('assistant', 78191))
    token_ids.append(newline_id)
    
    hf_tokenizer = None
    print(f"  Manual token sequence length: {len(token_ids)}")

seq_len = len(token_ids)

# === STEP 5: Get token embeddings ===
print("\n" + "=" * 60)
print("[STEP 5] Running embed_tokens...")
input_ids = np.array([token_ids], dtype=np.int64)  # [1, seq_len]
print(f"  input_ids shape: {input_ids.shape}")

t0 = time.time()
try:
    embed_outputs = embed_sess.run(None, {"input_ids": input_ids})
    embed_time = time.time() - t0
    embeddings = embed_outputs[0]  # [1, seq_len, HIDDEN_SIZE]
    print(f"  SUCCESS! Embeddings shape: {embeddings.shape}")
    print(f"  Embed time: {embed_time:.1f}s")
    print(f"  Embedding stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, "
          f"mean={embeddings.mean():.4f}")
except Exception as e:
    print(f"  FAILED! Error: {e}")
    sys.exit(1)

# === STEP 6: Merge vision features into embeddings ===
print("\n" + "=" * 60)
print("[STEP 6] Merging vision features into embeddings...")

vision_feature_idx = 0
for i, tid in enumerate(token_ids):
    if tid == IMAGE_PAD and vision_feature_idx < num_features:
        embeddings[0, i, :] = vision_features[vision_feature_idx].astype(np.float32)
        vision_feature_idx += 1

print(f"  Replaced {vision_feature_idx} IMAGE_PAD positions with vision features")
assert vision_feature_idx == num_features, f"Mismatch: replaced {vision_feature_idx} but have {num_features} features"

# === STEP 7: Chunked prefill ===
print("\n" + "=" * 60)
print("[STEP 7] Running chunked prefill through decoder...")

CHUNK_SIZE = 32

# Initialize cache states
conv_states = {i: np.zeros(CONV_SHAPE, dtype=np.float32) for i in DELTANET_LAYERS}
recurrent_states = {i: np.zeros(RECURRENT_SHAPE, dtype=np.float32) for i in DELTANET_LAYERS}
key_caches = {i: np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32) for i in FULL_ATTN_LAYERS}
value_caches = {i: np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float32) for i in FULL_ATTN_LAYERS}

# Get decoder output names
decoder_output_names = [o.name for o in decoder_sess.get_outputs()]
print(f"  Decoder has {len(decoder_output_names)} outputs")
print(f"  First 10 output names: {decoder_output_names[:10]}")

# Also get input names for validation
decoder_input_names = [i.name for i in decoder_sess.get_inputs()]
print(f"  Decoder has {len(decoder_input_names)} inputs")

processed = 0
last_logits = None
num_chunks = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"  seq_len={seq_len}, chunk_size={CHUNK_SIZE}, num_chunks={num_chunks}")

t0 = time.time()
prefill_success = True

while processed < seq_len:
    chunk_len = min(CHUNK_SIZE, seq_len - processed)
    chunk_embeds = embeddings[:, processed:processed+chunk_len, :]  # [1, chunk_len, 1024]
    
    # Attention mask: [1, past_seq_len + chunk_len]
    past_seq_len = key_caches[3].shape[2] if key_caches[3].shape[2] > 0 else 0
    attn_mask_len = past_seq_len + chunk_len
    attention_mask = np.ones((1, attn_mask_len), dtype=np.int64)
    
    # Position IDs: [3, 1, chunk_len] (M-RoPE: temporal, height, width all same)
    pos_start = past_seq_len
    position_ids = np.tile(
        np.arange(pos_start, pos_start + chunk_len).reshape(1, 1, -1),
        (3, 1, 1)
    ).astype(np.int64)
    
    # Build inputs
    inputs = {
        "inputs_embeds": chunk_embeds.astype(np.float32),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    
    for layer in range(NUM_LAYERS):
        if layer in DELTANET_LAYERS:
            inputs[f"past_conv.{layer}"] = conv_states[layer]
            inputs[f"past_recurrent.{layer}"] = recurrent_states[layer]
        else:
            inputs[f"past_key_values.{layer}.key"] = key_caches[layer]
            inputs[f"past_key_values.{layer}.value"] = value_caches[layer]
    
    # Run decoder
    try:
        outputs_list = decoder_sess.run(None, inputs)
        outputs = {name: val for name, val in zip(decoder_output_names, outputs_list)}
        
        # Update cache states
        for layer in DELTANET_LAYERS:
            conv_name = f"present_conv.{layer}"
            rec_name = f"present_recurrent.{layer}"
            if conv_name in outputs:
                conv_states[layer] = outputs[conv_name]
            if rec_name in outputs:
                recurrent_states[layer] = outputs[rec_name]
        
        for layer in FULL_ATTN_LAYERS:
            key_name = f"present.{layer}.key"
            val_name = f"present.{layer}.value"
            if key_name in outputs:
                key_caches[layer] = outputs[key_name]
                value_caches[layer] = outputs[val_name]
        
        # Get logits from the last chunk
        if "logits" in outputs:
            logits_out = outputs["logits"]
            last_logits = logits_out[0, -1, :]  # Last token's logits
        
        processed += chunk_len
        kv_seq = key_caches[3].shape[2]
        elapsed = time.time() - t0
        print(f"    Chunk {processed}/{seq_len} done | KV seq_len={kv_seq} | elapsed={elapsed:.1f}s")
        
    except Exception as e:
        print(f"  PREFILL FAILED at position {processed}: {e}")
        import traceback
        traceback.print_exc()
        prefill_success = False
        break

prefill_time = time.time() - t0
print(f"\n  Prefill {'SUCCEEDED' if prefill_success else 'FAILED'} in {prefill_time:.1f}s")

if prefill_success and last_logits is not None:
    print(f"  Final logits shape: {last_logits.shape}")
    print(f"  Logits stats: min={last_logits.min():.4f}, max={last_logits.max():.4f}")
    top5_ids = np.argsort(last_logits)[-5:][::-1]
    top5_probs = last_logits[top5_ids]
    print(f"  Top-5 token IDs: {top5_ids.tolist()}")
    print(f"  Top-5 logit values: {[f'{v:.3f}' for v in top5_probs]}")

# === STEP 8: Decode tokens ===
if prefill_success and last_logits is not None:
    print("\n" + "=" * 60)
    print("[STEP 8] Autoregressive token generation...")
    
    generated_tokens = []
    decode_times = []
    MAX_GENERATE = 30
    
    for step in range(MAX_GENERATE):
        t_step = time.time()
        
        # Greedy sampling (argmax)
        next_token_id = int(np.argmax(last_logits))
        generated_tokens.append(next_token_id)
        
        # Check for EOS
        if next_token_id in (EOS, IM_END):
            print(f"    Step {step}: EOS/IM_END token={next_token_id} - stopping")
            break
        
        # Decode token to text if possible
        if hf_tokenizer:
            token_text = hf_tokenizer.decode([next_token_id])
        else:
            token_text = f"[{next_token_id}]"
        
        # Get embedding for next token
        next_input_ids = np.array([[next_token_id]], dtype=np.int64)
        next_embeds = embed_sess.run(None, {"input_ids": next_input_ids})[0]
        
        # Build inputs for decode step
        past_kv_len = key_caches[3].shape[2]
        attention_mask = np.ones((1, past_kv_len + 1), dtype=np.int64)
        pos_id = past_kv_len  # next position
        position_ids = np.array([[[pos_id]]] * 3, dtype=np.int64)  # [3, 1, 1]
        
        inputs = {
            "inputs_embeds": next_embeds.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        
        for layer in range(NUM_LAYERS):
            if layer in DELTANET_LAYERS:
                inputs[f"past_conv.{layer}"] = conv_states[layer]
                inputs[f"past_recurrent.{layer}"] = recurrent_states[layer]
            else:
                inputs[f"past_key_values.{layer}.key"] = key_caches[layer]
                inputs[f"past_key_values.{layer}.value"] = value_caches[layer]
        
        try:
            outputs_list = decoder_sess.run(None, inputs)
            outputs = {name: val for name, val in zip(decoder_output_names, outputs_list)}
            
            # Update caches
            for layer in DELTANET_LAYERS:
                conv_name = f"present_conv.{layer}"
                rec_name = f"present_recurrent.{layer}"
                if conv_name in outputs:
                    conv_states[layer] = outputs[conv_name]
                if rec_name in outputs:
                    recurrent_states[layer] = outputs[rec_name]
            for layer in FULL_ATTN_LAYERS:
                key_name = f"present.{layer}.key"
                val_name = f"present.{layer}.value"
                if key_name in outputs:
                    key_caches[layer] = outputs[key_name]
                    value_caches[layer] = outputs[val_name]
            
            last_logits = outputs["logits"][0, -1, :]
            step_time = time.time() - t_step
            decode_times.append(step_time)
            
            print(f"    Step {step}: token={next_token_id:6d} | text='{token_text}' | "
                  f"time={step_time:.2f}s | KV_len={key_caches[3].shape[2]}")
            
        except Exception as e:
            print(f"    DECODE FAILED at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # === RESULTS ===
    print("\n" + "=" * 60)
    print("[RESULTS]")
    print(f"  Generated {len(generated_tokens)} tokens")
    print(f"  Token IDs: {generated_tokens}")
    
    if hf_tokenizer and generated_tokens:
        generated_text = hf_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\n  Generated text:\n  ---")
        print(f"  {generated_text}")
        print(f"  ---")
    
    if decode_times:
        print(f"\n  Decode timing:")
        print(f"    Average: {np.mean(decode_times):.2f}s/token")
        print(f"    Min: {np.min(decode_times):.2f}s, Max: {np.max(decode_times):.2f}s")
        print(f"    Total decode: {sum(decode_times):.1f}s")
    
    print(f"\n  Overall timing:")
    print(f"    Vision encoder: {vision_time:.1f}s")
    print(f"    Embed tokens: {embed_time:.1f}s")
    print(f"    Prefill: {prefill_time:.1f}s")
    if decode_times:
        print(f"    Decode ({len(decode_times)} tokens): {sum(decode_times):.1f}s")
        total_time = vision_time + embed_time + prefill_time + sum(decode_times)
        print(f"    TOTAL: {total_time:.1f}s")

else:
    print("\n[RESULTS] Prefill failed or no logits produced. Cannot decode.")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
