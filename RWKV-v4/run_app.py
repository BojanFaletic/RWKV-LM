import gradio as gr

import numpy as np
import math, os
import time
import types
import copy
import torch
from torch.nn import functional as F
from src.utils import TOKENIZER, Dataset
from src.model_run import RWKV_RNN


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.manual_seed(0)


TOKEN_MODE = 'pile' # char / bpe / pile

n_layer = 24
n_embd = 2048
ctx_len = 1024

top_p = 0.7
top_p_newline = 0.9 # only used in TOKEN_MODE = char

WORD_NAME = ['20B_tokenizer.json', '20B_tokenizer.json']
UNKNOWN_CHAR = None

#---> you can set MODEL_NAME to your fine-tuned model <---

MODEL_NAME = 'RWKV-4-Pile-1B5-20220903-8040'


os.environ['RWKV_FLOAT_MODE'] = 'fp16'  # 'bf16' / 'fp16' / 'fp32' (note: only using fp32 at this moment)
os.environ['RWKV_RUN_DEVICE'] = 'cuda'   # 'cpu' (already very fast) or 'cuda'
model_type = 'RWKV' # 'RWKV' or 'RWKV-ffnPre'

context = "Elon Musk "

model = RWKV_RNN(MODEL_NAME, os.environ['RWKV_RUN_DEVICE'], model_type, n_layer, n_embd, ctx_len)
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)



TEMPERATURE=1
NUM_TRIALS=1
LENGTH_PER_TRIAL=100


def predict(context):
    if tokenizer.charMode:
        context = tokenizer.refine_context(context)
        ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
    else:
        ctx = tokenizer.tokenizer.encode(context)
    src_len = len(ctx)
    src_ctx = ctx.copy()

    for TRIAL in range(NUM_TRIALS):
        t_begin = time.time_ns()
        print(('-' * 30) + context, end='')
        ctx = src_ctx.copy()
        model.clear()
        if TRIAL == 0:
            init_state = types.SimpleNamespace()
            for i in range(src_len):
                x = ctx[:i+1]
                if i == src_len - 1:
                    init_state.out = model.run(x)
                else:
                    model.run(x)
            model.save(init_state)
        else:
            model.load(init_state)

        for i in range(src_len, src_len + LENGTH_PER_TRIAL):
            x = ctx[:i+1]
            x = x[-ctx_len:]

            if i == src_len:
                out = copy.deepcopy(init_state.out)
            else:
                out = model.run(x)

            if TOKEN_MODE == 'pile':
                out[0] = -999999999  # disable <|endoftext|>

            char = tokenizer.sample_logits(out, x, ctx_len, temperature=TEMPERATURE,
                                        top_p_usual=top_p, top_p_newline=top_p_newline)
            char = char.item()
            if tokenizer.charMode:
                ch = tokenizer.itos[int(char)]
                context += ch
                print(ch, end='', flush=True)
            else:
                ch = tokenizer.tokenizer.decode(int(char))
                print(ch, end='', flush=True)
                context += ch
            ctx += [char]

        t_end = time.time_ns()
        print("\n----------", round((t_end - t_begin) / (10 ** 9), 2), end='s ')
        print(f"Message: {context}")
        return context

INPUTS = gr.inputs.Textbox()
INTERFACE = gr.Interface(fn=predict,
    inputs=INPUTS, outputs=INPUTS, title="RKWV-RNN",

    description="RKWV-RNN is GPT2 like model (1.5B)",
    thumbnail="RKWV-RNN",
    capture_session=False)

INTERFACE.launch(inbrowser=False)
