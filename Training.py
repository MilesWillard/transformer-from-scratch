import Transformer
import logging
import torch
from torch import nn
from tokenizers import Tokenizer

from pathlib import Path
import argparse
import csv
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to disk instead of opening a window (plt.show() would block an overnight run)
import matplotlib.pyplot as plt

torch._logging.set_logs(inductor=logging.ERROR)

EXPERIMENT = 'Speed' #Name of the save file in log.

START_FROM = None  # start the run from random weights (none), best of experiment ("best"), or final output of experiment ("last")

CONTEXT = 512
BATCH_SIZE = 100
STEPS = 3000
MAX_LEARNING_RATE = 1e-3

EVAL_INTERVAL = 50      # evaluate every N optimizer steps (plus step 0 and the final step)
TRAIN_EVAL_SEQS = 50    # fixed slice of training sequences, evaluated in eval mode, for a noise-free train curve

SEED = 70

dt = torch.float32

tokenizer = Tokenizer.from_file("data/tokenizer.json")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(SEED)

def token_batches(file, context):
    with open(file, 'r') as f:
        text = f.read()

    tokens = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    num_batches = (len(tokens) - 1) // context
    total_len = num_batches * context

    inputs = tokens[:total_len].view(num_batches, context).to(device)
    targets = tokens[1:total_len + 1].view(num_batches, context).to(device)

    # bytes per token for this file, used to convert loss to bits-per-byte (tokenizer-independent)
    bytes_per_token = len(text.encode('utf-8')) / len(tokens)

    return inputs, targets, bytes_per_token

@torch.no_grad()
def evaluate(model, inputs, targets, batch_size):
    """Mean per-token cross-entropy over ALL given sequences.
    Sums token losses and divides by the token count once, so a smaller last batch is weighted correctly."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for j in range(0, inputs.size(0), batch_size):
        logits = model(inputs[j:j+batch_size])
        tgt = targets[j:j+batch_size]
        total_loss += torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), tgt, reduction='sum').item()
        total_tokens += tgt.numel()
    if was_training:
        model.train()
    return total_loss / total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-lr", type=float, default=MAX_LEARNING_RATE)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--run-name", type=str, default=EXPERIMENT)
    args = parser.parse_args()

    run_dir = os.path.join("log", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("data/weights", exist_ok=True)

    INPUT, TARGET, train_bpt = token_batches('data/shakespeare_train.txt', CONTEXT)
    VAL_INPUT, VAL_TARGET, val_bpt = token_batches('data/shakespeare_test.txt', CONTEXT)
    # fixed subset of training data (taken before any shuffling) for an apples-to-apples train-vs-val comparison
    TRAIN_EVAL_INPUT = INPUT[:TRAIN_EVAL_SEQS].clone()
    TRAIN_EVAL_TARGET = TARGET[:TRAIN_EVAL_SEQS].clone()

    model = Transformer.Transformer().to(device)
    model.to(dtype=dt)
    model = torch.compile(model)
    if START_FROM:
        model.load_state_dict(torch.load(f'data/weights/{EXPERIMENT}_{START_FROM}.pth', map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), 0, weight_decay=.01)

    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = math.ceil(INPUT.size(0) / args.batch_size)
    config = {
        "run_name": args.run_name, "device": str(device), "params": n_params,
        "context": CONTEXT, "batch_size": args.batch_size, "steps": args.steps, "max_lr": args.max_lr,
        "weight_decay": 0.01, "eval_interval": args.eval_interval, "start_from": START_FROM,
        "train_sequences": INPUT.size(0), "val_sequences": VAL_INPUT.size(0),
        "train_tokens": INPUT.numel(), "val_tokens": VAL_INPUT.numel(), 
        "steps_per_epoch": steps_per_epoch, "train_bytes_per_token": train_bpt, "val_bytes_per_token": val_bpt,
        "vocab": Transformer.TOKENS, "dim": Transformer.DIM, "layers": Transformer.LAYERS, "heads": Transformer.HEADS,
        "seed":SEED, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2), flush=True)

    train_f = open(os.path.join(run_dir, "train_steps.csv"), "w", newline="")
    train_w = csv.writer(train_f)
    train_w.writerow(["step", "epoch", "tokens_seen", "train_loss", "grad_norm", "lr", "step_time_s"])
    eval_f = open(os.path.join(run_dir, "eval.csv"), "w", newline="")
    eval_w = csv.writer(eval_f)
    eval_w.writerow(["step", "epoch", "tokens_seen", "val_loss", "val_ppl", "val_bpb", "train_eval_loss", "wall_time_s"])

    t_start = time.time()
    best_val = float("inf")
    tokens_seen = 0

    def run_eval(step, epoch):
        nonlocal best_val
        val_loss = evaluate(model, VAL_INPUT, VAL_TARGET, args.batch_size)
        train_eval_loss = evaluate(model, TRAIN_EVAL_INPUT, TRAIN_EVAL_TARGET, args.batch_size)
        val_bpb = val_loss / math.log(2) / val_bpt
        eval_w.writerow([step, epoch, tokens_seen, val_loss, math.exp(val_loss), val_bpb, train_eval_loss, time.time() - t_start])
        eval_f.flush()
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'data/weights/{args.run_name}_best.pth')
        print(f"[eval] step {step} | val {val_loss:.4f} (ppl {math.exp(val_loss):.1f}, bpb {val_bpb:.3f}) | train(eval) {train_eval_loss:.4f}", flush=True)
    
    def lr_decay(step,max_steps,max_lr):
        if step < .05*max_steps:
            return max_lr/(int(.05*max_steps)) * step
        else:
            t = (step-int(.05*max_steps))/(max_steps-int(.05*max_steps)-1)
            return 1/2*max_lr*(1+math.cos(math.pi*t))
        
    def random_window_batch(inputs,context):
        in_tokens = torch.flatten(inputs)
        
        shift_value = torch.randint(0,context,(1,1)).item()
        in_shift = in_tokens[shift_value:]
        
        num_batches = (len(in_shift) - 1) // context
        total_len = num_batches * context

        inputs = in_shift[:total_len].view(num_batches, context).to(device)
        targets = in_shift[1:total_len + 1].view(num_batches, context).to(device)
        
        idx = torch.randperm(num_batches, device=INPUT.device)
        
        inputs = inputs[idx]
        targets = targets[idx]
        
        return inputs, targets
        
    model.train()
    num_of_steps = 0
    epoch = 0
    stop = False
    run_eval(0, 0)
    while not stop:
        inpts, tgts = random_window_batch(INPUT,CONTEXT)
        epoch_loss = 0
        k = 0
        for j in range(0, inpts.size()[0], args.batch_size):
            if num_of_steps >= args.steps:
                stop = True
                break
            t0 = time.time()
            optimizer.param_groups[0]['lr'] = lr_decay(num_of_steps,args.steps,args.max_lr)
            logits = model(inpts[j:j+args.batch_size])
            loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), tgts[j:j+args.batch_size])
            model.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1).item()  # measures, doesn't clip
            optimizer.step()
            loss_val = loss.item()
            tokens_seen += tgts[j:j+args.batch_size].numel()
            num_of_steps += 1
            epoch_loss += loss_val
            k += 1
            train_w.writerow([num_of_steps, epoch + 1, tokens_seen, loss_val, grad_norm, optimizer.param_groups[0]['lr'], time.time() - t0])
            if num_of_steps % args.eval_interval == 0 or num_of_steps == args.steps:
                train_f.flush()
                run_eval(num_of_steps, epoch + 1)
        if k > 0:
            epoch += 1
            print(f"average loss over epoch {epoch}: {epoch_loss/k:.4f} (step {num_of_steps})", flush=True)
            torch.save(model.state_dict(), f'data/weights/{args.run_name}_last.pth')

    train_f.close()
    eval_f.close()

    # quick-look plot; 
    steps, tr = [], []
    with open(os.path.join(run_dir, "train_steps.csv")) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"])); tr.append(float(row["train_loss"]))
    ev_steps, ev = [], []
    with open(os.path.join(run_dir, "eval.csv")) as f:
        for row in csv.DictReader(f):
            ev_steps.append(int(row["step"])); ev.append(float(row["val_loss"]))
    plt.figure(figsize=(9, 5))
    plt.plot(steps, tr, alpha=0.3, label="train (per step)")
    plt.plot(ev_steps, ev, marker="o", markersize=3, label="validation")
    plt.xlabel("step"); plt.ylabel("cross-entropy (nats/token)"); plt.yscale("log"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, "loss.png"), dpi=150, bbox_inches="tight")

    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')} after {time.time()-t_start:.0f}s, best val {best_val:.4f}\n")
    print(f"done. best val loss {best_val:.4f}. logs in {run_dir}", flush=True)

if __name__ == "__main__":
    main()
