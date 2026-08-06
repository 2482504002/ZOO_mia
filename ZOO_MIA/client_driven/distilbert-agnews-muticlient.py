"""
FedMeZO toy membership inference experiment (AG News + DistilBERT classification head).
Backup of zoo-toy-mlp-attack-fast.py; standalone: python federatedscope/1.py
"""
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class Config:
    seed: int = 42
    data_root: str = "data/agnews"
    model_name: str = "/home/zhike/JWH/model/distilbert-base-uncased/"
    max_length: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    num_clients: int = 100# 200,400,600,800,1000
    fl_samples: int = 500  # training samples per client
    local_steps: int = 10
    batch_size: int = 1
    lr: float = 1e-5
    zo_eps: float = 1e-3

    attack_samples: int = 500
    target_cache_path: str = "outputs/toy_mlp_target_cache.pt"

    round1_server_trials: int = 100
    round1_inject_step: int = 0

    adv_init_use: bool = True
    # With many clients, total anchors = num_clients * adv_init_anchors_per_client; backprop all anchors each step, so increase steps accordingly
    adv_init_steps: int = 140
    adv_init_lr: float = 1e-3
    adv_init_w_target: float = 1.0
    adv_init_w_anchor: float = 0.4
    adv_init_anchor_power: float = 2.0
    adv_init_w_anchor_max: float = 0.12
    adv_init_anchors_per_client: int = 2  # ~480 anchor batches with 20 clients for better coverage
    # Adversarial init anchor source: "clients"=per-client DataLoader (default); "attack_pool"=attack pool attack_batches
    adv_init_anchor_source: str = "attack_pool"
    adv_init_log_every: int = 20
    adv_init_bundle_path: str = "outputs/toy_mlp_adv_init_bundle.pt"
    adv_init_bundle_use: bool = True  # True: skip adversarial init if cache hit


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agnews_data(cfg):
    total_needed = cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    rng = random.Random(cfg.seed)

    def balanced_pick(texts, labels, target_n):
        label_to_idx = {}
        for i, y in enumerate(labels):
            label_to_idx.setdefault(int(y), []).append(i)
        if not label_to_idx:
            return [], []

        for y in label_to_idx:
            rng.shuffle(label_to_idx[y])

        classes = sorted(label_to_idx.keys())
        n_class = len(classes)
        base = target_n // n_class
        rem = target_n % n_class

        take_per_class = {y: min(base, len(label_to_idx[y])) for y in classes}
        need_extra = rem + (target_n - sum(take_per_class.values()) - rem)
        while need_extra > 0:
            progressed = False
            for y in classes:
                if need_extra <= 0:
                    break
                if take_per_class[y] < len(label_to_idx[y]):
                    take_per_class[y] += 1
                    need_extra -= 1
                    progressed = True
            if not progressed:
                break

        picked = []
        for y in classes:
            picked.extend(label_to_idx[y][: take_per_class[y]])
        rng.shuffle(picked)
        return [texts[i] for i in picked], [int(labels[i]) for i in picked]

    local_data_dir = os.path.join(cfg.data_root, "data")
    if os.path.isdir(local_data_dir):
        parquet_files = sorted(
            [
                os.path.join(local_data_dir, f)
                for f in os.listdir(local_data_dir)
                if f.endswith(".parquet")
            ]
        )
        if parquet_files:
            frames = [pd.read_parquet(fp) for fp in parquet_files]
            df = pd.concat(frames, ignore_index=True)
            if "text" in df.columns and "label" in df.columns:
                texts = df["text"].astype(str).tolist()
                labels = (
                    pd.to_numeric(df["label"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                    .tolist()
                )
                n = min(total_needed, len(texts))
                if n < total_needed:
                    print(f"Warning: local AG News has only {n}  samples, fewer than required  {total_needed}  samples.")
                return balanced_pick(texts, labels, n)
            print(f"Warning: local AG News missing text/label columns, actual columns: {list(df.columns)}")

    try:
        from datasets import load_dataset

        dataset = load_dataset("ag_news")
        train_data = dataset["train"]
        n = min(total_needed, len(train_data))
        train_texts = [item["text"] for item in train_data.select(range(n))]
        train_labels = [item["label"] for item in train_data.select(range(n))]
        return balanced_pick(train_texts, train_labels, n)
    except Exception:
        print("Could not load AG News from local or datasets, using synthetic data...")
        texts = [f"This is a sample news article number {i}" for i in range(total_needed)]
        labels = [i % 4 for i in range(total_needed)]
        return texts, labels


def get_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=4,
        ignore_mismatched_sizes=True,
    ).to(cfg.device)

    for name, param in model.named_parameters():
        if "classifier" not in name and "score" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    return model, tokenizer


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def flatten_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params]).float()


def assign_params(params, vec):
    offset = 0
    with torch.no_grad():
        for p in params:
            numel = p.numel()
            p.copy_(vec[offset : offset + numel].view_as(p))
            offset += numel


def compute_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    )
    return F.cross_entropy(outputs.logits, batch["label"].to(model.device))


def mezo_step(model, batch, params, theta, z, cfg):
    theta_orig = theta.clone()
    assign_params(params, theta_orig + cfg.zo_eps * z)
    loss_pos = compute_loss(model, batch)
    assign_params(params, theta_orig - cfg.zo_eps * z)
    loss_neg = compute_loss(model, batch)
    assign_params(params, theta_orig)
    g_dir = (loss_pos - loss_neg) / (2 * cfg.zo_eps)
    g_est = g_dir * z
    theta_new = theta_orig - cfg.lr * g_est
    return theta_new, g_est


def _grad_norm_classifier(model, batch, create_graph=True):
    params = get_trainable_params(model)
    dev = next(model.parameters()).device
    if not params:
        t = torch.zeros((), device=dev)
        return t.requires_grad_(True) if create_graph else t
    loss = compute_loss(model, batch)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
    return flat.norm()


def collect_anchor_batches(client_loaders, target_batch, cfg, seed):
    loaders = list(client_loaders)
    random.Random(seed).shuffle(loaders)
    anchors = []
    tid = None
    if target_batch is not None:
        tid = target_batch["input_ids"].detach().cpu()
    for loader in loaders:
        data_iter = iter(loader)
        count = 0
        tries = 0
        while count < cfg.adv_init_anchors_per_client and tries < 2000:
            tries += 1
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
            b = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "label": batch["label"],
            }
            if tid is not None and b["input_ids"].shape == tid.shape:
                if torch.equal(b["input_ids"].cpu(), tid):
                    continue
            anchors.append(b)
            count += 1
    return anchors


def collect_anchor_batches_from_attack_pool(attack_batches, target_batch, cfg, seed):
    """
    Sample anchor batches from attack pool list (skip items with same input_ids as target).
    Total count matches client mode: num_clients * adv_init_anchors_per_client.
    """
    rng = random.Random(seed)
    tid = target_batch["input_ids"].detach().cpu() if target_batch is not None else None
    candidates = []
    for b in attack_batches:
        bdict = {
            "input_ids": b["input_ids"],
            "attention_mask": b["attention_mask"],
            "label": b["label"],
        }
        if tid is not None and bdict["input_ids"].shape == tid.shape:
            if torch.equal(bdict["input_ids"].cpu(), tid):
                continue
        candidates.append(bdict)

    total_want = cfg.num_clients * cfg.adv_init_anchors_per_client
    if not candidates:
        print("[Adversarial init] Attack pool anchors: no candidates available (may contain only target).")
        return []

    rng.shuffle(candidates)
    if len(candidates) >= total_want:
        return candidates[:total_want]

    out = []
    while len(out) < total_want:
        rng.shuffle(candidates)
        for c in candidates:
            out.append(c)
            if len(out) >= total_want:
                break
    return out


def _adv_init_loss_terms(model, target_batch, anchor_batches, cfg):
    was_training = model.training
    model.eval()
    try:
        n_t = _grad_norm_classifier(model, target_batch, create_graph=False).item()
        n_div = max(1, len(anchor_batches))
        p = float(cfg.adv_init_anchor_power)
        if anchor_batches:
            norms = [
                _grad_norm_classifier(model, ab, create_graph=False).item() for ab in anchor_batches
            ]
            mean_anchor = float(np.mean(norms))
            max_anchor = float(max(norms))
            pen = cfg.adv_init_w_anchor / n_div * sum(x**p for x in norms)
            if cfg.adv_init_w_anchor_max > 0:
                pen += cfg.adv_init_w_anchor_max * max_anchor
        else:
            mean_anchor = 0.0
            max_anchor = 0.0
            pen = 0.0
        L = -cfg.adv_init_w_target * n_t + pen
    finally:
        model.train(was_training)
    return n_t, mean_anchor, max_anchor, L


def adversarial_sharpness_init(model, target_batch, anchor_batches, cfg):
    if target_batch is None:
        print("[Adversarial init] No target sample, skipping.")
        return
    params = get_trainable_params(model)
    if not params:
        print("[Adversarial init] No trainable parameters, skipping.")
        return

    model.train()
    opt = torch.optim.Adam(params, lr=cfg.adv_init_lr)
    n_anchor = max(1, len(anchor_batches))
    model.zero_grad(set_to_none=True)

    n_t0, mean_a0, max_a0, L0 = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    print(
        f"[Adversarial init] Start step 0/{cfg.adv_init_steps}: "
        f"||∇L_target||={n_t0:.6f}, anchor mean||∇L||={mean_a0:.6f}, max||∇L||={max_a0:.6f}, "
        f"L={L0:.6f} (=-w_t||g_tgt||+(w_a/n)Σ||g||^{cfg.adv_init_anchor_power:g}"
        f"+{'w_max·max||g||' if cfg.adv_init_w_anchor_max > 0 else '0'}) "
        f"(num anchors={len(anchor_batches)})"
    )

    log_every = max(0, int(cfg.adv_init_log_every))
    p = float(cfg.adv_init_anchor_power)
    for step in tqdm(range(cfg.adv_init_steps), desc="Adversarial init (classifier head)"):
        opt.zero_grad(set_to_none=True)
        n_t = _grad_norm_classifier(model, target_batch)
        loss = -cfg.adv_init_w_target * n_t
        anchor_norms = []
        for ab in anchor_batches:
            na = _grad_norm_classifier(model, ab)
            anchor_norms.append(na)
            loss = loss + (cfg.adv_init_w_anchor / n_anchor) * na.pow(p)
        if cfg.adv_init_w_anchor_max > 0 and anchor_norms:
            loss = loss + cfg.adv_init_w_anchor_max * torch.stack(anchor_norms).max()
        loss.backward()
        opt.step()

        done = step + 1
        if log_every > 0 and (done % log_every == 0 or done == cfg.adv_init_steps):
            nt, ma, mxa, Lm = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
            print(
                f"[Adversarial init]  step {done:4d}/{cfg.adv_init_steps}: "
                f"||∇L_target||={nt:.6f}, anchor mean||∇L||={ma:.6f}, max||∇L||={mxa:.6f}, L={Lm:.6f}"
            )

    model.eval()
    model.zero_grad(set_to_none=True)
    n_tf, mean_af, max_af, Lf = _adv_init_loss_terms(model, target_batch, anchor_batches, cfg)
    rel = (n_tf - n_t0) / (n_t0 + 1e-12) * 100.0
    print(
        f"[Adversarial init] Done: ||∇L_target|| {n_t0:.6f} -> {n_tf:.6f} "
        f"(relative {rel:+.2f}%), anchor mean||∇L|| {mean_a0:.6f} -> {mean_af:.6f}, "
        f"max||∇L|| {max_a0:.6f} -> {max_af:.6f}, L {L0:.6f} -> {Lf:.6f}"
    )


def client_update_with_inject_step(model, dataloader, target_batch, cfg, seed, inject_at_step=None):
    params = get_trainable_params(model)
    theta = flatten_params(params)
    g_sum = torch.zeros_like(theta)

    if theta.is_cuda:
        gen = torch.Generator(device=theta.device)
        gen.manual_seed(seed)
    else:
        gen = torch.Generator().manual_seed(seed)

    data_iter = iter(dataloader)
    for step in range(cfg.local_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if inject_at_step is not None and step == inject_at_step and target_batch is not None:
            batch = target_batch

        z = torch.randn(theta.shape, generator=gen, dtype=theta.dtype, device=theta.device)
        theta, g_est = mezo_step(model, batch, params, theta, z, cfg)
        g_sum += g_est.detach()

    assign_params(params, theta)
    return g_sum


def server_pick_member_by_grad_norm(g_sums):
    scores = [float(g.float().norm().item()) for g in g_sums]
    return int(np.argmax(scores)), scores


def compute_true_gradient(model, batch):
    model.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    grad = []
    for p in get_trainable_params(model):
        grad.append(p.grad.detach().flatten())
    return torch.cat(grad).float()


def select_target_sample(model, samples, cfg):
    if len(samples) == 0:
        print("Warning: attack pool is empty, cannot select target sample.")
        return None

    best_norm = -1.0
    best_batch = None
    for i in tqdm(range(len(samples)), desc="Selecting target sample"):
        batch = samples[i]
        g = compute_true_gradient(model, batch)
        norm = g.norm().item()
        if norm > best_norm:
            best_norm = norm
            best_batch = batch

    print(f"Target sample gradient norm: {best_norm:.6f}")
    return best_batch


def create_dataloader(texts, labels, tokenizer, cfg, shuffle=True):
    dataset = AGNewsDataset(texts, labels, tokenizer, cfg.max_length)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=True)


def _adv_init_meta(cfg):
    if not cfg.adv_init_use:
        return {"use": False}
    return {
        "use": True,
        "steps": cfg.adv_init_steps,
        "lr": cfg.adv_init_lr,
        "w_target": cfg.adv_init_w_target,
        "w_anchor": cfg.adv_init_w_anchor,
        "anchor_power": cfg.adv_init_anchor_power,
        "w_anchor_max": cfg.adv_init_w_anchor_max,
        "anchors_per_client": cfg.adv_init_anchors_per_client,
        "anchor_source": cfg.adv_init_anchor_source,
    }


def _build_step45_meta(cfg, attack_texts, attack_labels):
    attack_sig = hashlib.sha1(
        json.dumps(
            {"texts": attack_texts, "labels": attack_labels},
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "num_clients": cfg.num_clients,
        "fl_samples": cfg.fl_samples,
        "attack_samples": cfg.attack_samples,
        "local_steps": cfg.local_steps,
        "grad_stat_mode": "sum",
        "server_attack": "round1_g_sum_l2",
        "zo_eps": cfg.zo_eps,
        "lr": cfg.lr,
        "attack_sig": attack_sig,
        "adv_init": _adv_init_meta(cfg),
    }


def _target_batch_match(a, b):
    if a is None or b is None:
        return a is None and b is None
    try:
        return torch.equal(a["input_ids"].cpu(), b["input_ids"].cpu()) and torch.equal(
            a["label"].cpu(), b["label"].cpu()
        )
    except Exception:
        return False


def try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch):
    if not (cfg.adv_init_use and cfg.adv_init_bundle_use):
        return False
    path = cfg.adv_init_bundle_path
    if not os.path.isfile(path):
        return False
    try:
        obj = torch.load(path, map_location=cfg.device)
    except Exception as e:
        print(f"[Adversarial init cache] Read failed, will retrain: {e}")
        return False
    if not isinstance(obj, dict) or obj.get("meta") != cache_meta_45:
        print("[Adversarial init cache] meta mismatch, will retrain.")
        return False
    tb = obj.get("target_batch")
    if not _target_batch_match(tb, target_batch):
        print("[Adversarial init cache] target sample differs from current, will retrain.")
        return False
    sd = obj.get("state_dict")
    if not isinstance(sd, dict):
        return False
    model.load_state_dict(sd, strict=True)
    print(f"[Adversarial init cache] Loaded model and target: {path}")
    return True


def save_adv_init_bundle(model, cfg, cache_meta_45, target_batch):
    os.makedirs(os.path.dirname(cfg.adv_init_bundle_path) or ".", exist_ok=True)
    torch.save(
        {
            "meta": cache_meta_45,
            "state_dict": model.state_dict(),
            "target_batch": target_batch,
        },
        cfg.adv_init_bundle_path,
    )
    print(f"[Adversarial init cache] Saved: {cfg.adv_init_bundle_path}")


def main():
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Device: {cfg.device}")
    print(f"Config: {cfg}")

    print("\n[1] Loading data...")
    all_texts, all_labels = load_agnews_data(cfg)
    fl_texts = all_texts[: cfg.fl_samples * cfg.num_clients]
    fl_labels = all_labels[: cfg.fl_samples * cfg.num_clients]
    attack_texts = all_texts[
        cfg.fl_samples * cfg.num_clients : cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    ]
    attack_labels = all_labels[
        cfg.fl_samples * cfg.num_clients : cfg.fl_samples * cfg.num_clients + cfg.attack_samples
    ]

    client_data = []
    per_client = cfg.fl_samples
    for i in range(cfg.num_clients):
        start = i * per_client
        end = (i + 1) * per_client
        client_data.append((fl_texts[start:end], fl_labels[start:end]))

    print(f"Number of clients: {cfg.num_clients}, samples per client: {per_client}")
    print(f"Attack pool samples: {cfg.attack_samples}")

    print("\n[2] Loading model...")
    model, tokenizer = get_model_and_tokenizer(cfg)

    print("\n[3] Creating data loaders...")
    client_loaders = []
    for texts, labels in client_data:
        client_loaders.append(create_dataloader(texts, labels, tokenizer, cfg))

    attack_batches = []
    for i in range(len(attack_texts)):
        enc = tokenizer(
            attack_texts[i],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        attack_batches.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": torch.tensor([attack_labels[i]], dtype=torch.long),
            }
        )

    cache_meta_45 = _build_step45_meta(cfg, attack_texts, attack_labels)

    print("\n[4] Selecting target sample...")
    target_batch = None
    if os.path.exists(cfg.target_cache_path):
        try:
            target_obj = torch.load(cfg.target_cache_path, map_location="cpu")
            if isinstance(target_obj, dict) and target_obj.get("meta") == cache_meta_45:
                target_batch = target_obj.get("target_batch")
                print(f"Target sample cache hit: {cfg.target_cache_path}")
        except Exception as e:
            print(f"Failed to read target sample cache, will recompute: {e}")
    if target_batch is None:
        target_batch = select_target_sample(model, attack_batches, cfg)
        os.makedirs(os.path.dirname(cfg.target_cache_path) or ".", exist_ok=True)
        torch.save({"meta": cache_meta_45, "target_batch": target_batch}, cfg.target_cache_path)
        print(f"Saved target sample cache: {cfg.target_cache_path}")

    if cfg.adv_init_use:
        loaded = try_load_adv_init_bundle(model, cfg, cache_meta_45, target_batch)
        if not loaded:
            print("\n[4.6] Adversarial init (classifier head only, amplify target gradient norm)...")
            src = (cfg.adv_init_anchor_source or "clients").strip().lower()
            if src == "attack_pool":
                anchor_batches = collect_anchor_batches_from_attack_pool(
                    attack_batches, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adversarial init] Anchor source: attack pool ( {len(anchor_batches)} batches)")
            elif src == "clients":
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
                print(f"[Adversarial init] Anchor source: per-client training data ( {len(anchor_batches)} batches)")
            else:
                print(f"Warning: adv_init_anchor_source={cfg.adv_init_anchor_source!r} invalid, using clients instead.")
                anchor_batches = collect_anchor_batches(
                    client_loaders, target_batch, cfg, seed=cfg.seed + 7001
                )
            adversarial_sharpness_init(model, target_batch, anchor_batches, cfg)
            save_adv_init_bundle(model, cfg, cache_meta_45, target_batch)

    print("\n[5] Round 1 federated simulation: single-step target injection + server ||g_sum|| detection…")
    r1_success = 0
    r1_rng = random.Random(cfg.seed + 1009)
    if cfg.local_steps < 1:
        print("Skipping [5]: local_steps must be >= 1.")
        r1_trials_eff = 0
    else:
        r1_trials_eff = cfg.round1_server_trials
        max_s = cfg.local_steps - 1
        eff_inject = int(np.clip(cfg.round1_inject_step, 0, max_s))
        if cfg.round1_inject_step != eff_inject:
            print(f"Warning: round1_inject_step={cfg.round1_inject_step} clipped to [0,{max_s}]: {eff_inject}")

        for trial in range(cfg.round1_server_trials):
            true_member = r1_rng.randrange(cfg.num_clients)
            theta0 = copy.deepcopy(model)

            g_uploads = []
            for cid in range(cfg.num_clients):
                mloc = copy.deepcopy(theta0)
                inject_step = eff_inject if cid == true_member else None
                g_sum = client_update_with_inject_step(
                    mloc,
                    client_loaders[cid],
                    target_batch,
                    cfg,
                    seed=cfg.seed + 50000 + trial * 1000 + cid * 100,
                    inject_at_step=inject_step,
                )
                g_uploads.append(g_sum)

            pred_member, srv_scores = server_pick_member_by_grad_norm(g_uploads)
            hit_r1 = pred_member == true_member
            r1_success += hit_r1
            print(
                f"Round1 trial {trial+1}: true={true_member}, pred={pred_member}, "
                f"||g_sum||={[round(s, 4) for s in srv_scores]}, {'✓' if hit_r1 else '✗'}"
            )

    print("\n" + "=" * 60)
    r1_rate = (r1_success / r1_trials_eff) if r1_trials_eff else float("nan")
    if r1_trials_eff:
        print(f"[5] Round 1 + server detection success rate: {r1_success}/{r1_trials_eff} = {r1_rate:.1%}")
    else:
        print("[5] Round 1 + server detection: skipped")
    print("=" * 60)

    return r1_rate


if __name__ == "__main__":
    main()
