#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, time, math, random, zipfile, sys

# --- Optional deps: install if available ---
try:
    import yaml  # pip install pyyaml
except Exception:
    print("Note: for YAML manifests, please `pip install pyyaml`.", file=sys.stderr)
    raise

try:
    import psutil  # pip install psutil
except Exception:
    psutil = None

# --- Try to use your real sequential tester first ---
USING_LOCAL_TESTER = False
try:
    # Your codebase path; adjust if needed
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode  # type: ignore
    HAVE_USER_TESTER = True
except Exception:
    HAVE_USER_TESTER = False
    USING_LOCAL_TESTER = True

# --- Minimal fallback tester (only if import fails) ---
if USING_LOCAL_TESTER:
    class ModeParams:
        def __init__(self, alpha, gamma, eta, delta_star, eps_diff, n_min, n_max):
            self.alpha=alpha; self.gamma=gamma; self.eta=eta
            self.delta_star=delta_star; self.eps_diff=eps_diff
            self.n_min=n_min; self.n_max=n_max

    class TestingMode:
        QUICK = ModeParams(0.025, 0.15, 0.50, 0.80, 0.15, 10, 120)
        AUDIT = ModeParams(0.010, 0.10, 0.50, 1.00, 0.10, 30, 400)
        EXTENDED = ModeParams(0.001, 0.08, 0.40, 1.10, 0.08, 50, 800)

    class _Welford:
        def __init__(self): self.n=0; self.mean=0.0; self.m2=0.0
        def push(self,x:float):
            self.n+=1; d=x-self.mean; self.mean+=d/self.n; self.m2+=d*(x-self.mean)
        @property
        def var(self): return 0.0 if self.n<2 else self.m2/(self.n-1)

    def _eb_halfwidth(var: float, n: int, delta: float) -> float:
        if n<=1: return float("inf")
        import math
        logt=math.log(1.0/max(min(delta, 0.999999),1e-12))
        a=math.sqrt(max(0.0, 2.0*var*logt/n))
        b=7.0*logt/(3.0*max(n-1,1))
        return a+b

    def _spend(alpha: float, n: int) -> float:
        return alpha/(n*(n+1))

    def bounded_difference(a: str, b: str) -> float:
        import difflib
        def norm(t: str) -> str: return " ".join(t.strip().lower().split())
        ratio = difflib.SequenceMatcher(a=norm(a), b=norm(b)).ratio()
        return max(0.0, min(1.0, 1.0 - ratio))

    class EnhancedSequentialTester:
        def __init__(self, params): self.params=params
        def run(self, prompts, ref_generate, cand_generate):
            steps=[]; w=_Welford(); verdict="UNDECIDED"
            for i, p in enumerate(prompts, start=1):
                ref=ref_generate(p); cand=cand_generate(p)
                score = bounded_difference(ref, cand)
                w.push(score)
                delta_n=_spend(self.params.alpha, w.n)
                h=_eb_halfwidth(w.var, w.n, delta_n)
                if w.n>=self.params.n_min:
                    if (w.mean + h) <= self.params.gamma and h <= self.params.eta*self.params.gamma:
                        verdict="SAME"
                    elif (w.mean >= self.params.delta_star) and (h/max(w.mean,1e-12) <= self.params.eps_diff):
                        verdict="DIFFERENT"
                    else:
                        verdict="UNDECIDED"
                steps.append({
                    "index": i, "prompt": p, "ref_output": ref, "cand_output": cand,
                    "score": score, "mean": w.mean, "var": w.var, "halfwidth": h,
                    "delta_n": delta_n, "verdict_so_far": verdict
                })
                if verdict in ("SAME","DIFFERENT") or w.n>=self.params.n_max:
                    break
            return {"verdict": verdict, "n_used": len(steps),
                    "params": {"alpha": self.params.alpha, "gamma": self.params.gamma, "eta": self.params.eta,
                               "delta_star": self.params.delta_star, "eps_diff": self.params.eps_diff,
                               "n_min": self.params.n_min, "n_max": self.params.n_max},
                    "steps": steps}

# --- Model adapters ---
def _echo(tag: str):
    return type("Echo", (), {
        "generate": staticmethod(lambda prompt, _tag=tag: f"[{_tag}] {prompt}"),
        "name": staticmethod(lambda _tag=tag: f"echo:{_tag}")
    })

# Try to import real model loaders
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

class HFLocalModel:
    def __init__(self, model_path: str, model_name: str = None):
        if not HAVE_TRANSFORMERS:
            raise ImportError("transformers not installed; using echo stub")
        self.model_path = model_path
        self.model_name = model_name or model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Simplified loading to avoid bus errors
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, temperature: float = 0.0, max_length: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            if temperature < 0.01:  # Deterministic
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
    
    def name(self) -> str:
        return f"hf_local:{self.model_name}"

def make_model(spec: dict, role: str, robust: dict):
    t = spec.get("type")
    if t == "echo":
        return _echo(f"{role}-echo")
    
    if t == "hf_local":
        if HAVE_TRANSFORMERS:
            model_path = spec.get("model_path", spec.get("model_name", "gpt2"))
            model_name = spec.get("model_name", model_path)
            try:
                print(f"Loading {role} model from {model_path}...", file=sys.stderr)
                model = HFLocalModel(model_path, model_name)
                # Wrap with temperature from robustness config
                temp = float(robust.get("temperature", 0.0))
                if temp > 0:
                    orig_gen = model.generate
                    model.generate = lambda p: orig_gen(p, temperature=temp)
                print(f"Successfully loaded {role} model", file=sys.stderr)
                return model
            except Exception as e:
                import traceback
                print(f"Warning: Failed to load {model_path}: {e}", file=sys.stderr)
                traceback.print_exc()
                print(f"Falling back to echo stub for {role}", file=sys.stderr)
                return _echo(f"{role}-hf_fallback")
        else:
            print(f"Transformers not available, using echo stub for {role}", file=sys.stderr)
            return _echo(f"{role}-hf_stub")
    
    if t == "api":
        # API adapter stub - would need actual implementation
        return _echo(f"{role}-api_stub")
    
    raise ValueError(f"Unknown model type: {t}")

# --- HMAC seeding + minimal prompt mapper (replace with your templates) ---
import hmac, hashlib
def _hmac_hex(key_hex: str, msg: str) -> str:
    key=bytes.fromhex(key_hex); return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _gen_seeds(key_hex: str, run_id: str, n: int) -> list[str]:
    return [_hmac_hex(key_hex, f"{run_id}|{i}") for i in range(1, n+1)]

def _prompt_from_seed(seed_hex: str) -> str:
    which = int(seed_hex[:2], 16) % 3
    if which==0: return f"Summarize in one sentence: seed={seed_hex[:12]}"
    if which==1: return f"Compute 23*19 and explain briefly. seed={seed_hex[:12]}"
    return f"Define 'entropy' in 2 lines. seed={seed_hex[:12]}"

# --- metrics sampler (RSS/CPU/IO) ---
class MetricsSampler:
    def __init__(self, interval: float=0.5):
        self.interval=interval; self.samples=[]; self._stop=False; self._thr=None
        self._proc = psutil.Process(os.getpid()) if psutil else None
    def start(self):
        if not self._proc: return
        import threading
        self._thr=threading.Thread(target=self._loop, daemon=True); self._thr.start()
    def _loop(self):
        while not self._stop:
            try:
                rss = self._proc.memory_info().rss
                cpu = self._proc.cpu_percent(interval=None)
                io  = self._proc.io_counters() if hasattr(self._proc, "io_counters") else None
                rb = getattr(io, "read_bytes", None) if io else None
                wb = getattr(io, "write_bytes", None) if io else None
                self.samples.append({"t": time.time(), "rss": rss, "cpu_percent": cpu, "read_bytes": rb, "write_bytes": wb})
            except Exception:
                pass
            time.sleep(self.interval)
    def stop(self):
        if not self._proc: return
        self._stop=True
        if self._thr: self._thr.join(timeout=1.0)

# --- micro-robustness wrappers ---
def wrap_paraphrase(text: str) -> str:
    toks = text.split()
    return " ".join(toks[:2] + ["..."] + toks[-3:]) if len(toks)>6 else text

def tokenizer_overlap_shim(text: str, keep: float) -> str:
    if keep>=0.999: return text
    rnd = random.Random(0xC0FFEE)
    toks = text.split()
    kept = [t for t in toks if rnd.random() <= keep]
    if not kept: kept = toks[: max(1, int(math.ceil(len(toks)*0.1))) ]
    return " ".join(kept)

# --- utilities ---
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def write_json(p: str, obj): ensure_dir(os.path.dirname(p)); json.dump(obj, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
def write_ndjson(p: str, rows: list[dict]):
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def pack_bundle(zip_path: str, files: list[str], extra: dict):
    ensure_dir(os.path.dirname(zip_path))
    manifest = {"files":[os.path.basename(x) for x in files], **extra}
    man_path = os.path.join(os.path.dirname(zip_path), "bundle_manifest.json")
    write_json(man_path, manifest)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(man_path, arcname="bundle_manifest.json")
        for fp in files:
            if os.path.exists(fp) and os.path.basename(fp) != "bundle_manifest.json":
                z.write(fp, arcname=os.path.basename(fp))

# --- core run ---
def run_single(exp_cfg: dict, outdir: str) -> dict:
    mode_name = (exp_cfg.get("mode") or "AUDIT").upper()
    Mode = getattr(TestingMode, mode_name)
    n_challenges = int(exp_cfg.get("n_challenges"))
    run_id = exp_cfg["run_id"]; key_hex = exp_cfg["hmac_key_hex"]
    robust = exp_cfg.get("robustness", {})

    seeds = _gen_seeds(key_hex, run_id, n_challenges)
    prompts = [_prompt_from_seed(s) for s in seeds]

    ref = make_model(exp_cfg["ref"], "ref", robust)
    cand = make_model(exp_cfg["cand"], "cand", robust)

    do_wrap = bool(robust.get("wrapper", False))
    keep_frac = float(robust.get("tokenizer_overlap", 1.0) or 1.0)

    def ref_gen(p: str) -> str:
        out = ref.generate(p)
        return tokenizer_overlap_shim(out, keep_frac) if keep_frac < 0.999 else out

    def cand_gen(p: str) -> str:
        out = cand.generate(p)
        if do_wrap: out = wrap_paraphrase(out)
        return tokenizer_overlap_shim(out, keep_frac) if keep_frac < 0.999 else out

    sampler = MetricsSampler(interval=0.5); sampler.start()
    tester = EnhancedSequentialTester(Mode)
    result = tester.run(prompts=prompts, ref_generate=ref_gen, cand_generate=cand_gen)
    sampler.stop()
    metrics = sampler.samples

    ensure_dir(outdir)
    tpath = os.path.join(outdir, "transcript.ndjson")
    spath = os.path.join(outdir, "summary.json")
    mpath = os.path.join(outdir, "metrics.json")
    wpath = os.path.join(outdir, "manifest.json")
    zpath = os.path.join(outdir, "evidence_bundle.zip")

    write_ndjson(tpath, result["steps"] if isinstance(result, dict) else result.steps)  # type: ignore
    write_json(spath, {
        "verdict": (result["verdict"] if isinstance(result, dict) else result.verdict),
        "n_used": (result["n_used"] if isinstance(result, dict) else result.n_used),
        "params": (result["params"] if isinstance(result, dict) else result.params),
    })
    write_json(mpath, metrics)
    write_json(wpath, {
        "exp": {"id": exp_cfg["id"], "mode": mode_name, "n_challenges": n_challenges, "run_id": run_id},
        "models": {"ref": ref.name(), "cand": cand.name()},
        "robustness": robust,
        "using_user_tester": HAVE_USER_TESTER and not USING_LOCAL_TESTER,
    })
    pack_bundle(zpath, [tpath, spath, mpath, wpath], {"run_id": run_id})

    return {"outdir": outdir,
            "verdict": (result["verdict"] if isinstance(result, dict) else result.verdict),
            "n_used": (result["n_used"] if isinstance(result, dict) else result.n_used)}

# --- bootstrap power from transcripts (no extra inference) ---
def load_transcript(run_dir: str) -> list[dict]:
    p=os.path.join(run_dir, "transcript.ndjson")
    rows=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def bootstrap_power(run_dirs: list[str], B: int=1000) -> dict:
    # Simple CI proxy: resample observed scores; report CI of mean & proxy diff rate.
    scores = []
    for rd in run_dirs:
        for row in load_transcript(rd):
            s=row.get("score")
            if isinstance(s,(int,float)): scores.append(float(s))
    if not scores: return {"error":"no scores found"}
    rnd=random.Random(42); N=len(scores); diff_calls=0; means=[]
    for _ in range(B):
        samp=[scores[rnd.randrange(N)] for _ in range(min(64, N))]
        mu=sum(samp)/len(samp); means.append(mu)
        if mu>=0.8: diff_calls+=1
    means.sort()
    lo=means[int(0.025*B)]; hi=means[int(0.975*B)-1]
    return {"bootstrap_B": B, "mean_CI95": [lo, hi], "diff_call_rate_proxy": diff_calls/B}

# --- CLI ---
def main():
    ap=argparse.ArgumentParser(description="PoT reviewer-friendly runner")
    sub=ap.add_subparsers(dest="cmd", required=True)

    p_batch=sub.add_parser("batch", help="Run a manifest of experiments")
    p_batch.add_argument("--manifest", required=True)
    p_batch.add_argument("--out", "--outdir", dest="outdir", default=None)

    p_run=sub.add_parser("run", help="Run a single experiment by id")
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--id", required=True)
    p_run.add_argument("--out", "--outdir", dest="outdir", default=None)

    p_pow=sub.add_parser("power", help="Bootstrap CIs from a run dir or parent")
    p_pow.add_argument("--run", required=True)
    p_pow.add_argument("--B", type=int, default=1000)

    p_bun=sub.add_parser("bundle", help="Repack evidence bundle for a run dir")
    p_bun.add_argument("--run", required=True)

    args=ap.parse_args()

    if args.cmd=="batch":
        man=yaml.safe_load(open(args.manifest, "r", encoding="utf-8"))
        run_id=man["run_id"]; key_hex=man["hmac_key_hex"]
        g=man.get("global", {})
        out_root=args.outdir or g.get("out_root","runs/batch")
        n_chal=int(g.get("n_challenges", 64))
        results=[]
        for exp in man["experiments"]:
            exp_cfg={
                "id": exp["id"],
                "mode": exp.get("mode", g.get("mode","AUDIT")).upper(),
                "n_challenges": exp.get("n_challenges", n_chal),
                "run_id": run_id,
                "hmac_key_hex": key_hex,
                "ref": exp["ref"],
                "cand": exp["cand"],
                "robustness": g.get("robustness", {}),
            }
            outdir=os.path.join(out_root, exp_cfg["id"])
            print(f"==> {exp_cfg['id']} (mode={exp_cfg['mode']}, n={exp_cfg['n_challenges']})")
            res=run_single(exp_cfg, outdir)
            print(f"    verdict={res['verdict']}  n_used={res['n_used']}  out={res['outdir']}")
            results.append({"id":exp_cfg["id"], **res})
        os.makedirs(out_root, exist_ok=True)
        json.dump({"results":results, "manifest": os.path.basename(args.manifest)},
                  open(os.path.join(out_root, "batch_summary.json"),"w",encoding="utf-8"), indent=2)
    elif args.cmd=="run":
        man=yaml.safe_load(open(args.manifest,"r",encoding="utf-8"))
        run_id=man["run_id"]; key_hex=man["hmac_key_hex"]; g=man.get("global", {})
        tgt = next((e for e in man["experiments"] if e["id"]==args.id), None)
        if not tgt: sys.exit(f"Experiment id not found: {args.id}")
        exp_cfg={
            "id": tgt["id"],
            "mode": tgt.get("mode", g.get("mode","AUDIT")).upper(),
            "n_challenges": tgt.get("n_challenges", g.get("n_challenges",64)),
            "run_id": run_id, "hmac_key_hex": key_hex,
            "ref": tgt["ref"], "cand": tgt["cand"],
            "robustness": g.get("robustness", {}),
        }
        out_root=args.outdir or g.get("out_root","runs/batch")
        outdir=os.path.join(out_root, exp_cfg["id"])
        print(json.dumps(run_single(exp_cfg, outdir), indent=2))
    elif args.cmd=="power":
        root=args.run
        if os.path.isdir(os.path.join(root,"exp_001")):
            dirs=[os.path.join(root, d) for d in os.listdir(root) if d.startswith("exp_")]
        else:
            dirs=[root]
        print(json.dumps(bootstrap_power(dirs, B=int(args.B)), indent=2))
    elif args.cmd=="bundle":
        rd=args.run
        t=os.path.join(rd, "transcript.ndjson")
        s=os.path.join(rd, "summary.json")
        m=os.path.join(rd, "manifest.json")
        met=os.path.join(rd, "metrics.json")
        z=os.path.join(rd, "evidence_bundle.zip")
        pack_bundle(z, [t,s,m,met], {"repacked": True})
        print(json.dumps({"bundle": z, "exists": os.path.exists(z)}, indent=2))

if __name__=="__main__":
    main()