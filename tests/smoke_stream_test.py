import sys, os, time, threading, types, re
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../webui"))
sys.argv = [sys.argv[0]]  # argparse対策

# --- minimal stubs for PIL / numpy (tests may run without these deps) ---
if 'PIL' not in sys.modules:
    pil_pkg = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    class _Img:
        def convert(self, *a, **k):
            return self
        @staticmethod
        def open(*a, **k):
            return _Img()
        @staticmethod
        def fromarray(*a, **k):
            return _Img()
    image_mod.open = _Img.open
    image_mod.fromarray = _Img.fromarray
    png_mod = types.ModuleType("PIL.PngImagePlugin")
    class _PngInfo:
        def add_text(self, *a, **k):
            pass
    png_mod.PngInfo = _PngInfo
    pil_pkg.Image = image_mod
    pil_pkg.PngImagePlugin = png_mod
    sys.modules['PIL'] = pil_pkg
    sys.modules['PIL.Image'] = image_mod
    sys.modules['PIL.PngImagePlugin'] = png_mod

if 'numpy' not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.array = lambda x, *a, **k: x
    sys.modules['numpy'] = np_stub

if 'tqdm' not in sys.modules:
    tqdm_mod = types.ModuleType("tqdm")
    class _Tqdm:
        @staticmethod
        def write(*a, **k):
            pass
    tqdm_mod.tqdm = _Tqdm
    sys.modules['tqdm'] = tqdm_mod

if 'yaml' not in sys.modules:
    yaml_mod = types.ModuleType("yaml")
    yaml_mod.safe_load = lambda *a, **k: {}
    yaml_mod.dump = lambda *a, **k: ""
    sys.modules['yaml'] = yaml_mod

# ---- 必要モジュールのダミー化 ----
# torch / safetensors / huggingface_hub / gradio など重い依存をスタブする
torch_stub = types.ModuleType("torch")
class _Cuda:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def empty_cache():
        pass
torch_stub.cuda = _Cuda()
torch_stub.float16 = object()
torch_stub.bfloat16 = object()
torch_stub.from_numpy = lambda x: x
class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass
    def __call__(self, func):
        def wrapped(*a, **k):
            return func(*a, **k)
        return wrapped
torch_stub.no_grad = lambda: _NoGrad()
sys.modules['torch'] = torch_stub

hf_stub = types.ModuleType("huggingface_hub")
hf_stub.snapshot_download = lambda *a, **k: None
sys.modules['huggingface_hub'] = hf_stub

safetensors_pkg = types.ModuleType("safetensors")
safetensors_torch = types.ModuleType("safetensors.torch")
safetensors_pkg.torch = safetensors_torch
sys.modules['safetensors'] = safetensors_pkg
sys.modules['safetensors.torch'] = safetensors_torch

class _Comp:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def queue(self): return self
    def launch(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass
    def __getattr__(self, name):
        def _m(*a, **k):
            return self
        return _m

gradio_stub = types.ModuleType("gradio")
gradio_stub.update = lambda **k: ("update", k)
gradio_stub.skip = lambda: ("skip",)
def _getattr(name):
    return _Comp
gradio_stub.__getattr__ = _getattr
sys.modules['gradio'] = gradio_stub

diffusers_helper_pkg = types.ModuleType("diffusers_helper")
diffusers_helper_pkg.__path__ = []
hf_login_mod = types.ModuleType("diffusers_helper.hf_login")
hf_login_mod.login = lambda *a, **k: None
gradio_pkg = types.ModuleType("diffusers_helper.gradio")
gradio_pkg.__path__ = []
progress_bar_mod = types.ModuleType("diffusers_helper.gradio.progress_bar")
progress_bar_mod.make_progress_bar_css = lambda *a, **k: ""
progress_bar_mod.make_progress_bar_html = lambda *a, **k: ""
gradio_pkg.progress_bar = progress_bar_mod
diffusers_helper_pkg.hf_login = hf_login_mod
diffusers_helper_pkg.gradio = gradio_pkg
sys.modules['diffusers_helper'] = diffusers_helper_pkg
sys.modules['diffusers_helper.hf_login'] = hf_login_mod
sys.modules['diffusers_helper.gradio'] = gradio_pkg
sys.modules['diffusers_helper.gradio.progress_bar'] = progress_bar_mod

# other diffusers_helper submodules used
def _dummy_module(name):
    mod = types.ModuleType(name)
    return mod

hunyuan_mod = _dummy_module('diffusers_helper.hunyuan')
hunyuan_mod.encode_prompt_conds = lambda *a, **k: None
hunyuan_mod.vae_decode = lambda *a, **k: None
hunyuan_mod.vae_encode = lambda *a, **k: None
hunyuan_mod.vae_decode_fake = lambda *a, **k: None
sys.modules['diffusers_helper.hunyuan'] = hunyuan_mod

utils_mod = _dummy_module('diffusers_helper.utils')
utils_mod.save_bcthw_as_mp4 = lambda *a, **k: None
utils_mod.crop_or_pad_yield_mask = lambda *a, **k: None
utils_mod.soft_append_bcthw = lambda *a, **k: None
utils_mod.resize_and_center_crop = lambda *a, **k: None
utils_mod.state_dict_weighted_merge = lambda *a, **k: None
utils_mod.state_dict_offset_merge = lambda *a, **k: None
utils_mod.generate_timestamp = lambda *a, **k: None
sys.modules['diffusers_helper.utils'] = utils_mod

models_mod = _dummy_module('diffusers_helper.models.hunyuan_video_packed')
class _HVTPacked:
    @staticmethod
    def load_config(path):
        return {}
    @staticmethod
    def from_config(cfg, torch_dtype=None):
        class _Dummy:
            def to(self, *a, **k):
                return self
        return _Dummy()
models_mod.HunyuanVideoTransformer3DModelPacked = _HVTPacked
sys.modules['diffusers_helper.models.hunyuan_video_packed'] = models_mod

pipelines_mod = _dummy_module('diffusers_helper.pipelines.k_diffusion_hunyuan')
pipelines_mod.sample_hunyuan = lambda *a, **k: None
sys.modules['diffusers_helper.pipelines.k_diffusion_hunyuan'] = pipelines_mod

memory_mod = _dummy_module('diffusers_helper.memory')
memory_mod.cpu = memory_mod.gpu = lambda *a, **k: None
memory_mod.gpu_complete_modules = lambda *a, **k: None
memory_mod.get_cuda_free_memory_gb = lambda *a, **k: 0
memory_mod.move_model_to_device_with_memory_preservation = lambda *a, **k: None
memory_mod.offload_model_from_device_for_memory_preservation = lambda *a, **k: None
memory_mod.fake_diffusers_current_device = lambda model, device: None
memory_mod.DynamicSwapInstaller = type('DynamicSwapInstaller', (), {})
memory_mod.unload_complete_models = lambda *a, **k: None
memory_mod.load_model_as_complete = lambda *a, **k: None
sys.modules['diffusers_helper.memory'] = memory_mod

thread_utils_mod = _dummy_module('diffusers_helper.thread_utils')
thread_utils_mod.AsyncStream = type('AsyncStream', (), {})
thread_utils_mod.async_run = lambda func, *a, **k: func(*a, **k)
sys.modules['diffusers_helper.thread_utils'] = thread_utils_mod

clip_vision_mod = _dummy_module('diffusers_helper.clip_vision')
clip_vision_mod.hf_clip_vision_encode = lambda *a, **k: None
sys.modules['diffusers_helper.clip_vision'] = clip_vision_mod

bucket_tools_mod = _dummy_module('diffusers_helper.bucket_tools')
bucket_tools_mod.find_nearest_bucket = lambda *a, **k: None
bucket_tools_mod.SAFE_RESOLUTIONS = []
sys.modules['diffusers_helper.bucket_tools'] = bucket_tools_mod

diffusers_mod = types.ModuleType('diffusers')
diffusers_mod.AutoencoderKLHunyuanVideo = type('AutoencoderKLHunyuanVideo', (), {})
sys.modules['diffusers'] = diffusers_mod

transformers_mod = types.ModuleType('transformers')
class _Base:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()
    def to(self, *a, **k):
        return self
transformers_mod.LlamaModel = type('LlamaModel', (), {})
transformers_mod.CLIPTextModel = type('CLIPTextModel', (), {})
transformers_mod.LlamaTokenizerFast = type('LlamaTokenizerFast', (_Base,), {})
transformers_mod.CLIPTokenizer = type('CLIPTokenizer', (_Base,), {})
transformers_mod.SiglipImageProcessor = type('SiglipImageProcessor', (_Base,), {})
transformers_mod.SiglipVisionModel = type('SiglipVisionModel', (_Base,), {})
sys.modules['transformers'] = transformers_mod

accelerate_mod = types.ModuleType('accelerate')
class _Ctx:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass
accelerate_mod.init_empty_weights = lambda *a, **k: _Ctx()
sys.modules['accelerate'] = accelerate_mod

einops_stub = types.ModuleType("einops")
einops_stub.rearrange = lambda *a, **k: None
sys.modules['einops'] = einops_stub

# ==== ダミー gradio ====
class _DummyGr:
    @staticmethod
    def update(**kwargs): return ("update", kwargs)
    @staticmethod
    def skip(): return ("skip",)
gr = _DummyGr()

# oneframe_ichi を読み込み（本体の gr をダミーに入れ替え）
import importlib
one = importlib.import_module("webui.oneframe_ichi")
one.gr = gr  # monkey patch

timestamp_re = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _set_progress_counts():
    one.progress_ref_total = 2
    one.progress_img_total = 3
    one.progress_ref_idx = 2
    one.progress_img_idx = 3


def producer_complete(ctx):
    _set_progress_counts()
    ts = one.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx.bus.publish(('progress', (None, one.translate("開始しています... ") + ts, '')))
    time.sleep(0.05)
    dummy = os.path.abspath("dummy.png")
    open(dummy, "wb").write(b"\x89PNG\r\n\x1a\n")
    ctx.bus.publish(('file', dummy))
    time.sleep(0.05)
    ts = one.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = one.translate("【全バッチ処理完了】プロセスが完了しました - ") + ts + " - " + \
          f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total}"
    ctx.bus.publish(('progress', (None, msg, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()


def producer_abort(ctx):
    _set_progress_counts()
    ts = one.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctx.bus.publish(('progress', (None, one.translate("開始しています... ") + ts, '')))
    time.sleep(0.05)
    ts = one.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = one.translate("バッチ処理が中断されました") + " - " + \
          f"参考画像 {one.progress_ref_idx}/{one.progress_ref_total} ,イメージ {one.progress_img_idx}/{one.progress_img_total} - " + ts
    ctx.bus.publish(('progress', (None, msg, '')))
    ctx.bus.publish(('end', None))
    ctx.bus.close()


def consume(ctx):
    out = []
    for ui in one._stream_job_to_ui(ctx):
        out.append(ui[2])
    return out


def test_smoke_stream():
    ctx = one.JobContext()
    t = threading.Thread(target=producer_complete, args=(ctx,), daemon=True)
    t.start()
    descs = consume(ctx)
    print("---- STREAM DESC LOG ----")
    for d in descs:
        print(d)
    start_msgs = [d for d in descs if "開始しています" in (d or "")]
    end_msgs = [d for d in descs if "全バッチ処理完了" in (d or "")]
    assert start_msgs, "開始メッセージが流れていません"
    assert end_msgs, "終了サマリが流れていません"
    assert timestamp_re.search(start_msgs[0]), "開始メッセージに時刻が含まれていません"
    assert timestamp_re.search(end_msgs[-1]), "終了サマリに時刻が含まれていません"


def test_smoke_stream_interrupted():
    ctx = one.JobContext()
    t = threading.Thread(target=producer_abort, args=(ctx,), daemon=True)
    t.start()
    descs = consume(ctx)
    mid_msgs = [d for d in descs if "中断されました" in (d or "")]
    assert mid_msgs, "中断メッセージが流れていません"
    assert any("開始しています" in (d or "") for d in descs), "開始メッセージが流れていません"
    assert timestamp_re.search(mid_msgs[-1]), "中断メッセージに時刻が含まれていません"
