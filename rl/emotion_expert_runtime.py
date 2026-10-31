"""Native Omni forwards with decoded-media caching only, on physical GPU 2."""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import subprocess
import time

from .emotion_expert import SETTINGS, VERSION, atomic, digest, fingerprint, input_only, instruction


def normalize_quantized_biases(model, linear4bit, dtype):
    """Apply Linear4bit.forward's lazy BF16 bias cast before frozen auditing."""
    changed=[]
    for name,module in model.named_modules():
        if isinstance(module,linear4bit) and module.bias is not None and module.bias.dtype!=dtype:
            if module.bias.requires_grad:raise RuntimeError("Quantized base bias must be frozen")
            before=str(module.bias.dtype)
            module.bias.data=module.bias.data.to(dtype)
            changed.append({"module":name,"from":before,"to":str(dtype)})
    return changed


class DecodedMedia:
    def __init__(self, context):
        self.root=Path(context["data_root"]).resolve()
        self.cache=Path(context["run"])/"decoded-media"
        self.settings=context["settings"]
        self.ffmpeg=str(Path(context["data_root"]).parent/"media-tools/bin/ffmpeg")
        self.hashes=dict(context.get("media_sha256",{}))
        self.last_key=None;self.last=None

    def load(self,row):
        import numpy as np
        safe=input_only(row);path=(self.root/safe["video"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():raise ValueError("Missing/escaped media")
        if safe["video"] not in self.hashes:self.hashes[safe["video"]]=digest(path)
        key=fingerprint({"sha256":self.hashes[safe["video"]],"settings":self.settings,"decoder":VERSION})
        if key==self.last_key:return self.last
        target=self.cache/(key+".npz")
        if not target.exists():
            self.cache.mkdir(parents=True,exist_ok=True)
            values=self.decode(path)
            import os
            temp=target.with_name(target.name+f".partial-{os.getpid()}")
            with temp.open("wb") as f:np.savez(f,**values)
            temp.replace(target)
        with np.load(target,allow_pickle=False) as f:
            value={k:f[k] for k in f.files}
        value["metadata"]=json.loads(str(value["metadata"]))
        self.last_key,self.last=key,value
        return value

    def decode(self,path):
        import av
        import cv2
        import numpy as np
        s=self.settings
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            fps=float(stream.average_rate);total=int(stream.frames or 0)
            if total<=0:total=sum(1 for _ in container.decode(video=0))
        if not math.isfinite(fps) or fps<=0 or total<4:raise ValueError("Invalid source video timing")
        n=min(s["video_max_frames"],max(4,round(total/fps*s["video_fps"]/2)*2),total//2*2)
        indices=np.linspace(0,total-1,n).round().astype(int).tolist()
        face_indices={indices[i] for i in np.linspace(0,n-1,min(4,n)).round().astype(int)}
        detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt2.xml")
        if detector.empty():raise RuntimeError("Bundled face detector is unavailable")
        selected=set(indices);frames={};faces=[]
        with av.open(str(path)) as container:
            for i,frame in enumerate(container.decode(video=0)):
                if i in selected:
                    rgb=frame.to_ndarray(format="rgb24");h,w=rgb.shape[:2]
                    scale=min(1.,math.sqrt(s["video_pixels"]/(h*w)))
                    hh=max(28,int(h*scale)//28*28);ww=max(28,int(w*scale)//28*28)
                    while hh*ww>s["video_pixels"]:
                        if ww>=hh:ww-=28
                        else:hh-=28
                    frames[i]=cv2.resize(rgb,(ww,hh),interpolation=cv2.INTER_AREA)
                    if i in face_indices:
                        boxes=detector.detectMultiScale(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),
                            scaleFactor=1.1,minNeighbors=5,minSize=(32,32))
                        if len(boxes):
                            x,y,bw,bh=max(boxes,key=lambda b:int(b[2])*int(b[3]));pad=s["face_padding"]
                            x0=max(0,int(x-bw*pad));y0=max(0,int(y-bh*pad))
                            x1=min(w,int(x+bw*(1+pad)));y1=min(h,int(y+bh*(1+pad)))
                            crop=cv2.resize(rgb[y0:y1,x0:x1],(s["face_size"],s["face_size"]),interpolation=cv2.INTER_AREA)
                            faces.append(crop)
                if i>=indices[-1]:break
        if set(frames)!=selected:raise ValueError("Source frame metadata and decoded frames disagree")
        result=subprocess.run([self.ffmpeg,"-v","error","-i",str(path),"-vn","-ac","1","-ar","16000",
                               "-f","f32le","pipe:1"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        audio=np.frombuffer(result.stdout,dtype="<f4").copy() if result.returncode==0 else np.empty(0,dtype=np.float32)
        return {"frames":np.stack([frames[i] for i in indices]),
                "faces":np.stack(faces) if faces else np.empty((0,s["face_size"],s["face_size"],3),dtype=np.uint8),
                "audio":audio,"metadata":np.array(json.dumps({"total_num_frames":total,"fps":fps,"frames_indices":indices})),
                "media_sha256":np.array(digest(path))}


class ExpertPolicy:
    def __init__(self,context,adapter=None,*,training=False,reference=None):
        import torch
        from peft import LoraConfig,PeftModel,TaskType,get_peft_model,prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig,Qwen2_5OmniForConditionalGeneration,Qwen2_5OmniProcessor
        self.torch=torch;self.context=context;self.training=training;self.device="cuda:0"
        self.processor=Qwen2_5OmniProcessor.from_pretrained(context["model"],local_files_only=True)
        omni=Qwen2_5OmniForConditionalGeneration.from_pretrained(context["model"],local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
        omni.disable_talker();base=omni.thinker;del omni
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=training,
                                            gradient_checkpointing_kwargs={"use_reentrant":False})
        from bitsandbytes.nn import Linear4bit
        self.bias_normalization=normalize_quantized_biases(base,Linear4bit,torch.bfloat16)
        names=[name for name,module in base.named_modules() if name.startswith("model.layers.")
               and name.rsplit(".",1)[-1] in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")]
        if not names:raise RuntimeError("Native language decoder LoRA targets were not found")
        if adapter:
            self.model=PeftModel.from_pretrained(base,str(adapter),adapter_name="policy",is_trainable=training,local_files_only=True)
            self.verify_adapter(adapter,"policy")
        else:
            config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias="none",task_type=TaskType.CAUSAL_LM,target_modules=names)
            self.model=get_peft_model(base,config,adapter_name="policy")
        if reference:
            self.model.load_adapter(str(reference),adapter_name="reference",is_trainable=False,local_files_only=True)
            self.verify_adapter(reference,"reference")
        for module in self.model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        self.model.set_adapter("policy");self.set_trainability()
        self.media=DecodedMedia(context)
        ids=[self.processor.tokenizer.encode(x,add_special_tokens=False) for x in ("A","B","C","D")]
        if any(len(v)!=1 for v in ids) or len({v[0] for v in ids})!=4:raise ValueError("Distinct one-token choices required")
        self.choice_ids=[v[0] for v in ids]
        self.prepared_key=None;self.prepared_value=None

    def verify_adapter(self,path,name):
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        expected=load_file(str(Path(path)/"adapter_model.safetensors"))
        loaded=get_peft_model_state_dict(self.model,adapter_name=name)
        if set(expected)!=set(loaded):raise ValueError("Task adapter tensor names do not match")
        for key,value in expected.items():
            if not self.torch.equal(value.float(),loaded[key].detach().cpu().float()):raise ValueError("Task adapter mismatch: "+key)

    def set_trainability(self):
        for name,p in self.model.named_parameters():
            p.requires_grad=bool(self.training and ".policy." in name and "lora_" in name and
                                 not any(k in name for k in ("visual","audio","projector")))

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def inventory(self):
        return [{"name":n,"shape":list(p.shape),"count":p.numel(),"dtype":str(p.dtype)}
                for n,p in self.model.named_parameters() if p.requires_grad]

    def parameter_digest(self,which="policy"):
        import hashlib
        h=hashlib.sha256()
        for name,p in self.model.named_parameters():
            if ((which=="policy" and ".policy." in name) or
                (which=="reference" and ".reference." in name) or
                (which=="base" and "lora_" not in name)):
                h.update(name.encode());h.update(p.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    @contextmanager
    def adapter(self,name):
        if name=="base":
            with self.model.disable_adapter():yield
            return
        previous=self.model.active_adapter
        self.model.set_adapter(name);self.set_trainability()
        try:yield
        finally:self.model.set_adapter(previous);self.set_trainability()

    def prepare(self,row,task):
        from PIL import Image
        safe=input_only(row)
        key=fingerprint({"input":safe,"task":task,"settings":self.context["settings"]})
        if self.prepared_key==key:return self.prepared_value
        visual=task in ("E","U","V")
        audio=task in ("E","U","M") and safe["protocol"]=="VA"
        media=self.media.load(row) if visual or audio else None
        content=[];kwargs={"padding":True,"return_tensors":"pt"}
        if visual:
            content.append({"type":"video","video":"input"})
            kwargs["videos"]=[media["frames"]]
            kwargs["videos_kwargs"]={"use_audio_in_video":False,"do_sample_frames":False,"do_resize":False,
                "return_metadata":True,"video_metadata":[media["metadata"]],"fps":2.}
            if len(media["faces"]):
                content.extend({"type":"image","image":"face"} for _ in media["faces"])
                kwargs["images"]=[Image.fromarray(c) for c in media["faces"]]
                kwargs["images_kwargs"]={"do_resize":False}
        if audio:
            if not media["audio"].size:raise ValueError("VA audio cannot be missing")
            content.append({"type":"audio","audio":"input"})
            kwargs["audio"]=media["audio"]
            kwargs["audio_kwargs"]={"sampling_rate":16000}
        content.append({"type":"text","text":instruction(row,task)})
        messages=[{"role":"system","content":[{"type":"text","text":"你是多模态情感分类助手。严格区分各路输入与真实中性，只回答要求的选项。"}]},
                  {"role":"user","content":content}]
        rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        kwargs["text"]=rendered
        inputs={k:v for k,v in self.processor(**kwargs).items() if v is not None}
        returned=inputs.pop("video_metadata",None)
        inputs["use_audio_in_video"]=False
        has_video="pixel_values_videos" in inputs;has_audio="input_features" in inputs
        if has_video!=visual or has_audio!=audio:raise RuntimeError("Modality isolation failed")
        if not visual and any(k in inputs for k in ("pixel_values","image_grid_thw")):raise RuntimeError("Visual evidence leaked")
        if inputs["input_ids"].shape[-1]>self.context["settings"]["max_prompt_tokens"]:raise ValueError("Untruncated prompt budget exceeded")
        torch=self.torch
        inputs={k:v.to(device=self.device,dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                else v.to(self.device) if torch.is_tensor(v) else v for k,v in inputs.items()}
        self.prepared_key,self.prepared_value=key,inputs
        return inputs

    def scores(self,row,task,*,adapter="policy"):
        from .omni_runtime import OmniPolicy
        inputs=self.prepare(row,task)
        self.model.train(self.training)
        for name,module in self.model.named_modules():
            if any(k in name for k in ("visual","audio","projector")):module.eval()
        base=self.model.get_base_model()
        if hasattr(base,"rope_deltas"):base.rope_deltas=None
        count=2 if task=="U" else 3 if task=="E" else 4
        with self.adapter(adapter),OmniPolicy.suffix_projection(self,1) as projection,self.torch.autocast("cuda",dtype=self.torch.bfloat16):
            output=self.model(**inputs,**projection,use_cache=False,return_dict=True)
        return output.logits[0,-1,self.choice_ids[:count]].float()

    def save_adapter(self,path):
        self.model.save_pretrained(str(path),selected_adapters=["policy"],safe_serialization=True)
        result=Path(path)/"policy"
        if not (result/"adapter_model.safetensors").is_file():raise RuntimeError("Task adapter was not saved")
        return result
