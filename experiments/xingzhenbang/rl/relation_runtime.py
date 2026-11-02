"""Read-only Omni feature extraction and frozen-v4 question completion."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .finite_actions import omni_settings
from .finite_runtime import FinitePolicy
from .grounded_uncertainty import main_prefix
from .omni_io import prepare_omni_inputs, prepare_task_inputs
from .reward import VALID_EMOTIONS, parse_completion, question_is_consistent


class RelationExtractor(FinitePolicy):
    def __init__(self,model,adapter,settings):
        super().__init__(model,adapter,omni_settings(settings),training=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        tokenizer=self.processor.tokenizer
        self.value_ids={"full":[tokenizer.encode(v,add_special_tokens=False) for v in ("yes","no")],
                        "view":[tokenizer.encode(v,add_special_tokens=False) for v in VALID_EMOTIONS]}
        if any(len(v)!=1 for values in self.value_ids.values() for v in values):
            raise ValueError("Native field values must be single tokens")
        self.value_ids={k:[v[0] for v in values] for k,values in self.value_ids.items()}

    def extend(self,prefix,text):
        torch=self.torch
        ids=torch.tensor(self.processor.tokenizer.encode(text,add_special_tokens=False),
                         device=prefix["input_ids"].device,dtype=torch.long)[None,:]
        result=dict(prefix)
        result["input_ids"]=torch.cat((prefix["input_ids"],ids),-1)
        result["attention_mask"]=torch.cat((prefix["attention_mask"],torch.ones_like(ids)),-1)
        return result

    def hidden(self,prefix,features,mode,emotion=None):
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Omni must remain completely frozen")
        torch=self.torch
        inputs=self.extend(prefix,main_prefix(emotion) if mode=="full" else '{"pred_emotion":"')
        if not bool(inputs["attention_mask"][0,-1]):
            raise ValueError("Feature extraction position is padded")
        captured=[]
        head=self.model.get_base_model().get_output_embeddings()
        def capture(module,args):
            captured.append(args[0][0,-1,:].detach().float().clone())
        self.clear_positions()
        with self.adapter("reference"),self.cached_perception(features),self.suffix_projection(1) as projection:
            handle=head.register_forward_pre_hook(capture)
            try:
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    output=self.model(**inputs,**projection,use_cache=False,return_dict=True)
            finally:
                handle.remove()
        if len(captured)!=1 or captured[0].shape!=(2048,) or not torch.isfinite(captured[0]).all():
            raise RuntimeError("Unexpected final-hidden-state capture")
        values=output.logits[0,-1,self.value_ids["full" if mode=="full" else "view"]].float().detach()
        return captured[0],values

    def prepare_view_reusing(self,row,c,mode,original,features):
        prefix=prepare_task_inputs(self.processor,row,c["data_root"],self.settings,mode)
        if mode=="text":
            return prefix,{}
        key="video" if mode=="vision" else "audio"
        if key not in features:
            raise ValueError("Cannot reuse missing modality features")
        torch=self.torch
        for name in (("video_grid_thw",) if mode=="vision" else ("feature_attention_mask",)):
            if name not in prefix or name not in original or not torch.equal(prefix[name],original[name]):
                raise ValueError("Isolated input changed frozen encoder metadata")
        token=getattr(self.model.get_base_model().config,key+"_token_id")
        if int((prefix["input_ids"]==token).sum()) != features[key].shape[0]:
            raise ValueError("Isolated feature/token count changed")
        return prefix,{key:features[key]}

    def inputs(self,row,split,c,source,baseline):
        from .grounded_store import task_key
        if split!="test":
            full,features=source.load(row,split,"full")
        else:
            full=prepare_omni_inputs(self.processor,row,c["data_root"],self.settings)
            features=self.perception(full)
        result={"full":(full,features)}
        for mode in ("vision","text" if row["protocol"]=="VT" else "audio"):
            if split!="test" and task_key(row["sample_id"],mode) in source.entries:
                result[mode]=source.load(row,split,mode)
            else:
                result[mode]=self.prepare_view_reusing(row,c,mode,full,features)
        return result

    def extract(self,row,split,c,source,baseline):
        started=time.monotonic()
        tasks=self.inputs(row,split,c,source,baseline)
        outputs,diagnostics=[],{}
        for mode in ("vision","text" if row["protocol"]=="VT" else "audio","full"):
            h,logits=self.hidden(*tasks[mode],mode,baseline["pred_emotion"] if mode=="full" else None)
            outputs.append(h)
            diagnostics[mode]={"logits":logits.cpu().tolist(),"tokens":int(tasks[mode][0]["input_ids"].shape[1])}
        diagnostics["seconds"]=time.monotonic()-started
        return self.torch.stack(outputs).cpu(),diagnostics

    def answer(self,row,split,c,source,baseline,uncertain):
        from .conflict_runtime import generated_question
        if uncertain not in ("yes","no"):
            raise ValueError("Invalid selected relation decision")
        if uncertain=="no":
            question="none"
        elif baseline["is_uncertain"]=="yes":
            question=baseline["clarifying_question"]
        else:
            if split=="test":
                prefix=prepare_omni_inputs(self.processor,row,c["data_root"],self.settings)
                features=self.perception(prefix)
            else:
                prefix,features=source.load(row,split,"full")
            selected=main_prefix(baseline["pred_emotion"])+uncertain+'","clarifying_question":'
            for opening in ("",'"'):
                with self.cached_perception(features):
                    _,tail=self.generate(self.extend(prefix,selected+opening),adapter="reference")
                result=generated_question(tail,opened_quote=bool(opening))
                if result is not None:
                    question=result[0]
                    break
            else:
                raise RuntimeError("Frozen v4 failed to generate a valid question for "+row["sample_id"])
        value={"pred_emotion":baseline["pred_emotion"],"is_uncertain":uncertain,"clarifying_question":question}
        if parse_completion(json.dumps(value)) is None or not question_is_consistent(value):
            raise ValueError("Frozen question serialization failed")
        return json.dumps(value,ensure_ascii=False,separators=(",",":"))
