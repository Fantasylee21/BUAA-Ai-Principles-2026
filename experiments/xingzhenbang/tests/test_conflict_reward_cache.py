"""Reproduce the actual A/B cache-lock race with separate OS processes."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.joint_affect_pipeline import atomic,digest,read,run_lock,immutable
from scripts.train_conflict_reward import MediaCache
from rl.cache_io import cache_writer_lock
from rl.reference_cache_recovery import import_reference_cache,REFERENCE_SOURCES
from rl.experiment import fingerprint


def cache_fixture(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    media=folder/"video.mp4";media.write_bytes(b"fake media; no real decoder used by this locking test")
    row={"sample_id":"one","protocol":"VT","video":"video.mp4","text_content":"same input"}
    c={"batch_root":str(folder),"data_root":str(folder),"base_identity":"frozen",
       "manifest_value":{"media":{"video.mp4":digest(media)}}}
    return c,row


def cpu_policy(folder):
    import torch
    def prepare(*args):
        with (Path(folder)/"decode-count.txt").open("a") as f:f.write("decode\n")
        time.sleep(.15)
        return {"input_ids":torch.tensor([[1,2]]),"attention_mask":torch.ones(1,2,dtype=torch.long)}
    return SimpleNamespace(torch=torch,prepare=prepare,perception=lambda _: {"video":torch.ones(1,2)})


def process_load(c,row,barrier,result):
    try:
        cache=MediaCache(c,cpu_policy(c["batch_root"]))
        barrier.wait(timeout=30)
        with patch("rl.finite_runtime.unpack_cache",side_effect=lambda torch,packed:(packed["prefix"],packed["features"])):
            for _ in range(4):
                prefix,features=cache.load(row)
                if prefix["input_ids"].tolist()!=[[1,2]]:raise AssertionError("Changed immutable payload")
        result.put({"ok":True})
    except BaseException as exc:
        result.put({"ok":False,"error":repr(exc)})


class CacheConcurrency(unittest.TestCase):
    def test_completed_reader_ignores_both_writer_and_controller_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            c,row=cache_fixture(folder);cache=MediaCache(c,cpu_policy(folder))
            with patch("rl.finite_runtime.unpack_cache",side_effect=lambda torch,packed:(packed["prefix"],packed["features"])):
                cache.load(row)
                key,_,_=cache.key_for(row);root=cache.root/key
                # The previous implementation raises BlockingIOError here.
                with run_lock(root),cache_writer_lock(root):
                    MediaCache(c,cpu_policy(folder)).load(row)
            self.assertEqual((Path(folder)/"decode-count.txt").read_text().splitlines(),["decode"])

    def run_processes(self,completed):
        with tempfile.TemporaryDirectory() as folder:
            c,row=cache_fixture(folder);cache=MediaCache(c,cpu_policy(folder))
            if completed:
                with patch("rl.finite_runtime.unpack_cache",side_effect=lambda torch,packed:(packed["prefix"],packed["features"])):
                    cache.load(row)
            ctx=multiprocessing.get_context("spawn");barrier=ctx.Barrier(2);results=ctx.Queue()
            children=[ctx.Process(target=process_load,args=(c,row,barrier,results)) for _ in range(2)]
            key,_,_=cache.key_for(row)
            with run_lock(cache.root/key):
                for child in children:child.start()
                received=[results.get(timeout=60) for _ in children]
                for child in children:child.join(timeout=10)
            try:
                self.assertTrue(all(x["ok"] for x in received),received)
                self.assertTrue(all(child.exitcode==0 for child in children))
                self.assertEqual((Path(folder)/"decode-count.txt").read_text().splitlines(),["decode"])
            finally:
                for child in children:
                    if child.is_alive():child.terminate();child.join()

    @unittest.skipIf(os.name=="nt","Actual POSIX process race is verified on training server")
    def test_two_processes_read_same_completed_entry_under_legacy_lock(self):
        self.run_processes(True)

    @unittest.skipIf(os.name=="nt","Actual POSIX writer race is verified on training server")
    def test_two_processes_create_same_missing_entry_only_once(self):
        self.run_processes(False)

    def test_writer_wait_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as folder:
            def cancel():raise InterruptedError("requested stop")
            with cache_writer_lock(folder):
                with self.assertRaises(InterruptedError):
                    with cache_writer_lock(folder,check=cancel):self.fail("Should not acquire")


class Recovery(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);manifest={"train":[{"sample_id":"a"},{"sample_id":"b"}],
            "folds":{"content":{"folds":[{"train":[0],"held":[1]}]}}}
        contexts=[]
        for name in ("old","new"):
            base=root/name;immutable(base/"manifest.json",manifest)
            scientific={"settings":{"same":True},"source_sha256":{p:"same" for p in REFERENCE_SOURCES}}
            plan={"identity":scientific,"identity_hash":fingerprint(scientific)};immutable(base/"plan.json",plan)
            c={"batch_root":str(base),"batch_identity":plan["identity_hash"],"cache_root":str(base/"cache"),
               "manifest":str(base/"manifest.json"),"manifest_sha256":digest(base/"manifest.json"),
               "grouping":"content","fold":0,"reference":"same adapter","reference_artifact_hash":"same weights"}
            immutable(base/"context.json",c);contexts.append(base/"context.json")
        old,new=contexts;entries=[]
        for sid,split in (("a","train"),("b","held")):
            entries.append({"sample_id":sid,"split":split,"context_sha256":digest(old),
                "baseline":{"sample_id":sid,"pred_emotion":"positive","is_uncertain":"no","clarifying_question":"none"},
                "reference_log_odds":-2.5})
        path=root/"old/cache/shard-0.json";immutable(path,{"context_sha256":digest(old),"entries":entries})
        immutable(path.parent/"complete.json",{"context_sha256":digest(old),"files":{path.name:digest(path)}})
        return old,new,entries

    def test_recovery_rewraps_provenance_without_changing_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            old,new,entries=self.fixture(folder);result=import_reference_cache(new,old)
            recovered=read(Path(read(new)["cache_root"])/"shard-0.json")["entries"]
            self.assertTrue(result["reference_values_unchanged"]);self.assertFalse(result["policy_checkpoints_imported"])
            for a,b in zip(entries,recovered):
                for key in ("sample_id","split","baseline","reference_log_odds"):self.assertEqual(a[key],b[key])

    def test_recovery_rejects_changed_reference_or_inference_source(self):
        with tempfile.TemporaryDirectory() as folder:
            old,new,_=self.fixture(folder)
            value=read(new);value["reference_artifact_hash"]="other weights";atomic(new,value)
            with self.assertRaises(ValueError):import_reference_cache(new,old)

    def test_recovery_rejects_modified_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            old,new,_=self.fixture(folder)
            p=Path(read(old)["cache_root"])/"shard-0.json";value=read(p);value["entries"][0]["reference_log_odds"]=99.;atomic(p,value)
            with self.assertRaises(ValueError):import_reference_cache(new,old)

    def test_recovery_rejects_changed_prompt_source_even_with_valid_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            old,new,_=self.fixture(folder)
            path=Path(read(new)["batch_root"])/"plan.json";plan=read(path)
            plan["identity"]["source_sha256"]["sft_prompt.py"]="different prompt"
            plan["identity_hash"]=fingerprint(plan["identity"]);atomic(path,plan)
            c=read(new);c["batch_identity"]=plan["identity_hash"];atomic(new,c)
            with self.assertRaises(ValueError):import_reference_cache(new,old)


if __name__=="__main__":unittest.main()
