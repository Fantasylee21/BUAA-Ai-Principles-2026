"""Convex gradients, isolated evidence, split safety, recovery and quality gates."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from rl.evidence_bottleneck import (DEFAULTS, calibration_objective, decision_scores, disagreement,
    evidence_probabilities, feature_key, fit_calibration, fit_head, head_objective,
    layer_norm, log_odds, safe_input, settings_from, softmax, split_design, tasks_for, view_examples)
from rl.evidence_store import feature_entry, save_feature, saved_fit, save_fit
from rl.evidence_training import train_only
from rl.experiment import ROOT, write_json
from rl.grounded_uncertainty import load_training
from rl.omni_io import task_messages
from scripts.evidence_dashboard import render


def row(sid="s", protocol="VT", conflict=False):
    return {"sample_id": sid, "video": f"videos/{sid}.mp4", "protocol": protocol,
        "text_content": "A supplied transcript", "gt_label": "negative",
        "sample_type": "Conflict" if conflict else "Aligned", "source_dataset": "ch_sims_v2",
        "split_group_id": sid, "natural": True,
        "view_targets": {"vision": {"label": "negative", "clear": True},
            "text" if protocol == "VT" else "audio": {"label": "positive" if conflict else "negative", "clear": True}}}


class EvidenceTests(unittest.TestCase):
    def finite_difference(self, objective, theta):
        loss, gradient = objective(theta)
        for i in range(len(theta)):
            shift = np.zeros_like(theta); shift[i] = 1e-6
            numerical = (objective(theta+shift)[0]-objective(theta-shift)[0])/2e-6
            self.assertAlmostEqual(numerical, gradient[i], places=6)
        self.assertTrue(np.isfinite(loss))

    def test_head_gradient_and_confident_error_direction(self):
        x = layer_norm(np.array([[1., 2., 4.], [2., 1., 3.]]))
        native = np.array([[1000., -1000., 0.], [0., 0., 0.]])
        y = np.array([1, 2])
        objective = lambda theta: head_objective(theta, x, native, y)
        self.finite_difference(objective, np.arange(12, dtype=float)*.01)
        _, grad = objective(np.zeros(12))
        self.assertLess(grad.reshape(3,4)[1,-1], 0)
        self.assertGreater(grad.reshape(3,4)[0,-1], 0)

    def test_calibration_gradient_and_exact_class_costs(self):
        odds = np.array([-3., 2., 1., -.5, .2])
        protocol = np.array([0, 0, 1, 1, 1])
        y = np.array([1, 0, 0, 0, 0], dtype=bool)
        objective = lambda b: calibration_objective(b, odds, protocol, y)
        b = np.array([.2, -.1])
        self.finite_difference(objective, b)
        d = odds+b[protocol]
        expected = .6*np.logaddexp(0,-d[y]).mean()+.4*np.logaddexp(0,d[~y]).mean()+.005*np.dot(b,b)
        self.assertAlmostEqual(objective(b)[0], expected, places=13)

    def test_zero_residual_matches_native_and_normalizes(self):
        h = np.random.default_rng(0).normal(size=(4,8))
        native = np.arange(12,dtype=float).reshape(4,3)
        zero = {"parameters": np.zeros((3,9)).tolist()}
        np.testing.assert_array_equal(evidence_probabilities(h,native,zero), softmax(native))
        np.testing.assert_allclose(evidence_probabilities(h,native).sum(1), 1)

    def test_convex_emotion_fit_reduces_loss_and_recovers_exactly(self):
        rng = np.random.default_rng(0)
        h = rng.normal(size=(45,8)); y = h[:,:3].argmax(1)
        native = np.zeros((45,3))
        a, b = fit_head(h,native,y), fit_head(h,native,y)
        self.assertLess(a['solver']['data_loss'], a['solver']['initial_loss']*.6)
        np.testing.assert_array_equal(a['parameters'], b['parameters'])

    def test_nonconvergence_and_nonfinite_fail_closed(self):
        fake = types.SimpleNamespace(success=False,message='budget',nit=300,nfev=301,x=np.zeros(2))
        from rl.evidence_bottleneck import solve
        with patch('rl.evidence_bottleneck.minimize',return_value=fake):
            with self.assertRaisesRegex(RuntimeError,'did not converge'):
                solve(lambda p:(0.,np.ones(2)),2)
        with self.assertRaises(FloatingPointError):
            solve(lambda p:(float('nan'),np.ones(2)),2)

    def test_relation_endpoints_and_monotone_offsets(self):
        p = np.array([[1.,0.,0.],[1.,0.,0.]])
        m = np.array([[1.,0.,0.],[0.,1.,0.]])
        np.testing.assert_array_equal(disagreement(p,m), [0.,1.])
        rows = [row(str(i)) for i in range(51)]
        q = np.linspace(0,1,51)
        calibrated = decision_scores(q,rows,{'offsets':[3.,-2.]})
        self.assertEqual(calibrated[0],0);self.assertEqual(calibrated[-1],1)
        self.assertTrue((np.diff(calibrated)>0).all())
        with self.assertRaises(ValueError):disagreement(p*2,m)

    def test_calibration_preserves_evidence_and_no_double_weighting(self):
        rows = [row(str(i),conflict=i<3) for i in range(20)]
        q = np.full(20,.1); unchanged = q.copy()
        fitted = fit_calibration(q,rows)
        np.testing.assert_array_equal(q,unchanged)
        self.assertTrue(fitted['solver']['success'])
        d = decision_scores(q,rows,fitted)
        # Equal evidence must yield the same within-protocol decision, irrespective of label.
        self.assertTrue(np.all(d==d[0]))
        self.assertGreater(d[0],.5)

    def test_annotations_and_second_modality_cannot_change_visual_input(self):
        original = row()
        changed = {**original,'gt_label':'positive','sample_type':'garbage','view_targets':{},
            'source_dataset':'garbage','text_content':'something else','protocol':'VA','is_clear':False}
        self.assertEqual(safe_input(original,'vision'),safe_input(changed,'vision'))
        self.assertEqual(task_messages(original,Path('a.mp4'),'vision'),task_messages(changed,Path('a.mp4'),'vision'))
        self.assertEqual(feature_key('train',original,'vision'),feature_key('train',changed,'vision'))
        self.assertNotEqual(feature_key('train',original,'vision'),feature_key('val',original,'vision'))

    def test_visual_dedup_and_unclear_retained_for_relation(self):
        a,b = row(),row('other','VA');b['video']=a['video'];b['split_group_id']=a['split_group_id']
        self.assertEqual(len(tasks_for([a,b],'train')),3)
        self.assertEqual(len(view_examples([a,b],'vision')),1)
        b['view_targets']['audio']['clear']=False
        self.assertEqual(view_examples([a,b],'audio'),[])
        self.assertEqual(len(tasks_for([a,b],'train')),3)
        b['view_targets']['vision']['label']='positive'
        with self.assertRaises(ValueError):view_examples([a,b],'vision')

    def test_real_data_nested_partitions_and_supervision_counts(self):
        rows = load_training(ROOT/'Dataset/train.jsonl')
        self.assertEqual({m:len(view_examples(rows,m)) for m in ('vision','text','audio')},
                         {'vision':2084,'text':1730,'audio':1561})
        a,b = split_design(rows,42),split_design(rows,43)
        self.assertNotEqual(a['hash'],b['hash'])
        held = []
        for part in a['nested']:
            self.assertEqual([len(part[k]) for k in ('train','calibration','held')],[1788,448,1118])
            held.extend(part['held'])
            groups = [{rows[i]['video'] for i in part[k]} for k in ('train','calibration','held')]
            self.assertFalse(any(groups[i]&groups[j] for i in range(3) for j in range(i)))
        self.assertEqual(sorted(held),list(range(3354)))

    def test_validation_rows_cannot_reach_training_objectives(self):
        a,b=row('train'),row('val')
        train_only({'train':[a]},[a])
        with self.assertRaises(ValueError):train_only({'train':[a]},[b])
        with self.assertRaises(ValueError):train_only({'train':[a]},[{**a,'sample_type':'Conflict'}])

    def test_feature_checksums_disk_roundtrip_and_model_identity(self):
        r=row();task=tasks_for([r],'train')[0];c={'identity_hash':'base-only'}
        with tempfile.TemporaryDirectory() as d:
            h=np.arange(2048,dtype=np.float32);native=np.array([1.,2.,3.],dtype=np.float32)
            save_feature(d,c,task,h,native,.2)
            entry,values=feature_entry(d,c,task,required=True)
            np.testing.assert_array_equal(h,values[0])
            with self.assertRaises(ValueError):feature_entry(d,{'identity_hash':'v4'},task,required=True)
            path=Path(d)/'features/train'/entry['file'];path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):feature_entry(d,c,task,required=True)

    def test_incomplete_fit_recomputed_and_complete_fit_identity_checked(self):
        c={'identity_hash':'new-algorithm'};value={'solver':{'success':True},'parameters':[0.]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'head.json';write_json(path,value)
            self.assertIsNone(saved_fit(path,c,'signature'))
            save_fit(path,c,'signature',value)
            self.assertEqual(saved_fit(path,c,'signature'),value)
            with self.assertRaises(ValueError):saved_fit(path,c,'other-data')

    def test_short_dashboard_retains_loss_score_status_and_controls(self):
        with tempfile.TemporaryDirectory() as d,patch('scripts.evidence_dashboard.gpu_summary',return_value='GPU TEST'):
            root=Path(d)
            write_json(root/'pipeline.status.json',{'phase':'complete','reason':'primary_gate_not_met'})
            write_json(root/'results/baseline.json',{'metrics':{'score':.712673}})
            with patch('scripts.evidence_dashboard.shutil.get_terminal_size',return_value=types.SimpleNamespace(columns=150,lines=16)):
                text=render(root)
            self.assertLessEqual(len(text.splitlines()),15)
            for term in ('LOSS','0.712673','正常结束','Ctrl+B'):self.assertIn(term,text)

    def test_primary_failure_stops_before_replication_questions_and_test(self):
        import rl.train_evidence_bottleneck as app
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[row('c',conflict=True),row('a')]
            c={'identity_hash':'unit','started':0,'val':rows,
               'baselines':{'val':{r['sample_id']:{'pred_emotion':'negative','is_uncertain':'no','clarifying_question':'none'} for r in rows}}}
            base=app.extra_report(app.baseline_records(c))['score'];calls=[]
            def jobs(args,jobs,cpu=False):
                for command,role,extra in jobs:
                    calls.append((command,role,cpu))
                    if command=='preflight':write_json(root/'preflight.json',{'identity_hash':'unit','passed':True,'active_adapter':None})
                    if command=='fit':
                        write_json(root/'development.json',{'task_component':.4})
                        write_json(root/'results/evidence-seed42.json',{'identity_hash':'unit','seed':42,'metrics':{'score':base+.001}})
            fake=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
            args=types.SimpleNamespace(run_dir=root)
            with patch.dict('sys.modules',{'fcntl':fake}),patch.object(app,'initialize',return_value=c),\
                 patch.object(app,'run_jobs',side_effect=jobs),patch.object(app,'publish'):
                app.pipeline(args);n=len(calls);app.pipeline(args);self.assertEqual(n,len(calls))
            self.assertEqual(json.loads((root/'final-report.json').read_text())['reason'],'primary_gate_not_met')
            self.assertFalse(any(command in ('questions','export') or '43' in role or role.startswith('test-') for command,role,_ in calls))

    def test_pending_outputs_cannot_authorize_test(self):
        from rl.train_evidence_bottleneck import qualified
        with tempfile.TemporaryDirectory() as d:
            rows=[row('c',conflict=True),row('a')]
            c={'identity_hash':'unit','val':rows,'baselines':{'val':{r['sample_id']:{'pred_emotion':'negative','is_uncertain':'no','clarifying_question':'none'} for r in rows}}}
            for seed in (42,43):write_json(Path(d)/f'results/evidence-seed{seed}.json',{'identity_hash':'unit','seed':seed,
                'metrics':{'score':1.,'invalid_outputs':None,'inconsistent_questions':None,'emotion_changes':0}})
            with self.assertRaises(ValueError):qualified(d,c)

    def test_fixed_configuration_rejects_unapproved_sweeps(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';write_json(path,{'regularization':.2})
            with self.assertRaises(ValueError):settings_from(path)

    def test_nested_oof_final_fit_and_resume_end_to_end(self):
        from rl.evidence_training import development, final_candidate
        rows=[]
        for i in range(36):
            for protocol in ('VT','VA'):
                r=row(f'{i}-{protocol}',protocol,i%2==0)
                r.update(video=f'videos/{i}.mp4',split_group_id=str(i))
                rows.append(r)
        val=[]
        for i in range(12):
            r=row(f'v{i}','VT' if i%2 else 'VA',i%3==0)
            r.update(video=f'videos/{100+i}.mp4')
            val.append(r)
        class Bank:
            split='train'
            def get(self, selected, mode):
                h=[]
                for r in selected:
                    # A deterministic fake encoder using only an input path.
                    n=int(Path(r['video']).stem)
                    label=1 if mode=='vision' else (0 if n%2==0 else 1)
                    v=np.zeros(8);v[label]=3;v[3:]=np.arange(5)*.1;h.append(v)
                return np.array(h),np.zeros((len(h),3))
        c={'identity_hash':'integration','settings':DEFAULTS,'train':rows,'val':val,
           'designs':{'42':split_design(rows,42),'43':split_design(rows,43)},
           'baselines':{s:{r['sample_id']:{'pred_emotion':'negative','is_uncertain':'no','clarifying_question':'none'}
               for r in rr} for s,rr in (('train',rows),('val',val))}}
        with tempfile.TemporaryDirectory() as d,patch('rl.evidence_training.FeatureBank',return_value=Bank()):
            dev=development(d,c,Bank(),lambda *a,**kw:None)
            out=final_candidate(d,c,Bank(),42,lambda *a,**kw:None)
            self.assertEqual(len(dev['records']),72)
            self.assertEqual(len(out['val_probabilities']),12)
            self.assertTrue(out['metrics']['output_checks_pending'])
            self.assertEqual(final_candidate(d,c,Bank(),42,lambda *a,**kw:None),out)
            train_ids={r['sample_id'] for r in rows}
            for path in Path(d).glob('heads/*/*.json'):
                if path.name.endswith('.complete.json'):continue
                fit=json.loads(path.read_text(encoding='utf-8'))
                self.assertTrue(set(fit['examples'])<=train_ids)
            oof=json.loads((Path(d)/'calibration-oof-42.json').read_text(encoding='utf-8'))
            self.assertEqual(oof['ids'],[r['sample_id'] for r in rows])


if __name__=='__main__':unittest.main()
