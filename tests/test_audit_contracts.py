import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import load_config
from data.dataset import JointNormalizer
from training.engine import Trainer, compare_scene_tables
from main import MODEL_CLASSES


class AuditContracts(unittest.TestCase):
    def test_shipped_config_is_runnable(self):
        cfg = load_config()
        self.assertEqual(cfg.training['device'], 'cuda')
        for variant, model in MODEL_CLASSES.items():
            self.assertEqual(cfg.model['architectures'][variant], model.architecture_version)

    def test_invalid_normalization_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'norm.json'
            for mean,std in [([0]*6,[0]*6),([0]*6,[-1]*6),([0]*6,[float('nan')]*6),([0]*5,[1]*6),([float('inf')]*6,[1]*6)]:
                path.write_text(json.dumps({'joint':{'mean':mean,'std':std}}))
                with self.assertRaises(ValueError): JointNormalizer(path)

    def test_valid_normalization_preserves_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'norm.json'
            path.write_text(json.dumps({'joint':{'mean':[2]*6,'std':[4]*6}}))
            image=np.full((6,2,2),10,dtype=np.float32)
            image[0,0,0]=np.nan
            actual=JointNormalizer(path)(image)
            self.assertEqual(actual[0,0,0],0)
            self.assertTrue((actual[1:]==2).all())

    def test_missing_geography_is_rejected(self):
        trainer=object.__new__(Trainer)
        dataset=SimpleNamespace(frame=pd.DataFrame([{'scene_id':'S1'}]))
        with self.assertRaises(KeyError): trainer._scene_metadata_lookup(dataset)

    def test_comparison_rejects_mismatched_protocol_and_duplicates(self):
        row=dict(scene_id='S1',province='AH',spatial_group_id='G1',scale_bin='<5%',
                 valid_mask_sha256='mask',target_sha256='target',threshold=.5,precision='torch.float32',
                 tp=2,fp=1,fn=1,scene_iou=.5)
        for key in ('valid_mask_sha256','target_sha256','threshold','precision','spatial_group_id'):
            other=copy.copy(row);other[key]='different'
            with self.assertRaises(ValueError):compare_scene_tables([row],[other],'a','b',20)
        with self.assertRaises(ValueError):compare_scene_tables([row,row],[row],'a','b',20)
        legacy=copy.copy(row);legacy.pop('valid_mask_sha256')
        with self.assertRaises(ValueError):compare_scene_tables([row],[legacy],'a','b',20)

    def test_identical_comparison_is_zero(self):
        row=dict(scene_id='S1',province='AH',spatial_group_id='G1',scale_bin='<5%',
                 valid_mask_sha256='mask',target_sha256='target',threshold=.5,precision='torch.float32',
                 tp=2,fp=1,fn=1,scene_iou=.5)
        result=compare_scene_tables([row],[row],'a','b',20)
        self.assertEqual(result['overall']['delta_micro_iou_pp_ci95'],[0.,0.])
