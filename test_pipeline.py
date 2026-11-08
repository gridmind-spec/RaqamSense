from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_model import build_data
from app import prepare_input,predict_image,predict_sketch,MODEL,METRICS


def test_cleaning_and_shape():
    df, columns, cleaning = build_data()
    assert len(df)==1797
    assert len(columns)==64
    assert df.isna().sum().sum()==0
    assert df.duplicated().sum()==0
    assert list(df['digit'].unique())


def test_final_metrics_are_grounded():
    assert METRICS['test_samples']==360
    assert 0.7 < METRICS['test_accuracy'] < 1
    assert METRICS['validation_accuracy_at_selection'] > 0.7
    assert len(METRICS['tuning']) == 6


def test_ten_dataset_examples():
    preds=[]
    for label in range(10):
        image=Image.open(ROOT/'app_assets'/'samples'/f'digit_{label}.png')
        vec, preview=prepare_input(image)
        assert vec.shape==(1,64)
        assert preview.size==(240,240)
        assert np.min(vec)>=0 and np.max(vec)<=16
        outcome,_=predict_image(image)
        preds.append(int(outcome.split('outcome-digit">')[1].split('<')[0]))
    assert sum(i==p for i,p in enumerate(preds))>=8


def test_hand_drawn_image_path():
    img=Image.new('RGB',(256,256),'black')
    from PIL import ImageDraw
    d=ImageDraw.Draw(img)
    d.line([(120,50),(120,210)],fill='white',width=24)
    outcome,preview=predict_sketch({'composite':img})
    assert 'outcome-digit' in outcome
    assert preview.size==(240,240)


def test_missing_input_rejected():
    with pytest.raises(Exception):
        predict_image(None)
