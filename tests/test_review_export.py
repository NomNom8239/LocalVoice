"""Offline review CSV application to immutable AST browse-library v2."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil

import pytest

from localvoice.style import batch, review_export


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'PROJECT_ROOT', tmp_path)
    profile, run_id = 'Ui_Shigure', 'run_test'
    run = tmp_path / 'work' / run_id
    v1 = tmp_path / 'outputs' / 'candidates' / profile / 'v1'
    run.mkdir(parents=True)
    (v1 / 'audio').mkdir(parents=True)
    inputs, predictions, manifest = [], [], []
    original = tmp_path / 'data' / 'training_audio' / profile / 'audio'
    original.mkdir(parents=True)
    categories = ['07_息をのむ', '04_笑い', '01_通常会話', '05_喘ぎ_うめき']
    for i, cat in enumerate(categories, 1):
        sid = f'c{i:05d}'
        wav = original / f'{i}.wav'
        wav.write_bytes(f'RIFF-fake-wav-{i}'.encode())
        sha = hashlib.sha256(wav.read_bytes()).hexdigest()
        row = {'source_id':sid,'profile':profile,'source_path':str(wav.resolve()),
               'relative_path':wav.name,'source_sha256':sha,'size_bytes':wav.stat().st_size}
        inputs.append(row)
        pred = {**row,'status':'scored','category_dir':cat}
        predictions.append(pred)
        relative = f'audio/{cat}/{sid}_{sha[:8]}.wav'
        target = v1 / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav,target)
        manifest.append({**pred,'output_path':relative,'output_sha256':sha,
                         'library_tier':'UNVERIFIED_AST_CANDIDATE','human_approved':False})
    digest = batch._fingerprint(inputs)
    run_state = {'run_id':run_id,'status':'completed','profile':profile,'samples':4,
                 'input_sha256':digest}
    (run/'run_manifest.json').write_text(json.dumps(run_state),encoding='utf-8')
    (run/'input_manifest.jsonl').write_text(''.join(batch._json(x)+'\n' for x in inputs),encoding='utf-8')
    (run/'style_predictions.jsonl').write_text(''.join(batch._json(x)+'\n' for x in predictions),encoding='utf-8')
    (v1/'manifest.jsonl').write_text(''.join(batch._json(x)+'\n' for x in manifest),encoding='utf-8')
    (v1/'summary.json').write_text(json.dumps({
        'profile':profile,'run_id':run_id,'version':'v1','input_sha256':digest,
        'source_count':4,'tier':'UNVERIFIED_AST_CANDIDATE',
    }),encoding='utf-8')
    csv_path=tmp_path/'review.csv'
    reviews = [
       dict(source_id='c00001',source_sha256=inputs[0]['source_sha256'],ast_category=categories[0],
            manual_category='02_囁き',decision='keep',identity='self',quality='ok',
            reject_reason='',memo='確認',updated_at='2026-10-09T11:00:00Z'),
       dict(source_id='c00002',source_sha256=inputs[1]['source_sha256'],ast_category=categories[1],
            manual_category='90_その他',decision='keep',identity='unverified',quality='unverified',
            reject_reason='',memo='',updated_at='2026-10-09T11:00:00Z'),
       dict(source_id='c00004',source_sha256=inputs[3]['source_sha256'],ast_category=categories[3],
            manual_category='',decision='reject',identity='other',quality='bad_audio',
            reject_reason='other_voice',memo='',updated_at='2026-10-09T11:00:00Z'),
    ]
    def write_csv(items):
        with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(review_export.COLUMNS));w.writeheader();w.writerows(items)
    write_csv(reviews)
    return tmp_path,run_id,v1,csv_path,inputs,manifest,reviews,write_csv


def test_dry_run_and_full_candidate_v2_preserve_v1(prepared):
    root,run_id,v1,csv_file,inputs,manifest,rows,write=prepared
    before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in v1.rglob('*') if x.is_file()}
    report=review_export.apply_review(run_id,csv_file,dry_run=True)
    assert report['source_count']==4 and report['review_records']==3
    assert (report['reviewed_usable'],report['needs_attention'],report['rejected'],report['unreviewed'])==(1,1,1,1)
    assert not (v1.parent/'v2').exists() and not list(v1.parent.glob('.building*'))
    actual=review_export.apply_review(run_id,csv_file)
    assert report==actual
    v2=v1.parent/'v2'
    assert (v2/'audio'/'02_囁き').exists()
    assert (v2/'audio'/review_export.CAUTION).exists()
    assert (v2/'audio'/review_export.REJECTED).exists()
    assert len(list((v2/'audio').rglob('*.wav')))==4
    result=batch._read_jsonl(v2/'manifest.jsonl')
    assert len(result)==4
    entries={x['source_id']:x for x in result}
    assert entries['c00001']['category_dir']=='02_囁き'
    assert entries['c00001']['review_status']=='reviewed_usable'
    assert entries['c00002']['category_dir']==review_export.CAUTION
    assert entries['c00002']['effective_category']=='90_その他'
    assert entries['c00003']['review_status']=='unreviewed'
    assert entries['c00004']['category_dir']==review_export.REJECTED
    assert all(x['human_approved'] is False for x in result)
    with (v2/'reviewed_usable.csv').open(encoding='utf-8-sig',newline='') as f:
        assert [r['source_id'] for r in csv.DictReader(f)]==['c00001']
    with (v2/'needs_attention.csv').open(encoding='utf-8-sig',newline='') as f:
        assert {r['source_id'] for r in csv.DictReader(f)}=={'c00002','c00004'}
    assert before=={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in v1.rglob('*') if x.is_file()}
    assert not (root/'outputs'/'datasets').exists()
    with pytest.raises(FileExistsError,match='already exists'):
        review_export.apply_review(run_id,csv_file)


def test_csv_sha_duplicate_unknown_and_category_fail_closed(prepared):
    root,run_id,v1,csv_file,inputs,manifest,rows,write=prepared
    modified=[dict(x) for x in rows]
    modified[0]['source_sha256']='0'*64
    write(modified)
    with pytest.raises(ValueError,match='SHA/AST'):
        review_export.apply_review(run_id,csv_file)
    write([rows[0],rows[0]])
    with pytest.raises(ValueError,match='duplicate'):
        review_export.apply_review(run_id,csv_file)
    modified=[dict(rows[0],manual_category='777_not_defined')]
    write(modified)
    with pytest.raises(ValueError,match='Invalid review'):
        review_export.apply_review(run_id,csv_file)
    modified=[dict(rows[0],ast_category='01_通常会話')]
    write(modified)
    with pytest.raises(ValueError,match='SHA/AST'):
        review_export.apply_review(run_id,csv_file)
    assert not (v1.parent/'v2').exists()


def test_tampered_v1_wav_blocks_without_creating_v2(prepared):
    root,run_id,v1,csv_file,inputs,manifest,rows,write=prepared
    (v1/manifest[1]['output_path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA mismatch'):
        review_export.apply_review(run_id,csv_file)
    assert not (v1.parent/'v2').exists()


def test_resume_partial_stage_does_not_duplicate_or_overwrite(prepared,monkeypatch):
    root,run_id,v1,csv_file,inputs,manifest,rows,write=prepared
    original=review_export.shutil.copyfileobj
    calls=0
    def interrupted(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2: raise OSError('disk interrupted')
        return original(*a,**kw)
    monkeypatch.setattr(review_export.shutil,'copyfileobj',interrupted)
    with pytest.raises(OSError,match='interrupted'):
        review_export.apply_review(run_id,csv_file)
    stage=v1.parent/f'.building-v2-review-{run_id}'
    assert stage.is_dir()
    monkeypatch.setattr(review_export.shutil,'copyfileobj',original)
    with pytest.raises(FileExistsError,match='Incomplete stage'):
        review_export.apply_review(run_id,csv_file)
    with csv_file.open('ab') as out: out.write(b'\n')
    with pytest.raises(ValueError,match='Staged review evidence mismatch'):
        review_export.apply_review(run_id,csv_file,resume_export=True)
    write(rows)
    assert review_export.apply_review(run_id,csv_file,resume_export=True)['source_count']==4
    assert not stage.exists() and (v1.parent/'v2').exists()
    assert len(list((v1.parent/'v2'/'audio').rglob('*.wav')))==4


def test_reject_without_reason_and_hold_are_not_ready(prepared):
    root,run_id,v1,csv_file,inputs,manifest,rows,write=prepared
    changed=[dict(rows[0], decision='reject')]
    write(changed)
    with pytest.raises(ValueError,match='Invalid review'):
        review_export.apply_review(run_id,csv_file)
    changed=[dict(rows[0],decision='hold',identity='self',quality='ok')]
    write(changed)
    report=review_export.apply_review(run_id,csv_file)
    assert report['held']==1 and report['reviewed_usable']==0
    assert (v1.parent/'v2'/'audio'/review_export.HELD).exists()
