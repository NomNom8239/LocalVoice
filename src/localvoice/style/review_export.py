"""Apply browser CSV decisions to an immutable AST candidate library release.

This is a *candidate* reclassification, not promotion to approved datasets.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from . import batch

SCHEMA = 'localvoice.ast-review-export.v1'
COLUMNS = (
    'source_id', 'source_sha256', 'ast_category', 'manual_category',
    'decision', 'identity', 'quality', 'reject_reason', 'memo', 'updated_at',
)
CATEGORIES = set(batch.JAPANESE_DIRS.values()) | {
    batch.UNKNOWN_DIR, '10_あくび', '90_その他',
}
IDENTITIES = {'unverified', 'self', 'other', 'overlap', 'uncertain'}
QUALITIES = {'unverified', 'ok', 'noise', 'bgm', 'clipping', 'overlap', 'bad_audio', 'other'}
REASONS = {'', 'other_voice', 'mixed', 'noise', 'bgm', 'quality', 'not_useful', 'other'}
CAUTION = '96_要確認_使用不可'
HELD = '97_保留'
REJECTED = '98_除外'


def _read_csv(path: Path, originals: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, str]], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Invalid or linked review CSV: {path}')
    raw = path.read_bytes()
    with io.StringIO(raw.decode('utf-8-sig'), newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError('Review CSV columns do not match the browser export schema')
        result = {}
        for row in reader:
            source_id = row.get('source_id', '')
            if any(v is None for v in row.values()) or None in row:
                raise ValueError(f'Malformed review CSV row: {source_id}')
            if source_id not in originals or source_id in result:
                raise ValueError(f'Unknown/duplicate source_id in review CSV: {source_id}')
            src = originals[source_id]
            if row['source_sha256'] != src['source_sha256'] or row['ast_category'] != src['category_dir']:
                raise ValueError(f'WAV SHA/AST category mismatch: {source_id}')
            if (row['manual_category'] not in CATEGORIES | {''} or
                row['decision'] not in {'', 'keep', 'hold', 'reject'} or
                row['identity'] not in IDENTITIES or
                row['quality'] not in QUALITIES or
                row['reject_reason'] not in REASONS or
                (row['decision'] == 'reject') != bool(row['reject_reason'])):
                raise ValueError(f'Invalid review category/decision/identity/quality/reason: {source_id}')
            if len(row['memo']) > 2000 or len(row['updated_at']) > 80:
                raise ValueError(f'Review memo/timestamp too long: {source_id}')
            result[source_id] = row
    return result, hashlib.sha256(raw).hexdigest()


def _safe_audio_file(root: Path, record: dict[str, Any]) -> Path:
    """Validate *exactly* the v1 audio location before reading or copying."""
    rel = record['output_path']
    if not isinstance(rel, str) or '\\' in rel:
        raise ValueError('Invalid v1 audio path')
    parts = PurePosixPath(rel).parts
    expected = ('audio', record['category_dir'],
                f"{record['source_id']}_{record['source_sha256'][:8]}.wav")
    if parts != expected or any(part in ('.', '..') for part in parts):
        raise ValueError(f'Unexpected v1 audio path: {rel}')
    path = root.joinpath(*parts)
    if any(x.is_symlink() for x in (root, path.parent.parent, path.parent, path)):
        raise ValueError(f'Linked candidate WAV or folder: {path}')
    if not path.is_file() or batch._sha256(path) != record['source_sha256']:
        raise ValueError(f'Candidate WAV SHA mismatch/missing: {path}')
    return path


def _preflight(run_id: str, source_version: str, csv_path: Path):
    if not batch.SAFE_NAME.fullmatch(source_version):
        raise ValueError('Unsafe source version')
    run = batch._work(run_id)
    state = batch._state(run)
    if state.get('run_id') != run_id or state.get('status') not in ('completed', 'completed_with_errors'):
        raise ValueError('AST source run is not complete')
    profile = state['profile']
    if not batch.SAFE_NAME.fullmatch(profile):
        raise ValueError('Unsafe profile')
    root = batch.PROJECT_ROOT / 'outputs' / 'candidates' / profile / source_version
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'No immutable source library: {root}')
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if (summary.get('profile'), summary.get('run_id'), summary.get('version'),
        summary.get('input_sha256'), summary.get('source_count')) != (
        profile, run_id, source_version, state['input_sha256'], state['samples']
    ) or summary.get('tier') != 'UNVERIFIED_AST_CANDIDATE':
        raise ValueError('v1 library provenance does not match source AST run')
    rows = batch._read_jsonl(root / 'manifest.jsonl')
    inputs = batch._read_jsonl(run / 'input_manifest.jsonl')
    predictions = batch._audit(inputs, batch._read_jsonl(run / 'style_predictions.jsonl'))
    if len(rows) != len(inputs) or len(predictions) != len(inputs):
        raise ValueError('Incomplete input/library/prediction records')
    if batch._fingerprint(inputs) != state['input_sha256']:
        raise ValueError('Frozen input fingerprint mismatch')
    expected = {x['source_id']: x for x in inputs}
    originals = {}
    paths = {}
    for record in rows:
        sid = record['source_id']
        if sid not in expected or sid in originals:
            raise ValueError(f'Duplicate/unknown candidate source_id: {sid}')
        original = expected[sid]
        prediction = predictions[sid]
        if any(record.get(k) != original[k] for k in ('source_path', 'source_sha256', 'relative_path')):
            raise ValueError(f'Candidate input provenance mismatch: {sid}')
        if record.get('category_dir') != prediction['category_dir'] or record.get('output_sha256') != original['source_sha256']:
            raise ValueError(f'Candidate category/output SHA mismatch: {sid}')
        paths[sid] = _safe_audio_file(root, record)
        originals[sid] = record
    decisions, csv_sha = _read_csv(csv_path, originals)
    return state, root, rows, paths, decisions, csv_sha


def _outcome(record: dict[str, Any], review: dict[str, str] | None):
    cat = review['manual_category'] if review and review['manual_category'] else record['category_dir']
    if review is None:
        return cat, 'unreviewed'
    if review['decision'] == 'reject':
        return REJECTED, 'rejected'
    if review['decision'] == 'hold':
        return HELD, 'held'
    if review['decision'] == 'keep' and review['identity'] == 'self' and review['quality'] == 'ok':
        return cat, 'reviewed_usable'
    return CAUTION, 'needs_attention'


def _write_metadata(path: Path, text: str, *, resume: bool, encoding: str = 'utf-8'):
    if path.is_symlink() or (path.exists() and (not resume or not path.is_file())):
        raise ValueError(f'Unexpected stage metadata path: {path}')
    mode = 'w' if path.exists() else 'x'
    with path.open(mode, encoding=encoding, newline='') as file:
        file.write(text)


def _safe_csv(v: Any) -> str:
    value = str(v if v is not None else '')
    if value.lstrip().startswith(('=', '+', '-', '@')):
        value = "'" + value
    return value


def _csv_text(rows: list[dict[str, Any]], columns: list[str]) -> str:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows([{k: _safe_csv(x.get(k)) for k in columns} for x in rows])
    return stream.getvalue()


def apply_review(run_id: str, csv_path: Path, source_version: str = 'v1',
                 version: str = 'v2', dry_run: bool = False, resume_export: bool = False) -> dict[str, Any]:
    if not batch.SAFE_NAME.fullmatch(version) or source_version == version:
        raise ValueError('Use a safe new version different from the source')
    state, _source_root, records, sources, reviews, csv_sha = _preflight(run_id, source_version, csv_path)
    profile = state['profile']
    parent = batch.PROJECT_ROOT / 'outputs' / 'candidates' / profile
    final = parent / version
    stage = parent / f'.building-{version}-review-{run_id}'
    if final.exists() or final.is_symlink():
        raise FileExistsError(f'Immutable reviewed candidate version already exists: {final}')
    count = Counter()
    categories = Counter()
    mapped = []
    for record in records:
        review = reviews.get(record['source_id'])
        folder, status = _outcome(record, review)
        count[status] += 1
        categories[folder] += 1
        mapped.append((record, review, folder, status))
    report = {'source_count': len(records), 'review_records': len(reviews),
              'reviewed_usable': count['reviewed_usable'],
              'needs_attention': count['needs_attention'], 'held': count['held'],
              'rejected': count['rejected'], 'unreviewed': count['unreviewed'],
              'categories': dict(sorted(categories.items())),
              'review_csv_sha256': csv_sha}
    if dry_run:
        return report
    if resume_export:
        if not stage.is_dir() or stage.is_symlink():
            raise ValueError('No matching incomplete staging export to resume')
        stamp = json.loads((stage / 'review_source.json').read_text(encoding='utf-8'))
        if (stamp.get('run_id'), stamp.get('source_version'), stamp.get('review_csv_sha256')) != (
            run_id, source_version, csv_sha
        ):
            raise ValueError('Staged review evidence mismatch; resume blocked')
    else:
        if stage.exists() or stage.is_symlink():
            raise FileExistsError(f'Incomplete stage exists; use --resume-export: {stage}')
        if parent.is_symlink():
            raise ValueError('Linked output parent')
        required = sum(sources[r['source_id']].stat().st_size for r in records)
        if shutil.disk_usage(parent).free < required + 32 * 1024 * 1024:
            raise OSError('Insufficient free space for independent v2 WAV copies')
        stage.mkdir(exist_ok=False)
        _write_metadata(stage / 'review_source.json', json.dumps({
            'run_id': run_id, 'source_version': source_version,
            'review_csv_sha256': csv_sha, 'source_input_sha256': state['input_sha256'],
        }, ensure_ascii=False, indent=2) + '\n', resume=False)
    out_manifest = []
    out_index = []
    for record, review, folder, review_status in mapped:
        sid = record['source_id']
        new_rel = PurePosixPath('audio') / folder / PurePosixPath(record['output_path']).name
        destination = stage.joinpath(*new_rel.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink() or destination.parent.parent.is_symlink() or destination.is_symlink():
            raise ValueError(f'Linked staged destination: {destination}')
        if destination.exists():
            if not resume_export or not destination.is_file() or batch._sha256(destination) != record['source_sha256']:
                raise ValueError(f'Invalid existing staged copy: {destination}')
        else:
            temp = destination.with_name(destination.name + '.copying')
            if temp.exists() or temp.is_symlink():
                if not resume_export or temp.is_symlink() or not temp.is_file():
                    raise ValueError(f'Unexpected incomplete copy: {temp}')
                temp.unlink()
            with sources[sid].open('rb') as src, temp.open('xb') as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            if batch._sha256(temp) != record['source_sha256']:
                raise ValueError(f'Copied WAV SHA mismatch: {sid}')
            temp.rename(destination)
        row = {**record,
               'ast_category': record['category_dir'],
               'effective_category': review['manual_category'] or record['category_dir'] if review else record['category_dir'],
               'category_dir': folder, 'output_path': new_rel.as_posix(),
               'review_status': review_status, 'review_decision': review['decision'] if review else '',
               'review_identity': review['identity'] if review else 'unverified',
               'review_quality': review['quality'] if review else 'unverified',
               'manual_category': review['manual_category'] if review else '',
               'review_reason': review['reject_reason'] if review else '',
               'review_memo': review['memo'] if review else '',
               'review_updated_at': review['updated_at'] if review else '',
               'eligible_for_later_promotion': review_status == 'reviewed_usable',
               'human_approved': False, 'library_tier': 'UNVERIFIED_REVIEWED_CANDIDATE'}
        out_manifest.append(row)
        out_index.append({k: row.get(k, '') for k in (
            'source_id', 'category_dir', 'effective_category', 'ast_category',
            'output_path', 'relative_path', 'source_sha256', 'review_status',
            'review_decision', 'review_identity', 'review_quality', 'manual_category',
            'review_reason', 'review_memo', 'eligible_for_later_promotion')})
    _write_metadata(stage / 'manifest.jsonl', ''.join(batch._json(x)+'\n' for x in out_manifest), resume=resume_export)
    columns = list(out_index[0])
    _write_metadata(stage / 'index.csv', _csv_text(out_index, columns), resume=resume_export, encoding='utf-8-sig')
    _write_metadata(stage / 'reviewed_usable.csv', _csv_text([r for r in out_index if r['review_status'] == 'reviewed_usable'], columns), resume=resume_export, encoding='utf-8-sig')
    _write_metadata(stage / 'needs_attention.csv', _csv_text([r for r in out_index if r['review_status'] in ('needs_attention','held','rejected')], columns), resume=resume_export, encoding='utf-8-sig')
    _write_metadata(stage / 'summary.json', json.dumps({**report,
        'schema_version': SCHEMA, 'tier': 'UNVERIFIED_REVIEWED_CANDIDATE',
        'profile': profile, 'run_id': run_id,
        'source_version': source_version, 'version': version,
        'input_sha256': state['input_sha256'],
    }, ensure_ascii=False, indent=2)+'\n', resume=resume_export)
    _write_metadata(stage / 'README.txt',
        '手動レビューCSV反映済み・暫定候補ライブラリ（未承認）\n'
        '================================================\n'
        f'profile={profile} / AST run={run_id} / source={source_version} / version={version}\n'
        f'レビューCSV SHA256={csv_sha}\n'
        f'総数={len(records)}、レビュー適合={count["reviewed_usable"]}、要確認={count["needs_attention"]}、保留={count["held"]}、除外={count["rejected"]}、未レビュー={count["unreviewed"]}\n'
        'レビュー適合は候補の本人性self・音質ok・keepであり、正式な学習素材承認ではありません。\n'
        '96_要確認_使用不可 / 97_保留 / 98_除外 は使用を避けてください。\n'
        '残る未レビュー音声はAST候補分類のまま。元v1・data・outputs/datasetsは不変。\n'
        '詳細はindex.csv、manifest.jsonl、reviewed_usable.csv、needs_attention.csv。\n', resume=resume_export)
    stage.rename(final)
    return report
