"""Blind annotation state and validated atomic canonical JSON persistence."""
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from uuid import uuid4

REASONS = {
    'defect': 'Брак / поломка',
    'refund': 'Возврат / деньги',
    'delivery': 'Не то / некомплект (ошибка продавца)',
    'question': 'Вопрос к продавцу',
    'safety': 'Здоровье / безопасность',
    'misinfo': 'Ложная информация о товаре',
    'strong_negative': 'Сильный негатив с конкретикой',
    'wb_logistics': 'Доставка Wildberries',
}
_SAVE_LOCK = RLock()


@contextmanager
def _process_lock(path):
    """Serialize read/modify/replace across local UI processes, including Windows."""
    folder = path.parent / '.annotation-backups'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (path.name + '.lock')).open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if not lock.tell():
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError('Review id must be a nonempty string or integer')
    return str(value)


def _validate_label(record):
    if not isinstance(record, dict):
        raise ValueError('Each label must be an object')
    record = dict(record)
    record['id'] = _id(record.get('id'))
    if type(record.get('need')) is not bool:
        raise ValueError('Выберите: нужен человек или хватит шаблона')
    reasons = record.get('reasons')
    if (not isinstance(reasons, list) or any(not isinstance(r, str) or r not in REASONS for r in reasons)
            or len(reasons) != len(set(reasons))):
        raise ValueError('Reasons must be unique known reason keys')
    if type(record.get('unsure', False)) is not bool:
        raise ValueError('Unsure must be boolean')
    return record


def _read_labels(path):
    if not path.exists():
        return {'labels': []}
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('labels'), list):
        raise ValueError('Expected canonical JSON object with labels array')
    records = [_validate_label(record) for record in payload['labels']]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate label ids')
    return {**payload, 'labels': records}


def _write_durable(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_save(path, payload):
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = path.parent / '.annotation-backups' / f'{path.name}.{stamp}.{uuid4().hex[:8]}.bak'
            _write_durable(backup, path.read_bytes())
        os.replace(temporary, path)
        # Windows cannot fsync directories; the file itself is flushed above.
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class AnnotationSession:
    def __init__(self, items_path, labels_path):
        self.labels_path = Path(labels_path)
        payload = json.loads(Path(items_path).read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or not isinstance(payload.get('items'), list) or not payload['items']:
            raise ValueError('Expected a nonempty items array in batch JSON')
        self.items = []
        for item in payload['items']:
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                raise ValueError('Each item must have review text')
            # Model drafts and seller metadata never reach the UI state.
            self.items.append({'id': _id(item.get('id')), 'text': item['text']})
        if len({item['id'] for item in self.items}) != len(self.items):
            raise ValueError('Duplicate item ids')
        _read_labels(self.labels_path)

    def _done(self):
        return {r['id']: r for r in _read_labels(self.labels_path)['labels']}

    def first_open(self, start=0):
        done = self._done()
        start = start % len(self.items)
        for i in list(range(start, len(self.items))) + list(range(start)):
            if self.items[i]['id'] not in done:
                return i
        return start

    def show(self, index):
        item = self.items[index]
        done = self._done()
        saved = done.get(item['id'], {})
        return {**item, 'need': saved.get('need'), 'reasons': list(saved.get('reasons', [])),
                'unsure': saved.get('unsure', False),
                'completed': sum(item['id'] in done for item in self.items), 'total': len(self.items)}

    def save(self, index, need, reasons, unsure):
        record = _validate_label({'id': self.items[index]['id'], 'need': need,
                                  'reasons': reasons, 'unsure': unsure,
                                  'at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')})
        with _SAVE_LOCK, _process_lock(self.labels_path):
            payload = _read_labels(self.labels_path)
            for i, old in enumerate(payload['labels']):
                if old['id'] == record['id']:
                    payload['labels'][i] = {**old, **record}
                    break
            else:
                payload['labels'].append(record)
            _atomic_save(self.labels_path, payload)
