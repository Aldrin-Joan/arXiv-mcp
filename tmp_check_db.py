from src.workflows.db import DatabaseClient
from pathlib import Path
p = Path('tmp_phase6_test.db')
if p.exists():
    p.unlink()

db = DatabaseClient(str(p))
print('db created?', p.exists())
now = '2026-03-22T00:00:00Z'
db.execute('INSERT INTO reading_list(arxiv_id,title,authors,year,abstract,tags,notes,read_status,added_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)', ('id1', 't', '[]', 2026, 'a', '[]', '', 'unread', now, now))
row = db.fetchone('SELECT arxiv_id FROM reading_list WHERE arxiv_id=?', ('id1',))
print('row', row)
db.close()

db2 = DatabaseClient(str(p))
row2 = db2.fetchone('SELECT arxiv_id FROM reading_list WHERE arxiv_id=?', ('id1',))
print('row2', row2)
db2.close()
