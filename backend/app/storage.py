import hashlib, json, sqlite3
from datetime import datetime, timezone
from .config import settings
from .models import Chunk, ContractMeta, Finding, ChatMessage, Evidence
class LocalStore:
    def __init__(self):
        self.root=settings.data_dir; self.db=self.root/"metadata.db"
        for n in ("contracts","extracted","pages","chunks","embeddings","indexes","reports","audit","cache"): (self.root/n).mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.db) as c:
            c.execute("create table if not exists contracts (id text primary key, data text not null)")
            c.execute("create table if not exists findings (id text primary key, contract_id text not null, data text not null, decision text)")
            c.execute("create table if not exists messages (id integer primary key autoincrement, contract_id text not null, role text not null, content text not null, created_at text not null, citations text not null)")
            c.execute("create table if not exists workspace_messages (id integer primary key autoincrement, workspace_id text not null, role text not null, content text not null, created_at text not null, citations text not null)")
    def contract_id(self, content, workspace_id="default"):
        digest=hashlib.sha256(content).hexdigest(); scope=hashlib.sha256(workspace_id.encode()).hexdigest()[:8]; return f"contract_{scope}_{digest[:12]}",digest
    def save_contract(self, meta, content):
        (self.root/"contracts"/f"{meta.contract_id}.pdf").write_bytes(content)
        with sqlite3.connect(self.db) as c: c.execute("insert or replace into contracts values (?,?)",(meta.contract_id,meta.model_dump_json()))
    def get_meta(self,cid):
        with sqlite3.connect(self.db) as c: row=c.execute("select data from contracts where id=?",(cid,)).fetchone()
        if not row: raise KeyError(cid)
        return ContractMeta.model_validate_json(row[0])
    def all_meta(self,workspace_id="default"):
        with sqlite3.connect(self.db) as c: rows=c.execute("select data from contracts order by rowid desc").fetchall()
        if workspace_id: rows=[row for row in rows if ContractMeta.model_validate_json(row[0]).workspace_id==workspace_id]
        return [ContractMeta.model_validate_json(row[0]) for row in rows]
    def update_meta(self,meta):
        with sqlite3.connect(self.db) as c: c.execute("update contracts set data=? where id=?",(meta.model_dump_json(),meta.contract_id))
    def save_chunks(self,cid,chunks): (self.root/"chunks"/f"{cid}.json").write_text(json.dumps([x.model_dump() for x in chunks]),encoding="utf-8")
    def load_chunks(self,cid): return [Chunk.model_validate(x) for x in json.loads((self.root/"chunks"/f"{cid}.json").read_text(encoding="utf-8"))]
    def save_embeddings(self,cid,vectors):
        import numpy as np
        np.save(self.root/"embeddings"/f"{cid}.npy",np.asarray(vectors,dtype="float32"))
    def load_embeddings(self,cid):
        import numpy as np
        return np.load(self.root/"embeddings"/f"{cid}.npy")
    def save_findings(self,cid,findings):
        with sqlite3.connect(self.db) as c:
            for f in findings: c.execute("insert or replace into findings values (?,?,?,coalesce((select decision from findings where id=?),null))",(f.finding_id,cid,f.model_dump_json(),f.finding_id))
    def findings(self,cid):
        with sqlite3.connect(self.db) as c: rows=c.execute("select data,decision from findings where contract_id=?",(cid,)).fetchall()
        return [(Finding.model_validate_json(d),dec) for d,dec in rows]
    def decide(self,fid,decision):
        with sqlite3.connect(self.db) as c: c.execute("update findings set decision=? where id=?",(decision,fid))
    def save_message(self,cid,role,content,citations=None):
        citations = citations or []
        now=datetime.now(timezone.utc)
        with sqlite3.connect(self.db) as c:
            c.execute("insert into messages(contract_id,role,content,created_at,citations) values (?,?,?,?,?)",(cid,role,content,now.isoformat(),json.dumps([x.model_dump() if hasattr(x,'model_dump') else x for x in citations])))
        return ChatMessage(role=role,content=content,created_at=now,citations=[Evidence.model_validate(x) for x in citations])
    def messages(self,cid):
        with sqlite3.connect(self.db) as c: rows=c.execute("select role,content,created_at,citations from messages where contract_id=? order by id",(cid,)).fetchall()
        return [ChatMessage(role=r,content=content,created_at=datetime.fromisoformat(ts),citations=[Evidence.model_validate(x) for x in json.loads(cits)]) for r,content,ts,cits in rows]
    def save_workspace_message(self,role,content,citations=None,workspace_id="default"):
        citations = citations or []
        now=datetime.now(timezone.utc)
        with sqlite3.connect(self.db) as c:
            c.execute("insert into workspace_messages(workspace_id,role,content,created_at,citations) values (?,?,?,?,?)",(workspace_id,role,content,now.isoformat(),json.dumps([x.model_dump() if hasattr(x,'model_dump') else x for x in citations])))
        return ChatMessage(role=role,content=content,created_at=now,citations=[Evidence.model_validate(x) for x in citations])
    def workspace_messages(self,workspace_id="default"):
        with sqlite3.connect(self.db) as c: rows=c.execute("select role,content,created_at,citations from workspace_messages where workspace_id=? order by id",(workspace_id,)).fetchall()
        return [ChatMessage(role=r,content=content,created_at=datetime.fromisoformat(ts),citations=[Evidence.model_validate(x) for x in json.loads(cits)]) for r,content,ts,cits in rows]
    def audit(self,event):
        event={**event,"timestamp":datetime.now(timezone.utc).isoformat()}
        with (self.root/"audit"/"events.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(event)+"\n")
store=LocalStore()
