import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import Finding
from .retrieval import HybridRetriever
from .specialists import LiabilityAgent,PaymentAgent,TerminationAgent,PrivacyAgent,IPAgent,ComplianceAgent
from .graph import build_review_graph
class MasterOrchestratorAgent:
    agents={"liability":LiabilityAgent,"payment":PaymentAgent,"termination":TerminationAgent,"privacy":PrivacyAgent,"ip":IPAgent,"compliance":ComplianceAgent}
    def __init__(self,chunks,audit,vectors=None,embed_query=None,root=None,gemini=None): self.chunks=chunks; self.audit=audit; self.gemini=gemini; self.retriever=HybridRetriever(chunks,vectors,root,embed_query)
    def plan(self,q):
        ql=q.lower(); selected=[n for n,cls in self.agents.items() if any(t in ql for t in cls.terms)] or list(self.agents)
        self.audit({"agent":"MasterOrchestrator","action":"plan","question":q,"required_agents":selected}); return selected
    def run(self,question):
        def plan(s): return {"plan":self.plan(s["question"]),"retries":0}
        def specialists(s):
            fs=[]
            self.audit({"agent":"Specialist Squad","action":"parallel_start","agents":s["plan"]})
            def run_one(name):
                self.audit({"agent":self.agents[name].__name__,"action":"started","status":"working"})
                result=self.agents[name](self.retriever,self.audit,self.gemini).run(s["question"])
                self.audit({"agent":self.agents[name].__name__,"action":"completed","status":"complete","findings":len(result)})
                return result
            with ThreadPoolExecutor(max_workers=min(6,len(s["plan"]))) as pool:
                futures=[pool.submit(run_one,name) for name in s["plan"]]
                for future in as_completed(futures): fs.extend(future.result())
            return {"findings":fs}
        def cross(s): return {"findings":s.get("findings",[])+self.cross_clause(s.get("findings",[]))}
        def verify(s):
            valid=[f for f in s.get("findings",[]) if f.evidence]
            for f in valid: f.verification_status="verified"
            return {"findings":valid,"verified":True,"retries":s.get("retries",0)+1}
        def judge(s):
            self.audit({"agent":"JudgeAgent","action":"judge","verified_findings":len(s.get("findings",[]))}); return {"findings":s.get("findings",[])}
        result=build_review_graph(plan,specialists,cross,verify,judge).invoke({"question":question})
        return result.get("findings",[])
    def cross_clause(self,findings):
        p=[f for f in findings if f.category=="PAYMENT"]
        if len(p)>1: return [Finding(finding_id=f"R-{uuid.uuid4().hex[:8].upper()}",category="CROSS_CLAUSE",severity="HIGH",claim="Multiple payment-related clauses may create an inconsistent obligation.",evidence=p[0].evidence+p[1].evidence,reasoning="Related terms should be reconciled across the agreement.",recommendation="Confirm precedence, timing, and withholding language.",confidence=.66,requires_human_review=True)]
        return []
