import React, {useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import {ArrowUpRight, ChevronRight, FileText, Info, Plus, Send, ShieldCheck, Sparkles, Upload, X} from 'lucide-react';
import './style.css';
import './progress.css';

const API = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api`;
const REVIEW_QUESTION = 'Review financial, liability, privacy, intellectual property, compliance, and termination risks.';

type Contract = {
  contract_id: string; original_filename: string; processing_status: string;
  page_count: number; chunk_count: number; indexing_stage: string; indexing_progress: number;
  analysis_status: string; analysis_progress: number; analysis_stage: string;
};
type Citation = {page: number; clause?: string; quote: string; chunk_id: string};
type Message = {role: string; content: string; citations?: Citation[]; model?: string};
type Finding = {finding_id: string; category: string; severity: string; claim: string; recommendation: string; proposed_language?: string};
type Review = {findings: Finding[]; summary?: {score?: number | null; rationale?: string}};
type Model = {id: string; name: string; provider: string};

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [active, setActive] = useState<Contract>();
  const [review, setReview] = useState<Review>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [analysisModel, setAnalysisModel] = useState('');
  const [chatModel, setChatModel] = useState('');
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [party, setParty] = useState('');
  const [perspective, setPerspective] = useState('');
  const [contractType, setContractType] = useState('');
  const [jurisdiction, setJurisdiction] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const cancelPoll = useRef<() => void>(() => {});
  const workspaceId = useMemo(() => {
    const key = 'ps9-workspace-id';
    let id = sessionStorage.getItem(key);
    if (!id) { id = crypto.randomUUID(); sessionStorage.setItem(key, id); }
    return id;
  }, []);
  const readyIds = useMemo(() => new Set(contracts.filter(c => c.processing_status === 'ready').map(c => c.contract_id)), [contracts]);
  const ready = active?.processing_status === 'ready';
  const fetchApi = (input: RequestInfo | URL, init: RequestInit = {}) =>
    fetch(input, {...init, headers: {...(init.headers || {}), 'X-Workspace-ID': workspaceId}});

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetchApi(`${API}/contracts`).then(r => r.json()),
      fetchApi(`${API}/models`).then(r => r.json()),
    ]).then(([loadedContracts, available]) => {
      if (cancelled) return;
      setContracts(loadedContracts);
      setSelectedIds(loadedContracts.filter((c: Contract) => c.processing_status === 'ready').map((c: Contract) => c.contract_id));
      setModels(available.models || []);
      setAnalysisModel(available.default || '');
      setChatModel(available.default || '');
    }).catch(error => console.error('Could not load workspace', error));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!active || ready || active.processing_status === 'failed') return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      const response = await fetchApi(`${API}/contracts/${active.contract_id}`);
      if (!cancelled && response.ok) {
        const next: Contract = await response.json();
        setActive(next);
        setContracts(items => items.map(item => item.contract_id === next.contract_id ? next : item));
        if (next.processing_status === 'ready') setSelectedIds(ids => ids.includes(next.contract_id) ? ids : [...ids, next.contract_id]);
      }
    }, 700);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active?.contract_id, active?.processing_status]);

  useEffect(() => () => cancelPoll.current(), [active?.contract_id]);

  const load = async (contract: Contract) => {
    setAnalysisOpen(false);
    const [contractResponse, reviewResponse, messagesResponse] = await Promise.all([
      fetchApi(`${API}/contracts/${contract.contract_id}`),
      fetchApi(`${API}/contracts/${contract.contract_id}/review`),
      fetchApi(`${API}/workspace/messages`),
    ]);
    const next = contractResponse.ok ? await contractResponse.json() : contract;
    setActive(next);
    setContracts(items => items.map(item => item.contract_id === next.contract_id ? next : item));
    setReview(reviewResponse.ok ? await reviewResponse.json() : {findings: []});
    setMessages(messagesResponse.ok ? await messagesResponse.json() : []);
  };

  const upload = async () => {
    if (!files.length) return;
    setBusy(true);
    try {
      for (const file of files) {
        const form = new FormData();
        form.append('file', file);
        const response = await fetchApi(`${API}/contracts`, {method: 'POST', body: form});
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || `Upload failed (${response.status})`);
        }
        const contract: Contract = await response.json();
        setContracts(items => [contract, ...items.filter(item => item.contract_id !== contract.contract_id)]);
        setActive(contract);
        setReview({findings: []});
      }
      setFiles([]);
    } catch (error) {
      console.error('Upload failed', error);
      window.alert(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setBusy(false);
    }
  };

  const ask = async (event?: React.FormEvent) => {
    event?.preventDefault();
    if (!ready || !question.trim() || !selectedIds.some(id => readyIds.has(id)) || chatBusy) return;
    const text = question.trim();
    setQuestion('');
    setChatBusy(true);
    setMessages(items => [...items, {role: 'user', content: text}]);
    try {
      const response = await fetchApi(`${API}/workspace/chat`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: text, contract_ids: selectedIds.filter(id => readyIds.has(id)), model: chatModel || null, party, perspective, contract_type: contractType, jurisdiction}),
      });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Chat request failed (${response.status})`);
      const reply: Message = await response.json();
      setMessages(items => [...items, reply]);
    } catch (error) {
      setMessages(items => [...items, {role: 'assistant', content: `Model API error: ${error instanceof Error ? error.message : String(error)}`}]);
    } finally { setChatBusy(false); }
  };

  const runReview = async () => {
    if (!active || !ready || busy || !analysisModel) return;
    setBusy(true);
    try {
      const response = await fetchApi(`${API}/contracts/${active.contract_id}/review`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: REVIEW_QUESTION, model: analysisModel, party, perspective, contract_type: contractType, jurisdiction}),
      });
      if (!response.ok) throw new Error(`Review request failed (${response.status})`);
      cancelPoll.current();
      let cancelled = false;
      cancelPoll.current = () => { cancelled = true; };
      const poll = async () => {
        if (cancelled) return;
        const statusResponse = await fetchApi(`${API}/contracts/${active.contract_id}`);
        if (statusResponse.ok) {
          const next: Contract = await statusResponse.json();
          setActive(next);
          setContracts(items => items.map(item => item.contract_id === next.contract_id ? next : item));
          if (next.analysis_status === 'complete' || next.analysis_status === 'failed') {
            const report = await fetchApi(`${API}/contracts/${active.contract_id}/review`);
            setReview(await report.json());
            setBusy(false);
            return;
          }
        }
        window.setTimeout(poll, 700);
      };
      poll();
    } catch (error) {
      setBusy(false);
      console.error('Review request failed', error);
      window.alert(error instanceof Error ? error.message : 'Review request failed');
    }
  };

  const findings = review?.findings || [];
  const evidence = [...messages].reverse().find(message => message.role === 'assistant' && message.citations?.length)?.citations || [];
  const counts = {
    high: findings.filter(f => ['HIGH', 'CRITICAL'].includes(f.severity)).length,
    medium: findings.filter(f => f.severity === 'MEDIUM').length,
    low: findings.filter(f => f.severity === 'LOW').length,
  };
  return <div className="app"><Header onNew={() => inputRef.current?.click()}/><div className="shell">
    <ContractSidebar contracts={contracts} active={active} selectedIds={selectedIds} onToggle={id => setSelectedIds(ids => ids.includes(id) ? ids.filter(item => item !== id) : [...ids, id])} files={files} busy={busy} inputRef={inputRef} onFiles={setFiles} onPick={() => inputRef.current?.click()} onUpload={upload} onLoad={load}/>
    <main>{!active ? <Welcome onPick={() => inputRef.current?.click()}/> : <div className="chat-layout">
      <ChatColumn active={active} ready={ready} messages={messages} evidence={evidence} question={question} setQuestion={setQuestion} ask={ask} chatBusy={chatBusy} analysisOpen={analysisOpen} setAnalysisOpen={setAnalysisOpen} party={party} setParty={setParty} perspective={perspective} setPerspective={setPerspective} contractType={contractType} setContractType={setContractType} jurisdiction={jurisdiction} setJurisdiction={setJurisdiction} model={chatModel} setModel={setChatModel} models={models}/>
      <ReportRail findings={findings} counts={counts} score={review?.summary?.score} scoreRationale={review?.summary?.rationale} busy={busy} onRun={runReview} analysisModel={analysisModel} setAnalysisModel={setAnalysisModel} models={models} analysisStatus={active.analysis_status} analysisProgress={active.analysis_progress} analysisStage={active.analysis_stage}/>
    </div>}</main>
  </div></div>;
}

function Header({onNew}: {onNew: () => void}) { return <header><div className="brand"><ShieldCheck size={21}/><span>PS-9</span></div><div className="product">Contract intelligence <span className="status"><i/> Local workspace</span></div><button className="new" onClick={onNew}><Plus size={16}/> New contract</button></header>; }
function Welcome({onPick}: {onPick: () => void}) { return <div className="welcome"><div className="welcome-mark"><ShieldCheck size={30}/></div><h1>Review contracts<br/><em>with confidence.</em></h1><p>Upload a tender, agreement, or policy and get a grounded first pass with page-level evidence.</p><button className="primary" onClick={onPick}><Upload size={17}/> Upload contracts</button></div>; }

function ContractSidebar({contracts, active, selectedIds, onToggle, files, busy, inputRef, onFiles, onPick, onUpload, onLoad}: {contracts: Contract[]; active?: Contract; selectedIds: string[]; onToggle: (id: string) => void; files: File[]; busy: boolean; inputRef: React.RefObject<HTMLInputElement | null>; onFiles: (files: File[]) => void; onPick: () => void; onUpload: () => void; onLoad: (contract: Contract) => void}) {
  return <aside className="contracts"><div className="side-title">Workspace <span>{contracts.length}</span></div><label className="drop-mini" onClick={onPick}><Upload size={18}/><strong>Upload contracts</strong><small>PDF files</small></label><input ref={inputRef} hidden type="file" multiple accept="application/pdf" onChange={event => onFiles(Array.from(event.target.files || []))}/>{files.length > 0 && <div className="queued"><div>{files.length} file{files.length > 1 ? 's' : ''} ready</div><button onClick={onUpload} disabled={busy}>{busy ? 'Indexing...' : 'Start indexing'}</button></div>}<div className="side-title recent">Your contracts</div>{contracts.map(contract => <div className="contract-row" key={contract.contract_id}><input type="checkbox" disabled={contract.processing_status !== 'ready'} checked={selectedIds.includes(contract.contract_id)} onChange={() => onToggle(contract.contract_id)} aria-label={`Use ${contract.original_filename} in chat`}/><button className={'contract ' + (active?.contract_id === contract.contract_id ? 'selected' : '')} onClick={() => onLoad(contract)}><FileText size={17}/><span><b>{contract.original_filename}</b><small>{contract.processing_status === 'ready' ? `${contract.page_count} pages` : `${contract.indexing_progress || 0}% · ${contract.indexing_stage}`}</small></span><ChevronRight size={15}/></button></div>)}</aside>;
}

function ChatColumn(props: {active: Contract; ready: boolean; messages: Message[]; evidence: Citation[]; question: string; setQuestion: (value: string) => void; ask: (event?: React.FormEvent) => void; chatBusy: boolean; analysisOpen: boolean; setAnalysisOpen: (value: boolean) => void; party: string; setParty: (value: string) => void; perspective: string; setPerspective: (value: string) => void; contractType: string; setContractType: (value: string) => void; jurisdiction: string; setJurisdiction: (value: string) => void; model: string; setModel: (value: string) => void; models: Model[]}) {
  return <section className="chat-column"><div className="chat-heading"><div><span className="eyebrow">CONTRACT CHAT</span><h1>{props.active.original_filename}</h1><p>{props.ready ? `${props.active.page_count} pages · ${props.active.chunk_count} indexed evidence chunks` : `${props.active.indexing_stage} · ${props.active.indexing_progress}% complete`}</p></div><div className="chat-actions"><button className="analysis-button" disabled={!props.ready} onClick={() => props.setAnalysisOpen(!props.analysisOpen)}><Info size={15}/> Analysis {props.analysisOpen && <X size={14}/>}</button><a className="document-link" href={`${API}/contracts/${props.active.contract_id}/document`} target="_blank"><FileText size={15}/> Open PDF <ArrowUpRight size={13}/></a></div></div>{props.analysisOpen && <EvidencePanel evidence={props.evidence}/>}<div className="messages">{!props.ready ? <div className="chat-empty"><h2>Preparing your contract</h2><p>{props.active.indexing_stage} · {props.active.indexing_progress}% complete.</p></div> : props.messages.length === 0 ? <EmptyChat onSuggest={props.setQuestion}/> : props.messages.map((message, index) => <div className={'message ' + message.role} key={index}><div className="avatar">{message.role === 'user' ? 'You' : 'PS'}</div><div className="message-body"><ReactMarkdown components={{a: ({href, children}) => <a href={href} target="_blank" rel="noreferrer">{children}</a>}}>{message.content}</ReactMarkdown>{message.citations?.length ? <button className="source-link" onClick={() => props.setAnalysisOpen(true)}><Info size={12}/> {message.citations.length} sources retrieved</button> : null}</div></div>)}{props.chatBusy && <div className="typing">Reviewing the contract evidence...</div>}</div><form className="composer" onSubmit={props.ask}><div className="composer-top"><input disabled={!props.ready} value={props.question} onChange={event => props.setQuestion(event.target.value)} placeholder="Message PS-9 about this contract..."/><button disabled={!props.ready} aria-label="Send message"><Send size={17}/></button></div><div className="composer-context"><input value={props.party} onChange={event => props.setParty(event.target.value)} placeholder="Your party"/><input value={props.perspective} onChange={event => props.setPerspective(event.target.value)} placeholder="Perspective"/><input value={props.contractType} onChange={event => props.setContractType(event.target.value)} placeholder="Contract type"/><input value={props.jurisdiction} onChange={event => props.setJurisdiction(event.target.value)} placeholder="Jurisdiction"/></div><div className="composer-meta"><span>{props.ready ? 'Ready contracts only' : 'Indexing in background'}</span><label>Model <select value={props.model} onChange={event => props.setModel(event.target.value)}><option value="">Local fallback</option>{props.models.map(model => <option key={model.id} value={model.id}>{model.name} · {model.provider}</option>)}</select></label></div></form></section>;
}

function EvidencePanel({evidence}: {evidence: Citation[]}) { return <div className="analysis-popover"><div className="analysis-title"><span><Sparkles size={15}/> Retrieved evidence</span><b>Top {Math.min(5, evidence.length)}</b></div>{evidence.slice(0, 5).map((item, index) => <div className="evidence-row" key={item.chunk_id}><span>{index + 1}</span><div><a href={`${API}/contracts/${item.chunk_id.split('_C')[0]}/document#page=${item.page}`} target="_blank" rel="noreferrer"><b>Page {item.page}{item.clause ? ` · Clause ${item.clause}` : ''}</b></a><p>{item.quote}</p></div></div>)}</div>; }
function EmptyChat({onSuggest}: {onSuggest: (value: string) => void}) { return <div className="chat-empty"><div className="empty-icon"><Sparkles size={20}/></div><h2>What would you like to know?</h2><div className="suggestions"><button onClick={() => onSuggest('What should I negotiate?')}>What should I negotiate?</button><button onClick={() => onSuggest('Summarize payment obligations')}>Summarize payment terms</button></div></div>; }
function ReportRail({findings, counts, score, scoreRationale, busy, onRun, analysisModel, setAnalysisModel, models, analysisStatus, analysisProgress, analysisStage}: {findings: Finding[]; counts: {high: number; medium: number; low: number}; score?: number | null; scoreRationale?: string; busy: boolean; onRun: () => void; analysisModel: string; setAnalysisModel: (value: string) => void; models: Model[]; analysisStatus: string; analysisProgress: number; analysisStage: string}) { return <aside className="report-rail"><div className="report-head"><div><span className="eyebrow">RISK REPORT</span><h2>Contract health</h2></div><span className="report-score" title={scoreRationale || ''}>{score == null ? '--' : score}<small>/100</small></span></div><label className="analysis-model">Analysis model <select value={analysisModel} onChange={event => setAnalysisModel(event.target.value)}><option value="">Select model</option>{models.map(model => <option key={model.id} value={model.id}>{model.name} · {model.provider}</option>)}</select></label>{(busy || analysisStatus === 'running' || analysisStatus === 'queued') && <div className="analysis-progress"><strong>{analysisStage || 'Starting review'}</strong><span>{analysisProgress || 0}%</span></div>}<div className="risk-strip"><div><strong className="high-text">{counts.high}</strong><small>High risk</small></div><div><strong className="medium-text">{counts.medium}</strong><small>Review</small></div><div><strong className="low-text">{counts.low}</strong><small>Low</small></div></div><div className="report-divider"/>{findings.slice(0, 5).map(finding => <article className="risk-item" key={finding.finding_id}><div className="risk-item-head"><span className={'risk-dot ' + finding.severity.toLowerCase()}/><span>{finding.category}</span><b>{finding.severity}</b></div><h3>{finding.claim}</h3><p>{finding.recommendation}</p></article>)}{!findings.length && <div className="report-empty"><ShieldCheck size={22}/><p>{score == null ? 'Run analysis when indexing is complete to calculate health.' : 'No clause-level risks were found.'}</p></div>}<button className="run-review" onClick={onRun} disabled={busy || !analysisModel}>{busy ? 'Analyzing...' : 'Run analysis'} <Sparkles size={14}/></button></aside>; }

createRoot(document.getElementById('root')!).render(<App/>);
