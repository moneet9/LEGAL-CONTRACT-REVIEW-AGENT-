import React, {useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import {Activity, AlertTriangle, ArrowUpRight, CheckCircle2, ChevronRight, FileText, Info, Maximize2, Plus, RotateCcw, Send, ShieldCheck, Sparkles, Upload, X, ZoomIn, ZoomOut} from 'lucide-react';
import './style.css';
import './progress.css';
import './agent-activity.css';

const API = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api`;
const REVIEW_QUESTION = 'Review financial, liability, privacy, intellectual property, compliance, and termination risks.';
let currentContractId = '';
const CURRENT_AGENTS = new Set(['Ingestion','DocumentConfigAgent','ChunkOptimizer','InitialScan','GraphBuilder','GraphValidator','GraphReviewerAgent','FixerAgent','TemplateReviewAgent','CrossChecker','ContractStructurer','SignatureVerifier','IntentAgent','DeadlineAgent','ObligationAgent','CoverageAgent','DocumentQualityAgent','SensitiveDataAgent','ContractScorer','RetrieverAgent','GraphRetriever','AnswerGenerator']);
const AGENT_DESCRIPTIONS: Record<string,string> = {
  Ingestion:'Extracts PDF text, applies finalized chunk settings, and builds local embeddings/indexes.', DocumentConfigAgent:'Profiles this document and chooses bounded indexing, retrieval, and review settings.', ChunkOptimizer:'Finalizes chunk size and overlap before indexing.', InitialScan:'Scans the complete contract for evidence-backed risk clauses.', GraphBuilder:'Extracts named contract entities and evidence-backed relations with the selected model, then verifies each quote against the source.', GraphValidator:'Checks graph integrity and removes invalid links.', GraphReviewerAgent:'Reviews graph links and reports disconnected or malformed relationships.', FixerAgent:'Removes unsupported findings and repairs invalid or duplicate graph links.', TemplateReviewAgent:'Compares Indian property deeds with the included checklists; reports gaps and red flags.', CrossChecker:'Checks findings against exact evidence and removes duplicates.', ContractStructurer:'Groups chunks into clauses, sections, pages, and coverage.', SignatureVerifier:'Checks signature, seal, witness, and authority signals.', IntentAgent:'Detects suspicious instruction-like or hidden-intent signals.', DeadlineAgent:'Extracts dates and checks deadline context.', ObligationAgent:'Extracts shall/must duties and responsible parties.', CoverageAgent:'Checks expected legal and commercial clause families.', DocumentQualityAgent:'Checks extraction quality and OCR-like noise.', SensitiveDataAgent:'Detects sensitive-data patterns locally.', ContractScorer:'Calculates the explainable contract health score.', RetrieverAgent:'Retrieves contract chunks to answer questions.', GraphRetriever:'Connects retrieved chunks to graph facts.', AnswerGenerator:'Produces grounded answers or abstains when retrieved evidence is insufficient.'
};

type Contract = {
  contract_id: string; original_filename: string; processing_status: string;
  page_count: number; chunk_count: number; indexing_stage: string; indexing_progress: number;
  analysis_status: string; analysis_progress: number; analysis_stage: string;
};
type Citation = {page: number; clause?: string; quote: string; chunk_id: string};
type RetrievalTrace = {query?: string; queries?: Record<string,string>; candidate_chunk_ids?: string[]; answer_chunk_ids?: string[]; graph_facts?: string};
type Message = {role: string; content: string; citations?: Citation[]; model?: string; retrieval_trace?: RetrievalTrace};
type Finding = {finding_id: string; category: string; severity: string; claim: string; recommendation: string; proposed_language?: string};
type ReviewResult = {findings: Finding[]; summary?: {score?: number | null; rationale?: string}};
type Review = ReviewResult & {perspective_reviews?: Record<string,ReviewResult>};
type Model = {id: string; name: string; provider: string};
type GraphProof = {chunk_id: string; page: number; quote: string};
type GraphNode = {id: string; label: string; kind: string; evidence?: GraphProof[]};
type GraphEdge = {source: string; target: string; relation: string; confidence?: number; term?: string; entity_type?: string; chunk_id?: string; page?: number; evidence?: string; evidence_sources?: GraphProof[]};
type ContractGraph = {graph?: {nodes: GraphNode[]; edges: GraphEdge[]; extraction?: {method?: string; model?: string; warning?: string; input_chars?: number}; document_name?: string; validation?: {valid: boolean; node_count: number; edge_count: number; duplicate_edges_removed?: number; invalid_edges_removed?: number}; legend?: Record<string, string>}; document_config?: {embedding_model?: string; embedding_dimension?: number; embedding_batch_size?: number; embedding_max_tokens?: number; recommended_chunk_size?: number; recommended_overlap?: number; top_k_bm25?: number; top_k_vector?: number; top_k_rerank?: number; review_context_chars?: number; max_retries?: number; complexity?: string}; structure?: {section_count?: number; chunk_count?: number}; signature?: {status?: string; signatory_candidates?: string[]; seal_present?: boolean; authority_concerns?: string[]; notes?: string}; document_quality?: {status?: string; warnings?: string[]}; coverage?: {present?: string[]; missing?: string[]}; deadlines?: {dates?: {value: string; page: number}[]; deadline_signals?: unknown[]}; obligations?: {obligations?: unknown[]; unassigned_count?: number}; graph_review?: {status?: string; relationship_count?: number; issues?: string[]; isolated_nodes?: string[]}; template_review?: {status?: string; detected_type?: string; detected_type_name?: string; detection_confidence?: string; candidate_types?: string[]; match_score?: number; verdict?: string; summary?: string; missing_clauses?: {title: string; severity: string; guidance?: string}[]; red_flags?: {title: string; why: string; evidence: string}[]; present_clauses?: {title: string; strength: string}[]; sources?: string[]}; fixer?: {findings_removed?: number; edges_removed?: number; nodes_removed?: number; status?: string}};
type AgentTelemetry = {master?: {status: string; accessible_log: boolean; supervises?: number}; agents: Record<string, {runs: number; failures: number; retries: number; last_status: string; contribution?: string; last_detail?: string}>; events: {agent: string; action: string; status?: string; attempt?: number; duration_ms?: number; error?: string; timestamp?: string; contract_id?: string; contribution?: string; detail?: string}[]};

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [active, setActive] = useState<Contract>();
  const [review, setReview] = useState<Review & {agents?: ContractGraph}>();
  const [contractGraph, setContractGraph] = useState<ContractGraph>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [analysisModel, setAnalysisModel] = useState('');
  const [chatModel, setChatModel] = useState('');
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const [reviewPerspectives, setReviewPerspectives] = useState<Record<string,string>>({});
  const [telemetry, setTelemetry] = useState<AgentTelemetry>({master: {status: 'waiting', accessible_log: true}, agents: {}, events: []});
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
  const reviewPerspective = active ? reviewPerspectives[active.contract_id] || 'contract_giver' : 'contract_giver';
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
    if (!active || active.processing_status === 'failed' || (ready && !['queued', 'running'].includes(active.analysis_status))) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      const response = await fetchApi(`${API}/contracts/${active.contract_id}`);
      if (!cancelled && response.ok) {
        const next: Contract = await response.json();
        setActive(next);
        setContracts(items => items.map(item => item.contract_id === next.contract_id ? next : item));
        if (next.processing_status === 'ready') setSelectedIds(ids => ids.includes(next.contract_id) ? ids : [...ids, next.contract_id]);
        if (next.analysis_status === 'complete' && active.analysis_status !== 'complete') {
          const report = await fetchApi(`${API}/contracts/${next.contract_id}/review`);
          if (!cancelled && report.ok) {
            const loaded = await report.json();
            setReview(loaded);
            setContractGraph(loaded.agents || undefined);
          }
        }
      }
    }, 700);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active?.contract_id, active?.processing_status, active?.analysis_status]);

  useEffect(() => () => cancelPoll.current(), [active?.contract_id]);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const loadTelemetry = async () => {
      const response = await fetchApi(`${API}/agents/telemetry?contract_id=${encodeURIComponent(active.contract_id)}`);
      if (!cancelled && response.ok) {
        const next: AgentTelemetry = await response.json();
        const agents=Object.fromEntries([...CURRENT_AGENTS].map(name => [name, {...((next.agents || {})[name] || {}), runs:(next.agents || {})[name]?.runs || 0, failures:(next.agents || {})[name]?.failures || 0, retries:(next.agents || {})[name]?.retries || 0, last_status:(next.agents || {})[name]?.last_status || 'waiting', contribution:(next.agents || {})[name]?.contribution || AGENT_DESCRIPTIONS[name], last_detail:(next.agents || {})[name]?.last_detail || 'Waiting for this agent to run.'}]));
        setTelemetry({
          ...next,
          agents,
          events: (next.events || []).filter(event => CURRENT_AGENTS.has(event.agent) && (!event.contract_id || event.contract_id === active.contract_id)),
        });
      }
    };
    loadTelemetry();
    const timer = window.setInterval(loadTelemetry, active.analysis_status === 'running' || active.analysis_status === 'queued' ? 900 : 3000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active?.contract_id, active?.analysis_status]);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    fetchApi(`${API}/contracts/${active.contract_id}/graph`).then(response => response.ok ? response.json() : null)
      .then(value => { if (!cancelled && value) setContractGraph(value); })
      .catch(() => { if (!cancelled) setContractGraph(undefined); });
    return () => { cancelled = true; };
  }, [active?.contract_id, active?.analysis_status]);

  const load = async (contract: Contract) => {
    setAnalysisOpen(false);
    setGraphOpen(false);
    const [contractResponse, reviewResponse, messagesResponse] = await Promise.all([
      fetchApi(`${API}/contracts/${contract.contract_id}`),
      fetchApi(`${API}/contracts/${contract.contract_id}/review`),
      fetchApi(`${API}/workspace/messages`),
    ]);
    const next = contractResponse.ok ? await contractResponse.json() : contract;
    setActive(next);
    setContracts(items => items.map(item => item.contract_id === next.contract_id ? next : item));
    const loadedReview = reviewResponse.ok ? await reviewResponse.json() : {findings: []};
    setReview(loadedReview);
    setContractGraph(loadedReview.agents || undefined);
    setMessages(messagesResponse.ok ? await messagesResponse.json() : []);
  };

  const upload = async () => {
    if (!files.length) return;
    setBusy(true);
    try {
      for (const file of files) {
        const form = new FormData();
        form.append('file', file);
        form.append('model', analysisModel || 'local');
        form.append('perspective', 'contract_giver');
        const response = await fetchApi(`${API}/contracts`, {method: 'POST', body: form});
        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          throw new Error(detail.detail || `Upload failed (${response.status})`);
        }
        const contract: Contract = await response.json();
        setContracts(items => [contract, ...items.filter(item => item.contract_id !== contract.contract_id)]);
        setActive(contract);
        setReview({findings: []});
        setContractGraph(undefined);
        setAnalysisOpen(true);
        setGraphOpen(false);
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
        body: JSON.stringify({message: text, contract_ids: selectedIds.filter(id => readyIds.has(id)), model: chatModel || null}),
      });
      if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || `Chat request failed (${response.status})`);
      const reply: Message = await response.json();
      setMessages(items => [...items, reply]);
    } catch (error) {
      setMessages(items => [...items, {role: 'assistant', content: `Model API error: ${error instanceof Error ? error.message : String(error)}`}]);
    } finally { setChatBusy(false); }
  };

  const runReview = async () => {
    if (!active || !ready || busy) return;
    setBusy(true);
    try {
      const response = await fetchApi(`${API}/contracts/${active.contract_id}/review`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: REVIEW_QUESTION, model: analysisModel || 'local', perspectives: ['contract_giver','contract_acceptor']}),
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
            const loadedReport = await report.json();
            setReview(loadedReport);
            setContractGraph(loadedReport.agents || undefined);
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

  const hasBothPerspectives = Boolean(review?.perspective_reviews?.contract_giver && review?.perspective_reviews?.contract_acceptor);
  const selectedReview = review?.perspective_reviews ? review.perspective_reviews[reviewPerspective] : reviewPerspective === 'contract_giver' ? review : undefined;
  const findings = selectedReview?.findings || [];
  const evidence = [...messages].reverse().find(message => message.role === 'assistant' && message.citations?.length)?.citations || [];
  const counts = {
    high: findings.filter(f => ['HIGH', 'CRITICAL'].includes(f.severity)).length,
    medium: findings.filter(f => f.severity === 'MEDIUM').length,
    low: findings.filter(f => f.severity === 'LOW').length,
  };
  return <div className="app"><Header onNew={() => inputRef.current?.click()}/><div className="shell">
    <ContractSidebar contracts={contracts} active={active} selectedIds={selectedIds} onToggle={id => setSelectedIds(ids => ids.includes(id) ? ids.filter(item => item !== id) : [...ids, id])} files={files} busy={busy} inputRef={inputRef} onFiles={setFiles} onPick={() => inputRef.current?.click()} onUpload={upload} onLoad={load} models={models} reviewModel={analysisModel} setReviewModel={setAnalysisModel}/>
    <main>{!active ? <Welcome onPick={() => inputRef.current?.click()}/> : <div className="chat-layout">
      <ChatColumn active={active} ready={ready} messages={messages} evidence={evidence} graph={contractGraph} telemetry={telemetry} question={question} setQuestion={setQuestion} ask={ask} chatBusy={chatBusy} analysisOpen={analysisOpen} setAnalysisOpen={setAnalysisOpen} graphOpen={graphOpen} setGraphOpen={setGraphOpen} model={chatModel} setModel={setChatModel} models={models}/>
      <ReportRail findings={findings} counts={counts} score={selectedReview?.summary?.score} scoreRationale={selectedReview?.summary?.rationale} graph={contractGraph} busy={busy} analysisStatus={active.analysis_status} analysisProgress={active.analysis_progress} analysisStage={active.analysis_stage} perspective={reviewPerspective} hasBothPerspectives={hasBothPerspectives} onPerspectiveChange={value => setReviewPerspectives(previous => ({...previous,[active.contract_id]:value}))} onOpenAnalysis={() => { setAnalysisOpen(true); setGraphOpen(false); }} onRetry={runReview}/>
    </div>}</main>
  </div></div>;
}

function Header({onNew}: {onNew: () => void}) { return <header><div className="brand"><ShieldCheck size={21}/><span>PS-9</span></div><div className="product">Contract intelligence <span className="status"><i/> Local workspace</span></div><button className="new" onClick={onNew}><Plus size={16}/> New contract</button></header>; }
function Welcome({onPick}: {onPick: () => void}) { return <div className="welcome"><div className="welcome-mark"><ShieldCheck size={30}/></div><h1>Review contracts<br/><em>with confidence.</em></h1><p>Upload a tender, agreement, or policy and get a grounded first pass with page-level evidence.</p><button className="primary" onClick={onPick}><Upload size={17}/> Upload contracts</button></div>; }

function ContractSidebar({contracts, active, selectedIds, onToggle, files, busy, inputRef, onFiles, onPick, onUpload, onLoad, models, reviewModel, setReviewModel}: {contracts: Contract[]; active?: Contract; selectedIds: string[]; onToggle: (id: string) => void; files: File[]; busy: boolean; inputRef: React.RefObject<HTMLInputElement | null>; onFiles: (files: File[]) => void; onPick: () => void; onUpload: () => void; onLoad: (contract: Contract) => void; models: Model[]; reviewModel: string; setReviewModel: (value: string) => void}) {
  return <aside className="contracts"><div className="side-title">Workspace <span>{contracts.length}</span></div><label className="drop-mini" onClick={onPick}><Upload size={18}/><strong>Upload contracts</strong><small>PDF files</small></label><input ref={inputRef} hidden type="file" multiple accept="application/pdf" onChange={event => onFiles(Array.from(event.target.files || []))}/>{files.length > 0 && <div className="queued"><div className="queued-count">{files.length} file{files.length > 1 ? 's' : ''} ready</div><label className="upload-model">Review model<select value={reviewModel} onChange={event => setReviewModel(event.target.value)} disabled={busy}><option value="">Local review · no model</option>{models.map(model => <option key={model.id} value={model.id}>{model.name} · {model.provider}</option>)}</select></label><small className="model-hint">This model will analyze the contract after upload.</small><button onClick={onUpload} disabled={busy}>{busy ? 'Uploading and reviewing…' : 'Upload & review'}</button></div>}<div className="side-title recent">Your contracts</div>{contracts.map(contract => <div className="contract-row" key={contract.contract_id}><input type="checkbox" disabled={contract.processing_status !== 'ready'} checked={selectedIds.includes(contract.contract_id)} onChange={() => onToggle(contract.contract_id)} aria-label={`Use ${contract.original_filename} in chat`}/><button className={'contract ' + (active?.contract_id === contract.contract_id ? 'selected' : '')} onClick={() => onLoad(contract)}><FileText size={17}/><span><b>{contract.original_filename}</b><small>{contract.processing_status === 'ready' ? `${contract.page_count} pages` : `${contract.indexing_progress || 0}% · ${contract.indexing_stage}`}</small></span><ChevronRight size={15}/></button></div>)}</aside>;
}
function ChatColumn(props: {active: Contract; ready: boolean; messages: Message[]; evidence: Citation[]; graph?: ContractGraph; telemetry: AgentTelemetry; question: string; setQuestion: (value: string) => void; ask: (event?: React.FormEvent) => void; chatBusy: boolean; analysisOpen: boolean; setAnalysisOpen: (value: boolean) => void; graphOpen: boolean; setGraphOpen: (value: boolean) => void; model: string; setModel: (value: string) => void; models: Model[]}) {
  currentContractId=props.active.contract_id;
  const [agentOpen,setAgentOpen]=useState(false);
  const [selectedAgent,setSelectedAgent]=useState('');
  const retrievalTrace=[...props.messages].reverse().find(message => message.role === 'assistant' && message.retrieval_trace)?.retrieval_trace;
  return <section className="chat-column"><button className="agent-side-tab" disabled={!props.ready} onClick={() => setAgentOpen(!agentOpen)} aria-label="Open Gante agent status" aria-expanded={agentOpen}><Activity size={15}/><span>Gante</span>{agentOpen && <X size={14}/>}</button><div className="chat-heading"><div className="chat-title"><span className="eyebrow">CONTRACT CHAT</span><h1 title={props.active.original_filename}>{props.active.original_filename}</h1><p>{props.ready ? `${props.active.page_count} pages · ${props.active.chunk_count} indexed evidence chunks` : `${props.active.indexing_stage} · ${props.active.indexing_progress}% complete`}</p></div><div className="chat-actions"><button className={'analysis-button '+(props.analysisOpen?'active':'')} disabled={!props.ready} onClick={() => { props.setAnalysisOpen(!props.analysisOpen); props.setGraphOpen(false); }} aria-pressed={props.analysisOpen}><Info size={15}/> Analysis</button><button className={'analysis-button '+(props.graphOpen?'active':'')} disabled={!props.ready} onClick={() => { props.setGraphOpen(!props.graphOpen); props.setAnalysisOpen(false); }} aria-pressed={props.graphOpen}><Activity size={15}/> Graph</button><a className="document-link" href={`${API}/contracts/${props.active.contract_id}/document`} target="_blank" rel="noreferrer"><FileText size={15}/> Open PDF <ArrowUpRight size={13}/></a></div></div>{props.analysisOpen && <><EvidencePanel evidence={props.evidence} trace={retrievalTrace}/><AnalysisDetailsPanel graph={props.graph}/></>}{props.graphOpen && <GraphPanel graph={props.graph}/>}{agentOpen && <AgentActivityDashboard telemetry={props.telemetry} running={props.active.analysis_status === 'running' || props.active.analysis_status === 'queued'} selectedAgent={selectedAgent} onSelectAgent={setSelectedAgent}/>}<div className="messages">{!props.ready ? <div className="chat-empty"><h2>Preparing your contract</h2><p>{props.active.indexing_stage} · {props.active.indexing_progress}% complete.</p></div> : props.messages.length === 0 ? <EmptyChat onSuggest={props.setQuestion}/> : props.messages.map((message,index) => <div className={'message '+message.role} key={index}><div className="avatar">{message.role === 'user' ? 'You' : 'PS'}</div><div className="message-body"><ReactMarkdown components={{a: ({href,children}) => <a href={href} target="_blank" rel="noreferrer">{children}</a>}}>{message.content}</ReactMarkdown>{message.citations?.length ? <button className="source-link" onClick={() => { props.setAnalysisOpen(true); props.setGraphOpen(false); }}><Info size={12}/> {message.citations.length} sources retrieved</button> : null}</div></div>)}{props.chatBusy && <div className="typing">Checking the contract evidence…</div>}</div><form className="composer" onSubmit={props.ask}><div className="composer-top"><input disabled={!props.ready} value={props.question} onChange={event => props.setQuestion(event.target.value)} placeholder="Ask a question about this contract…"/><button disabled={!props.ready || props.chatBusy} aria-label="Send message"><Send size={17}/></button></div><div className="composer-meta"><span>{props.ready ? 'Answers use selected contracts and cited evidence' : 'Indexing in background'}</span><label>Chat model<select value={props.model} onChange={event => props.setModel(event.target.value)}><option value="">Local fallback</option>{props.models.map(model => <option key={model.id} value={model.id}>{model.name} · {model.provider}</option>)}</select></label></div></form></section>;
}
function EvidencePanel({evidence, trace}: {evidence: Citation[]; trace?: RetrievalTrace}) { return <><div className="analysis-popover"><div className="analysis-title"><span><Sparkles size={15}/> Retrieved evidence</span><b>Top {Math.min(5, evidence.length)}</b></div>{evidence.slice(0, 5).map((item, index) => <div className="evidence-row" key={item.chunk_id}><span>{index + 1}</span><div><a href={`${API}/contracts/${item.chunk_id.split('_C')[0]}/document#page=${item.page}`} target="_blank" rel="noreferrer"><b>Page {item.page}{item.clause ? ` · Clause ${item.clause}` : ''}</b></a><p>{item.quote}</p></div></div>)}</div>{trace && <div className="retrieval-trace"><strong>Retrieval trace</strong><span>Query: {trace.query || Object.values(trace.queries || {}).join(' · ')}</span><span>Candidate chunks: {(trace.candidate_chunk_ids || []).join(', ') || 'none'}</span><span>Answer chunks: {(trace.answer_chunk_ids || []).join(', ') || 'none'}</span><span>Graph facts: {trace.graph_facts || 'none connected to retrieved chunks'}</span></div>}</>; }

function AnalysisDetailsPanel({graph}: {graph?: ContractGraph}) {
  const signature=graph?.signature; const quality=graph?.document_quality; const coverage=graph?.coverage; const config=graph?.document_config; const template=graph?.template_review; const graphReview=graph?.graph_review;
  return <div className="analysis-popover review-details"><div className="analysis-title"><span><ShieldCheck size={15}/> Review findings & checks</span></div><div className="analysis-signals"><div><small>Document quality</small><strong>{quality?.status || 'Pending'}</strong>{quality?.warnings?.[0] && <span>{quality.warnings[0]}</span>}</div><div><small>Clause coverage</small><strong>{coverage ? `${coverage.present?.length || 0} present · ${coverage.missing?.length || 0} gaps` : 'Pending'}</strong>{coverage?.missing?.length ? <span>Missing: {coverage.missing.join(', ')}</span> : null}</div><div><small>Signature text check</small><strong>{signature?.status === 'found' ? 'Signature language found' : signature?.status === 'not_found' ? 'No signature language found' : 'Pending'}</strong>{signature?.signatory_candidates?.length ? <span>Names to verify: {signature.signatory_candidates.join(', ')}</span> : null}{signature?.notes && <span>{signature.notes}</span>}</div><div><small>Deadlines & duties</small><strong>{graph ? `${graph.deadlines?.dates?.length || 0} date mentions · ${graph.obligations?.obligations?.length || 0} obligations` : 'Pending'}</strong>{graph?.obligations?.unassigned_count ? <span>{graph.obligations.unassigned_count} obligations have no clear actor</span> : null}</div><div className="config-signal"><small>Document configuration</small><strong>{config ? `${config.recommended_chunk_size}/${config.recommended_overlap} char chunks · ${config.embedding_batch_size} batch` : 'Pending'}</strong>{config && <span>{config.embedding_dimension}-dimension embeddings · {config.embedding_max_tokens} max tokens · retrieval {config.top_k_bm25}/{config.top_k_vector}/{config.top_k_rerank} · review batches up to {config.review_context_chars} chars</span>}</div></div>{template && <div className="template-review"><div className="analysis-title"><span><FileText size={14}/> Template review · India property deeds</span><b>{template.match_score == null ? 'NEEDS TEXT' : `${template.match_score}/100`}</b></div><p>{template.summary || 'Template comparison complete.'}</p>{template.detection_confidence==='low' && <p className="template-note">Type detection is uncertain. Candidate types: {(template.candidate_types || []).join(', ') || 'none'}. Confirm the deed type before relying on this checklist.</p>}{!!template.missing_clauses?.length && <div><strong>Clause gaps</strong><span>{template.missing_clauses.slice(0,5).map(clause=>clause.title).join(' · ')}{template.missing_clauses.length>5?' …':''}</span></div>}{!!template.red_flags?.length && <div className="template-flags"><strong>{template.red_flags.length} red flag{template.red_flags.length===1?'':'s'}</strong><span>{template.red_flags.slice(0,3).map(flag=>`${flag.title}: ${flag.evidence}`).join(' · ')}</span></div>}<small>Keyword checklist screen; confirm state-specific rules and review cited wording.</small></div>}{graphReview && <div className="graph-review-note"><strong>Graph reviewer</strong><span>{graphReview.issues?.length ? graphReview.issues.join(' ') : `No relationship issues found across ${graphReview.relationship_count || 0} links.`}</span></div>}</div>;
}

function forceGraphLayout(nodes: GraphNode[], edges: GraphEdge[]) {
  const width=1100, height=620, center={x:width/2,y:height/2};
  const points=new Map<string,{x:number;y:number;vx:number;vy:number}>();
  const ordered=[...nodes].sort((a,b)=>a.kind.localeCompare(b.kind)||a.label.localeCompare(b.label));
  ordered.forEach((node,index)=>{const angle=index*2.399963;const radius=65+Math.sqrt(index)*35;points.set(node.id,{x:center.x+Math.cos(angle)*radius,y:center.y+Math.sin(angle)*radius,vx:0,vy:0})});
  for(let step=0;step<260;step++){
    const force=new Map<string,{x:number;y:number}>();ordered.forEach(node=>force.set(node.id,{x:(center.x-points.get(node.id)!.x)*.004,y:(center.y-points.get(node.id)!.y)*.004}));
    for(let i=0;i<ordered.length;i++)for(let j=i+1;j<ordered.length;j++){
      const a=points.get(ordered[i].id)!,b=points.get(ordered[j].id)!;let dx=b.x-a.x,dy=b.y-a.y;let distance=Math.max(1,Math.hypot(dx,dy));
      const push=1100/(distance*distance)+Math.max(0,105-distance)*.018;const fx=dx/distance*push,fy=dy/distance*push;
      force.get(ordered[i].id)!.x-=fx;force.get(ordered[i].id)!.y-=fy;force.get(ordered[j].id)!.x+=fx;force.get(ordered[j].id)!.y+=fy;
    }
    for(const edge of edges){const a=points.get(edge.source),b=points.get(edge.target);if(!a||!b)continue;const dx=b.x-a.x,dy=b.y-a.y,distance=Math.max(1,Math.hypot(dx,dy));const pull=(distance-175)*.0018,fx=dx/distance*pull,fy=dy/distance*pull;force.get(edge.source)!.x+=fx;force.get(edge.source)!.y+=fy;force.get(edge.target)!.x-=fx;force.get(edge.target)!.y-=fy;}
    for(const node of ordered){const point=points.get(node.id)!,f=force.get(node.id)!;point.vx=(point.vx+f.x)*.78;point.vy=(point.vy+f.y)*.78;point.x=Math.max(72,Math.min(width-72,point.x+point.vx));point.y=Math.max(58,Math.min(height-58,point.y+point.vy));}
  }
  return Object.fromEntries([...points].map(([id,p])=>[id,{x:p.x,y:p.y}]));
}

function GraphPanel({graph}: {graph?: ContractGraph}) {
  const data=graph?.graph; const nodes=data?.nodes || []; const edges=data?.edges || [];
  const [selected,setSelected]=useState<GraphNode>(); const [zoom,setZoom]=useState(1); const [pan,setPan]=useState({x:0,y:0});
  const drag=useRef<{x:number;y:number;panX:number;panY:number}|null>(null);
  const positions=useMemo(()=>forceGraphLayout(nodes,edges),[nodes,edges]);
  const colors:Record<string,string>={document:'#294f46',chunk:'#8aa39a',clause:'#3c8c73',topic:'#c29442',party:'#6b78b4',date:'#a66a99',amount:'#b96854',obligation:'#ce704b',deliverable:'#427da1',term:'#9270a8'};
  const kinds=[...new Set(nodes.map(node=>node.kind))];
  const connected=selected?edges.filter(edge=>edge.source===selected.id||edge.target===selected.id):[];
  const wheel=(event:React.WheelEvent<SVGSVGElement>)=>{event.preventDefault();setZoom(value=>Math.max(.45,Math.min(2.8,value*(event.deltaY<0?1.12:.89))))};
  const startDrag=(event:React.PointerEvent<SVGSVGElement>)=>{drag.current={x:event.clientX,y:event.clientY,panX:pan.x,panY:pan.y};event.currentTarget.setPointerCapture(event.pointerId)};
  const moveDrag=(event:React.PointerEvent<SVGSVGElement>)=>{if(!drag.current)return;const box=event.currentTarget.getBoundingClientRect();setPan({x:drag.current.panX+(event.clientX-drag.current.x)*1100/box.width,y:drag.current.panY+(event.clientY-drag.current.y)*620/box.height})};
  const label=(value:string)=>value.length>26?`${value.slice(0,25)}…`:value;
  const method=data?.extraction?.method;
  return <div className="analysis-popover graph-panel"><div className="analysis-title"><span><Activity size={15}/> {data?.document_name || 'Contract'} · relationship graph</span><b>{nodes.length} entities · {edges.length} relations</b></div><div className="graph-toolbar"><span>{method==='whole-contract-model-evidence-checked'?`Whole-contract graph · ${data?.extraction?.model || 'selected model'}`:method==='local-fallback'?'Local fallback graph · model extraction unavailable':'Local-rule graph · choose a review model for named-entity extraction'}{data?.extraction?.warning&&<small className="graph-warning">{data.extraction.warning}</small>}</span>{nodes.length>0&&<div className="graph-controls"><button aria-label="Zoom out" title="Zoom out" onClick={()=>setZoom(value=>Math.max(.45,value/1.2))}><ZoomOut size={14}/></button><span>{Math.round(zoom*100)}%</span><button aria-label="Zoom in" title="Zoom in" onClick={()=>setZoom(value=>Math.min(2.8,value*1.2))}><ZoomIn size={14}/></button><button aria-label="Reset graph view" title="Reset view" onClick={()=>{setZoom(1);setPan({x:0,y:0})}}><Maximize2 size={14}/></button></div>}</div>{nodes.length>0?<><div className="graph-scroll"><svg className="contract-graph" viewBox="0 0 1100 620" onWheel={wheel} onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={()=>{drag.current=null}} onPointerCancel={()=>{drag.current=null}} role="group" aria-label="Interactive contract entity relationship graph"><title>Drag to pan, scroll to zoom, and select an entity to inspect its evidence and relationships</title><g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>{edges.map((edge,index)=>{const from=positions[edge.source],to=positions[edge.target];if(!from||!to)return null;const emphasized=!selected||edge.source===selected.id||edge.target===selected.id;const color=emphasized?'#789c8a':'#d5e2db';return <g key={`${edge.source}-${edge.target}-${edge.relation}-${index}`} opacity={emphasized?1:.42}><line x1={from.x} y1={from.y} x2={to.x} y2={to.y} className="graph-edge" style={{stroke:color}}/><text x={(from.x+to.x)/2} y={(from.y+to.y)/2-6} className="graph-edge-label">{label(edge.relation.replaceAll('_',' ').toLowerCase())}</text></g>})}{nodes.map(node=>{const point=positions[node.id];const active=node.id===selected?.id;const radius=active?19:node.kind==='chunk'?8:13;return <g key={node.id} className={'graph-node '+(active?'selected':'')} tabIndex={0} role="button" aria-label={`${node.kind}: ${node.label}`} onPointerDown={event=>event.stopPropagation()} onClick={()=>setSelected(node)} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();setSelected(node)}}}><title>{node.label}</title><circle cx={point.x} cy={point.y} r={radius} fill={colors[node.kind]||'#84948d'}/><text x={point.x} y={point.y+radius+17} textAnchor="middle" className="graph-node-label">{label(node.label)}</text></g>})}</g></svg></div><div className="graph-legend">{kinds.map(kind=><span key={kind}><i style={{background:colors[kind]||'#84948d'}}/>{data?.legend?.[kind]||kind}</span>)}</div>{selected&&<div className="graph-inspector"><div><strong>{selected.label}</strong><span>{data?.legend?.[selected.kind]||selected.kind} · {connected.length} linked relation{connected.length===1?'':'s'}</span></div>{connected.length>0?<ul>{connected.map((edge,index)=>{const other=nodes.find(node=>node.id===(edge.source===selected.id?edge.target:edge.source));return <li key={`${edge.source}-${edge.target}-${index}`}><b>{edge.relation.replaceAll('_',' ').toLowerCase()}</b><span>{edge.source===selected.id?'→':'←'} {other?.label||'Unknown entity'}</span>{(edge.evidence_sources||((edge.page&&edge.evidence)?[{chunk_id:edge.chunk_id||'',page:edge.page,quote:edge.evidence}]:[])).map((proof,proofIndex)=><small key={`${proof.chunk_id}-${proofIndex}`}>Page {proof.page}: {proof.quote} <a href={`${API}/contracts/${currentContractId}/document#page=${proof.page}`} target="_blank" rel="noreferrer">Open evidence</a></small>)}</li>})}</ul>:selected.evidence?.map((proof,index)=><p className="graph-proof" key={`${proof.chunk_id}-${index}`}>Page {proof.page}: “{proof.quote}” <a href={`${API}/contracts/${currentContractId}/document#page=${proof.page}`} target="_blank" rel="noreferrer">Open source</a></p>)}</div>}</>:<div className="graph-empty"><Activity size={20}/><p>{data?.extraction?.warning||'No evidence-backed entities or relationships were extracted for this contract.'}</p></div>}</div>;
}

function AgentActivityDashboard({telemetry, running, selectedAgent, onSelectAgent}: {telemetry: AgentTelemetry; running: boolean; selectedAgent: string; onSelectAgent: (name: string) => void}) {
  const names=Object.keys(telemetry.agents);
  const latest=[...telemetry.events].reverse().slice(0,8);
  const chosen=selectedAgent || names[0] || '';
  const item=chosen ? telemetry.agents[chosen] : undefined;
  const latestForAgent=[...telemetry.events].reverse().find(event => event.agent===chosen);
  return <div className="analysis-popover agent-panel"><div className="analysis-title"><span><Activity size={15}/> Agent activity</span><b className={running ? 'live-badge' : ''}>{running ? 'LIVE' : 'LATEST RUN'}</b></div><div className="master-health"><CheckCircle2 size={13}/> Master supervisor: {telemetry.master?.status || 'unknown'} · {telemetry.master?.supervises || names.length} agents monitored</div><div className="agent-grid">{names.length ? names.map(name => { const agent=telemetry.agents[name]; const failed=agent.last_status==='degraded'||agent.failures>0; const status=failed?'Needs attention':agent.last_status==='waiting'?'Waiting':agent.last_status==='healthy'?'Complete':agent.last_status; return <button type="button" className={'agent-card '+(chosen===name?'active':'')} key={name} onClick={() => onSelectAgent(name)} aria-pressed={chosen===name}><span className="agent-card-title">{failed ? <AlertTriangle size={13}/> : <CheckCircle2 size={13}/>}<strong>{name}</strong><span>{status}</span></span><small>{agent.runs} runs · {agent.retries} retries · {agent.failures} failures</small></button>; }) : <p className="muted">Agents will appear here when analysis starts.</p>}</div>{item && <div className="agent-selected"><div><strong>{chosen}</strong><span>{item.last_status==='waiting'?'Waiting to run':item.last_status==='degraded'?'Needs attention':'Status: '+item.last_status}</span></div><p>{item.contribution || 'Supervised review task.'}</p><p>{latestForAgent?.detail || item.last_detail || 'Waiting for output.'}</p>{latestForAgent?.error && <p className="agent-error">{latestForAgent.error}</p>}</div>}<div className="agent-log"><div className="agent-log-head"><span>Recent activity</span><RotateCcw size={12}/></div>{latest.length ? latest.map((event,index) => <div className="agent-log-row" key={`${event.timestamp}-${index}`}><span className={'log-dot '+(event.action==='failed'?'failed':event.action==='retry'?'retry':'done')}/><div><strong>{event.agent}</strong><small>{event.action}{event.attempt?` · attempt ${event.attempt}`:''}{event.duration_ms?` · ${event.duration_ms} ms`:''}</small><p className="agent-event-detail">{event.detail || event.contribution || event.error || ''}</p></div></div>) : <p className="muted">Waiting for the first agent event…</p>}</div></div>;
}
function EmptyChat({onSuggest}: {onSuggest: (value: string) => void}) { return <div className="chat-empty"><div className="empty-icon"><Sparkles size={20}/></div><h2>What would you like to know?</h2><div className="suggestions"><button onClick={() => onSuggest('What should I negotiate?')}>What should I negotiate?</button><button onClick={() => onSuggest('Summarize payment obligations')}>Summarize payment terms</button></div></div>; }
function ReportRail({findings, counts, score, scoreRationale, graph, busy, analysisStatus, analysisProgress, analysisStage, perspective, hasBothPerspectives, onPerspectiveChange, onOpenAnalysis, onRetry}: {findings: Finding[]; counts: {high: number; medium: number; low: number}; score?: number | null; scoreRationale?: string; graph?: ContractGraph; busy: boolean; analysisStatus: string; analysisProgress: number; analysisStage: string; perspective: string; hasBothPerspectives: boolean; onPerspectiveChange: (value: string) => void; onOpenAnalysis: () => void; onRetry: () => void}) {
  const analyzing=analysisStatus === 'running' || analysisStatus === 'queued';
  const template=graph?.template_review;
  const scoreMessage=score!=null ? (scoreRationale || 'Based on identified risks and missing protections in the contract.') : analyzing ? 'Review is running. The health score appears when it finishes.' : analysisStatus==='failed' ? (analysisStage || 'The review stopped before an overall score was saved.') : analysisStatus==='complete' ? 'No overall score was saved for this review.' : 'The overall review score is not available yet.';
  return <aside className="report-rail"><div className="report-head"><div><span className="eyebrow">RISK REPORT</span><h2>Contract health</h2></div><span className="report-score" title={scoreRationale || scoreMessage}>{score == null ? '—' : score}<small>/100</small></span></div><p className="score-explainer">{scoreMessage}</p><section className="perspective-control"><span>Review perspective</span><div role="tablist" aria-label="Choose contract review perspective"><button role="tab" type="button" aria-selected={perspective==='contract_giver'} className={perspective==='contract_giver'?'selected':''} disabled={busy||analyzing} onClick={()=>{if(perspective!=='contract_giver')onPerspectiveChange('contract_giver')}}>Contract giver</button><button role="tab" type="button" aria-selected={perspective==='contract_acceptor'} className={perspective==='contract_acceptor'?'selected':''} disabled={busy||analyzing} onClick={()=>{if(perspective!=='contract_acceptor')onPerspectiveChange('contract_acceptor')}}>Contract acceptor</button></div><small>{hasBothPerspectives ? 'Both party reviews are ready; switch tabs to compare findings.' : 'Run the review to prepare both party views.'}</small></section>{(busy || analyzing) && <div className="analysis-progress"><strong>{analysisStage || 'Starting review'}</strong><span>{analysisProgress || 0}%</span><div><i style={{width:`${Math.max(4,analysisProgress||0)}%`}}/></div></div>}{score==null && !analyzing && <button className="retry-review" onClick={onRetry} disabled={busy}>{analysisStatus==='failed' || analysisStatus==='complete' ? 'Retry contract review' : 'Start contract review'}</button>}<div className="risk-strip"><div><strong className="high-text">{counts.high}</strong><small>High risk</small></div><div><strong className="medium-text">{counts.medium}</strong><small>Review</small></div><div><strong className="low-text">{counts.low}</strong><small>Low</small></div></div>{template && <section className="report-template"><div><strong>Template checklist match</strong><b>{template.match_score == null ? '—' : `${template.match_score}/100`}</b></div><p>{template.detected_type_name || template.summary}</p><small>{template.verdict?.replaceAll('_',' ') || template.status} · {(template.missing_clauses || []).length} checklist gaps · {(template.red_flags || []).length} red flags</small>{template.detection_confidence==='low' && <small className="template-score-note">Deed type is uncertain. Confirm it before relying on this match.</small>}<small className="template-score-note">Checklist match is separate from overall contract health; low scores point to clauses that need review.</small><button onClick={onOpenAnalysis}>View template findings</button></section>}<div className="report-divider"/>{findings.slice(0, 5).map(finding => <article className="risk-item" key={finding.finding_id}><div className="risk-item-head"><span className={'risk-dot '+finding.severity.toLowerCase()}/><span>{finding.category}</span><b>{finding.severity}</b></div><h3>{finding.claim}</h3><p>{finding.recommendation}</p></article>)}{!findings.length && <div className="report-empty"><ShieldCheck size={22}/><p>{score!=null ? 'No clause-level risks were found.' : 'No findings to show yet.'}</p></div>}<div className="autonomous-note"><Sparkles size={14}/> Master agent coordinates the full review automatically.</div></aside>;
}

createRoot(document.getElementById('root')!).render(<App/>);
