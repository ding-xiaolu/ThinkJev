"""The demo page `thinkjev-serve` answers at `/`: one decision through ThinkJev, drawn live.

System 1 answers first (calibrated probabilities against the gate threshold). If it is unsure, the
page opens a server-sent event stream and, as the model thinks, draws the answer read at each
checkpoint (the budget-t answer, Figure 1 of the report) and this decision's confidence curve over
budgets next to the calibration report's test-accuracy curve. Standard library only; the page is
inline HTML with no external assets, so it works offline on the machine that runs the model.
"""

DEMO_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ThinkJev demo</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--card:#fff;--s1:#2a78d6;--s2:#eb6834;--ref:#a8a29e;--ok:#16a34a}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--line:#292524;--card:#1c1917;--s1:#60a5fa;--s2:#fb923c;--ref:#57534e;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}textarea,input,select{width:100%;font:inherit;color:inherit;background:transparent;border:1px solid var(--line);border-radius:6px;padding:8px}
textarea{min-height:84px;resize:vertical}.row{display:flex;gap:12px;flex-wrap:wrap}.row>*{flex:1;min-width:160px}
button{font:inherit;border:0;border-radius:6px;padding:9px 16px;background:var(--fg);color:var(--bg);cursor:pointer;margin-top:12px}
button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line);margin:0 8px 8px 0}button:disabled{opacity:.5;cursor:default}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:99px;color:#fff}.b1{background:var(--s1)}.b2{background:var(--s2)}
.bar{display:flex;align-items:center;gap:8px;margin:4px 0}.bar .k{width:30%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar .t{flex:1;height:14px;background:var(--line);border-radius:4px;overflow:hidden}.bar .f{height:100%}
.bar .v{width:56px;text-align:right;font-variant-numeric:tabular-nums}.muted{color:var(--muted);font-size:13px}
details pre{white-space:pre-wrap;font-size:13px;max-height:320px;overflow:auto}.err{color:#dc2626}
.strip{display:flex;gap:10px;overflow-x:auto;padding:6px 2px 2px;align-items:flex-end}
.cp{min-width:84px;border:1px solid var(--line);border-radius:8px;padding:6px;text-align:center;font-size:12px;color:var(--muted)}
.cp.stop{border-color:var(--fg);box-shadow:0 0 0 1px var(--fg)}.cp.final{border-style:dashed}
.cp .cols{display:flex;align-items:flex-end;justify-content:center;gap:3px;height:56px;margin:4px 0}
.cp .col{width:12px;background:var(--s2);opacity:.35;border-radius:2px 2px 0 0}.cp .col.top{opacity:1}
.cp b{color:var(--fg)}.cp .lab{font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}svg text{font:12px system-ui,sans-serif;fill:var(--muted)}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);flex-wrap:wrap}.legend i{display:inline-block;width:18px;height:0;border-top:2px solid;vertical-align:middle;margin-right:4px}
.legend i.dash{border-top-style:dashed}
</style></head><body><main>
<h1>ThinkJev</h1><p class="sub">Typed decisions with calibrated probabilities. System 1 answers in one forward pass; when it is unsure, the same model thinks on the cached prefix, and the same answer slots are read at every think budget along the way.</p>
<div class="card">
<div id="presets"></div>
<label>State (text or JSON)</label><textarea id="state"></textarea>
<label>Question</label><input id="q">
<div class="row"><div><label>Type</label><select id="type"><option>choice</option><option>noul</option><option>score</option></select></div>
<div><label>Thinking</label><select id="think"><option value="auto">auto (think only when unsure)</option><option value="never">never (System 1 only)</option><option value="always">always (System 2)</option></select></div></div>
<label>Options, one per line as <code>id: description</code> (choice), levels in order (score), or <code>true:</code>/<code>false:</code> (noul, optional)</label><textarea id="opts"></textarea>
<button id="go" onclick="decide()">Decide</button> <span id="model" class="muted"></span>
</div>
<div id="out"></div>
</main><script>
const PRESETS=[
 {name:"Easy: route a ticket",state:"Hi, I was charged twice for my subscription this month. Please refund one of the charges.",q:"Which team should handle this ticket?",type:"choice",opts:"billing: Payments, invoices, refunds\ntechnical: Bugs, outages, integrations\nsales: Pricing, upgrades, new accounts"},
 {name:"Hard: business-day shipping date",state:"An order was placed on Thursday, August 5, 2027. Orders ship on the 5th business day after the order date. Business days are Monday to Friday; the order date itself does not count.",q:"On what date does the order ship?",type:"choice",opts:"aug11: August 11, 2027\naug13: August 13, 2027\naug12: August 12, 2027\naug10: August 10, 2027"},
 {name:"Hard: invoice deadline",state:"Invoice 9876 was issued on September 15, 2026 with net-60 payment terms, meaning payment is due 60 calendar days after the issue date. A courtesy reminder was sent on October 17, 2026. The customer's payment was received on November 16, 2026. A payment is late only if it is received after the due date.",q:"Was this payment late under the stated terms?",type:"choice",opts:"on_time: The payment was received on or before the due date\nlate: The payment was received after the due date"},
 {name:"Hard: conditional count",state:"Order log: #1 Lyon, refunded, web, 120; #2 Oslo, shipped, app, 45; #3 Lyon, shipped, web, 310; #4 Oslo, refunded, web, 80; #5 Riga, shipped, app, 95; #6 Lyon, cancelled, web, 60; #7 Oslo, shipped, web, 200; #8 Riga, refunded, app, 15.",q:"How many shipped orders were placed through the web with an amount above 100?",type:"choice",opts:"0: 0 orders\n1: 1 order\n2: 2 orders\n3: 3 orders"}];
const $=id=>document.getElementById(id);let INFO=null;
function preset(i){const p=PRESETS[i];$("state").value=p.state;$("q").value=p.q;$("type").value=p.type;$("opts").value=p.opts;}
$("presets").innerHTML=PRESETS.map((p,i)=>`<button class="ghost" onclick="preset(${i})">${esc(p.name)}</button>`).join("");
function parseState(t){try{const v=JSON.parse(t);if(typeof v==="object"&&v)return v}catch(e){}return t}
function question(){const type=$("type").value,lines=$("opts").value.split("\n").map(s=>s.trim()).filter(Boolean),q={type,instructions:$("q").value};
 if(type==="score"){q.criteria=lines}else{const c={};for(const l of lines){const i=l.indexOf(":");const k=(i<0?l:l.slice(0,i)).trim(),v=i<0?null:l.slice(i+1).trim();c[k]=v||null}
 if(type==="choice"||Object.keys(c).length)q.criteria=c}return q}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function pct(v){return (v*100).toFixed(1)+"%"}
function bars(p,color){return Object.entries(p).map(([k,v])=>`<div class="bar"><span class="k" title="${esc(k)}">${esc(k)}</span><span class="t"><span class="f" style="display:block;width:${(v*100).toFixed(1)}%;background:${color}"></span></span><span class="v">${pct(v)}</span></div>`).join("")}
function head(a){return a.type==="choice"?`<b>${esc(a.choice)}</b> · confidence ${a.confidence.toFixed(2)}`:a.type==="score"?`<b>score ${a.score.toFixed(2)}</b> · confidence ${a.confidence.toFixed(2)}`:`<b>P(yes) = ${a.noul.toFixed(3)}</b>`}
function argmax(p){let b=null;for(const k in p)if(b===null||p[k]>p[b])b=k;return b}
// One checkpoint as a small group of vertical bars, the top option in full color (Figure 1's reads).
function cpCard(read,label,cls){const top=argmax(read.probabilities);const cols=Object.entries(read.probabilities).map(([k,v])=>`<span class="col${k===top?" top":""}" style="height:${Math.max(2,v*56).toFixed(0)}px" title="${esc(k)} ${pct(v)}"></span>`).join("");
 return `<div class="cp${cls}"><div class="lab">t = ${esc(label)}</div><div class="cols">${cols}</div><b>${esc(top)}</b> ${read.confidence.toFixed(2)}</div>`}
// This decision's confidence over budgets (orange) beside the report's test accuracy per budget (gray, dashed).
function chart(reads,tau,finalRead){const xs=(INFO&&INFO.checkpoints?INFO.checkpoints.slice():[]).concat(["full"]);
 const W=640,H=200,L=44,R=12,T=14,B=28,x=i=>L+(W-L-R)*i/Math.max(1,xs.length-1),y=v=>T+(H-T-B)*(1-v);
 const pts=reads.map(r=>{const i=xs.indexOf(r.t);return i<0?null:[x(i),y(r.confidence)]}).filter(Boolean);
 if(finalRead)pts.push([x(xs.length-1),y(finalRead.confidence)]);
 const ref=(INFO&&INFO.reference_curve||[]).map(c=>{const i=xs.indexOf(c.t);return i<0?null:[x(i),y(c.accuracy)]}).filter(Boolean);
 const path=p=>p.map((q,i)=>(i?"L":"M")+q[0].toFixed(1)+","+q[1].toFixed(1)).join(" ");
 let g=`<svg viewBox="0 0 ${W} ${H}">`;
 for(const v of [0,0.25,0.5,0.75,1])g+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L-6}" y="${y(v)+4}" text-anchor="end">${(v*100).toFixed(0)}%</text>`;
 xs.forEach((t,i)=>{g+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle">${t}</text>`});
 if(tau)g+=`<line x1="${L}" x2="${W-R}" y1="${y(tau)}" y2="${y(tau)}" stroke="var(--fg)" stroke-dasharray="2 4"/><text x="${W-R}" y="${y(tau)-4}" text-anchor="end">anytime τ = ${tau.toFixed(2)}</text>`;
 if(ref.length)g+=`<path d="${path(ref)}" fill="none" stroke="var(--ref)" stroke-width="2" stroke-dasharray="6 4"/>`;
 if(pts.length)g+=`<path d="${path(pts)}" fill="none" stroke="var(--s2)" stroke-width="2.5"/>`+pts.map(q=>`<circle cx="${q[0]}" cy="${q[1]}" r="3.5" fill="var(--s2)"/>`).join("");
 const stop=reads.find(r=>r.anytime_stop);if(stop){const i=xs.indexOf(stop.t);if(i>=0)g+=`<circle cx="${x(i)}" cy="${y(stop.confidence)}" r="8" fill="none" stroke="var(--fg)" stroke-width="1.5"/>`}
 g+=`</svg><div class="legend"><span><i style="border-color:var(--s2)"></i>this decision: top probability at budget t</span>${ref.length?`<span><i class="dash" style="border-color:var(--ref)"></i>report: test accuracy at budget t (${esc(INFO.model)})</span>`:""}${stop?`<span>◯ where the anytime policy would stop</span>`:""}</div>`;
 return g}
function render(st){const out=[];
 if(st.s1){out.push(`<div class="card"><span class="badge b1">System 1</span> <span class="muted">${(st.s1.seconds*1000).toFixed(0)} ms · calibrated confidence ${st.s1.confidence.toFixed(3)} vs threshold ${st.s1.threshold.toFixed(3)}</span><p>${st.s1.escalate?"Not sure, so the model thinks on the cached prefix.":"Confident enough: answered by System 1, no thinking."}</p>${bars(st.s1.probabilities,"var(--s1)")}</div>`)}
 else{out.push('<div class="card muted">System 1…</div>')}
 if(st.s1&&st.s1.escalate){const reads=st.reads,fin=st.final;
  const status=st.answer?`${st.answer.trace.think_tokens} thought tokens${st.answer.trace.thought_closed?", closed by itself":", stopped at the budget"} · ${st.answer.trace.seconds.toFixed(1)} s`:`thinking… ${st.tokens} tokens`;
  let strip=reads.map(r=>cpCard(r,r.t,r.anytime_stop?" stop":"")).join("");
  if(fin)strip+=cpCard(fin,"full"," final"+(st.answer&&st.answer.trace.anytime_stop==="full"?" stop":""));
  out.push(`<div class="card"><span class="badge b2">System 2</span> <span class="muted">${status}</span><p class="muted">The same answer slots read after the first t thought tokens: each group is the answer under a think budget of t. The outlined one is where the anytime policy would stop (confidence ≥ τ).</p><div class="strip">${strip||'<span class="muted">waiting for the first checkpoint…</span>'}</div>${chart(reads,INFO&&INFO.anytime_tau,fin)}</div>`)}
 if(st.answer){const a=st.answer.answer,t=st.answer.trace;out.push(`<div class="card"><span class="badge ${t.system==="s2"?"b2":"b1"}">Answer</span> <span class="muted">by ${t.system==="s2"?"System 2":"System 1"}${t.system==="s2"&&t.anytime_stop!==undefined?` · anytime policy would answer at t = ${t.anytime_stop}`:""}</span><p>${head(a)}</p>${a.probabilities?bars(a.probabilities,t.system==="s2"?"var(--s2)":"var(--s1)"):""}${t.thought?`<details><summary class="muted">Thought (${t.think_tokens} tokens)</summary><pre>${esc(t.thought)}</pre></details>`:""}</div>`)}
 if(st.error)out.push(`<div class="card err">${esc(st.error)}</div>`);
 $("out").innerHTML=out.join("")}
async function decide(){const st={s1:null,reads:[],tokens:0,final:null,answer:null,error:null};render(st);$("go").disabled=true;
 try{const r=await fetch("/v1/systemone",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({state:parseState($("state").value),questions:{decision:question()},thinkjev:{think:$("think").value,include_thought:true,stream:true}})});
  if(!r.ok){const j=await r.json();throw new Error(j.detail||r.status)}
  const reader=r.body.getReader(),dec=new TextDecoder();let buf="";
  while(true){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
   while((i=buf.indexOf("\n\n"))>=0){const chunk=buf.slice(0,i);buf=buf.slice(i+2);const line=chunk.split("\n").find(l=>l.startsWith("data:"));if(!line)continue;
    const ev=JSON.parse(line.slice(5));
    if(ev.type==="s1")st.s1=ev;else if(ev.type==="thinking")st.tokens=ev.tokens;else if(ev.type==="checkpoint")st.reads.push(ev);
    else if(ev.type==="answer"){st.answer=ev;const t=ev.trace;if(t.system==="s2"&&ev.answer.probabilities)st.final={probabilities:ev.answer.probabilities,confidence:t.s2_confidence||Math.max(...Object.values(ev.answer.probabilities))};}
    else if(ev.type==="error")st.error=ev.detail;
    render(st)}}
 }catch(e){st.error=e.message;render(st)}finally{$("go").disabled=false}}
fetch("/v1/thinkjev/info").then(r=>r.json()).then(i=>{INFO=i;$("model").textContent=`${i.model} · gate threshold ${i.threshold.toFixed(3)} · checkpoints ${i.checkpoints.join(", ")||"none"} · budget ${i.think_budget}`}).catch(()=>{});
preset(1);
</script></body></html>
"""
