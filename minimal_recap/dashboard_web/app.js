'use strict';
const $ = id => document.getElementById(id);
const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n, d=2) => n == null || !Number.isFinite(Number(n)) ? '—' : Number(n).toFixed(d);
const pct = n => n == null ? '—' : fmt(n*100,1)+'%';
const names = {initial_bc:'Initial BC', aggregate_bc:'Aggregate BC', success_bc:'Success BC', conditioned:'RECAP · Conditioned'};
const colors = ['#237c68','#d9a34f','#82a0c4','#b78591'];
let summary, episodes=[], filtered=[], episode, selected='', page=0, frame=0, playing=false, last=0, elapsed=0, loadVersion=0, groupVersion=0, epVersion=0;
async function api(path, params={}) {
  const response=await fetch('/api/'+path+'?'+new URLSearchParams(params));
  if(!response.ok) throw new Error('数据读取失败（HTTP '+response.status+'），请检查实验文件或刷新重试。');
  return response.json();
}
function error(ex){$('notice').textContent=ex.message; $('notice').className='error';}
function options(id, items){$(id).innerHTML=items.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('');}
function stopped(){playing=false; $('play').textContent='▶ 播放';}
function selection(){return {run:$('run').value};}
async function loadRun(){
  const version=++loadVersion; ++groupVersion; ++epVersion; stopped(); episode=null; selected='';
  $('play').disabled=true; $('notice').className=''; $('notice').textContent='正在读取实验…';
  try {
    const data=await api('run',selection()); if(version!==loadVersion)return; summary=data;
    const count=summary.groups.reduce((n,g)=>n+g.count,0), expert=summary.groups.find(g=>g.name==='expert')?.count||0;
    const evaluated=Object.entries(summary.metrics).filter(([,m])=>m.id?.success_rate!=null);
    const best=evaluated.reduce((a,b)=>!a||b[1].id.success_rate>a[1].id.success_rate?b:a,null);
    const stats=[['轨迹总量',count.toLocaleString(),'已保存的专家、rollout 与评估轨迹','↗'],['最佳 ID 成功率',best?pct(best[1].id.success_rate):'—',best?(names[best[0]]||best[0])+' · 当前 run':'尚无评估结果','◉'],['专家示范',expert.toLocaleString(),'已保存的专家轨迹 · 非额外交互预算','⌁'],['评估策略',evaluated.length,'同 run 对照 · 95% Wilson 区间','▥']];
    $('stats').innerHTML=stats.map(([a,b,c,d])=>`<div class="stat"><div class="stat-label">${esc(a)}</div><span class="stat-icon">${d}</span><div class="value">${esc(b)}</div><div class="desc">${esc(c)}</div></div>`).join('');
    const groupName=name=>name==='expert'?'专家示范':name==='diagnostics'?'诊断轨迹':name.startsWith('evaluation/')?`评估 / ${names[name.split('/')[1]]||name.split('/')[1]}`:name.startsWith('rollouts/')?`Rollout / ${name.split('/')[1]}`:name;
    options('group',[['__all__',`全部来源 · ${count}`],...summary.groups.map(g=>[g.name,`${groupName(g.name)} · ${g.count}`])]);
    if(summary.label_groups?.length) $('group').value=summary.label_groups[0];
    else if(summary.groups.some(g=>g.name==='expert')) $('group').value='expert';
    options('history',Object.keys(summary.histories).map(n=>[n,n]));
    const distributions=new Set();
    Object.values(summary.metrics).forEach(method=>Object.entries(method).forEach(([key,value])=>{
      if(value?.success_rate!=null)distributions.add(key);
      else if(key==='recovery'&&value&&typeof value==='object')Object.entries(value).forEach(([kind,m])=>{if(m?.success_rate!=null)distributions.add(`recovery.${kind}`);});
    }));
    const distributionNames={id:'ID · 统一分布','recovery.empty':'恢复 · 空抓','recovery.drop':'恢复 · 掉落'};
    options('distribution',[...distributions].map(key=>[key,distributionNames[key]||key]));
    drawHistory(); drawMetrics();
    $('provenance').textContent=`training seed ${summary.manifest.seed?.training??'—'} · ${summary.config.model?.hidden_layers??'—'} × ${summary.config.model?.hidden_units??'—'} MLP · 本地只读`;
    const totalFor=prefix=>summary.groups.filter(g=>g.name===prefix||g.name.startsWith(prefix+'/')).reduce((sum,g)=>sum+g.count,0);
    $('notice').textContent=summary.groups.length?`${summary.name} · 专家 ${totalFor('expert')} / Rollout ${totalFor('rollouts')} / 评估 ${totalFor('evaluation')} / 诊断 ${totalFor('diagnostics')} · ${summary.label_count??0} 份 Value / Advantage 标签`:'当前 run 尚无轨迹文件。';
    $('search').value='';$('result-filter').value='all';$('labeled-only').checked=false;
    await loadGroup();
  }catch(ex){if(version===loadVersion)error(ex);}
}
async function loadGroup(){
  const version=++groupVersion; ++epVersion; stopped(); episode=null; selected=''; $('play').disabled=true;
  $('ep-title').textContent='选择轨迹'; $('ep-result').textContent=''; $('ep-meta').textContent=''; $('frame-stats').textContent=''; $('signal-chart').textContent=''; $('time').textContent='0.00 s'; $('phase').textContent='—'; $('scene-info').textContent=''; frame=0; drawScene();
  $('episode-list').innerHTML='<div class="chart-empty">正在索引轨迹…</div>';
  drawValuePanels();
  try{
    const data=$('group').value?await api('episodes',{...selection(),group:$('group').value}):[];
    if(version!==groupVersion)return;
    episodes=data;
    page=0; filterEpisodes();
    if(filtered.length)await loadEpisode(filtered[0].path);
  }catch(ex){if(version===groupVersion)error(ex);}
}
function filterEpisodes(){
  const query=$('search').value.trim().toLowerCase(), result=$('result-filter').value;
  filtered=episodes.filter(e=>{
    const m=e.metadata, text=[e.path,m.episode_id,m.seed,m.scene?.layout].join(' ').toLowerCase();
    return (!$('labeled-only').checked||e.has_labels)&&text.includes(query)&&(result==='all'||(result==='success'&&m.success&&m.status==='complete')||(result==='failure'&&!m.success&&m.status==='complete')||(result==='aborted'&&m.status==='aborted'));
  });
  page=Math.min(page,Math.max(0,Math.ceil(filtered.length/10)-1)); renderList();
}
function renderList(){
  $('ep-count').textContent=`${filtered.length} 条`;
  $('episode-list').innerHTML=filtered.slice(page*10,page*10+10).map(e=>{
    const m=e.metadata, result=m.status==='aborted'?'中断':m.success?'成功':'失败';
    return `<button class="episode-item ${selected===e.path?'active':''}" data-path="${esc(e.path)}"><span class="row"><strong>${esc(m.episode_id||e.path)} ${e.has_labels?'<span class="label-chip">V/A</span>':''}</strong><span class="result ${m.success?'':'fail'}">● ${result}</span></span><span class="row"><small>${esc(m.scene?.layout||'—')} · seed ${esc(m.seed)}</small><small>${e.length} steps</small></span>${$('group').value==='__all__'?`<small>${esc(e.group)}</small>`:''}</button>`;
  }).join('')||'<div class="chart-empty">没有匹配的轨迹</div>';
  $('page').textContent=`${filtered.length?page+1:0} / ${Math.ceil(filtered.length/10)}`;
  $('prev').disabled=page===0; $('next').disabled=(page+1)*10>=filtered.length;
}
async function loadEpisode(path){
  const version=++epVersion; stopped(); $('play').disabled=true;
  try{
    const data=await api('episode',{...selection(),path}); if(version!==epVersion)return;
    episode=data; selected=path; frame=0; elapsed=0; renderList();
    const m=episode.metadata;
    $('ep-title').textContent=m.episode_id||path;
    $('ep-result').textContent=m.status==='aborted'?'中断':m.success?'成功完成':'任务失败'; $('ep-result').className='badge '+(m.success?'':'fail');
    $('scrub').max=Math.max(0,episode.frames.length-1); $('scrub').value=0;
    $('play').disabled=episode.frames.length<2;
    const items=[['来源',m.source],['场景 / seed',`${m.scene?.layout??'—'} / ${m.seed??'—'}`],['终止原因',m.reason],['累计奖励',fmt(episode.reward.reduce((a,b)=>a+b,0),0)],['Value / Advantage',episode.labels.value?.length?`${episode.labels.value.length} 步已标注`:'未保存标签'],['文件',path]];
    $('ep-meta').innerHTML=items.map(([a,b])=>`<div><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`).join('');
    updateFrame();
  }catch(ex){if(version===epVersion)error(ex);}
}
function drawScene(){
  const canvas=$('scene'), rect=canvas.getBoundingClientRect(), w=rect.width,h=rect.height,dpr=devicePixelRatio||1;
  canvas.width=w*dpr; canvas.height=h*dpr; const c=canvas.getContext('2d'); c.scale(dpr,dpr);
  if(!episode){c.fillStyle='#899c90';c.font='12px Segoe UI';c.textAlign='center';c.fillText('选择一条轨迹开始探索',w/2,h/2);return;}
  const env=summary.config.environment, cols=episode.columns, obs=episode.frames[frame], g=cols.gripper,o=cols.object,ped=cols.pedestal,go=cols.goal;
  if([g,o,ped,go].some(x=>x==null)){c.fillText('当前观测未包含回放所需特征',30,80);return;}
  const xmin=Math.min(env.workspace_x[0]-.05,...episode.frames.map(f=>f[o]-.03)), xmax=Math.max(env.workspace_x[1]+.05,env.arm_base[0]+.08,...episode.frames.map(f=>f[o]+.03));
  const maxz=Math.max(env.max_height+.07,...episode.frames.map(f=>f[g+1]+.08));
  const scale=Math.min((w-85)/(xmax-xmin),(h-70)/(maxz+.025));
  const left=(w-(xmax-xmin)*scale)/2, base=h-39, X=x=>left+(x-xmin)*scale, Z=z=>base-z*scale;
  c.lineWidth=1;c.font='9px Consolas';c.textAlign='right';
  for(let z=0;z<=maxz;z+=.05){c.strokeStyle='#e7ede5';c.beginPath();c.moveTo(X(xmin),Z(z));c.lineTo(X(xmax),Z(z));c.stroke();c.fillStyle='#9eada0';c.fillText(z.toFixed(2),X(xmin)-8,Z(z)+3);}
  for(let x=Math.ceil(xmin*10)/10;x<xmax;x+=.1){c.strokeStyle='#e7ede5';c.beginPath();c.moveTo(X(x),Z(0));c.lineTo(X(x),Z(maxz));c.stroke();c.fillStyle='#9eada0';c.textAlign='center';c.fillText(x.toFixed(1),X(x),Z(0)+16);}
  c.fillStyle='#e3ebe1';c.fillRect(X(xmin),Z(0), (xmax-xmin)*scale,5);
  const goal=obs[go], half=env.target_half_width;
  c.fillStyle='#c5dbba';c.fillRect(X(goal-half),Z(.003),half*2*scale,6);c.fillStyle='#769467';c.font='9px Segoe UI';c.fillText('GOAL',X(goal),Z(0)+29);
  c.fillStyle='#c7cec4';c.fillRect(X(obs[ped]-obs[ped+1]),Z(obs[ped+2]),obs[ped+1]*scale,obs[ped+2]*scale);
  c.fillStyle='#8c9b87';c.fillText('PEDESTAL',X(obs[ped]-obs[ped+1]/2),Z(obs[ped+2]) - 7);
  if($('trail').checked){
    [[g,'#237c68'],[o,'#d9a34f']].forEach(([col,color])=>{
      c.strokeStyle=color+'35';c.lineWidth=1.5;c.setLineDash([3,4]);c.beginPath();episode.frames.forEach((f,i)=>i?c.lineTo(X(f[col]),Z(f[col+1])):c.moveTo(X(f[col]),Z(f[col+1])));c.stroke();c.setLineDash([]);
      c.strokeStyle=color;c.lineWidth=2;c.beginPath();episode.frames.slice(0,frame+1).forEach((f,i)=>i?c.lineTo(X(f[col]),Z(f[col+1])):c.moveTo(X(f[col]),Z(f[col+1])));c.stroke();
    });
  }
  const length=env.block_length,thickness=env.block_thickness;
  c.save();c.translate(X(obs[o]),Z(obs[o+1]));c.rotate(Math.atan2(obs[o+4],obs[o+5]));c.fillStyle='#d9a34f';c.fillRect(-length*scale/2,-thickness*scale/2,length*scale,thickness*scale);c.strokeStyle='#ba883f';c.strokeRect(-length*scale/2,-thickness*scale/2,length*scale,thickness*scale);c.restore();
  const gx=obs[g],gz=obs[g+1],opening=obs[g+7],fl=env.finger_length,ft=env.finger_thickness;
  if(cols.arm!=null){
    let x=env.arm_base[0],z=env.arm_base[1],q=0;
    c.strokeStyle='#899eae';c.lineWidth=10;c.lineCap='round';c.beginPath();c.moveTo(X(x),Z(z));
    for(let k=0;k<3;k++){q+=obs[cols.arm+k];x+=env.arm_lengths[k]*Math.sin(q);z+=env.arm_lengths[k]*Math.cos(q);c.lineTo(X(x),Z(z));}
    c.stroke();c.lineCap='butt';
  }
  c.save();c.translate(X(gx),Z(gz));c.rotate(Math.atan2(obs[g+4],obs[g+5]));
  c.fillStyle='#237c68';c.fillRect(-.055*scale,-(fl/2+.017)*scale,.11*scale,.016*scale);
  for(const x of [-opening/2-ft,opening/2])c.fillRect(x*scale,-fl/2*scale,ft*scale,fl*scale);
  c.beginPath();c.arc(0,0,2.5,0,Math.PI*2);c.fill();c.restore();

}
function updateFrame(){
  if(!episode)return;
  const obs=episode.frames[frame],g=episode.columns.gripper,o=episode.columns.object,hz=episode.metadata.control_hz||summary.config.environment.control_hz;
  $('scrub').value=frame; $('time').textContent=fmt(frame/hz)+' s';
  $('phase').textContent=episode.phases[frame]||'NETWORK POLICY'; $('scene-info').textContent=`STEP ${frame} / ${episode.reward.length}`;
  const vals=[['夹爪 X / Z',g==null?'—':`${fmt(obs[g],3)} / ${fmt(obs[g+1],3)} m`],['物块 X / Z',o==null?'—':`${fmt(obs[o],3)} / ${fmt(obs[o+1],3)} m`],['夹爪开度',g==null?'—':fmt(obs[g+7]*100,2)+' cm'],['当前奖励',frame<episode.reward.length?fmt(episode.reward[frame],0):'终止状态']];
  $('frame-stats').innerHTML=vals.map(([a,b])=>`<div>${a}<strong>${b}</strong></div>`).join('');drawScene();drawSignal();drawValuePanels();
}
function chart(id, series, {cursor=null,xlabel='step',range=null,reference=null}={}){
  const host=$(id); series=series.map(s=>({...s,points:s.points.filter(p=>p[1]!=null&&Number.isFinite(p[1]))})).filter(s=>s.points.length);
  if(!series.length){host.innerHTML='<div class="chart-empty">暂无对应数据</div>';return;}
  const w=Math.max(300,host.clientWidth),h=host.clientHeight||160,L=53,R=15,T=24,B=30;
  const xs=series.flatMap(s=>s.points.map(p=>p[0])), ys=series.flatMap(s=>s.points.map(p=>p[1]));
  const minx=Math.min(...xs),maxx=Math.max(...xs), miny=range?range[0]:Math.min(...ys),maxy=range?range[1]:Math.max(...ys);
  const X=x=>L+(x-minx)/(maxx-minx||1)*(w-L-R), Y=y=>h-B-(y-miny)/(maxy-miny||1)*(h-T-B);
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(id)}">`;
  for(let i=0;i<=4;i++){const y=miny+(maxy-miny)*i/4;svg+=`<line x1="${L}" y1="${Y(y)}" x2="${w-R}" y2="${Y(y)}" stroke="#e8eee7"/><text x="${L-8}" y="${Y(y)+3}" text-anchor="end" fill="#91a095" font-size="9">${Math.abs(y)<.01&&y!==0?y.toExponential(1):fmt(y,Math.abs(maxy-miny)>10?0:3)}</text>`;}
  for(let i=0;i<=4;i++){const x=minx+(maxx-minx)*i/4;svg+=`<text x="${X(x)}" y="${h-9}" text-anchor="middle" fill="#91a095" font-size="9">${Math.round(x)}</text>`;}
  series.forEach((s,i)=>{const color=s.color||colors[i%colors.length]; svg+=`<path d="${s.points.map((p,j)=>`${j?'L':'M'}${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="1.8"/><circle cx="${10+i*105}" cy="8" r="2.5" fill="${color}"/><text x="${17+i*105}" y="11" fill="#6e8476" font-size="9">${esc(s.name)}</text>`;});
  if(reference!=null&&reference>=miny&&reference<=maxy)svg+=`<line x1="${L}" x2="${w-R}" y1="${Y(reference)}" y2="${Y(reference)}" stroke="#b78591" stroke-dasharray="4 4"/>`;
  if(cursor!=null)svg+=`<line x1="${X(cursor)}" x2="${X(cursor)}" y1="${T}" y2="${h-B}" stroke="#8fae9a" stroke-dasharray="3 3"/>`;
  svg+=`<text x="${w-R}" y="${h-9}" fill="#91a095" font-size="8" text-anchor="end">${xlabel}</text></svg>`;host.innerHTML=svg;
}
function drawSignal(){
  if(!episode)return;
  const type=$('signal').value,g=episode.columns.gripper,o=episode.columns.object,pts=arr=>arr.map((v,i)=>[i,v]);let series=[];
  if(type==='position')series=[['夹爪 X',g],['夹爪 Z',g==null?null:g+1],['物块 X',o],['物块 Z',o==null?null:o+1]].filter(([,col])=>col!=null).map(([name,col])=>({name,points:episode.frames.map((f,i)=>[i,f[col]])}));
  if(type==='action')series=['dx','dz','dtheta','opening'].map((name,k)=>({name,points:episode.action.map((a,i)=>[i,a[k]])}));
  if(type==='return')series=[{name:'Reward',points:pts(episode.reward)},{name:'MC Return',points:pts(episode.returns||[])}];
  if(type==='value'){drawValueComparison('signal-chart');return;}
  chart('signal-chart',series,{cursor:frame});
  if(type==='value'&&!series.some(s=>s.points.length))$('signal-chart').innerHTML='<div class="chart-empty">此轨迹没有 Value / Advantage 标签，可选择 rollouts/round_00。</div>';
}
function drawValuePanels(){
  const labels=episode?.labels||{}, value=labels.value||[], advantage=labels.advantage||[], ret=episode?.returns||labels.returns||[];
  const threshold=summary?.config.recap?.threshold??0, n=episode?.label_info?.n_step??summary?.config.recap?.n_step;
  const available=value.length>0&&advantage.length>0, at=frame<value.length;
  $('open-labeled').disabled=!summary?.label_groups?.length;
  $('value-note').textContent=available?`${episode.metadata.episode_id} · ${value.length} 步 · n = ${n??'—'} · 曲线随回放进度同步。V 是行为数据下的未来回报估计。`:'当前轨迹未保存 Value / Advantage 标签。专家、评估与未标注轮次不会自动生成预测，请点击“查看带标签轨迹”。';
  const mae=value.length&&ret.length===value.length?value.reduce((sum,v,i)=>sum+Math.abs(v-ret[i]),0)/value.length:null;
  const currentA=advantage[frame], hasA=Number.isFinite(currentA), high=hasA&&currentA>threshold;
  const stats=[['当前 V(s)',at?fmt(value[frame]):'—'],['真实 MC Return',frame<ret.length?fmt(ret[frame]):'—'],['当前 Advantage',hasA?`${fmt(currentA)} <span class="adv-badge ${high?'high-adv':'low-adv'}">${high?'高优势':'低优势'}</span>`:'—'],['高优势步占比',advantage.length?pct(advantage.filter(a=>a>threshold).length/advantage.length):'—'],['本轨迹 Value / MC MAE',fmt(mae)]];
  $('value-stats').innerHTML=stats.map(([a,b])=>`<div>${a}<strong>${b}</strong></div>`).join('');
  drawValueComparison('value-comparison');
  $('analysis-scrub').max=Math.max(0,(episode?.frames.length||1)-1);$('analysis-scrub').value=frame;$('analysis-scrub').disabled=!episode;
  $('analysis-step').textContent=episode?`step ${frame} / ${episode.frames.length-1}`:'—';
  $('advantage-rule').textContent=`绿色高优势：A > ${threshold}；红色低优势：A ≤ ${threshold}（等于阈值也属低优势）。虚线为 Advantage 阈值，底部色带逐步标注分类。左右轴独立刻度；点击图表或拖动滑块与轨迹同步，末帧无动作标签。`;
}
function drawValueComparison(id){
  const host=$(id),v=episode?.labels.value||[],a=episode?.labels.advantage||[],r=episode?.returns||episode?.labels.returns||[];
  if(!v.length&&!a.length&&!r.length){host.innerHTML='<div class="chart-empty">暂无 Value / Return / Advantage 数据，请选择带 V/A 标签的轨迹。</div>';return;}
  const threshold=summary?.config.recap?.threshold??0, high='#237c68',low='#c86461';
  const w=Math.max(300,host.clientWidth),h=host.clientHeight||350,L=55,R=55,T=15,B=45,bottom=h-B,maxStep=Math.max(1,(episode?.frames.length||1)-1);
  const finite=arr=>arr.filter(Number.isFinite), left=finite([...v,...r]), right=finite([...a,threshold]);
  const bounds=arr=>{const lo=Math.min(...arr),hi=Math.max(...arr),pad=Math.max((hi-lo)*.08,1);return [lo-pad,hi+pad];};
  const [lo,hi]=bounds(left.length?left:[0]),[alo,ahi]=bounds(right),X=i=>L+i/maxStep*(w-L-R),Y=y=>bottom-(y-lo)/(hi-lo)*(bottom-T),A=y=>bottom-(y-alo)/(ahi-alo)*(bottom-T),zero=A(threshold);
  const path=(arr,scale)=>arr.map((val,i)=>Number.isFinite(val)?`${i&&Number.isFinite(arr[i-1])?'L':'M'}${X(i).toFixed(2)},${scale(val).toFixed(2)}`:'').join(' ');
  const clipHigh=id+'-high',clipLow=id+'-low';
  let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Value、真实 MC Return、绿色高优势和红色低优势"><defs><clipPath id="${clipHigh}"><rect x="${L}" y="${T}" width="${w-L-R}" height="${zero-T}"/></clipPath><clipPath id="${clipLow}"><rect x="${L}" y="${zero}" width="${w-L-R}" height="${bottom-zero}"/></clipPath></defs>`;
  for(let i=0;i<=4;i++){const y=T+(bottom-T)*i/4;svg+=`<line x1="${L}" x2="${w-R}" y1="${y}" y2="${y}" stroke="#e8eee7"/><text x="${L-7}" y="${y+3}" text-anchor="end" fill="#507ca9" font-size="9">${fmt(hi-(hi-lo)*i/4,0)}</text><text x="${w-R+7}" y="${y+3}" fill="#83958b" font-size="9">${fmt(ahi-(ahi-alo)*i/4,1)}</text>`;}
  svg+=`<line x1="${L}" x2="${w-R}" y1="${zero}" y2="${zero}" stroke="#b78591" stroke-dasharray="5 4"/><text x="${w-R-3}" y="${zero-5}" text-anchor="end" fill="#a46c78" font-size="9">A = ${threshold}</text>`;
  svg+=`<path data-series="Value" d="${path(v,Y)}" fill="none" stroke="#507ca9" stroke-width="2"/><path data-series="MC Return" d="${path(r,Y)}" fill="none" stroke="#d9a34f" stroke-width="2"/>`;
  const advPath=path(a,A);
  svg+=`<path data-series="高优势" d="${advPath}" clip-path="url(#${clipHigh})" fill="none" stroke="${high}" stroke-width="1.6"/><path data-series="低优势" d="${advPath}" clip-path="url(#${clipLow})" fill="none" stroke="${low}" stroke-width="1.6"/>`;
  a.forEach((val,i)=>{if(!Number.isFinite(val))return;const color=val>threshold?high:low;svg+=`<rect data-advantage="${val>threshold?'high':'low'}" x="${X(i)}" y="${bottom+7}" width="${Math.max(.4,(w-L-R)/maxStep)}" height="7" fill="${color}"><title>step ${i}: A=${fmt(val)} · ${val>threshold?'高优势':'低优势'}</title></rect>`;});
  for(let i=0;i<=4;i++){const step=Math.round(maxStep*i/4);svg+=`<text x="${X(step)}" y="${h-10}" text-anchor="middle" fill="#91a095" font-size="9">${step}</text>`;}
  svg+=`<line x1="${X(frame)}" x2="${X(frame)}" y1="${T}" y2="${bottom+16}" stroke="#506d61" stroke-dasharray="3 3"/>`;
  [[v,Y,'#507ca9'],[r,Y,'#d9a34f'],[a,A,a[frame]>threshold?high:low]].forEach(([arr,scale,color])=>{if(Number.isFinite(arr[frame]))svg+=`<circle cx="${X(frame)}" cy="${scale(arr[frame])}" r="3" fill="${color}" stroke="white"/>`;});
  host.innerHTML=svg+'</svg>';
}
function drawHistory(){
  const history=summary?.histories[$('history').value]||[];
  const useSteps=history.length>0&&history.every(h=>h.gradient_step!=null), x=h=>useSteps?h.gradient_step:h.epoch, xlabel=useSteps?'gradient step':'epoch';
  chart('loss-chart',[{name:'Train loss',points:history.map(h=>[x(h),h.train_loss])},{name:'Validation loss',points:history.map(h=>[x(h),h.validation_loss])}],{xlabel});
  chart('validation-chart',[{name:'ID success %',points:history.map(h=>[x(h),h.id_validation?.success_rate==null?null:h.id_validation.success_rate*100])}],{xlabel,range:[0,100]});
}
function drawMetrics(){
  const key=$('distribution').value;
  const rows=Object.entries(summary?.metrics||{}).map(([name,m])=>[name,key.split('.').reduce((v,k)=>v?.[k],m)]).filter(([,m])=>m?.success_rate!=null);
  $('bars').innerHTML=rows.map(([name,m])=>`<div class="bar-row ${name==='conditioned'?'highlight':''}"><span>${esc(names[name]||name)}</span><div class="bar-track"><div class="bar-fill" style="width:${m.success_rate*100}%"></div>${m.success_ci95?`<span class="ci" style="left:${Math.max(0,m.success_ci95[0])*100}%;width:${(Math.min(1,m.success_ci95[1])-Math.max(0,m.success_ci95[0]))*100}%"></span>`:''}</div><strong>${pct(m.success_rate)}</strong></div>`).join('')||'<div class="chart-empty">该分布尚无评估结果</div>';
  $('metrics-table').innerHTML=rows.map(([name,m])=>`<tr><td>${esc(names[name]||name)}</td><td>${m.episodes}</td><td>${pct(m.success_rate)}</td><td>${pct(m.first_grasp_success_rate)}</td><td>${fmt(m.mean_return)}</td><td>${m.success_completion_seconds==null?'—':fmt(m.success_completion_seconds)+' s'}</td></tr>`).join('');
}
async function init(){
  try{const old=$('run').value;const runs=await api('runs');options('run',runs.map(n=>[n,n]));if(runs.includes(old))$('run').value=old;if(runs.length)await loadRun();else{$('notice').textContent='runs 目录下暂无实验。';}}catch(ex){error(ex);}
}
$('run').onchange=loadRun; $('refresh').onclick=init; $('group').onchange=loadGroup;
$('search').oninput=$('result-filter').onchange=$('labeled-only').onchange=()=>{page=0;filterEpisodes();if(filtered.length&&!filtered.some(e=>e.path===selected))loadEpisode(filtered[0].path);};
$('prev').onclick=()=>{page--;renderList();}; $('next').onclick=()=>{page++;renderList();};
$('episode-list').onclick=e=>{const item=e.target.closest('[data-path]');if(item)loadEpisode(item.dataset.path);};
$('history').onchange=drawHistory; $('distribution').onchange=drawMetrics; $('signal').onchange=drawSignal; $('trail').onchange=drawScene;
$('scrub').oninput=()=>{stopped();frame=Number($('scrub').value);updateFrame();};
$('analysis-scrub').oninput=()=>{stopped();frame=Number($('analysis-scrub').value);updateFrame();};
for(const id of ['value-comparison','signal-chart'])$(id).onclick=event=>{
  if(!episode||(id==='signal-chart'&&$('signal').value!=='value'))return;
  const rect=$(id).getBoundingClientRect(),w=Math.max(300,rect.width),x=(event.clientX-rect.left)/rect.width*w;
  stopped();frame=Math.max(0,Math.min(episode.frames.length-1,Math.round((x-55)/(w-110)*(episode.frames.length-1))));updateFrame();
};
$('value-comparison').onkeydown=event=>{
  if(!episode||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();stopped();frame=event.key==='Home'?0:event.key==='End'?episode.frames.length-1:Math.max(0,Math.min(episode.frames.length-1,frame+(event.key==='ArrowRight'?1:-1)));updateFrame();
};
$('restart').onclick=()=>{stopped();frame=0;elapsed=0;updateFrame();};
$('play').onclick=()=>{if(!episode)return;if(playing){stopped();return;}if(frame>=episode.frames.length-1)frame=0;playing=true;last=performance.now();elapsed=0;$('play').textContent='Ⅱ 暂停';requestAnimationFrame(tick);};
function tick(t){if(!playing||!episode)return;elapsed+=(t-last)*Number($('speed').value);last=t;const step=1000/(episode.metadata.control_hz||summary.config.environment.control_hz);if(elapsed>=step){frame=Math.min(episode.frames.length-1,frame+Math.floor(elapsed/step));elapsed%=step;updateFrame();if(frame===episode.frames.length-1)stopped();}if(playing)requestAnimationFrame(tick);}
$('open-labeled').onclick=async()=>{if(!summary?.label_groups?.length)return;$('search').value='';$('result-filter').value='all';$('labeled-only').checked=true;$('group').value=summary.label_groups[0];await loadGroup();};
new ResizeObserver(()=>{drawScene();if(summary)drawHistory();if(episode)drawSignal();drawValuePanels();}).observe($('scene'));
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>{document.querySelectorAll('nav a').forEach(x=>x.classList.toggle('active',x===a));});
init();
