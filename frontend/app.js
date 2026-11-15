'use strict';
const $ = id => document.getElementById(id);
const colors = {train:'#367d64',validation:'#90b29a',id:'#b1c8b6',inner_goal_ood:'#91a5b8',outer_goal_ood:'#6688a8',size_ood:'#c3ab83',inner_composition_ood:'#af9db6',outer_composition_ood:'#8e72a1',goal_ood:'#91a5b8',composition_ood:'#af9db6'};
const names = {all:'全部数据',train:'Train',validation:'Validation',id:'ID',inner_goal_ood:'Inner Goal OOD',outer_goal_ood:'Outer Goal OOD',size_ood:'Size OOD',inner_composition_ood:'Inner Composition OOD',outer_composition_ood:'Outer Composition OOD',goal_ood:'Goal OOD',composition_ood:'Composition OOD'};
const difficultyColors = {easy:'#5c987b',medium:'#cbb682',hard:'#ba8b80'};
let data, scenes, selected, detail, split='all', difficulty='all', currentPage='overview', frame=0, playing=false, lastTick=0, sortDirection=0, requestId=0;
const escapeHTML = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n,d=2) => Number(n).toFixed(d);
const mean = values => values.length ? values.reduce((a,b)=>a+b,0)/values.length : 0;
const dt = () => data.config.env.timestep * data.config.env.substeps;
const filteredScenes = () => Object.values(scenes).filter(s=>(split==='all'||s.split===split)&&(difficulty==='all'||s.metrics.difficulty===difficulty));
const filteredEpisodes = () => {const ids=new Set(filteredScenes().map(s=>s.scene_id));return data.episodes.filter(e=>ids.has(e.scene_id));};
function download(value, filename, type='application/json') {const url=URL.createObjectURL(new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function showReadError(target, retryAction){
  target.textContent='无法连接本地数据服务。请确认服务正在运行（python frontend/serve.py）。 ';
  const retry=document.createElement('button');retry.className='text-button';retry.textContent='重新读取 ↻';retry.onclick=retryAction;target.appendChild(retry);
}
function navigate() {
  currentPage=['overview','episodes','config'].includes(location.hash.slice(1))?location.hash.slice(1):'overview';
  document.querySelectorAll('.page').forEach(e=>e.classList.toggle('active',e.id===currentPage));
  document.querySelectorAll('nav a').forEach(e=>e.classList.toggle('active',e.dataset.page===currentPage));
  const titles={overview:['从轨迹，看见策略。','探索二维机械臂的避障与到达，用真实实验数据理解每一次运动。','实验概览'],episodes:['每条轨迹，都有迹可循。','在独立场景之间切换，检查专家的动作、路径与最终结果。','轨迹浏览器'],config:['实验的每一个设定。','从环境动力学到训练超参数，完整呈现这一次实验的配置。','实验配置']};
  $('page-title').textContent=titles[currentPage][0];$('page-subtitle').textContent=titles[currentPage][1];$('breadcrumb').textContent=titles[currentPage][2];
  document.querySelector('.filterbar').hidden=currentPage==='config';
  (currentPage==='episodes'?$('browser-player-slot'):$('overview-player-slot')).appendChild($('trajectory-workspace'));
  pause();
  if(data && currentPage==='episodes') renderEpisodeNav();
}
function renderTabs(){ $('split-tabs').innerHTML=['all',...Object.keys(data.scenes)].map(key=>`<button data-split="${key}" class="${split===key?'active':''}">${names[key]}<span>${key==='all'?Object.keys(scenes).length:(data.scenes[key]||[]).length}</span></button>`).join(''); }
function updateFilters(){renderTabs();renderOverview();renderEpisodeNav();const rows=filteredEpisodes();$('replay-empty').hidden=rows.length>0;if(!selected||!rows.some(e=>e.path===selected.path)){if(rows.length) selectEpisode(rows[0]);else{requestId++;pause();detail=null;selected=null;$('player').hidden=true;}}}
function renderOverview(){
  const ss=filteredScenes(), es=filteredEpisodes(), steps=es.map(e=>e.steps);
  const stats=[['场景总数',ss.length,'个',split==='all'?`${Object.keys(data.scenes).length} 个独立数据划分`:`${names[split]} 数据划分`,'◫'],['轨迹总数',es.length,'条',`${[...new Set(es.map(e=>e.source))].join(' / ')||'暂无'} · 已保存 episode`,'⌁'],['平均轨迹长度',es.length?fmt(mean(steps),1):'—','步',es.length?`平均时长 ${fmt(mean(steps)*dt())} s`:'当前筛选暂无轨迹','↔'],['累计控制步',steps.reduce((a,b)=>a+b,0).toLocaleString(),'步',`控制频率 ${fmt(1/dt(),0)} Hz`,'⌘']];
  $('stats').innerHTML=stats.map(([label,value,unit,foot,icon])=>`<div class="stat"><div class="stat-label">${label}</div><span class="stat-icon">${icon}</span><div class="stat-value">${value}<small>${unit}</small></div><div class="stat-foot">${escapeHTML(foot)}</div></div>`).join('');
  renderScatter(ss);
  $('split-description').textContent=`${Object.keys(data.scenes).length} 个实际数据划分 · ${Object.keys(scenes).length} 个场景`;
  renderQuotas();
  $('split-bars').innerHTML=Object.keys(data.scenes).map(key=>`<div class="split-row"><span>${names[key]}</span><div class="bar-track"><div style="width:${(data.scenes[key]||[]).length/Math.max(...Object.values(data.scenes).map(a=>a.length),1)*100}%;background:${colors[key]};opacity:${split==='all'||split===key?1:.3}"></div></div><strong>${(data.scenes[key]||[]).length}</strong></div>`).join('');
  const ds=Object.keys(difficultyColors).map(key=>[key,ss.filter(s=>s.metrics.difficulty===key).length]);
  $('difficulty-total').textContent=`${ss.length} SCENES`;
  $('difficulty-bar').innerHTML=ds.map(([key,n])=>`<div style="width:${ss.length?n/ss.length*100:0}%;background:${difficultyColors[key]};border-radius:2px" title="${key}: ${n}"></div>`).join('');
  $('difficulty-legend').innerHTML=ds.map(([key,n])=>`<span><i style="background:${difficultyColors[key]}"></i>${key[0].toUpperCase()+key.slice(1)} <b>${n}</b></span>`).join('');
  renderHistogram(steps);
  const labels={accepted:'已接受场景',initial_collision:'初始碰撞',planning_or_execution:'规划 / 执行未通过',difficulty_quota_full:'难度配额已满',construction_failed:'候选构造失败',medium_detour_too_long:'Medium 绕行过长'};
  $('generation').innerHTML=Object.entries(data.generation).map(([key,value])=>`<div><div class="generation-value">${Number(value).toLocaleString()}</div><div class="generation-label">${escapeHTML(labels[key]||key)}</div><div class="generation-detail">${escapeHTML(key)}</div></div>`).join('')||'<p class="muted">目录中暂无生成统计</p>';

}
function renderQuotas(){
  const groups=Object.entries(data.difficulty||{}).filter(([key])=>split==='all'||key===split);
  $('quota-card').hidden=!groups.length;
  let actualTotal=0,targetTotal=0,quotaMatch=true;
  $('quota-grid').innerHTML=groups.map(([key,record])=>{
    const actual={easy:0,medium:0,hard:0};(data.scenes[key]||[]).forEach(s=>{if(s.metrics?.difficulty in actual)actual[s.metrics.difficulty]++;});
    return `<div class="quota-group"><h3>${escapeHTML(names[key]||key)}</h3>${Object.keys(difficultyColors).map(level=>{const target=record.target?.[level];actualTotal+=actual[level];targetTotal+=Number(target)||0;quotaMatch=quotaMatch&&target===actual[level];return `<div class="quota-row"><span><i style="background:${difficultyColors[level]}"></i>${level}</span><strong class="${target===actual[level]?'quota-ok':''}">${actual[level]} / ${target??'—'}</strong></div>`;}).join('')}<small>场景清单 / 目标配额</small><p>尝试 ${record.attempts??'—'} · 局部绕行 ${record.local_detours??'—'}</p></div>`;
  }).join('');
  $('quota-summary').textContent=`${actualTotal} / ${targetTotal} 场景 · ${quotaMatch?'配额一致':'配额待核对'}（全部难度）`;
}
function renderScatter(ss){
  const w=600,h=285,left=48,right=570,top=18,bottom=250,x=v=>left+(v+1)/2*(right-left),y=v=>bottom-(v+1)/2*(bottom-top);
  let svg='';for(let i=-1;i<=1.001;i+=.5){svg+=`<line x1="${x(i)}" y1="${top}" x2="${x(i)}" y2="${bottom}" stroke="#eaf0e9"/><line x1="${left}" y1="${y(i)}" x2="${right}" y2="${y(i)}" stroke="#eaf0e9"/><text x="${x(i)}" y="${bottom+18}" text-anchor="middle" class="axis-text">${fmt(i,1)}</text><text x="${left-12}" y="${y(i)+3}" text-anchor="end" class="axis-text">${fmt(i,1)}</text>`;}
  svg+=`<ellipse cx="${x(0)}" cy="${y(0)}" rx="${(right-left)/2}" ry="${(bottom-top)/2}" fill="none" stroke="#dce6db" stroke-dasharray="3 5"/><text x="${right+9}" y="${bottom+3}" class="axis-text">x</text><text x="${left-5}" y="${top-7}" class="axis-text">y</text>`;
  svg+=ss.map(s=>`<circle class="scatter-point" tabindex="0" role="button" aria-label="查看 ${escapeHTML(s.scene_id)}" data-scene="${escapeHTML(s.scene_id)}" cx="${x(s.goal[0])}" cy="${y(s.goal[1])}" r="3.4" fill="${colors[s.split]||'#78977d'}" opacity=".75"><title>${escapeHTML(s.scene_id)} · ${s.metrics.difficulty}\nx=${fmt(s.goal[0],3)}, y=${fmt(s.goal[1],3)}</title></circle>`).join('');
  if(!ss.length) svg+='<text x="300" y="140" text-anchor="middle" class="axis-text">当前筛选暂无场景</text>';
  $('scatter').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="目标位置散点图">${svg}</svg>`;
  $('scatter-legend').innerHTML=Object.keys(data.scenes).map(key=>[key,colors[key]]).map(([key,color])=>`<span><i style="background:${color}"></i>${names[key]}</span>`).join('');
}
function renderHistogram(values){
  const max=Math.max(...values,20), binWidth=Math.ceil(max/8), bins=Array.from({length:8},(_,i)=>values.filter(v=>v>=i*binWidth &&(i===7?v<=(i+1)*binWidth:v<(i+1)*binWidth)).length), peak=Math.max(...bins,1);
  let svg='';[0,.5,1].forEach(n=>{const y=120-n*100;svg+=`<line x1="38" y1="${y}" x2="570" y2="${y}" stroke="#edf1e9"/><text x="28" y="${y+3}" text-anchor="end" class="axis-text">${Math.round(n*peak)}</text>`;});
  bins.forEach((n,i)=>{const x=49+i*64,y=120-n/peak*100;svg+=`<rect x="${x}" y="${y}" width="44" height="${120-y}" rx="3" fill="${i%2?'#8bb39a':'#5e957b'}"><title>${i*binWidth}–${(i+1)*binWidth} 步: ${n} 条</title></rect><text x="${x+22}" y="140" text-anchor="middle" class="axis-text">${i*binWidth}</text>`;});
  $('histogram').innerHTML=`<svg viewBox="0 0 600 155" aria-label="步数直方图">${svg}</svg>`;
}
function matchingEpisodes(){let es=filteredEpisodes();const q=$('search').value.trim().toLowerCase();if(q)es=es.filter(e=>(e.episode_id+' '+e.scene_id).toLowerCase().includes(q));if(sortDirection)es.sort((a,b)=>(a.steps-b.steps)*sortDirection);return es;}
function syncEpisodeNav(){
  const rows=matchingEpisodes(), position=rows.findIndex(e=>e.path===selected?.path);
  $('episode-position').textContent=rows.length ? `${position>=0?position+1:'—'} / ${rows.length}` : '0 条轨迹';
  $('prev-episode').disabled=position<=0;$('next-episode').disabled=!rows.length||position===rows.length-1;
  $('episode-list').querySelectorAll('[data-episode]').forEach(button=>{
    const active=button.dataset.episode===selected?.path;
    button.classList.toggle('selected',active);
    if(active)button.setAttribute('aria-current','true');else button.removeAttribute('aria-current');
  });
}
function renderEpisodeNav(){
  if(!data)return;const rows=matchingEpisodes(), scroll=$('episode-list').scrollTop;
  $('episode-count').textContent=rows.length;
  $('episode-list').innerHTML=rows.map(e=>`<button class="episode-item" data-episode="${escapeHTML(e.path)}" aria-label="回放 ${escapeHTML(e.episode_id)}，${e.steps} 步"><span class="episode-item-title"><strong>${escapeHTML(e.episode_id)}</strong><i class="episode-result ${e.success?'':'failed'}" title="${e.success?'成功':escapeHTML(e.failure_reason||'失败')}"></i></span><span class="episode-item-scene">${escapeHTML(e.scene_id)}</span><span class="episode-item-meta"><span style="color:${colors[e.split]}">${names[e.split]}</span><span>${e.steps} 步 · ${escapeHTML(scenes[e.scene_id]?.metrics.difficulty||'—')}</span></span></button>`).join('')||'<div class="nav-empty">没有匹配的轨迹<br>请调整搜索或筛选</div>';
  $('episode-list').scrollTop=scroll;syncEpisodeNav();
}
function stepEpisode(direction){
  const rows=matchingEpisodes(), position=rows.findIndex(e=>e.path===selected?.path), next=rows[position+direction];
  if(next){selectEpisode(next);const button=Array.from($('episode-list').querySelectorAll('[data-episode]')).find(b=>b.dataset.episode===next.path);button?.scrollIntoView({block:'nearest',inline:'nearest'});}
}
function kinematics(q){const elbow=[.5*Math.cos(q[0]),.5*Math.sin(q[0])];return [[0,0],elbow,[elbow[0]+.5*Math.cos(q[0]+q[1]),elbow[1]+.5*Math.sin(q[0]+q[1])]];}
async function selectEpisode(e){
  pause();const req=++requestId;selected=e;detail=null;frame=0;$('player').hidden=false;$('selected-label').textContent=`${e.scene_id} / ${e.episode_id} · 正在读取…`;$('arm-stage').innerHTML='';$('play').disabled=true;syncEpisodeNav();
  try{const response=await fetch(`/api/episode?path=${encodeURIComponent(e.path)}`);if(!response.ok)throw new Error('轨迹读取失败');const result=await response.json();if(req!==requestId)return;detail=result;
    $('play').disabled=false;$('selected-label').textContent=`${e.scene_id} / ${e.episode_id} · ${names[e.split]}`;$('episode-name').textContent=e.episode_id;$('outcome').innerHTML=`<span class="badge ${e.success?'':'fail'}">${e.success?'SUCCESS':'FAILED'}</span>`;
    const s=scenes[e.scene_id];$('episode-metrics').innerHTML=[['数据划分',names[e.split]],['轨迹来源',e.source],['轨迹步数',`${e.steps} steps`],['累计回报',e.total_return],['障碍半径',`${fmt(s.radius*100,1)} cm`],['场景难度',s.metrics.difficulty]].map(([a,b])=>`<div><dt>${a}</dt><dd>${escapeHTML(typeof b==='object'?JSON.stringify(b):b)}</dd></div>`).join('');
    detail.poses=[s.q0,...detail.qpos];detail.tips=detail.poses.map(q=>kinematics(q)[2]);detail.distances=detail.tips.map(p=>Math.hypot(p[0]-s.goal[0],p[1]-s.goal[1]));
    $('scrubber').max=detail.poses.length-1;$('scrubber').value=0;$('duration').textContent=`${fmt(e.steps*dt())} s`;
    drawFrame();
  }catch(error){if(req===requestId){showReadError($('selected-label'),()=>selectEpisode(e));$('arm-stage').innerHTML='<text x="300" y="220" text-anchor="middle" fill="#9c6c55">读取未完成，请重新读取轨迹</text>';}}
}
function lineChart(id, values, color, current){
  const width=320,height=48,pad=5,min=Math.min(0,...values),max=Math.max(...values,.001),x=i=>pad+i/Math.max(1,values.length-1)*(width-pad*2),y=v=>height-pad-(v-min)/(max-min)*(height-pad*2);
  const path=values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' '), cx=x(Math.min(current,values.length-1));
  $(id).innerHTML=`<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><path d="${path} L${x(values.length-1)},${height} L${pad},${height} Z" fill="${color}" opacity=".07"/><line x1="0" x2="320" y1="${y(0)}" y2="${y(0)}" stroke="#e6ece1"/><path d="${path}" fill="none" stroke="${color}" stroke-width="1.4"/><line x1="${cx}" x2="${cx}" y1="2" y2="46" stroke="#819a78" stroke-dasharray="2 3"/><circle cx="${cx}" cy="${y(values[Math.min(current,values.length-1)])}" r="2.8" fill="${color}"/></svg>`;
}
function drawFrame(){
  if(!detail)return;const s=scenes[selected.scene_id],x=v=>300+v*184,y=v=>219-v*184,q=detail.poses[frame], joints=kinematics(q), tip=detail.tips[frame];let svg='';
  for(let i=-1;i<=1.001;i+=.25){svg+=`<line x1="${x(i)}" x2="${x(i)}" y1="17" y2="421" stroke="#e7eee4" stroke-width=".7"/><line x1="98" x2="502" y1="${y(i)}" y2="${y(i)}" stroke="#e7eee4" stroke-width=".7"/>`;if(i%0.5===0)svg+=`<text x="${x(i)}" y="436" text-anchor="middle" class="axis-text">${fmt(i,1)}</text><text x="86" y="${y(i)+3}" text-anchor="end" class="axis-text">${fmt(i,1)}</text>`;}
  const poly=points=>points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ');
  svg+=`<circle cx="300" cy="219" r="184" fill="none" stroke="#dce6d7" stroke-dasharray="3 5"/><circle cx="${x(s.obstacle[0])}" cy="${y(s.obstacle[1])}" r="${s.radius*184}" fill="#eac6a9" fill-opacity=".75" stroke="#d7ad8a"/><circle cx="${x(s.goal[0])}" cy="${y(s.goal[1])}" r="${data.config.env.success_distance*184}" fill="#92c4ae" fill-opacity=".22" stroke="#72ad92" stroke-width="1.3"/><path d="M${x(s.goal[0])-4},${y(s.goal[1])}h8 M${x(s.goal[0])},${y(s.goal[1])-4}v8" stroke="#4c9476"/><polyline points="${poly(detail.tips)}" fill="none" stroke="#abc7b0" stroke-dasharray="3 4" stroke-width="1"/><polyline points="${poly(detail.tips.slice(0,frame+1))}" fill="none" stroke="#448b71" stroke-width="1.5" opacity=".65"/><polyline points="${poly(joints)}" fill="none" stroke="#277d66" stroke-width="${data.config.env.link_radius*2*184}" stroke-linecap="round" stroke-linejoin="round"/>`;
  joints.forEach((p,i)=>{svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="${i===0?7:i===1?5:3.6}" fill="${i===2?'#267b62':'#f6faf4'}" stroke="#267b62" stroke-width="2.3"/>`;});
  svg+=`<text x="512" y="223" class="axis-text">x</text><text x="295" y="11" class="axis-text">y</text><text x="494" y="39" text-anchor="end" class="axis-text">${fmt(frame*dt())} s</text>`;
  $('arm-stage').innerHTML=svg;$('time').textContent=`${fmt(frame*dt())} s`;$('frame').textContent=`FRAME ${frame} / ${detail.poses.length-1}`;$('scrubber').value=frame;
  lineChart('distance-chart',detail.distances,'#538e72',frame);lineChart('clearance-chart',detail.clearance,'#b5a077',Math.max(0,frame-1));
}
function pause(){playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','播放轨迹');}
function tick(time){if(playing&&detail){const interval=dt()*1000/Number($('speed').value);if(time-lastTick>=interval){const advance=Math.floor((time-lastTick)/interval);lastTick+=advance*interval;frame=Math.min(detail.poses.length-1,frame+advance);drawFrame();if(frame===detail.poses.length-1)pause();}}requestAnimationFrame(tick);}
function renderConfig(){const groups={experiment:{seed:data.config.seed,device:data.config.device},...Object.fromEntries(Object.entries(data.config).filter(([k,v])=>v&&typeof v==='object'))};$('config-grid').innerHTML=Object.entries(groups).map(([key,values])=>`<article class="card"><h2>${key}</h2><dl>${Object.entries(values).map(([a,b])=>`<div><dt>${escapeHTML(a)}</dt><dd>${escapeHTML(typeof b==='object'?JSON.stringify(b):b)}</dd></div>`).join('')}</dl></article>`).join('');}
$('split-tabs').addEventListener('click',event=>{const b=event.target.closest('button');if(b){split=b.dataset.split;updateFilters();}});
$('difficulty').addEventListener('change',()=>{difficulty=$('difficulty').value;updateFilters();});
function selectScene(event){const c=event.target.closest('[data-scene]');if(c){const e=filteredEpisodes().find(e=>e.scene_id===c.dataset.scene);if(e){selectEpisode(e);$('player').scrollIntoView({behavior:'smooth',block:'center'});}}}
$('scatter').addEventListener('click',selectScene);$('scatter').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectScene(e);}});
$('open-browser').onclick=()=>location.hash='episodes';
$('search').oninput=()=>{renderEpisodeNav();};$('sort-steps').onclick=()=>{sortDirection=sortDirection===1?-1:1;$('sort-steps').textContent=`步数 ${sortDirection===1?'↑':'↓'}`;renderEpisodeNav();};
$('episode-list').onclick=event=>{const b=event.target.closest('[data-episode]');if(b)selectEpisode(data.episodes.find(e=>e.path===b.dataset.episode));};
$('episode-list').onkeydown=event=>{if(!['ArrowDown','ArrowUp'].includes(event.key))return;const buttons=Array.from($('episode-list').querySelectorAll('[data-episode]')), position=buttons.indexOf(event.target), next=buttons[position+(event.key==='ArrowDown'?1:-1)];if(next){event.preventDefault();next.focus({preventScroll:true});next.scrollIntoView({block:'nearest'});next.click();}};
$('prev-episode').onclick=()=>stepEpisode(-1);$('next-episode').onclick=()=>stepEpisode(1);
$('play').onclick=()=>{if(!detail)return;if(playing)pause();else{if(frame===detail.poses.length-1)frame=0;playing=true;lastTick=performance.now();$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暂停轨迹');}};
$('restart').onclick=()=>{pause();frame=0;drawFrame();};$('scrubber').oninput=()=>{pause();frame=Number($('scrubber').value);drawFrame();};$('speed').onchange=()=>lastTick=performance.now();
$('download-episode').onclick=()=>{if(detail){const {poses,tips,distances,...raw}=detail;download({...raw,scene:scenes[selected.scene_id],control_dt:dt()},`${selected.episode_id}.json`);}};
$('export').onclick=()=>{const rows=filteredEpisodes();const fields=['episode_id','scene_id','split','source','steps','total_return','success','failure_reason'];const csv='\uFEFF'+[fields.join(','),...rows.map(e=>fields.map(k=>'"'+String(e[k]).replace(/"/g,'""')+'"').join(','))].join('\r\n');download(csv,`seed42-${split}-${difficulty}.csv`,'text/csv;charset=utf-8');};
window.addEventListener('hashchange',navigate);
document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();}});
let refreshing=false;
async function refreshData(){
  if(refreshing)return;refreshing=true;$('refresh-data').disabled=true;$('refresh-data').textContent='正在读取…';
  const previous=selected?.path;
  try{
    const response=await fetch('/api/data',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const next=await response.json();data=next;
    Object.keys(data.scenes).forEach((key,i)=>{names[key]??=key;colors[key]??=['#367d64','#90b29a','#91a5b8','#c3ab83','#af9db6'][i%5];});
    scenes=Object.fromEntries(Object.values(data.scenes).flat().map(s=>[s.scene_id,s]));
    if(split!=='all'&&!(split in data.scenes))split='all';
    requestId++;pause();detail=null;selected=null;
    $('nav-count').textContent=data.episodes.length;$('status').textContent='';
    $('data-summary').textContent=`runs/${data.run} · ${Object.keys(data.scenes).length} 个划分 · ${data.episodes.length} 条轨迹 · 更新于 ${new Date(data.refreshed_at).toLocaleTimeString('zh-CN',{hour12:false})}`;
    renderConfig();navigate();
    const rows=filteredEpisodes();selected=rows.find(e=>e.path===previous)||null;updateFilters();
    if(previous&&selected?.path===previous&&!detail)await selectEpisode(selected);
  }catch(error){showReadError($('status'),refreshData);}
  finally{refreshing=false;$('refresh-data').disabled=false;$('refresh-data').textContent='↻ 刷新目录数据';}
}
$('refresh-data').onclick=refreshData;
refreshData();requestAnimationFrame(tick);
