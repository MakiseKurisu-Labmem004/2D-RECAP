'use strict';
function renderLearningFrame(episode, index){
  const el=id=>document.getElementById(id), length=episode.reward.length;
  const valid=key=>Array.isArray(episode[key])&&episode[key].length===length&&episode[key].every(Number.isFinite);
  const available=valid('value_prediction')&&valid('td_advantage');
  const terminal=index===length, number=value=>Number.isFinite(value)?value.toFixed(5):'—';
  el('learning-index').textContent=terminal?`终止帧 t=${index}`:`t=${index} · 动作前状态`;
  if(!available){
    ['value-current','advantage-current','return-current','advantage-label'].forEach(id=>el(id).textContent='—');
    el('value-chart').innerHTML='';el('advantage-chart').innerHTML='';
    el('learning-source').textContent='此轨迹没有对应的 Value / Advantage 标注。基础 Expert / BC 数据没有自己的 Value 模型；可切换到改进实验的轮次查看对应模型的估计。';
    el('learning-alignment').textContent='缺失值不作零值填充，也不将回报冒充 Value。';
    return;
  }
  const threshold=episode.advantage_threshold??0;
  el('value-current').textContent=terminal?'—':number(episode.value_prediction[index]);
  el('advantage-current').textContent=terminal?'—':number(episode.td_advantage[index]);
  el('return-current').textContent=terminal?'—':number(episode.mc_return?.[index]);
  el('advantage-label').textContent=terminal?'无后续动作':episode.td_advantage[index]>threshold?'高于阈值':'不高于阈值';
  el('advantage-current').className=!terminal&&episode.td_advantage[index]>0?'positive':'nonpositive';
  const origin=episode.annotation_origin==='viewer_recomputed_from_saved_checkpoint'?'由已保存模型补算':'轨迹原有标注';
  el('learning-source').textContent=`${origin} · ${episode.value_checkpoint||'Value checkpoint 未记录路径'}${episode.value_checkpoint_sha256?' · SHA256 '+episode.value_checkpoint_sha256.slice(0,12):''} · n=${episode.n_step??'—'} · 归一化尺度 ${episode.return_scale??'—'} · 阈值 ${threshold}`;
  el('learning-alignment').textContent=`索引 t 对应 observation[t] / action[t]，与画面中动作前的姿态同步；T 是终止帧，无 V/Advantage 标注。TD 标签按阈值计算，不含专家或接管帧的强制高条件规则。${episode.within_task_horizon?.[index]===false?' 当前帧超过原任务时限，不在 Value 的时限内监督区间。':''}`;
  function chart(id, series){
    const finite=series.flatMap(s=>s[1]).filter(Number.isFinite),lo=Math.min(0,...finite),hi=Math.max(0,...finite),span=hi-lo||1;
    const w=500,h=160,left=49,right=483,top=13,bottom=133,x=i=>left+i/length*(right-left),y=v=>bottom-(v-lo)/span*(bottom-top);
    let svg='';for(let i=0;i<=3;i++){const value=lo+span*i/3;svg+=`<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" stroke="#e6ece2"/><text x="${left-8}" y="${y(value)+3}" class="axis-text" text-anchor="end">${value.toFixed(3)}</text>`;}
    svg+=`<line x1="${left}" x2="${right}" y1="${y(0)}" y2="${y(0)}" stroke="#aabda9" stroke-dasharray="3 3"/>`;
    series.forEach(([label,values,color])=>{if(!values)return;svg+=`<path d="${values.map((v,i)=>`${i?'L':'M'}${x(i)},${y(v)}`).join(' ')}" stroke="${color}" stroke-width="1.6" fill="none"/>`;if(!terminal)svg+=`<circle cx="${x(index)}" cy="${y(values[index])}" r="3.5" fill="${color}"><title>${label}[${index}] = ${number(values[index])}</title></circle>`;});
    svg+=`<line x1="${x(index)}" x2="${x(index)}" y1="${top}" y2="${bottom}" stroke="#456c56" stroke-dasharray="3 3"/>`;
    [0,Math.floor(length/2),length].forEach(i=>svg+=`<text x="${x(i)}" y="153" class="axis-text" text-anchor="middle">${i}</text>`);
    el(id).innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${series.map(s=>s[0]).join('、')}逐帧曲线">${svg}</svg><div class="legend">${series.map(([label,,color])=>`<span><i style="background:${color}"></i>${label}</span>`).join('')}<span>横轴：动作索引 t</span></div>`;
  }
  chart('value-chart',[['Value',episode.value_prediction,'#367d64'],...(valid('mc_return')?[['MC Return',episode.mc_return,'#b99970']]:[])]);
  chart('advantage-chart',[['TD Advantage',episode.td_advantage,'#758bb3']]);
}
