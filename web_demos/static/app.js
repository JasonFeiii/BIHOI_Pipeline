'use strict';
const $ = (id) => document.getElementById(id);
const states = {wapr: {files:{}, values:{}, job:null}};
let app = 'wapr', config, image, imageURL, drawing, uploadAbort;
const uploadCache = new WeakMap();
const definitions = {
  wapr: {name:'WAPR', tag:'POSE ESTIMATION', eyebrow:'RGB-D → 6D POSE', title:'让物体的位姿清晰可见',
    subtitle:'输入 RGB-D 与物体网格，估计相机坐标系中的旋转和平移，并在原图中查看位姿。',
    repo:'https://github.com/WangYuLin-SEU/WAPR', action:'开始位姿估计', preview:'RGB 图像',
    empty:'从一张 RGB 图像开始', emptyText:'上传后预览图像，并选择目标物体区域', output:'4 × 4 位姿矩阵 · 叠加图 · PLY',
    guide:'深度、网格和标定需要使用一致的尺度',
    guideText:'深度与网格会按所填系数转换为米。相机内参应对应原始 RGB 分辨率；Mask 与深度须与 RGB 同尺寸。直径填写真实物体直径（米），结果分数是排序信号。'}
};
function uploadField(key,title,types,accept,wide=false) {
  return `<label class="upload-card ${wide?'wide':''}" data-upload="${key}"><input type="file" data-file="${key}" accept="${accept}" aria-label="${title}"><span class="upload-symbol">+</span><strong>${title}</strong><small>${types}</small></label>`;
}
function numberField(name,label,value,min,max,step='any',placeholder='') {
  return `<div class="field"><label for="${name}">${label}</label><input id="${name}" name="${name}" type="number" value="${value}" min="${min}" max="${max}" step="${step}" placeholder="${placeholder}" required></div>`;
}
function gpuField(){return '<div class="field"><label for="device">推理设备</label><input id="device" name="device" value="cuda:0" pattern="cuda(:[0-9]+)?" required></div>';}
function waprFields(){return `
  <div class="fields"><div class="field-label">观测与物体模型<span>拖入文件或点击上传</span></div><div class="upload-grid">
    ${uploadField('rgb','RGB 图像','PNG / JPG','.png,.jpg,.jpeg')}
    ${uploadField('depth','深度图','PNG / NPY / TIFF','.png,.npy,.tif,.tiff')}
    ${uploadField('mesh','物体网格','PLY / OBJ / STL / GLB','.ply,.obj,.stl,.glb')}
    ${uploadField('mask','前景 Mask','可选 · 非零值为前景','.png,.jpg,.jpeg')}
  </div></div><div class="form-divider"></div>
  <div class="fields"><h3 class="section-title">目标物体区域</h3><div class="segmented"><label><input type="radio" name="region" value="box" checked>拖动框选</label><label><input type="radio" name="region" value="mask">使用 Mask</label></div>
    <div id="bbox-fields"><div class="field-row">${numberField('x','X / 左上角',0,0,100000,1)}${numberField('y','Y / 左上角',0,0,100000,1)}</div><div class="field-row">${numberField('w','宽度 / px','',1,100000,1)}${numberField('h','高度 / px','',1,100000,1)}</div><p class="hint">在右侧原图上拖动框选，也可直接输入像素坐标。</p></div>
    <p id="mask-help" class="hint" hidden>上传黑底白前景 Mask；将使用所有非零像素。</p>
  </div><div class="form-divider"></div>
  <div class="fields"><h3 class="section-title">相机内参 <span class="hint">/ 原图像素单位</span></h3><div class="field-row">${numberField('fx','fx','',.000001,1e7,'any','焦距 x')}${numberField('fy','fy','',.000001,1e7,'any','焦距 y')}</div><div class="field-row">${numberField('cx','cx','',-1e7,1e7,'any','主点 x')}${numberField('cy','cy','',-1e7,1e7,'any','主点 y')}</div></div>
  <details class="advanced" open><summary>尺度与推理设置</summary><div class="field-row">${numberField('depth_scale','深度 → 米',.001,1e-10,1e6)}${numberField('mesh_scale','网格 → 米',.001,1e-10,1e6)}</div><p class="hint">毫米数据填 0.001；已经是米的数据填 1。</p><div class="field-row">${numberField('diameter_m','物体直径 / m','',1e-10,1e6,'any','例如 0.12')}${gpuField()}</div><div class="field-row"><div class="field"><label for="backend">推理后端</label><select id="backend" name="backend"><option value="torch">PyTorch</option><option value="trt">TensorRT</option></select></div></div><p class="hint">首次运行会下载缺失的官方权重。TensorRT 需已准备匹配的运行环境。</p></details>`;}
function values(){return Object.fromEntries([...$('form').querySelectorAll('input[name],select[name]')].filter(el=>el.type!=='radio'||el.checked).map(el=>[el.name,el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value]));}
function message(text,success=false){$('notice').textContent=text;$('notice').className=success?'notice success':'notice';$('notice').hidden=!text;}
function switchView(view){for(const name of ['input','result','log'])$(name+'-view').hidden=name!==view;document.querySelectorAll('[data-view]').forEach(el=>{const selected=el.dataset.view===view;el.classList.toggle('active',selected);el.setAttribute('aria-selected',String(selected));});}
function regionChanged(){const box=$('form').elements.region?.value==='box';$('bbox-fields').hidden=!box;$('mask-help').hidden=box;for(const key of ['x','y','w','h'])$(key).disabled=!box;$('rgb-canvas').classList.toggle('box-mode',box);draw();}
function render(next){
  if(uploadAbort)return;
  if($('input-fields').children.length)states[app].values=values();
  app=next;const def=definitions[app];document.title=`${def.name} · Pose Lab`;
  for(const [id,value] of Object.entries({breadcrumb:def.name,eyebrow:def.eyebrow,subtitle:def.subtitle,'preview-label':def.preview,'empty-title':def.empty,'empty-text':def.emptyText,'output-type':def.output,'guide-title':def.guide,'guide-text':def.guideText}))$(id).textContent=value;
  $('title').replaceChildren(document.createTextNode(def.title),Object.assign(document.createElement('span'),{textContent:'。'}));
  $('model-tag').replaceChildren(document.createTextNode(def.name),Object.assign(document.createElement('span'),{textContent:def.tag}));
  $('repo-link').href=def.repo;
  document.querySelectorAll('[data-app]').forEach(el=>{el.classList.toggle('active',el.dataset.app===app);});
  $('input-fields').innerHTML=waprFields();
  for(const [name,value] of Object.entries(states[app].values))for(const el of $('form').querySelectorAll(`[name="${name}"]`)){if(el.type==='checkbox')el.checked=value;else if(el.type==='radio')el.checked=el.value===value;else el.value=value;}
  document.querySelectorAll('[data-file]').forEach(el=>el.addEventListener('change',()=>chooseFile(el.dataset.file,el.files[0])));
  document.querySelectorAll('[data-upload]').forEach(el=>{el.addEventListener('dragover',e=>{e.preventDefault();el.classList.add('drag');});el.addEventListener('dragleave',()=>el.classList.remove('drag'));el.addEventListener('drop',e=>{e.preventDefault();el.classList.remove('drag');chooseFile(el.dataset.upload,e.dataTransfer.files[0]);});});
  document.querySelectorAll('[name="region"]').forEach(el=>el.addEventListener('change',regionChanged));for(const key of ['x','y','w','h'])$(key).addEventListener('input',draw);regionChanged();
  $('environment').textContent=`服务已连接，模型可用性将在运行时检查。\n\nPython: ${config.python[app]}\n源码: ${config.repos[app]}\n\n通过启动参数指定对应模型的 Python 环境。`;
  $('run-note').textContent='首次加载模型和渲染组件可能需要几分钟。';
  message('');image=null;$('rgb-canvas').hidden=true;$('input-media').querySelector('.empty').hidden=false;$('dimensions').textContent='尚未载入';
  for(const [key,file] of Object.entries(states[app].files))updateUpload(key,file);
  refreshPreview();switchView('input');renderJob(states[app].job);
}
function updateUpload(key,file){const el=document.querySelector(`[data-upload="${key}"]`);if(!el)return;el.classList.add('loaded');el.querySelector('strong').textContent=file.name;el.querySelector('small').textContent=`${(file.size/1024/1024).toFixed(2)} MB · 点击更换`;el.querySelector('.upload-symbol').textContent='✓';}
function chooseFile(key,file){if(!file)return;if(file.size>config.max_upload){message('单个文件不能超过 2 GiB。');return;}states[app].files[key]=file;updateUpload(key,file);message('');if(key==='rgb')refreshPreview();}
function refreshPreview(){
  if(imageURL){URL.revokeObjectURL(imageURL);imageURL=null;}
  const file=states[app].files.rgb;if(!file)return;
  image=new Image();imageURL=URL.createObjectURL(file);const pending=image;
  image.onload=()=>{if(image!==pending)return;$('input-media').querySelector('.empty').hidden=true;const canvas=$('rgb-canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.hidden=false;$('dimensions').textContent=`${canvas.width} × ${canvas.height} px`;$('preview-help').textContent='选择「拖动框选」，在图像上拖出目标框。坐标始终对应原始图像。';draw();};
  image.onerror=()=>message('无法在浏览器读取这张 RGB 图像。');image.src=imageURL;
}
function draw(){if(!image?.complete||!image.naturalWidth)return;const canvas=$('rgb-canvas'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);if($('form').elements.region.value!=='box')return;const [x,y,w,h]=['x','y','w','h'].map(key=>Number($(key).value));if(w>0&&h>0){ctx.lineWidth=Math.max(2,canvas.width/350);ctx.fillStyle='#d6f38924';ctx.strokeStyle='#d6f389';ctx.fillRect(x,y,w,h);ctx.strokeRect(x,y,w,h);}}
function point(event){const r=$('rgb-canvas').getBoundingClientRect(),c=$('rgb-canvas');return [Math.max(0,Math.min(c.width,Math.round((event.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.round((event.clientY-r.top)*c.height/r.height)))];}
$('rgb-canvas').addEventListener('pointerdown',e=>{if($('form').elements.region.value!=='box')return;drawing=point(e);e.currentTarget.setPointerCapture(e.pointerId);});
$('rgb-canvas').addEventListener('pointermove',e=>{if(!drawing)return;const p=point(e);const vals=[Math.min(drawing[0],p[0]),Math.min(drawing[1],p[1]),Math.abs(p[0]-drawing[0]),Math.abs(p[1]-drawing[1])];['x','y','w','h'].forEach((key,i)=>$(key).value=vals[i]);draw();});
for(const event of ['pointerup','pointercancel'])$('rgb-canvas').addEventListener(event,()=>drawing=null);
async function api(path, options={}){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw Error(data.error||`请求失败 (${response.status})`);return data;}
function renderJob(job){
  const running=job&&['queued','running'].includes(job.status);$('run').disabled=!!running||!!uploadAbort;$('reset').disabled=!!running||!!uploadAbort;$('example').disabled=!!running||!!uploadAbort;$('cancel').hidden=!running&&!uploadAbort;
  $('run').innerHTML='<span>▶</span> '+definitions[app].action;
  $('status').className='status '+(job?.status||'');$('status').textContent=({queued:'等待执行',running:'推理进行中',succeeded:'推理完成',failed:'推理失败',cancelled:'已取消'})[job?.status]||'等待输入';
  $('log').textContent=job?.log||'暂无任务。开始推理后，日志会显示在这里。';
  $('result-empty').hidden=job?.status==='succeeded';$('result-content').hidden=job?.status!=='succeeded';
  if(job?.status==='succeeded')renderResult(job);
}
async function renderResult(job){
  const resultApp=app;
  try{const latest=await api(`/files/${job.id}/result.json?refresh=${Date.now()}`);if(app!==resultApp||states[app].job?.id!==job.id)return;job.result=latest;}catch(error){if(app!==resultApp)return;}
  const url=name=>`/files/${job.id}/${encodeURIComponent(name)}`;
  const choices=[['overlay.png','位姿叠加']];
  $('result-tabs').replaceChildren();const show=(name,button)=>{document.querySelectorAll('#result-tabs button').forEach(b=>b.classList.toggle('active',b===button));const media=document.createElement('img');media.src=url(name);media.alt='模型输出可视化';media.className='result-image';$('result-media').replaceChildren(media);};
  for(const [name,label] of choices.filter(([name])=>job.artifacts.includes(name))){const button=document.createElement('button');button.type='button';button.textContent=label;button.onclick=()=>show(name,button);$('result-tabs').append(button);}
  $('result-tabs').firstChild?.click();$('result-json').textContent=JSON.stringify(job.result,null,2);
  const r=job.result||{};const entries=[['排序分数',Number(r.score_6d).toFixed(4)],['耗时（含初始化）',`${r.elapsed_seconds_including_setup} s`],['坐标单位','米']];
  $('metrics').replaceChildren(...entries.map(([label,value])=>{const el=document.createElement('div');el.textContent=label;el.append(Object.assign(document.createElement('strong'),{textContent:String(value)}));return el;}));
  $('downloads').replaceChildren(...job.artifacts.filter(n=>n!=='run.log').map(name=>{const a=document.createElement('a');a.href=url(name);a.download=name;a.textContent=name==='results.zip'?'↓ 下载全部结果':`↓ ${name}`;if(name==='results.zip')a.title='首次下载时打包，后续下载使用缓存';return a;}));
}
$('form').addEventListener('submit',async e=>{
  e.preventDefault();const jobApp=app, required=['rgb','depth','mesh',...(values().region==='mask'?['mask']:[])];
  if(required.some(k=>!states[app].files[k])){message('请上传 RGB、深度、网格，以及所选模式需要的 Mask。');return;}
  const body={app,...values()};
  message('');const abort=new AbortController();uploadAbort=abort;renderJob(states[app].job);
  try{
    for(const key of required){const file=states[jobApp].files[key];$('status').textContent=`上传 ${file.name}…`;let token=uploadCache.get(file);if(!token){const result=await api('/api/upload',{method:'POST',headers:{'X-Filename':encodeURIComponent(file.name),'Content-Type':'application/octet-stream'},body:file,signal:abort.signal});token=result.token;uploadCache.set(file,token);}body[key]=token;}
    let job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(abort.signal.aborted)job=await api(`/api/jobs/${job.id}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    states[jobApp].job=job;localStorage.setItem(`pose-lab-${jobApp}`,job.id);if(app===jobApp){renderJob(job);switchView('log');}poll(jobApp,job.id);
  }catch(error){if(error.name!=='AbortError')message(error.message);else message('已取消上传。');}
  finally{uploadAbort=null;if(app===jobApp)renderJob(states[jobApp].job);}
});
async function poll(jobApp,id){try{const job=await api(`/api/jobs/${id}`);if(states[jobApp].job&&states[jobApp].job.id!==id)return;states[jobApp].job=job;if(app===jobApp){renderJob(job);if(job.status==='succeeded')switchView('result');if(job.status==='failed')message('推理失败，请查看运行日志中的具体原因。');}if(['running','queued'].includes(job.status))setTimeout(()=>poll(jobApp,id),1200);}catch(error){if(app===jobApp)message(`无法获取任务状态：${error.message}。刷新页面可重新连接。`);}}
$('cancel').addEventListener('click',async()=>{if(uploadAbort){uploadAbort.abort();return;}const job=states[app].job;if(!job)return;try{states[app].job=await api(`/api/jobs/${job.id}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});renderJob(states[app].job);}catch(error){message(error.message);}});
$('reset').addEventListener('click',()=>{states[app]={files:{},values:{},job:null};localStorage.removeItem(`pose-lab-${app}`);$('input-fields').replaceChildren();render(app);});
$('example').addEventListener('click',async()=>{
  const selectedApp=app;$('example').disabled=true;message('正在载入官方样例…',true);
  try{
    const sample=await api(`/api/example/${selectedApp}`);
    const files=await Promise.all(Object.entries(sample.files).map(async([key,info])=>{const response=await fetch(info.url);if(!response.ok)throw Error('无法读取样例文件');return [key,new File([await response.blob()],info.name)];}));
    states[selectedApp].files=Object.fromEntries(files);states[selectedApp].values={...states[selectedApp].values,...sample.values};
    if(app===selectedApp){$('input-fields').replaceChildren();render(app);message('官方样例已载入，可直接开始推理。',true);}
  }catch(error){if(app===selectedApp)message(error.message);}
  finally{if(app===selectedApp)$('example').disabled=false;}
});
document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>switchView(el.dataset.view)));
document.querySelectorAll('[data-app]').forEach(el=>el.addEventListener('click',e=>{e.preventDefault();if(uploadAbort)return;history.pushState({},'',el.getAttribute('href'));render(el.dataset.app);}));
window.addEventListener('popstate',()=>render('wapr'));
(async()=>{try{config=await api('/api/config');render('wapr');for(const key of ['wapr']){const id=localStorage.getItem(`pose-lab-${key}`);if(id&&/^[a-f0-9]{32}$/.test(id))poll(key,id);}}catch(error){message(`无法连接本地服务：${error.message}`);$('run').disabled=true;}})();
