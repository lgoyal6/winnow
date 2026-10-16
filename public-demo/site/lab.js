const $=id=>document.getElementById(id),v=id=>$(id).value,n=id=>Number(v(id));
let revision=0,pending;
function clear(message){$('lab-status').textContent=message;$('lab-output').textContent='';}
$('lab-form').addEventListener('input',()=>{revision++;pending?.abort();clear('Inputs changed. Run the checks again.');$('lab-run').disabled=false;});
$('lab-form').addEventListener('submit',async event=>{
 event.preventDefault();const current=++revision;pending?.abort();pending=new AbortController();const controller=pending;
 const timeout=setTimeout(()=>controller.abort(),15000);clear('Running...');$('lab-run').disabled=true;
 try {const response=await fetch('/api/public_demo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(({hardware:v('hardware'),batch:n('batch'),kv_heads:n('kv_heads'),cache_len:n('cache_len'),head_dim:n('head_dim'),bit_width:n('bit_width'),max_abs_error:n('max_abs_error'),backends:v('backends')})),signal:controller.signal});
 const result=await response.json();if(current!==revision)return;if(!response.ok)throw new Error(result.error||'No result is available.');
 $('lab-status').textContent='Checks completed for these inputs';$('lab-output').textContent=JSON.stringify(result,null,2);
 }catch(error){if(current===revision)clear(error.name==='AbortError'?'Request timed out. No result is available.':error.message);}
 finally{clearTimeout(timeout);if(current===revision)$('lab-run').disabled=false;}
});
