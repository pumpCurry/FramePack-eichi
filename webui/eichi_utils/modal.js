function setupOrigSize(){
  const modal=document.getElementById('orig_size_modal');
  const imgElem=document.getElementById('orig_size_img');
  const closeBtn=document.getElementById('orig_size_close');
  closeBtn.addEventListener('click',()=>{modal.classList.remove('visible');imgElem.src='';});
  function addButtons(){
    document.querySelectorAll('[data-testid="image"]').forEach(div=>{
      if(div.querySelector('.orig-size-btn')) return;
      const img=div.querySelector('img');
      const toolbar=div.querySelector('div.flex');
      if(!img||!toolbar) return;
      const btn=document.createElement('button');
      btn.innerText='原寸';
      btn.className='orig-size-btn';
      btn.addEventListener('click',()=>{imgElem.src=img.src;modal.classList.add('visible');});
      toolbar.insertBefore(btn, toolbar.firstChild);
    });
  }
  addButtons();
  const obs=new MutationObserver(addButtons);
  obs.observe(document.body,{childList:true,subtree:true});
}
if(document.readyState !== 'loading'){
  setupOrigSize();
} else {
  window.addEventListener('load', setupOrigSize);
}
