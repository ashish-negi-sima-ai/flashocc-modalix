import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

const $=id=>document.getElementById(id);
let config,renderer,scene,camera,controls,mesh,volumeGroup;
let labels=null,rawFrame=null,metadata=null,etag='',paused=false;
let needsGeometry=false,building=false,visibleCount=0,received=0;
const enabled=new Uint8Array(18).fill(1);enabled[17]=0;
const instanceVoxels=new Int32Array(640000);
const selected=new THREE.Vector2(),raycaster=new THREE.Raycaster();
const sourceToWorld=(x,y,z)=>new THREE.Vector3(-y,z,-x);

function sprite(text,color='#bacfe6'){
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=96;
  const context=canvas.getContext('2d');context.font='34px system-ui';context.fillStyle=color;context.fillText(text,14,60);
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  const result=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false,transparent:true}));
  result.scale.set(7.2,1.35,1);return result;
}
function sceneSetup(){
  renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.outputColorSpace=THREE.SRGBColorSpace;
  $('canvas').append(renderer.domElement);scene=new THREE.Scene();scene.background=new THREE.Color('#101925');
  scene.fog=new THREE.Fog('#101925',65,150);
  camera=new THREE.PerspectiveCamera(48,1,.1,300);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.09;
  controls.minDistance=3;controls.maxDistance=140;controls.maxPolarAngle=Math.PI*.9;
  scene.add(new THREE.HemisphereLight(0xddeaff,0x556071,2.3));
  const sun=new THREE.DirectionalLight(0xfff4df,2.6);sun.position.set(-15,40,20);scene.add(sun);
  volumeGroup=new THREE.Group();scene.add(volumeGroup);
  mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(.386,.386,.386),new THREE.MeshLambertMaterial({color:0xffffff}),640000);
  mesh.count=0;mesh.frustumCulled=false;mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.instanceColor=new THREE.InstancedBufferAttribute(new Float32Array(640000*3),3);mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  volumeGroup.add(mesh);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(80,80),new THREE.MeshLambertMaterial({color:0x142130,side:THREE.DoubleSide}));
  floor.rotation.x=-Math.PI/2;floor.position.y=-1.05;volumeGroup.add(floor);
  const grid=new THREE.GridHelper(80,16,0x415b73,0x273a4e);grid.position.y=-1.025;volumeGroup.add(grid);
  const bounds=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(-40,-1,-40),new THREE.Vector3(40,5.4,40)),0x3c536b);volumeGroup.add(bounds);
  const ego=new THREE.Mesh(new THREE.BoxGeometry(1.85,1.4,4.4),new THREE.MeshLambertMaterial({color:0xd6e6ed}));
  ego.position.y=.75;volumeGroup.add(ego);
  const cabin=new THREE.Mesh(new THREE.BoxGeometry(1.55,.6,2.1),new THREE.MeshLambertMaterial({color:0x45657b}));cabin.position.set(0,1.25,.15);volumeGroup.add(cabin);
  for(const x of [-1,1])for(const z of [-1.4,1.4]){
    const wheel=new THREE.Mesh(new THREE.BoxGeometry(.22,.5,.65),new THREE.MeshLambertMaterial({color:0x0a0c11}));wheel.position.set(x,.3,z);volumeGroup.add(wheel);
  }
  const egoLabel=sprite('EGO reference');egoLabel.position.set(0,3.3,1);volumeGroup.add(egoLabel);
  const axes=[[sourceToWorld(1,0,0),0xec737d,'+X forward'],[sourceToWorld(0,1,0),0x79d99b,'+Y left'],[sourceToWorld(0,0,1),0x729fff,'+Z up']];
  for(const [direction,color,name] of axes){
    volumeGroup.add(new THREE.ArrowHelper(direction,new THREE.Vector3(0,.04,0),7,color,.8,.4));
    const nameSprite=sprite(name,'#c5d7eb');nameSprite.position.copy(direction.clone().multiplyScalar(9));nameSprite.position.y+=.5;volumeGroup.add(nameSprite);
  }
  new ResizeObserver(()=>{const box=$('canvas').getBoundingClientRect();renderer.setSize(box.width,box.height);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();}).observe($('canvas'));
  setView('orbit');
  let pointer=null;
  renderer.domElement.addEventListener('pointerdown',event=>{pointer=[event.clientX,event.clientY];});
  renderer.domElement.addEventListener('pointerup',event=>{
    if(!pointer||Math.hypot(event.clientX-pointer[0],event.clientY-pointer[1])>4||!labels)return;
    const box=renderer.domElement.getBoundingClientRect();selected.set((event.clientX-box.left)/box.width*2-1,-(event.clientY-box.top)/box.height*2+1);
    raycaster.setFromCamera(selected,camera);const hit=raycaster.intersectObject(mesh,false)[0];if(!hit)return;
    const index=instanceVoxels[hit.instanceId],x=Math.floor(index/3200),y=Math.floor(index/16)%200,z=index%16;
    $('picked-class').textContent=config.classes[labels[index]].name;
    $('picked-position').textContent=`x ${(-39.8+x*.4).toFixed(1)} m · y ${(-39.8+y*.4).toFixed(1)} m · z ${(-.8+z*.4).toFixed(1)} m`;
  });
}
function setView(name){
  if(!camera)return;
  if(name==='top'){camera.position.set(0,75,.01);controls.target.set(0,0,0);}
  else if(name==='street'){camera.position.set(9,5,15);controls.target.set(0,1,-9);}
  else{camera.position.set(29,23,32);controls.target.set(0,.8,-3);}
  controls.update();
  for(const button of ['orbit','top','street'])$('view-'+button).classList.toggle('active',button===name);
}
function scheduleGeometry(){needsGeometry=true;}
function buildGeometry(){
  if(!labels||building)return;building=true;
  try{
    const min=Number($('z-min').value),max=Number($('z-max').value),radius=Number($('radius').value),stride=Number($('sampling').value);
    const surface=$('surface').checked,eligible=new Uint8Array(640000),r2=radius*radius;
    for(let x=0;x<200;x+=stride){const vx=-39.8+x*.4;
      for(let y=0;y<200;y+=stride){const vy=-39.8+y*.4;if(vx*vx+vy*vy>r2)continue;
        for(let z=min;z<=max;z++){const i=(x*200+y)*16+z;if(enabled[labels[i]])eligible[i]=1;}
      }
    }
    const matrices=mesh.instanceMatrix.array,colors=mesh.instanceColor.array;
    const palette=config.classes.map(item=>new THREE.Color(item.color));let count=0;
    for(let i=0;i<640000;i++){
      if(!eligible[i])continue;
      const x=Math.floor(i/3200),y=Math.floor(i/16)%200,z=i%16;
      if(surface&&x>0&&x<199&&y>0&&y<199&&z>0&&z<15&&eligible[i-3200]&&eligible[i+3200]&&eligible[i-16]&&eligible[i+16]&&eligible[i-1]&&eligible[i+1])continue;
      const offset=count*16;matrices[offset]=matrices[offset+5]=matrices[offset+10]=matrices[offset+15]=1;
      matrices[offset+12]=39.8-y*.4;matrices[offset+13]=-.8+z*.4;matrices[offset+14]=39.8-x*.4;
      const color=palette[labels[i]],c=count*3;colors[c]=color.r;colors[c+1]=color.g;colors[c+2]=color.b;
      instanceVoxels[count++]=i;
    }
    mesh.count=count;mesh.boundingSphere=null;visibleCount=count;
    for(const [attribute,size] of [[mesh.instanceMatrix,16],[mesh.instanceColor,3]]){
      attribute.clearUpdateRanges();if(count)attribute.addUpdateRange(0,count*size);attribute.needsUpdate=true;
    }
    volumeGroup.scale.y=Number($('z-scale').value);
    $('voxel-count').textContent=count.toLocaleString()+' voxels';
    const counts=new Uint32Array(18);for(const c of labels)counts[c]++;
    for(let id=0;id<17;id++)$('class-count-'+id).textContent=counts[id].toLocaleString();
  }finally{building=false;}
}
function setupControls(){
  for(const item of config.classes.filter(c=>c.id!==17)){
    const row=document.createElement('label');row.className='class-row';
    const input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.class=String(item.id);input.setAttribute('aria-label',item.name);
    input.addEventListener('change',()=>{enabled[item.id]=Number(input.checked);clearPreset();scheduleGeometry();});
    const swatch=document.createElement('span');swatch.className='class-color';swatch.style.background=item.color;
    const name=document.createElement('span');name.textContent=item.name;
    const count=document.createElement('span');count.className='class-count';count.id='class-count-'+item.id;count.textContent='—';count.title='Class count in the complete volume';
    row.append(input,swatch,name,count);$('classes').append(row);
  }
  function rangeLabels(){
    $('radius-value').textContent=$('radius').value+' m';
    $('z-min-value').textContent=(-1+Number($('z-min').value)*.4).toFixed(1)+' m';
    $('z-max-value').textContent=(-1+(Number($('z-max').value)+1)*.4).toFixed(1)+' m';
    const scale=Number($('z-scale').value);$('z-scale-value').textContent=scale===1?'1× · physical':scale+'× · exaggerated';
  }
  for(const name of ['radius','z-min','z-max','z-scale','sampling','surface'])$(name).addEventListener('input',()=>{
    if(Number($('z-min').value)>Number($('z-max').value))$(name==='z-min'?'z-max':'z-min').value=$(name).value;
    rangeLabels();scheduleGeometry();
  });rangeLabels();
  const preset=(name)=>{
    enabled.fill(0);
    const ids=name==='objects'?[1,2,3,4,5,6,7,8,9,10]:name==='ground'?[11,12,13,14]:Array.from({length:17},(_,i)=>i);
    for(const id of ids)enabled[id]=1;
    for(const input of document.querySelectorAll('[data-class]'))input.checked=!!enabled[Number(input.dataset.class)];
    clearPreset();$('preset-'+name).classList.add('active');scheduleGeometry();
  };
  for(const name of ['scene','objects','ground'])$('preset-'+name).onclick=()=>preset(name);
  $('all-classes').onclick=()=>preset('scene');$('no-classes').onclick=()=>{enabled.fill(0);for(const input of document.querySelectorAll('[data-class]'))input.checked=false;clearPreset();scheduleGeometry();};
  for(const name of ['orbit','top','street'])$('view-'+name).onclick=()=>setView(name);
  $('view-reset').onclick=()=>setView('orbit');
  $('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause';};
  $('save-image').onclick=()=>{renderer.render(scene,camera);download(renderer.domElement.toDataURL('image/png'),'flashocc-3d.png');};
  $('download').onclick=()=>{if(rawFrame){const url=URL.createObjectURL(new Blob([rawFrame],{type:'application/octet-stream'}));download(url,`flashocc-${metadata.token}.occ`);setTimeout(()=>URL.revokeObjectURL(url),1000);}};
}
function clearPreset(){for(const name of ['scene','objects','ground'])$('preset-'+name).classList.remove('active');}
function download(url,name){const anchor=document.createElement('a');anchor.href=url;anchor.download=name;anchor.click();}
async function poll(){
  if(!paused){
    try{
      const response=await fetch('/api/frame',{headers:etag?{'If-None-Match':etag}:{},cache:'no-store'});
      if(response.status!==304){
        if(!response.ok)throw new Error('Waiting for device snapshots');
        const buffer=await response.arrayBuffer();if(buffer.byteLength!==640064)throw new Error('Unexpected volume size');
        const bytes=new Uint8Array(buffer);if(new TextDecoder().decode(bytes.subarray(0,8))!=='FOCC3D01')throw new Error('Invalid volume header');
        const view=new DataView(buffer),sequence=Number(view.getBigUint64(8,true)),marker=Number(view.getBigUint64(16,true)),timestamp=Number(view.getBigUint64(24,true));
        metadata={sequence,marker,timestamp,token:new TextDecoder().decode(bytes.subarray(32,64))};
        rawFrame=buffer;labels=bytes.subarray(64);etag=response.headers.get('ETag')||'';received++;
        $('waiting').style.display='none';$('frame-info').textContent=`Modalix · frame ${marker} · group ${sequence} · ${metadata.token.slice(0,8)}`;
        scheduleGeometry();
      }
    }catch(error){$('status').textContent=labels?'Waiting for device update':error.message;}
  }
  setTimeout(poll,250);
}
let frameCounter=0,lastStats=performance.now();
function animate(){
  requestAnimationFrame(animate);
  if(needsGeometry){needsGeometry=false;buildGeometry();}
  controls.update();renderer.render(scene,camera);frameCounter++;
  const now=performance.now();if(now-lastStats>1000){
    $('render-info').textContent=`${Math.round(frameCounter*1000/(now-lastStats))} render FPS · ${received} volume updates`;
    frameCounter=0;lastStats=now;
    if(metadata){const age=(Date.now()-metadata.timestamp)/1000;
      $('age').textContent=paused?'Volume paused':age<0?'Device clock ahead':`Volume age ${age.toFixed(1)} s`;
      $('status').textContent=paused?'Paused':age<2?'Live from Modalix':'Device data is stale';
      $('status-dot').style.background=paused?'#bfa05d':age<2?'#85e9c9':'#dd8d73';
    }
  }
}
window.viewerDebug=()=>{
  let pickTarget=null;
  if(mesh&&camera)for(let i=0;i<mesh.count;i+=Math.max(1,Math.floor(mesh.count/100))){
    const offset=i*16,p=new THREE.Vector3(mesh.instanceMatrix.array[offset+12],mesh.instanceMatrix.array[offset+13],mesh.instanceMatrix.array[offset+14]);
    p.applyMatrix4(mesh.matrixWorld).project(camera);
    if(Math.abs(p.x)<.5&&Math.abs(p.y)<.5&&p.z>-1&&p.z<1){pickTarget=[p.x,p.y];break;}
  }
  return {received,visibleCount,metadata,camera:camera?.position.toArray(),pickTarget,
    triangles:renderer?.info.render.triangles,enabled:Array.from(enabled),zScale:volumeGroup?.scale.y,
    webgl:!!renderer,shape:config?.shape};
};
try{
  const response=await fetch('/api/config');if(!response.ok)throw new Error('Cannot load viewer configuration');
  config=await response.json();sceneSetup();setupControls();animate();poll();
}catch(error){$('fatal').hidden=false;$('fatal').textContent='The 3D viewer could not start: '+error.message;$('waiting').style.display='none';console.error(error);}
