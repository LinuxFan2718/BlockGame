const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
let config, state, selected = null, shape = [], anchor = null, token;
let legal = new Map(), busy = false, pollTimer, toastTimer, requestSerial = 0;
const cells = [];
const coord = (i) => `${String.fromCharCode(65 + i % 14)}${1 + Math.floor(i / 14)}`;
const key = (a) => [...a].sort((a, b) => a - b).join(",");
const normalize = (a) => {
  const r = Math.min(...a.map(x => x[0])), c = Math.min(...a.map(x => x[1]));
  return a.map(([y, x]) => [y - r, x - c]).sort((a,b) => a[0]-b[0] || a[1]-b[1]);
};
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").hidden = true, 4500);
}
async function api(path, data) {
  const response = await fetch(`/api/${path}`, data === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Blokus-Token": token},
    body: JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Could not connect to the local game.");
  return result;
}
function pieceSvg(points) {
  const norm = normalize(points), width = 1 + Math.max(...norm.map(x=>x[1]));
  const height = 1 + Math.max(...norm.map(x=>x[0]));
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `-0.08 -0.08 ${width + .16} ${height + .16}`);
  svg.setAttribute("aria-hidden", "true");
  for (const [row, col] of norm) {
    const rect = document.createElementNS(SVG, "rect");
    for (const [name, value] of Object.entries({x:col+.055,y:row+.055,width:.89,height:.89,rx:.08,fill:"currentColor"})) rect.setAttribute(name,value);
    svg.append(rect);
  }
  return svg;
}
function buildBoard() {
  for (let i = 0; i < 14; i++) {
    const col = document.createElement("span"); col.textContent = String.fromCharCode(65+i);
    document.querySelector(".column-labels").append(col);
    const row = document.createElement("span"); row.textContent = i+1;
    document.querySelector(".row-labels").append(row);
  }
  for (let i = 0; i < 196; i++) {
    const button = document.createElement("button");
    button.className = "cell"; button.type = "button"; button.setAttribute("role", "gridcell");
    button.tabIndex = i === 0 ? 0 : -1;
    button.addEventListener("pointerenter", () => { anchor = i; preview(); });
    button.addEventListener("focus", () => { anchor = i; preview(); });
    button.addEventListener("click", () => place(i));
    button.addEventListener("keydown", (e) => {
      const delta = {ArrowLeft:-1, ArrowRight:1, ArrowUp:-14, ArrowDown:14}[e.key];
      if (delta !== undefined) {
        e.preventDefault(); const next = Math.max(0, Math.min(195, i+delta));
        cells[i].tabIndex = -1; cells[next].tabIndex = 0; cells[next].focus();
      }
    });
    cells.push(button); $("board").append(button);
  }
  $("board").addEventListener("pointerleave", () => { anchor = null; preview(); });
}
function buildRack() {
  config.pieces.forEach((points, i) => {
    const button = document.createElement("button"); button.type = "button";
    button.className = "piece-button"; button.dataset.piece = i;
    button.setAttribute("aria-label", `Piece ${i+1}, ${points.length} square${points.length===1?"":"s"}`);
    button.setAttribute("aria-pressed", "false");
    button.append(pieceSvg(points));
    const number = document.createElement("small"); number.textContent = String(i+1).padStart(2,"0");
    button.append(number); button.addEventListener("click", () => selectPiece(i));
    $("piece-rack").append(button);
    $("opponent-rack").append(pieceSvg(points));
  });
}
function selectPiece(i) {
  if (busy || state.status !== "your_turn" || !state.remaining[state.human][i]) return;
  selected = i; shape = normalize(config.pieces[i]); renderRack(); preview();
}
function renderRack() {
  const count = state.remaining[state.human].filter(Boolean).length;
  $("rack-count").textContent = `${count} / 21`;
  for (const button of $("piece-rack").children) {
    const id = Number(button.dataset.piece);
    button.disabled = busy || state.status !== "your_turn" || !state.remaining[state.human][id];
    button.setAttribute("aria-pressed", String(selected === id));
    // The selected tile reflects its current rotation, not only its original shape.
    button.querySelector("svg").replaceWith(pieceSvg(selected === id ? shape : config.pieces[id]));
  }
  const enabled = selected !== null && state.status === "your_turn" && !busy;
  $("rotate").disabled = $("flip").disabled = !enabled;
  $("piece-caption").textContent = selected === null ? "Choose a shape to get started." :
    `Piece ${selected+1} · ${shape.length} squares · rotate or flip to fit.`;
}
function placementAt(i) {
  if (selected === null) return {squares:[], action:null};
  const r = Math.floor(i/14), c = i%14;
  const points = shape.map(([y,x]) => [y+r,x+c]);
  const squares = points.filter(([y,x]) => y>=0 && x>=0 && y<14 && x<14).map(([y,x]) => y*14+x);
  const action = squares.length === shape.length ? legal.get(`${selected}:${key(squares)}`) : undefined;
  return {squares, action: action ?? null};
}
function preview() {
  for (const cell of cells) cell.classList.remove("ghost", "invalid", "hint");
  if (!state || busy || state.status !== "your_turn" || selected === null) return;
  if ($("legal-hints").checked) {
    for (let i = 0; i < 196; i++) if (placementAt(i).action !== null) cells[i].classList.add("hint");
  }
  if (anchor !== null) {
    const {squares, action} = placementAt(anchor);
    for (const i of squares) { cells[i].classList.add("ghost"); if (action === null) cells[i].classList.add("invalid"); }
  }
}
function render(next) {
  state = next;
  $("new-game").disabled = busy;
  $("save-game").disabled = false;
  document.body.dataset.human = state.human;
  if (selected !== null && !state.remaining[state.human][selected]) { selected = null; shape = []; }
  legal = new Map(state.legal_moves.map(move => [`${move.piece}:${key(move.cells)}`, move.id]));
  const last = state.actions.at(-1);
  cells.forEach((cell, i) => {
    cell.dataset.color = state.board[i];
    cell.textContent = state.board[i]===0 && state.starts.includes(i) ? "✳" : "";
    cell.setAttribute("aria-label", `${coord(i)} ${state.board[i]===0 ? "empty" : state.board[i]===1 ? "teal" : "coral"}${state.starts.includes(i) ? " starting square" : ""}`);
    cell.classList.toggle("last-move", Boolean(last?.cells.includes(i)));
  });
  const human = state.human, bot = 1-human;
  $("human-score").textContent = state.scores[human]; $("bot-score").textContent = state.scores[bot];
  $("human-order").textContent = human===0 ? "FIRST" : "SECOND";
  $("bot-order").textContent = human===0 ? "SECOND" : "FIRST";
  $("human-remaining").textContent = `${state.remaining[human].filter(Boolean).length} pieces left`;
  $("bot-remaining").textContent = `${state.remaining[bot].filter(Boolean).length} pieces left`;
  $("opponent-count").textContent = state.remaining[bot].filter(Boolean).length;
  [...$("opponent-rack").children].forEach((svg,i)=>svg.classList.toggle("used",!state.remaining[bot][i]));
  const label = config.models.find(m=>m.id===state.model)?.label || "Neural bot";
  $("model-name").textContent = label;
  const backend = state.device === "mps" ? "Metal GPU" : state.device === "cuda" ? "CUDA GPU" : "CPU";
  $("device-badge").textContent = `${backend} · local`;
  $("runtime-detail").textContent = `Rust rules · ${backend} inference · private & local`;
  $("move-count").textContent = `MOVE ${String(state.actions.length+1).padStart(2,"0")}`;
  const thinking = state.status === "thinking";
  $("turn-dot").classList.toggle("thinking", thinking);
  $("status").textContent = state.status === "finished" ?
    (state.winner === "draw" ? "A draw. Nicely played." : state.winner === "human" ? "You win. Well played!" : "The bot wins. Another round?") :
    state.status === "error" ? "The bot needs attention" : thinking ? `Bot is thinking${state.thinking_seconds>1 ? ` · ${Math.floor(state.thinking_seconds)}s` : "…"}` : "Your turn";
  $("board-help").textContent = state.status === "finished" ? "Final scores include completion bonuses." :
    state.status === "error" ? state.error : thinking ? "Planning a path through the board…" :
    state.starts.length ? "Your first piece must cover a marked starting square." : "Select a piece, then click to place. R rotates · F flips.";
  $("undo").disabled = !state.can_undo || busy;
  const search = state.last_search;
  $("search-detail").textContent = search ? `${backend} · ${search.simulations ? search.simulations.toLocaleString()+" simulations" : "network only"} · ${search.seconds.toFixed(2)}s` :
    `${backend} · ${state.simulations ? state.simulations.toLocaleString()+" simulations per move" : "network only"}`;
  renderRack(); preview();
  clearTimeout(pollTimer);
  if (thinking) pollTimer = setTimeout(refresh, 600);
}
async function refresh() {
  if (busy) { pollTimer = setTimeout(refresh, 600); return; }
  const serial = ++requestSerial;
  try { const next = await api("state"); if (serial === requestSerial) render(next); }
  catch (error) { toast(error.message); pollTimer = setTimeout(refresh, 2500); }
}
async function mutate(path, data) {
  if (busy) return;
  busy = true; render(state);
  const serial = ++requestSerial;
  try {
    const next = await api(path, data); busy = false; if (serial === requestSerial) render(next);
  } catch (error) { busy = false; toast(error.message); await refresh(); }
}
function place(i) {
  if (busy || state?.status !== "your_turn") return;
  if (selected === null) { toast("Choose one of your pieces first."); return; }
  const {action} = placementAt(i);
  if (action === null) { toast("That shape doesn’t fit here. Try a different spot or rotation."); return; }
  mutate("move", {id:state.id,version:state.version,action});
}
function transform(kind) {
  if (selected === null || busy || state.status !== "your_turn") return;
  shape = normalize(shape.map(([y,x])=>kind==="rotate" ? [x,-y] : [y,-x]));
  renderRack(); preview();
}
$("rotate").addEventListener("click",()=>transform("rotate"));
$("flip").addEventListener("click",()=>transform("flip"));
$("legal-hints").addEventListener("change",preview);
$("undo").addEventListener("click",()=>mutate("undo",{id:state.id,version:state.version}));
$("new-game").addEventListener("click",()=>{
  $("seat").value = state.human; $("budget").value = state.simulations;
  $("model-select").value = state.model;
  $("device-select").value = state.device;
  $("settings").showModal();
});
$("close-settings").addEventListener("click",()=>$("settings").close());
$("new-game-form").addEventListener("submit", async (e)=>{
  e.preventDefault(); $("settings").close(); selected=null; shape=[]; anchor=null;
  await mutate("new",{human:Number($("seat").value),simulations:Number($("budget").value),model:$("model-select").value,device:$("device-select").value});
});
document.addEventListener("keydown",(e)=>{
  if ($("settings").open || e.ctrlKey || e.metaKey || e.altKey || ["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName)) return;
  if (["r","f"].includes(e.key.toLowerCase())) { e.preventDefault(); transform(e.key.toLowerCase()==="r" ? "rotate" : "flip"); }
  if (e.key==="Escape") { selected=null; shape=[]; renderRack(); preview(); }
});
$("save-game").addEventListener("click",()=>{
  const game = {format:"blokus-duo-human-v1",date:new Date().toISOString(),model:state.model,checkpoint_sha256:state.checkpoint_sha256,device:state.device,precision:"float32",simulations:state.simulations,human:state.human,scores:state.scores,finished:state.status==="finished",actions:state.actions.map(x=>x.action)};
  const url=URL.createObjectURL(new Blob([JSON.stringify(game,null,2)],{type:"application/json"}));
  const a=document.createElement("a"); a.href=url; a.download=`duo-${new Date().toISOString().slice(0,19).replaceAll(":","-")}.json`; a.click(); URL.revokeObjectURL(url);
});
window.addEventListener("focus",()=>{if(config && !busy) refresh();});
try {
  config=await api("config"); token=config.token;
  for(const model of config.models){const o=new Option(model.label,model.id);$("model-select").append(o);}
  for(const device of config.devices){$("device-select").append(new Option(device==="mps"?"Metal GPU":device.toUpperCase(),device));}
  buildBoard(); buildRack(); await refresh();
} catch(error) { $("status").textContent="Could not load the local game"; toast(error.message); }
